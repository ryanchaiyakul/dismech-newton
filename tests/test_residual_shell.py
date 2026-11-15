"""Phase 2 tests for 6D BAC ``Shells`` constitutive residual, $SE(3)$ invariance, adjoint FD, and FrameGraph."""

from functools import cache

import newton
import numpy as np
import pytest
import warp as wp
from fd import check_vjp_probe, make_probe_dirs
from rollouts import cloth, finalize_worlds, grid, simulate
from scipy.spatial.transform import Rotation
from utils.common import FrameGraph

from dismech_newton import Residual, RodResidual, ShellResidual, SolverDiSMech, flatten_state
from dismech_newton import solver as dn_solver
from dismech_newton.builder import Rod, add_rod, fix_segment, vec6f, vec12f
from dismech_newton.constants import ADJ_MODULE_OPTS
from dismech_newton.residuals.base import clamp_channel
from dismech_newton.residuals.rff import shell_rff_sum
from dismech_newton.solver import StepAdjoint


def _cantilever_plate(
    cx: int,
    cy: int,
    *,
    lx: float = 0.3,
    ly: float = 0.2,
    clamp_x: float = 0.1,
    tri_ke: float = 2.0e3,
    tri_ka: float = 2.0e3,
    tri_kd: float = 2.0,
    bend_stiffness: float = 0.12,
    damping: float = 0.1,
    density: float = 1.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -5.0),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot=None,
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[newton.ModelBuilder, np.ndarray, np.ndarray, np.ndarray]:
    v, f = grid(cx + 1, cy + 1, lx, ly)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity)
    ids = cloth(
        b,
        v,
        f,
        pos=pos,
        rot=rot,
        vel=vel,
        density=density,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        bend_stiffness=bend_stiffness,
        damping=damping,
    )
    fixed = v[:, 0] <= clamp_x + 1e-6
    for k in np.flatnonzero(fixed):
        b.particle_flags[ids[k]] &= ~newton.ParticleFlags.ACTIVE
    return b, v, f, fixed


@wp.kernel
def _shell_linear_probe_loss_kernel(
    p_out: wp.array[vec12f],
    r_out: wp.array[vec6f],
    w_p: wp.array[vec12f],
    w_r: wp.array[vec6f],
    loss: wp.array[float],
):
    f = wp.tid()
    s = float(0.0)
    dp = p_out[f]
    wp_f = w_p[f]
    for k in range(12):
        s += wp_f[k] * dp[k]
    dr = r_out[f]
    wr_f = w_r[f]
    for k in range(6):
        s += wr_f[k] * dr[k]
    wp.atomic_add(loss, 0, s)


@wp.kernel
def _shell_rollout_loss_kernel(
    q: wp.array[float],
    qd: wp.array[float],
    q_ref: wp.array[float],
    w_q: wp.array[float],
    w_qd: wp.array[float],
    dt: float,
    loss: wp.array[float],
):
    i = wp.tid()
    dq = q[i] - q_ref[i]
    vq = dt * qd[i]
    val = 0.5 * w_q[i] * dq * dq + w_qd[i] * vq + 0.1 * q[i] * vq
    wp.atomic_add(loss, 0, val)


def _rollout_loss(
    model: newton.Model,
    solver: SolverDiSMech,
    s_init: newton.State,
    q_ref: wp.array,
    w_q: wp.array,
    w_qd: wp.array,
    steps: int,
    dt: float,
    *,
    record_tape: bool,
) -> tuple[float, wp.Tape | None, wp.array | None]:
    dev = model.device
    states = [s_init] + [model.state(requires_grad=record_tape) for _ in range(steps)]
    for st in states[1:]:
        flatten_state(st)
    if record_tape:
        for arr in solver.residual.grad_inputs():
            if arr.grad is not None:
                arr.grad.zero_()
        loss_arr = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
        tape = wp.Tape()
        with tape:
            for k in range(steps):
                solver.step(states[k], states[k + 1], None, None, dt)
            wp.launch(
                _shell_rollout_loss_kernel,
                dim=s_init.dismech.q.shape[0],
                inputs=[states[-1].dismech.q, states[-1].dismech.qd, q_ref, w_q, w_qd, dt],
                outputs=[loss_arr],
                device=dev,
            )
        return float(loss_arr.numpy()[0]), tape, loss_arr
    for k in range(steps):
        solver.step(states[k], states[k + 1], None, None, dt)
    qn = states[-1].dismech.q.numpy().astype(np.float64)
    qdn = states[-1].dismech.qd.numpy().astype(np.float64)
    vqn = dt * qdn
    dq = qn - q_ref.numpy().astype(np.float64)
    val = float(
        np.sum(
            0.5 * w_q.numpy().astype(np.float64) * dq * dq
            + w_qd.numpy().astype(np.float64) * vqn
            + 0.1 * qn * vqn
        )
    )
    return val, None, None


@pytest.mark.parametrize("worlds", [1, 3])
def test_shell_zero_weights_bit_identical(device, worlds: int):
    sub, _, _, _ = _cantilever_plate(3, 4, gravity=(0.0, 0.0, -9.81))
    model = finalize_worlds(sub, worlds, device)
    model.gravity.assign((0.0, 0.0, -9.81))

    R = 32
    s_base = SolverDiSMech(model, iterations=12, self_contact=False, residual=None)
    assert s_base.shells.effective is None

    w_shape = (R, 6) if worlds == 1 else (worlds, R, 6)
    s_res = SolverDiSMech(
        model,
        iterations=12,
        self_contact=False,
        residual=ShellResidual(rff_dim=R, w_shell=np.zeros(w_shape, dtype=np.float32), seed=4),
    )
    assert s_res.shells.effective is not None
    assert s_res.residual.shell.w_shell.shape == (1 if worlds == 1 else worlds, R)

    dt = 0.01
    st0_b, _ = simulate(model, s_base, 10, dt)
    st0_r, _ = simulate(model, s_res, 10, dt)

    p_eff, r_eff = s_res.shells.effective(st0_r.dismech, dt)
    q_b, qd_b = st0_b.dismech.q.numpy(), st0_b.dismech.qd.numpy()
    q_r, qd_r = st0_r.dismech.q.numpy(), st0_r.dismech.qd.numpy()

    if wp.get_device(device).is_cpu:
        assert np.array_equal(p_eff.numpy(), s_res.shells.params.numpy())
        assert np.array_equal(r_eff.numpy(), s_res.shells.rest.numpy())
        assert np.array_equal(q_b, q_r)
        assert np.array_equal(qd_b, qd_r)
    else:
        assert np.array_equal(p_eff.numpy(), s_res.shells.params.numpy())
        assert np.array_equal(r_eff.numpy(), s_res.shells.rest.numpy())
        np.testing.assert_allclose(q_b, q_r, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(qd_b, qd_r, atol=1e-5, rtol=0.0)


def test_shell_origin_and_se3_invariance(device, rng):
    # 1) Curved unstrained rest mesh: origin stress == 0, AD & FD origin Jacobian == 0, stays put for 10 steps
    v, f = grid(5, 4, 0.4, 0.3, rng, jitter=0.15)
    v[:, 2] = 0.08 * np.sin(4.0 * v[:, 0]) + 0.05 * v[:, 1] ** 2
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cloth(sub, v, f, density=1.0, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0, bend_stiffness=0.08, damping=0.02)
    model = sub.finalize(device=device)

    R = 32
    w_nz = rng.normal(0.0, 0.2, size=(R, 6)).astype(np.float32)
    solver = SolverDiSMech(
        model,
        iterations=12,
        self_contact=False,
        residual=ShellResidual(rff_dim=R, w_shell=w_nz, seed=12),
    )

    dt = 0.01
    s_orig = model.state(requires_grad=True)
    flatten_state(s_orig)
    s_orig.dismech.q.requires_grad = True
    s_orig.dismech.qd.requires_grad = True
    p_out = wp.empty_like(solver.shells.params, requires_grad=True)
    r_out = wp.empty_like(solver.shells.rest, requires_grad=True)

    solver.shells.effective(s_orig.dismech, dt, p_out, r_out)
    np.testing.assert_allclose(p_out.numpy(), solver.shells.params.numpy(), atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(r_out.numpy(), solver.shells.rest.numpy(), atol=1e-7, rtol=0.0)

    # Exact AD origin Jacobian via linear probes wrt (q, qd)
    w_p = wp.array(rng.normal(size=(solver.shells.count, 12)).astype(np.float32), dtype=vec12f, device=device)
    w_r_np = rng.normal(size=(solver.shells.count, 6))
    w_r = wp.array(w_r_np.astype(np.float32), dtype=vec6f, device=device)
    w_r_phys = wp.array((w_r_np * solver.shells.params.numpy()[:, :6]).astype(np.float32), dtype=vec6f, device=device)
    loss_arr = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        solver.shells.effective(s_orig.dismech, dt, p_out, r_out)
        wp.launch(
            _shell_linear_probe_loss_kernel,
            dim=solver.shells.count,
            inputs=[p_out, r_out, w_p, w_r],
            outputs=[loss_arr],
            device=device,
        )
    tape.backward(loss_arr)
    g_q_orig = s_orig.dismech.q.grad.numpy().copy()
    g_qd_orig = s_orig.dismech.qd.grad.numpy().copy()
    tape.zero()
    assert np.abs(g_q_orig).max() < 1e-6, f"non-zero AD q origin grad: {np.abs(g_q_orig).max()}"
    assert np.abs(g_qd_orig).max() < 1e-6, f"non-zero AD qd origin grad: {np.abs(g_qd_orig).max()}"

    # Central FD physical stress/tangent origin Jacobian wrt (q, qd)
    q0_np = s_orig.dismech.q.numpy().copy()
    qd0_np = s_orig.dismech.qd.numpy().copy()
    dq_dir = rng.normal(size=q0_np.size)
    dq_dir /= np.linalg.norm(dq_dir)
    dqd_dir = rng.normal(size=qd0_np.size)
    dqd_dir /= np.linalg.norm(dqd_dir)

    p_base = solver.shells.params.numpy().astype(np.float64)
    r_base = solver.shells.rest.numpy().astype(np.float64)
    wr_phys_np = w_r_phys.numpy().astype(np.float64)

    def _probe_pr(q_vec: np.ndarray, qd_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st = model.state()
        flatten_state(st)
        st.dismech.q.assign(q_vec.astype(np.float32))
        st.dismech.qd.assign(qd_vec.astype(np.float32))
        solver.shells.effective(st.dismech, dt, p_out, r_out)
        dp = p_out.numpy().astype(np.float64) - p_base
        dr = r_out.numpy().astype(np.float64) - r_base
        return dp, dr

    h = 2.5e-5
    dp_qp, dr_qp = _probe_pr(q0_np + h * dq_dir, qd0_np)
    dp_qm, dr_qm = _probe_pr(q0_np - h * dq_dir, qd0_np)
    dp_vp, dr_vp = _probe_pr(q0_np, qd0_np + h * dqd_dir)
    dp_vm, dr_vm = _probe_pr(q0_np, qd0_np - h * dqd_dir)
    fd_q = float(np.sum(wr_phys_np * (dr_qp - dr_qm)) / (2.0 * h))
    fd_qd = float(np.sum(wr_phys_np * (dr_vp - dr_vm)) / (2.0 * h))
    assert abs(fd_q) < 1e-4, f"non-zero FD q origin Jacobian: {fd_q}"
    assert abs(fd_qd) < 1e-4, f"non-zero FD qd origin Jacobian: {fd_qd}"
    assert max(np.abs(dp_qp).max(), np.abs(dp_qm).max()) < 5e-5
    assert max(np.abs(dp_vp).max(), np.abs(dp_vm).max()) < 5e-5

    # Rest stability over 10 solver steps
    st_final, _ = simulate(model, solver, 10, dt)
    assert np.abs(st_final.particle_q.numpy() - v).max() < 1e-6
    assert np.abs(st_final.particle_qd.numpy()).max() < 1e-6

    # 2) Odd/even parity on (z_a, z_b) -> (-z_a, -z_b) AND physical small-deflection parity
    res = solver.residual.shell
    za_t = wp.empty(solver.shells.count, dtype=vec6f, device=device)
    zb_t = wp.empty(solver.shells.count, dtype=vec6f, device=device)
    za_np = rng.normal(0.0, 0.5, size=(solver.shells.count, 6)).astype(np.float32)
    zb_np = rng.normal(0.0, 0.5, size=(solver.shells.count, 6)).astype(np.float32)
    step_kernel = _get_shell_step_kernel()
    for sign, store in ((1.0, "pos"), (-1.0, "neg")):
        za_t.assign(sign * za_np)
        zb_t.assign(sign * zb_np)
        p_t = wp.empty_like(solver.shells.params)
        r_t = wp.empty_like(solver.shells.rest)
        stride = 1 if wp.get_device(device).is_cpu else 8
        wp.launch(
            step_kernel,
            dim=solver.shells.count * stride,
            inputs=[
                solver.shells.params,
                solver.shells.rest,
                res.d_rest,
                int(res.has_d_rest),
                solver.shells.area,
                res.rff_freq,
                res.w_shell,
                res.elem_world,
                za_t,
                zb_t,
                res.data.scales,
                stride,
                res.clamp_lo,
                res.clamp_hi,
                dt,
            ],
            outputs=[p_t, r_t],
            block_dim=1 if wp.get_device(device).is_cpu else 128,
            device=device,
        )
        if store == "pos":
            p_pos, r_pos = p_t.numpy(), r_t.numpy()
        else:
            p_neg, r_neg = p_t.numpy(), r_t.numpy()

    np.testing.assert_allclose(p_neg, p_pos, atol=1e-6, rtol=1e-6)
    dr_pos = r_pos - solver.shells.rest.numpy()
    dr_neg = r_neg - solver.shells.rest.numpy()
    assert np.abs(dr_pos).max() > 1e-6
    np.testing.assert_allclose(dr_neg, -dr_pos, atol=1e-6, rtol=1e-6)

    # Physical pure-bending parity on a flat plate for infinitesimal transverse deflection + velocity
    sub_f, vf, ff, _ = _cantilever_plate(3, 3, gravity=(0.0, 0.0, 0.0))
    mod_f = sub_f.finalize(device=device)
    sol_f = SolverDiSMech(mod_f, self_contact=False, residual=ShellResidual(rff_dim=R, w_shell=w_nz, seed=12))
    dz_bend = 1.0e-3 * (vf[:, 0] ** 2)
    vz_bend = 2.0e-3 * vf[:, 0]
    res_p, res_r = [], []
    for sgn in (1.0, -1.0):
        st = mod_f.state()
        st.particle_q.assign(vf + np.column_stack([np.zeros((len(vf), 2)), sgn * dz_bend]).astype(np.float32))
        st.particle_qd.assign(np.column_stack([np.zeros((len(vf), 2)), sgn * vz_bend]).astype(np.float32))
        flatten_state(st)
        pe, re = sol_f.shells.effective(st.dismech, dt)
        res_p.append(pe.numpy().copy())
        res_r.append((re.numpy() - sol_f.shells.rest.numpy()).copy())
    np.testing.assert_allclose(res_p[1], res_p[0], atol=1e-5, rtol=5e-3)
    np.testing.assert_allclose(res_r[1], -res_r[0], atol=1e-6, rtol=5e-3)

    # 3) Full SE(3) rotation + translation invariance with nonzero w_shell & shell_spatial_rest_delta
    rot_m = Rotation.random(random_state=rng).as_matrix()
    rot_q = tuple(Rotation.from_matrix(rot_m).as_quat().tolist())
    trans = np.array([1.7, -2.3, 4.1], dtype=np.float64)
    g_a = np.array([0.0, 0.0, -6.0], dtype=np.float64)
    g_b = rot_m @ g_a
    vel_a = np.array([0.1, -0.05, -0.2], dtype=np.float64)
    vel_b = rot_m @ vel_a

    sub_a, va, _, fixed_a = _cantilever_plate(3, 4, bend_stiffness=0.5, gravity=tuple(g_a), vel=tuple(vel_a))
    sub_b, _, _, _ = _cantilever_plate(
        3, 4, bend_stiffness=0.5, gravity=tuple(g_b), pos=tuple(trans), rot=wp.quat(*rot_q), vel=tuple(vel_b)
    )
    mod_a = sub_a.finalize(device=device)
    mod_b = sub_b.finalize(device=device)
    w_se3 = (rng.normal(0.0, 1.0, size=(R, 6)) * np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05], dtype=np.float32)).astype(
        np.float32
    )
    d_rest = (
        rng.normal(0.0, 1.0, size=(mod_a.tri_count, 6))
        * np.array([0.005, 0.005, 0.005, 0.02, 0.02, 0.02], dtype=np.float32)
    ).astype(np.float32)

    spec = ShellResidual(rff_dim=R, w_shell=w_se3, spatial_rest_delta=d_rest, seed=20)
    sol_a = SolverDiSMech(mod_a, iterations=15, self_contact=False, residual=spec)
    sol_b = SolverDiSMech(mod_b, iterations=15, self_contact=False, residual=spec)
    assert np.any(~fixed_a)

    sa, _ = simulate(mod_a, sol_a, 10, dt)
    sb, _ = simulate(mod_b, sol_b, 10, dt)
    qa_rot = sa.particle_q.numpy().astype(np.float64) @ rot_m.T + trans
    np.testing.assert_allclose(sb.particle_q.numpy(), qa_rot, atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(sb.particle_qd.numpy(), sa.particle_qd.numpy() @ rot_m.T, atol=1e-4, rtol=1e-3)


@cache
def _get_shell_step_kernel():
    @wp.kernel(module="unique", module_options=ADJ_MODULE_OPTS)
    def _shell_constit_step_kernel(
        pb: wp.array[vec12f],
        rb: wp.array[vec6f],
        dr: wp.array[vec6f],
        has_dr: int,
        ar: wp.array[float],
        fr: wp.array[vec12f],
        ws: wp.array2d[vec6f],
        ew: wp.array[wp.int32],
        za: wp.array[vec6f],
        zb: wp.array[vec6f],
        sc: vec12f,
        st: int,
        clo: float,
        chi: float,
        dt: float,
        pout: wp.array[vec12f],
        rout: wp.array[vec6f],
    ):
        f, lane = wp.tid() // st, wp.tid() % st
        rff_dim, wi, z_a = fr.shape[0], (ew[f] if ws.shape[0] > 1 else 0), za[f]
        sp, dk, dc = shell_rff_sum(
            z_a, zb[f], fr, ws, wi, rff_dim, lane, st, wp.sqrt(2.0 / float(rff_dim)) if rff_dim > 0 else 0.0
        )
        if lane == 0:
            pb_i = pb[f]
            r0 = rb[f] + dr[f] if has_dr != 0 else rb[f]
            sa = vec6f(sc[0], sc[1], sc[2], sc[3], sc[4], sc[5])
            sb = vec6f(sc[6], sc[7], sc[8], sc[9], sc[10], sc[11])
            er = wp.cw_mul(z_a, sa)
            ls = vec6f(ar[f])
            po, ro = vec12f(), vec6f()
            for j in range(6):
                r = clamp_channel(
                    pb_i[j],
                    pb_i[6 + j],
                    ls[j] * sp[j],
                    ls[j] * dk[j] / sa[j],
                    ls[j] * dc[j] / sb[j],
                    r0[j],
                    er[j],
                    0.0,
                    0.0,
                    0.0,
                    clo,
                    chi,
                    dt,
                )
                po[j] = r[0]
                po[6 + j] = r[1]
                ro[j] = r[2]
            pout[f] = po
            rout[f] = ro

    return _shell_constit_step_kernel


@pytest.mark.slow
def test_shell_continuum_refinement(device, rng):
    R = 64
    w = rng.normal(0.0, 0.08, size=(R, 6)).astype(np.float32)
    w *= np.float32(1.5 / max(float(np.linalg.norm(w)), 1e-8))

    steps, dt = 50, 0.02
    z_base, z_res, deltas = {}, {}, {}
    for cx, cy in ((3, 4), (6, 8), (12, 16)):
        n_tris = 2 * cx * cy
        sub, v, _, _ = _cantilever_plate(cx, cy, gravity=(0.0, 0.0, -5.0), bend_stiffness=0.12, damping=0.1)
        model = sub.finalize(device=device)
        tip = np.abs(v[:, 0] - 0.3) < 1e-6
        s_b = SolverDiSMech(model, iterations=15, self_contact=False, residual=None)
        s_r = SolverDiSMech(
            model,
            iterations=15,
            self_contact=False,
            residual=ShellResidual(rff_dim=R, w_shell=w, seed=1),
        )
        qb, _ = simulate(model, s_b, steps, dt)
        qr, _ = simulate(model, s_r, steps, dt)
        zb = float(qb.particle_q.numpy()[tip, 2].mean())
        zr = float(qr.particle_q.numpy()[tip, 2].mean())
        z_base[n_tris] = zb
        z_res[n_tris] = zr
        deltas[n_tris] = zr - zb

    diff_24_96 = abs(deltas[24] - deltas[96])
    diff_96_384 = abs(deltas[96] - deltas[384])
    tip_diff_24_96 = abs(z_res[24] - z_res[96])
    tip_diff_96_384 = abs(z_res[96] - z_res[384])
    print(
        f"\n[continuum refinement] deltas={deltas} |24-96|={diff_24_96:.6f} |96-384|={diff_96_384:.6f}"
        f" tip |24-96|={tip_diff_24_96:.6f} |96-384|={tip_diff_96_384:.6f}"
    )
    assert abs(deltas[96]) > 1e-4
    assert diff_24_96 < 2.0e-3
    assert diff_96_384 < max(diff_24_96, 1.0e-4)
    assert tip_diff_96_384 < tip_diff_24_96


@pytest.mark.parametrize("worlds", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_shell_step_adjoint_fd(device, rng, needs_cudss, monkeypatch, worlds: int):
    monkeypatch.setattr(dn_solver, "TOLERANCE", 1.0e-7)

    sub, v, _, fixed = _cantilever_plate(
        2,
        2,
        lx=0.4,
        ly=0.3,
        clamp_x=0.2,
        gravity=(0.0, 0.0, -9.81),
        tri_ke=200.0,
        tri_ka=200.0,
        tri_kd=0.3,
        bend_stiffness=0.5,
        damping=0.06,
    )
    model = finalize_worlds(sub, worlds, device)
    model.gravity.assign((0.0, 0.0, -9.81))

    R = 8
    F_tot = model.tri_count
    w_scale = np.array([0.2, 0.2, 0.2, 0.015, 0.015, 0.015], dtype=np.float32)
    d_scale = np.array([0.002, 0.002, 0.002, 0.015, 0.015, 0.015], dtype=np.float32)
    w0 = (rng.normal(size=(R, 6) if worlds == 1 else (worlds, R, 6)) * w_scale).astype(np.float32)
    w0_3d = w0[None] if w0.ndim == 2 else w0
    d0 = (rng.normal(size=(F_tot, 6)) * d_scale).astype(np.float32)

    spec = ShellResidual(rff_dim=R, w_shell=w0.copy(), spatial_rest_delta=d0.copy(), seed=14)
    solver = SolverDiSMech(model, iterations=30, exact=True, self_contact=False, residual=spec)
    assert isinstance(StepAdjoint(solver), StepAdjoint)

    def _base_state(grad: bool = True) -> newton.State:
        s = model.state(requires_grad=grad)
        pq = s.particle_q.numpy().copy()
        dx = np.maximum(pq[:, 0] - 0.2, 0.0)
        pq[:, 2] += 0.15 * dx**2
        s.particle_q.assign(pq)
        pqd = s.particle_qd.numpy().copy()
        pqd[:, 2] = -0.8 * dx
        s.particle_qd.assign(pqd)
        flatten_state(s)
        return s

    s_init = _base_state(grad=True)
    n_dof = s_init.dismech.q.shape[0]
    q_ref = wp.array(s_init.dismech.q.numpy() + 0.01, dtype=float, device=device)
    w_q = wp.array(rng.normal(0.0, 0.2, size=n_dof).astype(np.float32), dtype=float, device=device)
    w_qd = wp.array(rng.normal(0.0, 0.5, size=n_dof).astype(np.float32), dtype=float, device=device)

    steps, dt = 3, 0.008
    L0, tape, loss_arr = _rollout_loss(model, solver, s_init, q_ref, w_q, w_qd, steps, dt, record_tape=True)
    tape.backward(loss_arr)

    g_w = solver.residual.shell.w_shell.grad.numpy().reshape(-1).copy()
    g_d = solver.residual.shell.d_rest.grad.numpy().reshape(-1).copy()
    g_q0 = s_init.dismech.q.grad.numpy().copy()
    g_qd0 = s_init.dismech.qd.grad.numpy().copy()
    tape.zero()

    assert np.linalg.norm(g_w) > 1e-5, f"unexpected near-zero w_shell grad: {np.linalg.norm(g_w)}"
    assert np.linalg.norm(g_d) > 1e-5, f"unexpected near-zero d_rest_shell grad: {np.linalg.norm(g_d)}"

    tag = f"shell adjoint worlds={worlds}"
    print(f"\n[{tag}] L0={L0:.6f}")

    dirs_w = make_probe_dirs(rng, g_w)

    def _eval_w(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            delta = (a[:, None] * dirs_w).sum(axis=0).reshape(w0_3d.shape).astype(np.float32)
            solver.residual.shell.w_shell.assign(w0_3d + delta)
            val, _, _ = _rollout_loss(
                model, solver, _base_state(grad=False), q_ref, w_q, w_qd, steps, dt, record_tape=False
            )
            vals[row, 0] = val
        solver.residual.shell.w_shell.assign(w0_3d)
        return vals

    check_vjp_probe("w_shell", dirs_w, g_w, _eval_w, 1.0e-2, rtol=5.0e-3, tag=tag)

    dirs_d = make_probe_dirs(rng, g_d)

    def _eval_d(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            delta = (a[:, None] * dirs_d).sum(axis=0).reshape(d0.shape).astype(np.float32)
            solver.residual.shell.d_rest.assign(d0 + delta)
            val, _, _ = _rollout_loss(
                model, solver, _base_state(grad=False), q_ref, w_q, w_qd, steps, dt, record_tape=False
            )
            vals[row, 0] = val
        solver.residual.shell.d_rest.assign(d0)
        return vals

    check_vjp_probe("d_rest_shell", dirs_d, g_d, _eval_d, 2.0e-3, rtol=5.0e-3, tag=tag)

    q_p_base = s_init.particle_q.numpy().copy()
    v_p_base = s_init.particle_qd.numpy().copy()
    free_p = np.broadcast_to((~fixed)[None, :, None], (worlds, fixed.size, 3)).copy().astype(np.float64)
    free_mask = free_p.reshape(-1)

    dirs_q = make_probe_dirs(rng, g_q0[: q_p_base.size], mask=free_mask)

    def _eval_q0(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            s_pert = _base_state(grad=False)
            dq = (a[:, None] * dirs_q).sum(axis=0).reshape(q_p_base.shape).astype(np.float32)
            s_pert.particle_q.assign(q_p_base + dq)
            val, _, _ = _rollout_loss(model, solver, s_pert, q_ref, w_q, w_qd, steps, dt, record_tape=False)
            vals[row, 0] = val
        return vals

    check_vjp_probe("q0", dirs_q, g_q0[: q_p_base.size], _eval_q0, 3.0e-4, rtol=5.0e-3, tag=tag)

    dirs_v = make_probe_dirs(rng, g_qd0[: v_p_base.size], mask=free_mask)

    def _eval_v0(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            s_pert = _base_state(grad=False)
            dv = (a[:, None] * dirs_v).sum(axis=0).reshape(v_p_base.shape).astype(np.float32)
            s_pert.particle_qd.assign(v_p_base + dv)
            val, _, _ = _rollout_loss(model, solver, s_pert, q_ref, w_q, w_qd, steps, dt, record_tape=False)
            vals[row, 0] = val
        return vals

    check_vjp_probe("qd0", dirs_v, g_qd0[: v_p_base.size], _eval_v0, 8.0e-3, rtol=5.0e-3, tag=tag)


def test_batched_multiworld_rod_shell_independence(device, rng):
    W, R = 4, 16
    sub = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))

    # Rod at z = 0.4
    n_seg = 6
    pts = np.column_stack([np.linspace(0.0, 0.5, n_seg + 1), np.zeros(n_seg + 1), np.full(n_seg + 1, 0.4)])
    rod = Rod(pts, radius=0.01)
    add_rod(
        sub,
        rod=rod,
        stretch_stiffness=60.0,
        bend_stiffness=0.06,
        twist_stiffness=0.06,
        stretch_damping=0.05,
        bend_damping=1.0e-3,
        twist_damping=1.0e-3,
        collide=False,
        proxies=False,
    )
    fix_segment(sub, edge=0)

    # Disjoint shell at z = 0.0
    v, f = grid(5, 5, 0.3, 0.2)
    ids = cloth(sub, v, f, density=1.0, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0, bend_stiffness=0.08, damping=0.05)
    for k in np.flatnonzero(v[:, 0] <= 0.08 + 1e-6):
        sub.particle_flags[ids[k]] &= ~newton.ParticleFlags.ACTIVE

    model_b = finalize_worlds(sub, W, device)
    model_b.gravity.assign((0.0, 0.0, -9.81))

    w_rod = rng.normal(0.0, 0.06, size=(W, R, 5)).astype(np.float32)
    w_shell = rng.normal(0.0, 0.08, size=(W, R, 6)).astype(np.float32)
    dt, n_steps = 0.01, 8

    spec_b = Residual(
        rod=RodResidual(rff_dim=R, w_rod=w_rod, seed=23),
        shell=ShellResidual(rff_dim=R, w_shell=w_shell, seed=24),
    )
    solver_b = SolverDiSMech(model_b, iterations=15, self_contact=False, residual=spec_b)

    def _init_state(m: newton.Model) -> tuple[newton.State, newton.State]:
        s0, s1 = m.state(), m.state()
        flatten_state(s0)
        flatten_state(s1)
        return s0, s1

    s0, s1 = _init_state(model_b)
    if wp.get_device(device).is_cuda:
        def work():
            for _ in range(n_steps // 2):
                solver_b.step(s0, s1, None, None, dt)
                solver_b.step(s1, s0, None, None, dt)

        fg = FrameGraph(work, solver_b, enabled=True)
        for _ in range(4):
            fg()
            if fg.graph is not None:
                break
        assert fg.graph is not None
        s_fresh0, _ = _init_state(model_b)
        wp.copy(s0.dismech.q, s_fresh0.dismech.q)
        wp.copy(s0.dismech.qd, s_fresh0.dismech.qd)
        solver_b.reset(s0, world_mask=wp.ones(model_b.world_count + 1, dtype=wp.bool, device=device))
        fg()
    else:
        s0, _ = simulate(model_b, solver_b, n_steps, dt, state=s0)

    q_batched = s0.particle_q.numpy().reshape(W, -1, 3)

    # Compare every world against an independent single-world simulation
    model_1 = sub.finalize(device=device)
    model_1.gravity.assign((0.0, 0.0, -9.81))
    ref_qs = []
    for w in range(W):
        spec_1 = Residual(
            rod=RodResidual(rff_dim=R, w_rod=w_rod[w], seed=23),
            shell=ShellResidual(rff_dim=R, w_shell=w_shell[w], seed=24),
        )
        sol_1 = SolverDiSMech(model_1, iterations=15, self_contact=False, residual=spec_1)
        s1_out, _ = simulate(model_1, sol_1, n_steps, dt)
        ref_q = s1_out.particle_q.numpy()
        ref_qs.append(ref_q)
        np.testing.assert_allclose(q_batched[w], ref_q, atol=1e-5, rtol=1e-5)

    # Verify cross-world differentiation (both rod and shell weights produce distinct per-world trajectories)
    assert ids[0] == n_seg + 1
    for w in range(1, W):
        assert np.max(np.abs(q_batched[w, : n_seg + 1] - q_batched[0, : n_seg + 1])) > 1e-4
        assert np.max(np.abs(q_batched[w, n_seg + 1 :] - q_batched[0, n_seg + 1 :])) > 1e-4


def test_shell_residual_framegraph_capture(device, rng):
    if not wp.get_device(device).is_cuda:
        pytest.skip("FrameGraph capture test requires CUDA")

    worlds = 4
    sub, _, _, _ = _cantilever_plate(
        4, 4, lx=0.4, ly=0.4, clamp_x=0.1, tri_kd=0.05, bend_stiffness=0.08, damping=1.0e-3
    )
    model = finalize_worlds(sub, worlds, device)
    assert model.tri_count == worlds * 32

    R = 32
    w_nz = (rng.normal(0.0, 1.0, size=(R, 6)) * np.array([0.25, 0.25, 0.25, 0.002, 0.002, 0.002], dtype=np.float32)).astype(
        np.float32
    )
    dt, substeps = 0.01, 4

    def _capture_and_replay(solver: SolverDiSMech):
        s0, s1 = model.state(), model.state()
        flatten_state(s0)
        flatten_state(s1)

        def work():
            for _ in range(substeps // 2):
                solver.step(s0, s1, None, None, dt)
                solver.step(s1, s0, None, None, dt)

        fg = FrameGraph(work, solver, enabled=True)
        for _ in range(3):
            fg()
        assert fg.graph is not None

        s_init = model.state()
        flatten_state(s_init)
        wp.copy(s0.particle_q, s_init.particle_q)
        wp.copy(s0.particle_qd, s_init.particle_qd)
        for attr in ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q", "face_strain_q"):
            wp.copy(getattr(s0.dismech, attr), getattr(s_init.dismech, attr))
        solver.reset(s0)
        for _ in range(3):
            fg()
        return s0.particle_q.numpy(), s_init

    for sc in (False, True):
        s_base = SolverDiSMech(model, self_contact=sc, residual=None)
        s_res = SolverDiSMech(model, self_contact=sc, residual=ShellResidual(rff_dim=R, w_shell=w_nz, seed=6))
        q_base, _ = _capture_and_replay(s_base)
        q_res, s_init = _capture_and_replay(s_res)
        s_eager, _ = simulate(
            model,
            SolverDiSMech(model, self_contact=sc, residual=ShellResidual(rff_dim=R, w_shell=w_nz, seed=6)),
            3 * substeps,
            dt,
            state=s_init,
        )
        assert np.isfinite(q_base).all() and np.isfinite(q_res).all()
        assert np.max(np.abs(q_res - q_base)) > 1e-4
        np.testing.assert_allclose(q_res, s_eager.particle_q.numpy(), atol=1e-6)
