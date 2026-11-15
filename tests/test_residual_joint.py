"""Tests for robot-owned JointResidual in SolverCoupledProxy."""

import sys
from pathlib import Path

import fd
import newton
import numpy as np
import pytest
import warp as wp
from newton.solvers import SolverSemiImplicit

from dismech_newton import SolverDiSMech
from dismech_newton.coupling import SolverCoupledProxy
from dismech_newton.residual import JointResidual, JointResidualSpec, ResidualSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from utils.common import FrameGraph  # noqa: E402


@wp.kernel
def _vec_dot_kernel(a: wp.array[float], w: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, a[i] * w[i])


@wp.kernel
def _body_loss_kernel(bq: wp.array[wp.transform], bqd: wp.array[wp.spatial_vector], wq: wp.array[float],
                      wd: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    p, q, v = wp.transform_get_translation(bq[i]), wp.transform_get_rotation(bq[i]), bqd[i]
    acc = p[0] * wq[0] + p[1] * wq[1] + p[2] * wq[2] + q[0] * wq[3] + q[1] * wq[4] + q[2] * wq[5] + q[3] * wq[6]
    for k in range(6):
        acc += v[k] * wd[k]
    wp.atomic_add(out, 0, acc)


def _dir_fd(f, h: float) -> tuple[float, float]:
    def c(e):
        return (f(e) - f(-e)) / (2.0 * e)

    coarse, fine = c(h), c(0.5 * h)
    return (4.0 * fine - coarse) / 3.0, abs(coarse - fine)


def _build_coupled_worlds(device, n_worlds: int = 1, two_dof: bool = False, rod_ei: float = 1.0e-2,
                          arm_z: float = 0.2, collide_rod: bool = True, requires_grad: bool = False,
                          sphere: bool = False, arm_x: float = 0.08, n_rod_pts: int = 3):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverDiSMech.register_custom_attributes(b)
    rigid_b, rigid_j, rigid_sh, soft_p, soft_b, soft_sh, arm_b, prox_b = ([] for _ in range(8))
    add_sh = (lambda body, rad, hh: b.add_shape_sphere(body, radius=rad)) if sphere else (
        lambda body, rad, hh: b.add_shape_capsule(body, radius=rad, half_height=hh)
    )
    pts = [[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]] if n_rod_pts == 3 else (
        np.column_stack([np.linspace(-0.1, 0.15, n_rod_pts), np.zeros(n_rod_pts), np.zeros(n_rod_pts)]).tolist()
    )
    for _ in range(n_worlds):
        b.begin_world()
        link0 = b.add_link(mass=0.25, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
        sh0 = add_sh(link0, 0.02, 0.08)
        j0 = b.add_joint_revolute(
            parent=-1, child=link0, parent_xform=wp.transform((0.0, 0.0, arm_z), wp.quat_identity()),
            child_xform=wp.transform((-arm_x, 0.0, 0.0), wp.quat_identity()), axis=(0, 1, 0),
        )
        rigid_b.append(link0)
        rigid_j.append(j0)
        rigid_sh.append(sh0)
        arm_link = link0
        if two_dof:
            link1 = b.add_link(mass=0.2, inertia=wp.mat33(np.diag([0.015, 0.02, 0.025])))
            sh1 = add_sh(link1, 0.018, 0.06)
            j1 = b.add_joint_prismatic(
                parent=link0, child=link1, parent_xform=wp.transform((0.08, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), axis=(1, 0, 0),
            )
            b.add_articulation([j0, j1])
            rigid_b.append(link1)
            rigid_j.append(j1)
            rigid_sh.append(sh1)
            arm_link = link1
        else:
            b.add_articulation([j0])
        ap = b.add_link(is_kinematic=True, mass=0.0, inertia=wp.mat33(np.zeros((3, 3))))
        apsh = add_sh(ap, 0.02, 0.08)
        p0, s0_i = b.particle_count, b.shape_count
        rb = newton.Rod(points=pts, closed=False, radius=0.01)
        r_b, _ = SolverDiSMech.add_rod(
            b, rod=rb, bend_stiffness=rod_ei, stretch_stiffness=1.0e3,
            collide=collide_rod, proxies=collide_rod,
        )
        r_p = list(range(p0, b.particle_count))
        r_s = list(range(s0_i, b.shape_count))
        SolverDiSMech.fix_segment(b, body=r_b[0] if collide_rod else None, edge=0 if not collide_rod else None)
        b.end_world()
        soft_p.extend(r_p)
        soft_b.extend(r_b)
        soft_sh.append(apsh)
        soft_sh.extend(r_s)
        arm_b.append(arm_link)
        prox_b.append(ap)
    m = b.finalize(device=device, requires_grad=requires_grad)
    return m, (rigid_b, rigid_j, rigid_sh, soft_p, soft_b, soft_sh, arm_b, prox_b)


def _make_coupled(m, ids, spec: ResidualSpec | None):
    rigid_b, rigid_j, rigid_sh, soft_p, soft_b, soft_sh, arm_b, prox_b = ids
    entries = [
        SolverCoupledProxy.Entry(
            name="rigid", solver=lambda mv: SolverSemiImplicit(mv), bodies=rigid_b, joints=rigid_j, shapes=rigid_sh,
        ),
        SolverCoupledProxy.Entry(
            name="soft", solver=lambda mv: SolverDiSMech(mv, residual=spec),
            particles=soft_p, bodies=soft_b, shapes=soft_sh,
        ),
    ]
    proxies = [SolverCoupledProxy.Proxy(source="rigid", destination="soft", bodies=arm_b, proxy_bodies=prox_b)]
    return SolverCoupledProxy(m, entries=entries, coupling=SolverCoupledProxy.Config(proxies=proxies, iterations=1))


def _set_joint_dof_values(n_dof: int, world_size: int, idx_values: dict[int, float]) -> np.ndarray:
    arr = np.zeros(n_dof, np.float32)
    for k, v in idx_values.items():
        arr[k] = v
    return arr


def test_zero_weight_bitwise_identity(device):
    m, ids = _build_coupled_worlds(device, n_worlds=1, two_dof=True, arm_z=0.15, collide_rod=True)
    D = m.joint_dof_count
    spec = ResidualSpec(
        rod_rff_dim=0, shell_rff_dim=0,
        joint=JointResidualSpec(rff_dim=16, w_joint=np.zeros((1, D, 16), "f4"), coulomb=0.0, stribeck=0.0),
    )
    c_base, c_res = _make_coupled(m, ids, None), _make_coupled(m, ids, spec)
    s0_b, s1_b = m.state(), m.state()
    s0_r, s1_r = m.state(), m.state()
    q0 = _set_joint_dof_values(D, D, {0: 0.25, 1: -0.02})
    qd0 = _set_joint_dof_values(D, D, {0: 0.6, 1: -0.3})
    u0 = _set_joint_dof_values(D, D, {0: 1.4, 1: -0.8})
    for s in (s0_b, s0_r):
        s.joint_q.assign(q0)
        s.joint_qd.assign(qd0)
        newton.eval_fk(m, s.joint_q, s.joint_qd, s)
    ctrl_b, ctrl_r = m.control(), m.control()
    ctrl_b.joint_f = wp.array(u0, dtype=float, device=device)
    ctrl_r.joint_f = wp.array(u0.copy(), dtype=float, device=device)

    for _ in range(10):
        c_base.step(s0_b, s1_b, ctrl_b, None, 2.0e-3)
        c_res.step(s0_r, s1_r, ctrl_r, None, 2.0e-3)
        newton.eval_ik(m, s1_b, s1_b.joint_q, s1_b.joint_qd)
        newton.eval_ik(m, s1_r, s1_r.joint_q, s1_r.joint_qd)
        s0_b, s1_b = s1_b, s0_b
        s0_r, s1_r = s1_r, s0_r
    wp.synchronize()

    jres = c_res._entries["soft"].solver.joint_residual
    assert np.array_equal(jres.joint_f_eff.numpy(), u0)
    for attr in ("joint_q", "joint_qd", "body_q", "body_qd", "particle_q", "particle_qd"):
        assert np.array_equal(getattr(s0_r, attr).numpy(), getattr(s0_b, attr).numpy()), attr


def test_standstill_equilibrium_and_non_mutation(device, needs_cuda):
    rng = np.random.default_rng(3)
    m, ids = _build_coupled_worlds(device, n_worlds=1, two_dof=True, arm_z=0.25, collide_rod=True)
    D = m.joint_dof_count
    spec = ResidualSpec(
        rod_rff_dim=0, shell_rff_dim=0,
        joint=JointResidualSpec(
            rff_dim=32, coulomb=1.2, stribeck=0.8,
            w_joint=rng.normal(0.0, 2.0, (1, D, 32)).astype("f4"),
        ),
    )
    c = _make_coupled(m, ids, spec)
    soft_solver = c._entries["soft"].solver
    jres = soft_solver.joint_residual

    # Standstill exactness across multiple nonzero poses
    s0, s1, ctrl = m.state(), m.state(), m.control()
    ctrl.joint_f = wp.zeros(D, dtype=float, device=device)
    for q_pair in ((0.0, 0.0), (0.7, -0.05), (-1.1, 0.08)):
        s0.joint_q.assign(_set_joint_dof_values(D, D, {0: q_pair[0], 1: q_pair[1]}))
        s0.joint_qd.zero_()
        out_ctrl = jres.apply(s0, ctrl)
        wp.synchronize()
        assert np.max(np.abs(out_ctrl.joint_f.numpy())) == 0.0

    # Zero drift over 50 substeps under FrameGraph
    mh, idsh = _build_coupled_worlds(device, n_worlds=1, two_dof=False, arm_z=0.25, collide_rod=False)
    ch = _make_coupled(mh, idsh, ResidualSpec(
        rod_rff_dim=0, shell_rff_dim=0,
        joint=JointResidualSpec(
            rff_dim=32, coulomb=1.2, stribeck=0.8, w_joint=rng.normal(0.0, 2.0, (1, 1, 32)).astype("f4")),
    ))
    sh0, sh1, ch_ctrl = mh.state(), mh.state(), mh.control()
    ch_ctrl.joint_f = wp.zeros(1, dtype=float, device=device)
    sh0.joint_q.assign(np.array([0.35], "f4"))
    sh0.joint_qd.zero_()
    newton.eval_fk(mh, sh0.joint_q, sh0.joint_qd, sh0)
    bq_init = sh0.body_q.numpy().copy()
    pq_init = sh0.particle_q.numpy().copy()
    g_hold = FrameGraph(lambda: ch.step(sh0, sh1, ch_ctrl, None, 2.0e-3), ch._entries["soft"].solver)
    for _ in range(50):
        g_hold()
        sh0.body_q.assign(sh1.body_q)
        sh0.body_qd.assign(sh1.body_qd)
        sh0.particle_q.assign(sh1.particle_q)
        sh0.particle_qd.assign(sh1.particle_qd)
    wp.synchronize()
    assert np.max(np.abs(sh0.body_q.numpy() - bq_init)) < 1.0e-8
    assert np.max(np.abs(sh0.body_qd.numpy())) < 1.0e-8
    assert np.max(np.abs(sh0.particle_q.numpy() - pq_init)) < 1.0e-8
    assert np.max(np.abs(sh0.particle_qd.numpy())) < 1.0e-8

    # Caller control.joint_f must never be mutated across FrameGraph capture and replay
    sentinel = _set_joint_dof_values(D, D, {0: 1.25, 1: -0.75})
    u_caller = wp.array(sentinel.copy(), dtype=float, device=device)
    ctrl.joint_f = u_caller
    s0.joint_qd.assign(_set_joint_dof_values(D, D, {0: 0.4, 1: -0.2}))
    g_ctrl = FrameGraph(lambda: c.step(s0, s1, ctrl, None, 2.0e-3), soft_solver)
    for _ in range(5):
        g_ctrl()
    wp.synchronize()
    assert ctrl.joint_f is u_caller
    assert np.array_equal(u_caller.numpy(), sentinel)
    assert not np.array_equal(jres.joint_f_eff.numpy()[:2], sentinel[:2])


def test_ortho_viscous_standstill_slope(device):
    rng = np.random.default_rng(11)
    m, _ = _build_coupled_worlds(device, n_worlds=1, two_dof=True, arm_z=0.25, collide_rod=False)
    D = m.joint_dof_count
    wj = rng.normal(0.0, 1.5, (1, D, 32)).astype("f4")
    s0, ctrl = m.state(), m.control()
    s0.joint_q.assign(np.array([0.8, -0.3], "f4"))
    u_arr = wp.zeros(D, dtype=float, device=device)
    ctrl.joint_f = u_arr

    slopes = {}
    for ortho in (True, False):
        jres = JointResidual(m, JointResidualSpec(
            rff_dim=32, ortho_viscous=ortho, coulomb=0.0, stribeck=0.0, w_joint=wj,
        ), device=device)
        slopes[ortho] = []
        h = 1.0e-4
        for kind in ("qd", "u"):
            for idx in range(D):
                vals = []
                for sign in (1.0, -1.0):
                    s0.joint_qd.zero_()
                    u_arr.zero_()
                    target = s0.joint_qd if kind == "qd" else u_arr
                    vec = np.zeros(D, "f4")
                    vec[idx] = sign * h
                    target.assign(vec)
                    out = jres.apply(s0, ctrl)
                    wp.synchronize()
                    tau_res = out.joint_f.numpy()[idx] - (vec[idx] if kind == "u" else 0.0)
                    vals.append(float(tau_res))
                slopes[ortho].append(abs((vals[0] - vals[1]) / (2.0 * h)))

    assert max(slopes[True]) < 1.0e-6
    assert min(slopes[False]) > 1.0e-2


def test_actuator_object_seam_invariance(device):
    rng = np.random.default_rng(19)
    wj = rng.normal(0.0, 0.8, (1, 13, 16)).astype("f4")
    jfe_by_ei, wrench_by_ei = {}, {}
    for ei in (0.1, 1.0):
        m, ids = _build_coupled_worlds(
            device, n_worlds=1, two_dof=False, rod_ei=ei, arm_z=0.0299, collide_rod=True, sphere=True, arm_x=0.1,
        )
        D = m.joint_dof_count
        spec = ResidualSpec(
            rod_rff_dim=0, shell_rff_dim=0,
            joint=JointResidualSpec(rff_dim=16, coulomb=0.35, stribeck=0.25, w_joint=wj[:, :D, :]),
        )
        c = _make_coupled(m, ids, spec)
        s0, s1, ctrl = m.state(), m.state(), m.control()
        s0.joint_q.assign(_set_joint_dof_values(D, D, {0: 0.0}))
        s0.joint_qd.assign(_set_joint_dof_values(D, D, {0: 1.5}))
        newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
        ctrl.joint_f = wp.array(_set_joint_dof_values(D, D, {0: 2.0}), dtype=float, device=device)
        c.step(s0, s1, ctrl, None, 1.0e-2)
        wp.synchronize()
        soft_solver = c._entries["soft"].solver
        jfe_by_ei[ei] = soft_solver.joint_residual.joint_f_eff.numpy().copy()
        wrench_by_ei[ei] = soft_solver._proxy_arm.last_wrench.numpy()[0].copy()

    assert np.array_equal(jfe_by_ei[0.1], jfe_by_ei[1.0])
    assert np.all(np.isfinite(wrench_by_ei[0.1])) and np.all(np.isfinite(wrench_by_ei[1.0]))
    assert np.linalg.norm(wrench_by_ei[0.1]) > 1.0e-2
    rel_wrench_diff = np.linalg.norm(wrench_by_ei[1.0] - wrench_by_ei[0.1]) / np.linalg.norm(wrench_by_ei[0.1])
    assert rel_wrench_diff > 0.1


def test_tape_vjp_direct_and_coupled(device):
    rng = np.random.default_rng(23)

    # 1. Direct JointResidual.apply VJP wrt w_joint, coulomb, stribeck, joint_q, joint_qd, joint_f
    m_dir, _ = _build_coupled_worlds(device, n_worlds=1, two_dof=True, arm_z=0.2, collide_rod=False, requires_grad=True)
    D_dir = m_dir.joint_dof_count
    jres_dir = JointResidual(m_dir, JointResidualSpec(
        rff_dim=8, q_scale=0.5, qd_scale=0.2, u_scale=2.0, stribeck_v=0.5,
        coulomb=np.array([0.4, 0.3], "f4"), stribeck=np.array([0.3, 0.2], "f4"),
        w_joint=rng.normal(0.0, 0.6, (1, D_dir, 8)).astype("f4"),
    ), device=device)
    s_in, ctrl_dir = m_dir.state(), m_dir.control()
    s_in.joint_q.assign(np.array([0.45, -0.35], "f4"))
    s_in.joint_qd.assign(np.array([0.5, -0.4], "f4"))
    ctrl_dir.joint_f = wp.array(np.array([1.1, -0.7], "f4"), dtype=float, device=device, requires_grad=True)
    w_out = wp.array(rng.normal(0.0, 1.0, D_dir).astype("f4"), dtype=float, device=device)
    loss_dir = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        out = jres_dir.apply(s_in, ctrl_dir)
        loss_dir.zero_()
        wp.launch(_vec_dot_kernel, D_dir, device=device, inputs=[out.joint_f, w_out], outputs=[loss_dir])
    tape.backward(loss_dir)

    for name, arr in [
        ("w_joint", jres_dir.w_joint), ("coulomb", jres_dir.coulomb), ("stribeck", jres_dir.stribeck),
        ("joint_q", s_in.joint_q), ("joint_qd", s_in.joint_qd), ("joint_f", ctrl_dir.joint_f),
    ]:
        g = arr.grad.numpy().copy()
        x0 = arr.numpy().copy()
        d = rng.normal(size=x0.shape).astype("f4")
        d /= np.linalg.norm(d)
        actual = float(np.sum(g * d))

        def eval_dir(h, a=arr, base=x0, direction=d):
            a.assign((base + h * direction).astype("f4"))
            o = jres_dir.apply(s_in, ctrl_dir)
            loss_dir.zero_()
            wp.launch(_vec_dot_kernel, D_dir, device=device, inputs=[o.joint_f, w_out], outputs=[loss_dir])
            return float(loss_dir.numpy()[0])

        fd_v, est = _dir_fd(eval_dir, 5.0e-3)
        arr.assign(x0)
        fd.assert_derivative(actual, fd_v, est, rtol=1.0e-3, name=f"direct_{name}", resolved=2.0e-2)

    # 2. End-to-end coupled rigid+soft step wp.Tape VJP
    m_c, ids_c = _build_coupled_worlds(
        device, n_worlds=1, two_dof=False, arm_z=0.2, collide_rod=False, requires_grad=True)
    D_c = m_c.joint_dof_count
    spec_c = ResidualSpec(
        rod_rff_dim=0, shell_rff_dim=0,
        joint=JointResidualSpec(
            rff_dim=8, qd_scale=0.2, u_scale=2.0, stribeck_v=0.1,
            coulomb=0.35, stribeck=0.25, w_joint=rng.normal(0.0, 0.8, (1, D_c, 8)).astype("f4"),
        ),
    )
    c = _make_coupled(m_c, ids_c, spec_c)
    jres_c = c._entries["soft"].solver.joint_residual
    rigid_s1 = c._entries["rigid"].state_1
    s0, s1, ctrl_c = m_c.state(), m_c.state(), m_c.control()
    s0.joint_q.assign(np.array([0.4], "f4"))
    s0.joint_qd.assign(np.array([0.09], "f4"))
    newton.eval_fk(m_c, s0.joint_q, s0.joint_qd, s0)
    ctrl_c.joint_f = wp.array(np.array([1.5], "f4"), dtype=float, device=device, requires_grad=True)
    wq = wp.array(rng.normal(0.0, 1.0, 7).astype("f4"), dtype=float, device=device)
    wd = wp.array(rng.normal(0.0, 1.0, 6).astype("f4"), dtype=float, device=device)
    loss_c = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    tape_c = wp.Tape()
    with tape_c:
        c.step(s0, s1, ctrl_c, None, 2.0e-2)
        loss_c.zero_()
        wp.launch(
            _body_loss_kernel, rigid_s1.body_q.shape[0], device=device,
            inputs=[rigid_s1.body_q, rigid_s1.body_qd, wq, wd], outputs=[loss_c],
        )
    tape_c.backward(loss_c)

    for name, arr in [
        ("w_joint", jres_c.w_joint), ("coulomb", jres_c.coulomb),
        ("stribeck", jres_c.stribeck), ("joint_f", ctrl_c.joint_f),
    ]:
        g = arr.grad.numpy().copy()
        x0 = arr.numpy().copy()
        d = rng.normal(size=x0.shape).astype("f4")
        d /= np.linalg.norm(d)
        actual = float(np.sum(g * d))

        def eval_coupled(h, a=arr, base=x0, direction=d):
            a.assign((base + h * direction).astype("f4"))
            s1_t = m_c.state()
            c.step(s0, s1_t, ctrl_c, None, 2.0e-2)
            loss_c.zero_()
            wp.launch(
                _body_loss_kernel, rigid_s1.body_q.shape[0], device=device,
                inputs=[rigid_s1.body_q, rigid_s1.body_qd, wq, wd], outputs=[loss_c],
            )
            return float(loss_c.numpy()[0])

        fd_v, est = _dir_fd(eval_coupled, 5.0e-3)
        arr.assign(x0)
        fd.assert_derivative(actual, fd_v, est, rtol=2.0e-3, name=f"coupled_{name}", resolved=2.0e-2)


def test_multiworld_isolation_and_w256_overhead(device, needs_cuda):
    rng = np.random.default_rng(31)

    # Validation checks
    m0, _ = _build_coupled_worlds(device, n_worlds=1, two_dof=False, collide_rod=False)
    with pytest.raises(ValueError):
        JointResidual(m0, JointResidualSpec(tau_cap=0.0), device=device)

    # W=4 per-world weight isolation + enabled_dofs masking
    m4, ids4 = _build_coupled_worlds(device, n_worlds=4, two_dof=True, arm_z=0.2, collide_rod=True)
    D4 = m4.joint_dof_count // 4
    wj4 = np.zeros((4, D4, 16), "f4")
    cl4, st4 = np.zeros((4, D4), "f4"), np.zeros((4, D4), "f4")
    for w in range(1, 4):
        wj4[w] = rng.normal(0.0, 0.5 * w, (D4, 16)).astype("f4")
        cl4[w] = 0.25 * w
        st4[w] = 0.15 * w
    spec4 = ResidualSpec(
        rod_rff_dim=0, shell_rff_dim=0,
        joint=JointResidualSpec(rff_dim=16, w_joint=wj4, coulomb=cl4, stribeck=st4),
    )
    c4_base, c4_res = _make_coupled(m4, ids4, None), _make_coupled(m4, ids4, spec4)
    s0_b, s1_b = m4.state(), m4.state()
    s0_r, s1_r = m4.state(), m4.state()
    q4 = np.tile(_set_joint_dof_values(D4, D4, {0: 0.3, 1: -0.02}), 4)
    qd4 = np.tile(_set_joint_dof_values(D4, D4, {0: 0.5, 1: -0.2}), 4)
    u4 = np.tile(_set_joint_dof_values(D4, D4, {0: 1.2, 1: -0.6}), 4)
    for s in (s0_b, s0_r):
        s.joint_q.assign(q4)
        s.joint_qd.assign(qd4)
        newton.eval_fk(m4, s.joint_q, s.joint_qd, s)
    ctrl4 = m4.control()
    ctrl4.joint_f = wp.array(u4, dtype=float, device=device)
    c4_base.step(s0_b, s1_b, ctrl4, None, 2.0e-3)
    c4_res.step(s0_r, s1_r, ctrl4, None, 2.0e-3)
    wp.synchronize()

    jfe4 = c4_res._entries["soft"].solver.joint_residual.joint_f_eff.numpy()
    assert np.array_equal(jfe4[:D4], u4[:D4])
    assert np.array_equal(s1_r.body_qd.numpy()[:2], s1_b.body_qd.numpy()[:2])
    for w in range(1, 4):
        assert not np.array_equal(jfe4[w * D4:(w + 1) * D4], u4[w * D4:(w + 1) * D4])
    assert not np.allclose(jfe4[D4:2 * D4], jfe4[2 * D4:3 * D4])

    # enabled_dofs=(1,) leaves DOF 0 untouched
    jres_masked = JointResidual(m4, JointResidualSpec(
        rff_dim=16, w_joint=wj4, coulomb=cl4, stribeck=st4, enabled_dofs=(1,),
    ), device=device)
    out_masked = jres_masked.apply(s0_r, ctrl4)
    wp.synchronize()
    jfe_m = out_masked.joint_f.numpy()
    for w in range(4):
        assert jfe_m[w * D4] == u4[w * D4]
    assert jfe_m[1 * D4 + 1] != u4[1 * D4 + 1]

    # W=256 FrameGraph capture paired wp.Event(enable_timing=True) overhead check (<= +2.5%)
    m256, ids256 = _build_coupled_worlds(
        device, n_worlds=256, two_dof=False, arm_z=0.0299, collide_rod=True, sphere=True, arm_x=0.1, n_rod_pts=16,
    )
    D256 = m256.joint_dof_count // 256
    spec256 = ResidualSpec(
        rod_rff_dim=0, shell_rff_dim=0,
        joint=JointResidualSpec(
            rff_dim=32, coulomb=0.2, stribeck=0.1,
            w_joint=rng.normal(0.0, 0.2, (1, D256, 32)).astype("f4"),
        ),
    )
    s_base = _make_coupled(m256, ids256, None)
    s_res = _make_coupled(m256, ids256, spec256)
    s0_0, s1_0, c0 = m256.state(), m256.state(), m256.control()
    s0_1, s1_1, c1 = m256.state(), m256.state(), m256.control()
    newton.eval_fk(m256, s0_0.joint_q, s0_0.joint_qd, s0_0)
    newton.eval_fk(m256, s0_1.joint_q, s0_1.joint_qd, s0_1)
    c0.joint_f = wp.zeros(m256.joint_dof_count, dtype=float, device=device)
    c1.joint_f = wp.zeros(m256.joint_dof_count, dtype=float, device=device)
    g_base = FrameGraph(lambda: s_base.step(s0_0, s1_0, c0, None, 2.0e-3), s_base._entries["soft"].solver)
    g_res = FrameGraph(lambda: s_res.step(s0_1, s1_1, c1, None, 2.0e-3), s_res._entries["soft"].solver)

    for _ in range(100):
        g_base()
        g_res()
    wp.synchronize()

    iters, rounds = 20, 7
    ts_base, ts_res = [], []
    for r in range(rounds):
        order = (0, 1) if r % 2 == 0 else (1, 0)
        res = {}
        for k in order:
            g = g_base if k == 0 else g_res
            e0, e1 = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(e0)
            for _ in range(iters):
                g()
            wp.record_event(e1)
            wp.synchronize()
            res[k] = wp.get_event_elapsed_time(e0, e1) / iters
        ts_base.append(res[0])
        ts_res.append(res[1])

    med_base, med_res = float(np.median(ts_base)), float(np.median(ts_res))
    overhead_pct = (med_res - med_base) / med_base * 100.0
    print(f"\nW=256 FrameGraph: base={med_base:.4f} ms/step, res={med_res:.4f} ms/step, overhead={overhead_pct:+.2f}%")
    assert overhead_pct <= 2.5, f"Overhead {overhead_pct:.2f}% > +2.5%"
