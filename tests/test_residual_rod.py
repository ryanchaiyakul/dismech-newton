import time

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_close, assert_derivative, jacobian
from rollouts import simulate
from utils.common import FrameGraph

import dismech_newton as dn
import dismech_newton.solver as dn_solver
from dismech_newton.builder import vec5f, vec10f
from dismech_newton.residual import _ADJ_OPTS, ResidualSpec
from dismech_newton.rod import Rod, rod_dofs, rod_jacobian, vec11f
from dismech_newton.solver import SolverDiSMech
from dismech_newton.state import flatten_state

vec3f = wp.vec3f


@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _rod_strain_kernel(
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    rods: Rod,
    rest_base: wp.array(dtype=vec5f),
    params_base: wp.array(dtype=vec10f),
    rest: wp.array(dtype=vec5f),
    params: wp.array(dtype=vec10f),
    eps_out: wp.array(dtype=vec5f),
    eps_dot_out: wp.array(dtype=vec5f),
    sigma_out: wp.array(dtype=vec5f),
):
    t = wp.tid()
    idx = rod_dofs(rods, t)
    x = vec11f(
        q[idx[0]],
        q[idx[1]],
        q[idx[2]],
        q[idx[3]],
        q[idx[4]],
        q[idx[5]],
        q[idx[6]],
        q[idx[7]],
        q[idx[8]],
        q[idx[9]],
        q[idx[10]],
    )
    xd = vec11f(
        qd[idx[0]],
        qd[idx[1]],
        qd[idx[2]],
        qd[idx[3]],
        qd[idx[4]],
        qd[idx[5]],
        qd[idx[6]],
        qd[idx[7]],
        qd[idx[8]],
        qd[idx[9]],
        qd[idx[10]],
    )
    eps, J = rod_jacobian(x, rods, t)
    edot = J * xd
    r0 = rest[t]
    p = params[t]
    pb = params_base[t]
    rb = rest_base[t]
    sig = vec5f(0.0)
    for di in range(5):
        dk = p[di] - pb[di]
        sig[di] = p[di] * (rb[di] - r0[di]) + dk * (eps[di] - rb[di])
    eps_out[t] = eps
    eps_dot_out[t] = edot
    sigma_out[t] = sig


@wp.kernel(module="unique", module_options=_ADJ_OPTS)
def _dismech_loss_kernel(
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    q_ref: wp.array(dtype=float),
    w_q: wp.array(dtype=float),
    w_qd: wp.array(dtype=float),
    dt: float,
    loss: wp.array(dtype=float),
):
    i = wp.tid()
    dq = q[i] - q_ref[i]
    vq = dt * qd[i]
    val = 0.5 * dq * dq + w_q[i] * q[i] + w_qd[i] * vq + 0.05 * vq * vq
    wp.atomic_add(loss, 0, val)


def _quat_from_rotvec(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(0.5 * angle)
    c = np.cos(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _build_rod_builder(
    n_seg: int,
    *,
    length: float = 1.0,
    radius: float = 0.01,
    stretch: float = 50.0,
    shear: float = 20.0,
    bend: float = 0.05,
    twist: float = 0.05,
    stretch_damping: float = 0.0,
    bend_damping: float = 0.0,
    twist_damping: float = 0.0,
    dahl_eps_max: float = 0.0,
    dahl_tau: float = 0.0,
    n_fixed: int = 1,
    points: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    bend_stiffness: float | None = None,
    stretch_stiffness: float | None = None,
    twist_stiffness: float | None = None,
) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    dn.register_custom_attributes(b)
    if points is None:
        points = np.stack([np.linspace(0.0, length, n_seg + 1), np.zeros(n_seg + 1), np.zeros(n_seg + 1)], axis=-1)
    rod = newton.Rod(
        points,
        quaternions=quaternions,
        radius=radius,
        stretch_rigidity=stretch,
        shear_rigidity=shear,
        bend_rigidity=bend,
        twist_rigidity=twist,
    )
    dn.add_rod(
        b,
        rod=rod,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=stretch_damping,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
        dahl_eps_max=dahl_eps_max,
        dahl_tau=dahl_tau,
    )
    for edge in range(n_fixed):
        dn.fix_segment(b, edge=edge)
    return b


def _finalize_multiworld(sub_builder: newton.ModelBuilder, worlds: int, device) -> newton.Model:
    if worlds == 1:
        return sub_builder.finalize(device=device)
    main = newton.ModelBuilder()
    dn.register_custom_attributes(main)
    for _ in range(worlds):
        main.add_world(sub_builder)
    return main.finalize(device=device)


def _eval_direct(solver: SolverDiSMech, state: newton.State, dt: float) -> tuple[np.ndarray, ...]:
    flatten_state(state)
    triplets = solver.triplets
    triplets.effective(state.dismech, dt, triplets.params_step, triplets.rest_step)
    eps = wp.empty(triplets.count, dtype=vec5f, device=solver.model.device)
    edot = wp.empty(triplets.count, dtype=vec5f, device=solver.model.device)
    sigma = wp.empty(triplets.count, dtype=vec5f, device=solver.model.device)
    wp.launch(
        _rod_strain_kernel,
        dim=triplets.count,
        inputs=[
            state.dismech.q,
            state.dismech.qd,
            triplets.bind(state.dismech),
            triplets.rest,
            triplets.params,
            triplets.rest_step,
            triplets.params_step,
        ],
        outputs=[eps, edot, sigma],
        device=solver.model.device,
    )
    return (
        eps.numpy(),
        edot.numpy(),
        sigma.numpy(),
        triplets.params_step.numpy(),
        triplets.rest_step.numpy(),
    )


def _make_perturbed_state(model: newton.Model, dx_tip: float, alpha: float, theta: float, qd_scale: float):
    s = model.state()
    pq = s.particle_q.numpy().copy()
    bq = s.body_q.numpy().copy()
    pqd = np.zeros_like(pq)
    bqd = np.zeros_like(bq[:, :6])
    # Straight 2-segment rod along +x: (-L, 0, 0), (0, 0, 0), (+L, 0, 0) -> stretch/bend/twist independently.
    L0 = float(np.linalg.norm(pq[1] - pq[0]))
    l0 = L0 * (1.0 + dx_tip)
    l1 = L0 * (1.0 + 0.5 * dx_tip)
    pq[0] = np.array([-l0, 0.0, 0.0], dtype=np.float32)
    pq[1] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pq[2] = np.array([l1 * np.cos(alpha), l1 * np.sin(alpha), 0.0], dtype=np.float32)

    q_base = bq[0, 3:].copy()
    bq[0, :3] = 0.5 * (pq[0] + pq[1])
    bq[0, 3:] = q_base
    q_bend = _quat_from_rotvec(np.array([0.0, 0.0, 1.0]), alpha)
    q_twist = _quat_from_rotvec(np.array([1.0, 0.0, 0.0]), theta)
    # Wait: rotate tangent + twist cleanly; local z is tangent, world z is bend axis.
    bq[1, :3] = 0.5 * (pq[1] + pq[2])
    bq[1, 3:] = _quat_mul(q_bend, _quat_mul(q_twist, q_base))

    t1 = np.array([np.cos(alpha), np.sin(alpha), 0.0], dtype=np.float32)
    n1 = np.array([-np.sin(alpha), np.cos(alpha), 0.0], dtype=np.float32)
    pqd[0] = np.array([-qd_scale * 0.2, 0.0, 0.0], dtype=np.float32)
    pqd[2] = qd_scale * (0.3 * t1 + 0.5 * n1)
    bqd[1, 3:] = np.array([qd_scale * 0.4, 0.0, qd_scale * 0.3], dtype=np.float32)

    s.particle_q.assign(pq)
    s.body_q.assign(bq)
    s.particle_qd.assign(pqd)
    s.body_qd.assign(bqd)
    flatten_state(s)
    wt = qd_scale * 0.4
    s.dismech.edge_q.assign(np.array([-0.5 * theta, 0.5 * theta], dtype=np.float32))
    s.dismech.edge_qd.assign(np.array([-0.5 * wt, 0.5 * wt], dtype=np.float32))
    return s


@pytest.mark.parametrize("dahl", [False, True])
@pytest.mark.parametrize("worlds", [1, 3])
def test_zero_weights_bit_identical(device, dahl: bool, worlds: int):
    sub = _build_rod_builder(
        10,
        length=1.0,
        bend_damping=1.0e-3,
        stretch_damping=1.0e-2,
        twist_damping=1.0e-3,
        dahl_eps_max=0.02 if dahl else 0.0,
        dahl_tau=0.08 if dahl else 0.0,
    )
    model = _finalize_multiworld(sub, worlds, device)
    model.gravity.assign((0.0, 0.0, -9.81))

    def _init_state():
        s = model.state()
        pqd = s.particle_qd.numpy().copy()
        pqd[:, 1] = 0.35
        pqd[:, 2] = -0.25
        s.particle_qd.assign(pqd)
        return s

    R = 32
    w_shape = (R, 5) if worlds == 1 else (worlds, R, 5)
    spec = ResidualSpec(rod_rff_dim=R, w_rod=np.zeros(w_shape, dtype=np.float32), spatial_rest_delta=None)
    s_base = SolverDiSMech(model, residual=None)
    s_res = SolverDiSMech(model, residual=spec)

    dt = 0.01
    steps = 10
    fin_b, frames_b = simulate(model, s_base, steps, dt, every=1, state=_init_state())
    fin_r, frames_r = simulate(model, s_res, steps, dt, every=1, state=_init_state())

    pb, rb = (
        s_base.triplets.effective(fin_b.dismech, dt)
        if s_base.triplets.effective is not None
        else (s_base.triplets.params, s_base.triplets.rest)
    )
    pr, rr = s_res.triplets.effective(fin_b.dismech, dt)
    assert np.array_equal(pb.numpy(), pr.numpy())
    assert np.array_equal(rb.numpy(), rr.numpy())

    exact = wp.get_device(device).is_cpu
    for fb, fr in zip(frames_b, frames_r):
        assert np.array_equal(fb, fr) if exact else np.allclose(fb, fr, atol=1.0e-6, rtol=1.0e-6)
    for arr_b, arr_r in (
        (fin_b.particle_q.numpy(), fin_r.particle_q.numpy()),
        (fin_b.particle_qd.numpy(), fin_r.particle_qd.numpy()),
        (fin_b.body_q.numpy(), fin_r.body_q.numpy()),
        (fin_b.body_qd.numpy(), fin_r.body_qd.numpy()),
    ):
        assert np.array_equal(arr_b, arr_r) if exact else np.allclose(arr_b, arr_r, atol=1.0e-6, rtol=1.0e-6)


def test_origin_orthogonality_and_symmetries(device, rng):
    pts = np.array([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    b = _build_rod_builder(
        2,
        points=pts,
        stretch=50.0,
        shear=20.0,
        bend=0.05,
        twist=0.05,
        stretch_damping=1.0e-2,
        bend_damping=1.0e-4,
        twist_damping=1.0e-4,
        n_fixed=0,
    )
    model = b.finalize(device=device)
    R = 32
    w_nz = rng.normal(0.0, 0.4, size=(R, 5)).astype(np.float32)
    dt = 0.01

    solver_iso = SolverDiSMech(model, residual=ResidualSpec(rod_rff_dim=R, w_rod=w_nz, iso_bending=True, seed=7))
    s_rest = model.state()
    eps0, edot0, sig0, p0, r0 = _eval_direct(solver_iso, s_rest, dt)
    assert np.allclose(sig0, 0.0, atol=1e-7)
    assert np.allclose(p0, solver_iso.triplets.params.numpy(), atol=1e-7)
    assert np.allclose(r0, solver_iso.triplets.rest.numpy(), atol=1e-7)

    # Zero residual stress Jacobian at origin: exact AD check + small symmetric FD around rest.
    s_ad = model.state(requires_grad=True)
    flatten_state(s_ad)
    triplets = solver_iso.triplets
    p_ad = wp.empty(triplets.count, dtype=vec10f, device=device, requires_grad=True)
    r_ad = wp.empty(triplets.count, dtype=vec5f, device=device, requires_grad=True)
    eps_ad = wp.empty(triplets.count, dtype=vec5f, device=device, requires_grad=True)
    edot_ad = wp.empty(triplets.count, dtype=vec5f, device=device, requires_grad=True)
    sig_ad = wp.empty(triplets.count, dtype=vec5f, device=device, requires_grad=True)
    with wp.Tape() as tape_0:
        triplets.effective(s_ad.dismech, dt, p_ad, r_ad)
        wp.launch(
            _rod_strain_kernel,
            dim=triplets.count,
            inputs=[
                s_ad.dismech.q,
                s_ad.dismech.qd,
                triplets.bind(s_ad.dismech),
                triplets.rest,
                triplets.params,
                r_ad,
                p_ad,
            ],
            outputs=[eps_ad, edot_ad, sig_ad],
            device=device,
        )
    for d in range(5):
        seed_np = np.zeros((triplets.count, 5), dtype=np.float32)
        seed_np[:, d] = 1.0
        tape_0.backward(grads={sig_ad: wp.array(seed_np, dtype=vec5f, device=device)})
        assert np.max(np.abs(s_ad.dismech.q.grad.numpy())) < 1.0e-6
        assert np.max(np.abs(s_ad.dismech.qd.grad.numpy())) < 1.0e-6
        tape_0.zero()

    h = 5.0e-5
    for mode in ("stretch", "bend", "twist", "vel"):
        dx = h if mode == "stretch" else 0.0
        al = h if mode == "bend" else 0.0
        th = h if mode == "twist" else 0.0
        qd = h if mode == "vel" else 0.0
        _, _, sig_p, _, _ = _eval_direct(solver_iso, _make_perturbed_state(model, dx, al, th, qd), dt)
        _, _, sig_m, _, _ = _eval_direct(solver_iso, _make_perturbed_state(model, -dx, -al, -th, -qd), dt)
        d_sig = (sig_p - sig_m) / (2.0 * h)
        assert np.max(np.abs(d_sig)) < 3.0e-4, f"non-zero Jacobian at origin for {mode}: {d_sig}"

    # Verify solver step from rest with gravity=0 stays at rest.
    model.gravity.assign((0.0, 0.0, 0.0))
    s_next = model.state()
    solver_iso.step(model.state(), s_next, None, None, dt)
    assert np.max(np.abs(s_next.particle_q.numpy() - s_rest.particle_q.numpy())) < 1e-7

    # Odd parity of discrete residual stress and even parity of tangents under (-eps_rel, -edot).
    r_base = solver_iso.triplets.rest.numpy()
    for dx_s, al_s, th_s, qd_s in ((0.0, 0.04, 0.05, 0.08), (0.008, 0.0, 0.0, 0.0)):
        sp = _make_perturbed_state(model, dx_s, al_s, th_s, qd_s)
        sn = _make_perturbed_state(model, -dx_s, -al_s, -th_s, -qd_s)
        ep, edp, sigp, pp, _ = _eval_direct(solver_iso, sp, dt)
        em, edm, sigm, pm, _ = _eval_direct(solver_iso, sn, dt)
        assert_close(ep - r_base, -(em - r_base), 1e-5, name="odd_eps")
        assert_close(edp, -edm, 1e-5, name="odd_edot")
        assert np.max(np.abs(sigp)) > 1e-3
        assert_close(sigp, -sigm, 1e-4, name="odd_sigma")
        assert_close(pp, pm, 1e-5, name="even_tangents")

    s_pos = _make_perturbed_state(model, 0.008, 0.4, 0.05, 0.4)
    eps_p, edot_p, sig_p, param_p, _ = _eval_direct(solver_iso, s_pos, dt)

    # SO(2) cross-section bending equivariance & stretch/twist invariance for iso_bending=True.
    phi = 1.1
    R_phi_neg = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]], dtype=np.float32)

    def _roll_cross_section(state: newton.State) -> newton.State:
        sr = model.state()
        sr.particle_q.assign(state.particle_q.numpy())
        sr.particle_qd.assign(state.particle_qd.numpy())
        bq = state.body_q.numpy().copy()
        bqd = state.body_qd.numpy().copy()
        q_roll_loc = _quat_from_rotvec(np.array([1.0, 0.0, 0.0]), phi)
        for i in range(bq.shape[0]):
            bq[i, 3:] = _quat_mul(bq[i, 3:], q_roll_loc)
        sr.body_q.assign(bq)
        sr.body_qd.assign(bqd)
        flatten_state(sr)
        sr.dismech.edge_q.assign(state.dismech.edge_q.numpy() + phi)
        sr.dismech.edge_qd.assign(state.dismech.edge_qd.numpy())
        return sr

    s_roll = _roll_cross_section(s_pos)
    eps_r, edot_r, sig_r, param_r, _ = _eval_direct(solver_iso, s_roll, dt)
    # Check input curvature & curvature-rate really rotated by R(-phi).
    assert_close(eps_r[0, 2:4], R_phi_neg @ eps_p[0, 2:4], 1e-5, name="rotated_kappa")
    assert_close(edot_r[0, 2:4], R_phi_neg @ edot_p[0, 2:4], 1e-5, name="rotated_kappa_dot")
    # Invariance of stretch (0, 1) and twist (4) under bending rotation.
    for d in (0, 1, 4):
        assert_close(sig_r[0, d], sig_p[0, d], 1e-5, name=f"so2_inv_sigma_{d}")
        assert_close(param_r[0, d], param_p[0, d], 1e-5, name=f"so2_inv_k_{d}")
        assert_close(param_r[0, 5 + d], param_p[0, 5 + d], 1e-5, name=f"so2_inv_c_{d}")
    # Equivariance of bending moments (2, 3).
    assert_close(sig_r[0, 2:4], R_phi_neg @ sig_p[0, 2:4], 1e-5, name="so2_eqv_moment")

    # Verify non-isotropic mode activates full coupling when iso_bending=False.
    solver_aniso = SolverDiSMech(model, residual=ResidualSpec(rod_rff_dim=R, w_rod=w_nz, iso_bending=False, seed=7))
    _, _, sig_ap, _, _ = _eval_direct(solver_aniso, s_pos, dt)
    _, _, sig_ar, _, _ = _eval_direct(solver_aniso, s_roll, dt)
    assert np.linalg.norm(sig_ar[0, 2:4] - R_phi_neg @ sig_ap[0, 2:4]) > 1e-4


def test_se3_invariance_and_continuum_convergence(device, rng):
    # 1) SE(3) invariance on a curved, twisted rod with nonzero w_rod and spatial_rest_delta.
    n_seg = 8
    theta = np.linspace(0.0, 0.7, n_seg + 1)
    pts0 = np.stack([np.sin(theta), 0.25 * (1.0 - np.cos(theta)), 0.15 * theta], axis=-1).astype(np.float32)

    q_rand = rng.normal(size=4)
    q_rand /= np.linalg.norm(q_rand)
    R_wp = np.array(wp.quat_to_matrix(wp.quat(*q_rand.tolist())), dtype=np.float32).reshape(3, 3)
    t_shift = np.array([1.25, -0.8, 0.6], dtype=np.float32)

    b0 = _build_rod_builder(n_seg, points=pts0, bend_damping=1e-2, stretch_damping=1e-1, twist_damping=1e-2, n_fixed=1)
    m0 = b0.finalize(device=device)
    s0_in = m0.state()
    bq0 = s0_in.body_q.numpy().copy()
    quats0 = bq0[:, 3:].copy()
    quats1 = np.stack([_quat_mul(q_rand.astype(np.float32), qe) for qe in quats0], axis=0)
    pts1 = (pts0 @ R_wp.T) + t_shift

    b1 = _build_rod_builder(
        n_seg, points=pts1, quaternions=quats1, bend_damping=1e-2, stretch_damping=1e-1, twist_damping=1e-2, n_fixed=1
    )
    m1 = b1.finalize(device=device)

    g0 = np.array([0.0, -2.0, -9.81], dtype=np.float32)
    m0.gravity.assign(tuple(g0.tolist()))
    m1.gravity.assign(tuple((R_wp @ g0).tolist()))

    w_scale = np.array([0.08, 0.08, 0.015, 0.015, 0.015], dtype=np.float64)
    w_nz = (rng.normal(0.0, 1.0, size=(16, 5)) * w_scale).astype(np.float32)
    d_rest = rng.normal(0.0, 0.01, size=(n_seg - 1, 5)).astype(np.float32)
    spec0 = ResidualSpec(rod_rff_dim=16, w_rod=w_nz, spatial_rest_delta=d_rest, seed=3)
    spec1 = ResidualSpec(rod_rff_dim=16, w_rod=w_nz, spatial_rest_delta=d_rest, seed=3)

    s_ramp = np.linspace(0.0, 1.0, n_seg + 1, dtype=np.float32)[:, None]
    v0 = s_ramp * np.array([0.0, 0.25, -0.15], dtype=np.float32)
    s0_in.particle_qd.assign(v0)
    s1_in = m1.state()
    s1_in.particle_qd.assign(v0 @ R_wp.T)
    s0_b = m0.state()
    s0_b.particle_qd.assign(v0)

    fin_b, _ = simulate(m0, SolverDiSMech(m0, residual=None), 10, 0.01, state=s0_b)
    fin0, _ = simulate(m0, SolverDiSMech(m0, residual=spec0), 10, 0.01, state=s0_in)
    fin1, _ = simulate(m1, SolverDiSMech(m1, residual=spec1), 10, 0.01, state=s1_in)
    assert np.abs(fin0.particle_q.numpy() - fin_b.particle_q.numpy()).max() > 5e-4
    expected = (fin0.particle_q.numpy() @ R_wp.T) + t_shift
    assert_close(fin1.particle_q.numpy(), expected, 5e-5, name="se3_particle_q")

    # 2) Continuum mesh refinement convergence N=(8, 16, 32, 64) over fixed physical clamped zone [0, L/8].
    w_cont = rng.normal(0.0, 15.0, size=(16, 5)).astype(np.float32)
    tips_zero = {}
    tips_res = {}
    for n in (8, 16, 32, 64):
        h = 1.0 / n
        b = _build_rod_builder(
            n,
            length=1.0,
            radius=0.01,
            stretch=2000.0,
            shear=2000.0,
            bend=2.0,
            twist=2.0,
            bend_stiffness=2.0 / h,
            stretch_stiffness=2000.0 / h,
            twist_stiffness=2.0 / h,
            bend_damping=0.5 / h,
            stretch_damping=5.0 / h,
            twist_damping=0.5 / h,
            n_fixed=n // 8,
        )
        m = b.finalize(device=device)
        m.gravity.assign((0.0, 0.0, -9.81))
        fin_z, _ = simulate(m, SolverDiSMech(m, residual=None), 150, 0.01)
        sol_r = SolverDiSMech(m, residual=ResidualSpec(rod_rff_dim=16, w_rod=w_cont, seed=11))
        fin_r, _ = simulate(m, sol_r, 150, 0.01)
        tips_zero[n] = float(fin_z.particle_q.numpy()[-1, 2])
        tips_res[n] = float(fin_r.particle_q.numpy()[-1, 2])

    delta = {n: tips_res[n] - tips_zero[n] for n in (8, 16, 32, 64)}
    d8_16, d16_32, d32_64 = (
        abs(tips_res[8] - tips_res[16]),
        abs(tips_res[16] - tips_res[32]),
        abs(tips_res[32] - tips_res[64]),
    )
    dd8_16, dd16_32, dd32_64 = (
        abs(delta[8] - delta[16]),
        abs(delta[16] - delta[32]),
        abs(delta[32] - delta[64]),
    )
    print(
        f"\n[continuum] tips_res={tips_res} (d8_16={d8_16:.4e}, d16_32={d16_32:.4e}, d32_64={d32_64:.4e})"
        f"\n[continuum] residual_delta={delta} (dd16_32={dd16_32:.4e}, dd32_64={dd32_64:.4e})"
    )
    assert abs(tips_res[64]) > 0.05
    assert abs(delta[64]) > 5.0e-4
    assert d16_32 < 0.55 * d8_16 and d32_64 < 0.55 * d16_32
    assert dd16_32 < 0.55 * dd8_16 and dd32_64 < 0.55 * dd16_32 and dd32_64 < max(5.0e-4, 0.15 * abs(delta[64]))


def _rollout_loss(
    model: newton.Model,
    solver: SolverDiSMech,
    s0: newton.State,
    q_ref: wp.array,
    w_q: wp.array,
    w_qd: wp.array,
    steps: int,
    dt: float,
    *,
    record_tape: bool,
):
    states = [s0]
    for _ in range(steps):
        states.append(model.state(requires_grad=record_tape))
    for s in states:
        if record_tape:
            s.particle_q.requires_grad = True
            s.particle_qd.requires_grad = True
            s.body_q.requires_grad = True
            s.body_qd.requires_grad = True
        flatten_state(s)
        if record_tape:
            s.dismech.q.requires_grad = True
            s.dismech.qd.requires_grad = True

    if record_tape:
        loss = wp.zeros(1, dtype=float, device=model.device, requires_grad=True)
        with wp.Tape() as tape:
            for k in range(steps):
                solver.step(states[k], states[k + 1], None, None, dt)
            wp.launch(
                _dismech_loss_kernel,
                dim=states[-1].dismech.q.shape[0],
                inputs=[states[-1].dismech.q, states[-1].dismech.qd, q_ref, w_q, w_qd, dt],
                outputs=[loss],
                device=model.device,
            )
        return float(loss.numpy()[0]), tape, loss
    for k in range(steps):
        solver.step(states[k], states[k + 1], None, None, dt)
    q_np = states[-1].dismech.q.numpy().astype(np.float64)
    dq = q_np - q_ref.numpy().astype(np.float64)
    vq = dt * states[-1].dismech.qd.numpy().astype(np.float64)
    wq = w_q.numpy().astype(np.float64)
    wqd = w_qd.numpy().astype(np.float64)
    val = float(np.sum(0.5 * dq * dq + wq * q_np + wqd * vq + 0.05 * vq * vq))
    return val, None, None


@pytest.mark.parametrize("worlds", [1, 2])
@pytest.mark.parametrize("dahl", [False, True])
def test_rod_step_adjoint_fd(device, rng, needs_cudss, monkeypatch, worlds: int, dahl: bool):
    if worlds == 2 and dahl:
        pytest.skip("single-world Dahl + multi-world elastic cover the full matrix")
    monkeypatch.setattr(dn_solver, "TOLERANCE", 1.0e-7)
    monkeypatch.setattr(dn_solver, "ULP", 1.0e-8)

    n_seg = 4
    sub = _build_rod_builder(
        n_seg,
        length=0.8,
        radius=0.01,
        stretch=80.0,
        shear=30.0,
        bend=0.2,
        twist=0.2,
        stretch_damping=2.0,
        bend_damping=0.05,
        twist_damping=0.05,
        dahl_eps_max=0.015 if dahl else 0.0,
        dahl_tau=0.05 if dahl else 0.0,
        n_fixed=1,
    )
    model = _finalize_multiworld(sub, worlds, device)
    model.gravity.assign((0.0, 0.0, -9.81))

    R = 8
    T = model.state().dismech.triplet_strain_q.shape[0]
    w0 = rng.normal(0.0, 0.025, size=(R, 5) if worlds == 1 else (worlds, R, 5)).astype(np.float32)
    w0_3d = w0[None] if w0.ndim == 2 else w0
    d0 = rng.normal(0.0, 0.03, size=(T, 5)).astype(np.float32)
    spec = ResidualSpec(rod_rff_dim=R, w_rod=w0.copy(), spatial_rest_delta=d0.copy(), seed=13)
    solver = SolverDiSMech(model, iterations=30, exact=True, self_contact=False, residual=spec)

    def _base_state(grad: bool = True) -> newton.State:
        s = model.state(requires_grad=grad)
        pq = s.particle_q.numpy().copy()
        x = pq[:, 0]
        pq[:, 2] += (0.04 if dahl else 0.12) * np.maximum(x - 0.2, 0.0) ** 2
        s.particle_q.assign(pq)
        pqd = s.particle_qd.numpy().copy()
        pqd[:, 1] = (0.9 if dahl else 1.5) * np.maximum(x - 0.2, 0.0)
        pqd[:, 2] = (-1.6 if dahl else -1.2) * np.maximum(x - 0.2, 0.0)
        s.particle_qd.assign(pqd)
        bqd = s.body_qd.numpy().copy()
        bqd[:, 5] = 1.5
        s.body_qd.assign(bqd)
        flatten_state(s)
        return s

    s_init = _base_state(grad=True)
    n_dof = s_init.dismech.q.shape[0]
    q_ref = wp.array(s_init.dismech.q.numpy() + 0.01, dtype=float, device=device)
    w_q = wp.array(rng.normal(0.0, 0.2, size=n_dof).astype(np.float32), dtype=float, device=device)
    w_qd = wp.array(rng.normal(0.0, 0.2, size=n_dof).astype(np.float32), dtype=float, device=device)

    steps = 8
    dt = 0.005 if dahl else 0.01
    L0, tape, loss_arr = _rollout_loss(model, solver, s_init, q_ref, w_q, w_qd, steps, dt, record_tape=True)
    tape.backward(loss_arr)

    g_w = solver.residual.w_rod.grad.numpy().reshape(-1).copy()
    g_d = solver.residual.d_rest.grad.numpy().reshape(-1).copy()
    g_q0 = s_init.dismech.q.grad.numpy().copy()
    g_qd0 = s_init.dismech.qd.grad.numpy().copy()
    tape.zero()

    assert np.linalg.norm(g_w) > 1e-4, f"unexpected near-zero w_rod grad: {np.linalg.norm(g_w)}"
    assert np.linalg.norm(g_d) > 1e-4, f"unexpected near-zero d_rest grad: {np.linalg.norm(g_d)}"

    def _make_dirs(g: np.ndarray, mask: np.ndarray | None = None, k: int = 4) -> np.ndarray:
        g_m = g if mask is None else g * mask
        g_norm = float(np.linalg.norm(g_m))
        assert g_norm > 1e-5
        d = rng.normal(size=(k, g.size))
        if mask is not None:
            d *= mask
        d[0] = g_m / g_norm
        d[1:] /= np.linalg.norm(d[1:], axis=1, keepdims=True)
        return d

    def _check_probe(label: str, dirs: np.ndarray, g_vec: np.ndarray, eval_fn, h_fd: float, rtol: float = 1.0e-3):
        fd_j, est_j = jacobian(eval_fn, np.zeros(dirs.shape[0]), h_fd, error=True)
        adj = dirs @ g_vec
        scale = float(abs(adj[0]))
        rel0 = abs(adj[0] - fd_j[0, 0]) / max(abs(fd_j[0, 0]), 1.0e-12)
        max_rel = float(np.max(np.abs(adj - fd_j[0])) / max(scale, 1.0e-12))
        print(
            f"[adjoint worlds={worlds} dahl={dahl}] {label}: "
            f"ad0={adj[0]:+.6f} fd0={fd_j[0, 0]:+.6f} est0={est_j[0, 0]:.2e} rel0={rel0:.3e} max_rel={max_rel:.3e}"
        )
        for j in range(adj.shape[0]):
            assert_derivative(
                float(adj[j]),
                float(fd_j[0, j]),
                float(est_j[0, j]),
                rtol,
                name=f"{label}_{j}",
                scale=scale,
            )
        assert abs(adj[0] - fd_j[0, 0]) <= rtol * abs(fd_j[0, 0]) + 4.0 * est_j[0, 0]

    dirs_w = _make_dirs(g_w)

    def _eval_w(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            delta = (a[:, None] * dirs_w).sum(axis=0).reshape(w0_3d.shape).astype(np.float32)
            solver.residual.w_rod.assign(w0_3d + delta)
            s_cur = _base_state(grad=False)
            val, _, _ = _rollout_loss(model, solver, s_cur, q_ref, w_q, w_qd, steps, dt, record_tape=False)
            vals[row, 0] = val
        solver.residual.w_rod.assign(w0_3d)
        return vals

    print(f"\n[adjoint worlds={worlds} dahl={dahl}] L0={L0:.6f}")
    _check_probe("w_rod", dirs_w, g_w, _eval_w, 1.0e-2)

    dirs_d = _make_dirs(g_d)

    def _eval_d(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            delta = (a[:, None] * dirs_d).sum(axis=0).reshape(d0.shape).astype(np.float32)
            solver.residual.d_rest.assign(d0 + delta)
            s_cur = _base_state(grad=False)
            val, _, _ = _rollout_loss(model, solver, s_cur, q_ref, w_q, w_qd, steps, dt, record_tape=False)
            vals[row, 0] = val
        solver.residual.d_rest.assign(d0)
        return vals

    _check_probe("d_rest", dirs_d, g_d, _eval_d, 2.0e-3)

    q_p_base = s_init.particle_q.numpy().copy()
    v_p_base = s_init.particle_qd.numpy().copy()
    assert np.allclose(s_init.dismech.q.numpy()[: q_p_base.size], q_p_base.reshape(-1))
    free_p = np.ones((worlds, q_p_base.shape[0] // worlds, 3), dtype=np.float64)
    free_p[:, :2, :] = 0.0  # keep fixed edge 0 vertices unperturbed in each world
    free_mask = free_p.reshape(-1)

    dirs_q = _make_dirs(g_q0[: q_p_base.size], mask=free_mask)

    def _eval_q0(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            s_pert = _base_state(grad=False)
            dq = (a[:, None] * dirs_q).sum(axis=0).reshape(q_p_base.shape).astype(np.float32)
            s_pert.particle_q.assign(q_p_base + dq)
            val, _, _ = _rollout_loss(model, solver, s_pert, q_ref, w_q, w_qd, steps, dt, record_tape=False)
            vals[row, 0] = val
        return vals

    _check_probe("q0", dirs_q, g_q0[: q_p_base.size], _eval_q0, 2.0e-4, rtol=2.5e-3)

    dirs_v = _make_dirs(g_qd0[: v_p_base.size], mask=free_mask)

    def _eval_v0(alphas: np.ndarray) -> np.ndarray:
        vals = np.empty((alphas.shape[0], 1), dtype=np.float64)
        for row, a in enumerate(alphas):
            s_pert = _base_state(grad=False)
            dv = (a[:, None] * dirs_v).sum(axis=0).reshape(v_p_base.shape).astype(np.float32)
            s_pert.particle_qd.assign(v_p_base + dv)
            val, _, _ = _rollout_loss(model, solver, s_pert, q_ref, w_q, w_qd, steps, dt, record_tape=False)
            vals[row, 0] = val
        return vals

    _check_probe("qd0", dirs_v, g_qd0[: v_p_base.size], _eval_v0, 3.0e-2, rtol=2.5e-3)


def test_rod_residual_framegraph_w256(device, rng):
    if not wp.get_device(device).is_cuda:
        pytest.skip("FrameGraph benchmark requires CUDA")

    worlds = 256
    n_seg = 24
    sub = _build_rod_builder(
        n_seg,
        length=1.0,
        radius=0.01,
        stretch=50.0,
        shear=20.0,
        bend=0.05,
        twist=0.05,
        stretch_damping=1e-2,
        bend_damping=1e-3,
        twist_damping=1e-3,
        n_fixed=1,
    )
    model = _finalize_multiworld(sub, worlds, device)
    model.gravity.assign((0.0, 0.0, -9.81))

    R = 64
    w_nz = rng.normal(0.0, 0.05, size=(R, 5)).astype(np.float32)
    dt = 0.01
    warmup_iters, rounds, substeps = 15, 5, 4

    def _compile_warm(solver: SolverDiSMech):
        s0, s1 = model.state(), model.state()

        def work():
            for _ in range(substeps // 2):
                solver.step(s0, s1, None, None, dt)
                solver.step(s1, s0, None, None, dt)

        fg = FrameGraph(work, solver, enabled=True)
        for _ in range(warmup_iters):
            fg()
        assert fg.graph is not None
        return fg, s0

    def _run_block(fg: FrameGraph, iters: int = 10) -> float:
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fg()
        wp.synchronize()
        return 1000.0 * (time.perf_counter() - t0) / (substeps * iters)

    print()
    for sc in (False, True):
        s_base = SolverDiSMech(model, self_contact=sc, residual=None)
        s_res = SolverDiSMech(model, self_contact=sc, residual=ResidualSpec(rod_rff_dim=R, w_rod=w_nz, seed=5))
        fg_base, s0_base = _compile_warm(s_base)
        fg_res, s0_res = _compile_warm(s_res)
        times_base, times_res = [], []
        for _ in range(rounds):
            times_base.append(_run_block(fg_base))
            times_res.append(_run_block(fg_res))
        ms_base = float(np.median(times_base))
        ms_res = float(np.median(times_res))
        q_base, q_res = s0_base.particle_q.numpy(), s0_res.particle_q.numpy()
        overhead_pct = 100.0 * (ms_res - ms_base) / ms_base
        print(
            f"[FrameGraph W={worlds} N={n_seg} self_contact={sc}] "
            f"base={ms_base:.3f} ms/step, res={ms_res:.3f} ms/step, overhead={overhead_pct:+.2f}%"
        )
        assert np.isfinite(q_base).all()
        assert np.isfinite(q_res).all()
        assert np.max(np.abs(q_res - q_base)) > 1e-4
        assert overhead_pct <= 2.5, f"FrameGraph overhead {overhead_pct:+.2f}% > +2.5% (self_contact={sc})"
