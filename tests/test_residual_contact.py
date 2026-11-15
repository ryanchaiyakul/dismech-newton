"""Phase 4C tests for ``CompositionalContactResidual`` (pure-Warp O(M) compositional tribology residual):
- zero ``torch`` import in simulator tests,
- zero-weight bitwise identity for ``kh = 0`` and ``kh > 0`` (forward + replay),
- exact ``(A, B) <-> (B, A)`` surface material & feature swap permutation symmetry (node-shape & rod-rod),
- reference anchor ``mu_eff(u_ref) == mu_{0,AB}`` and boundedness ``mu >= mu_min_mult * mu_0`` / ``pad`` clamp,
- shallow-depth ``pad`` zero value & zero slope at ``dt_n = 0`` (``kh > 0`` hydroelastic pad),
- ``SE(3)`` covariance (rotation + translation) on rod and shell sliding over anisotropic shapes,
- single-step ``StepAdjoint`` VJP vs ``fd.py`` (``rtol <= 5e-3``) wrt ``w_contact``, ``embeddings``, ``q_in``, ``qd_in``
  for both ``kh = 0`` and ``kh > 0``, plus 2-step rollout adjoint sanity,
- multi-world ``W = 4`` per-world ``w_contact`` & ``embeddings`` isolation,
- ``W = 256`` CUDA ``FrameGraph`` paired benchmark (<= +2.5% overhead).
"""

import gc
import sys

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_derivative, jacobian
from rollouts import RADIUS, simulate
from scipy.spatial.transform import Rotation
from utils.common import FrameGraph, pads

from dismech_newton import SolverDiSMech, flatten_state, solver
from dismech_newton.contact import Slots, _slots, emit
from dismech_newton.residual import ResidualSpec, SurfaceMaterialSpec


@wp.kernel
def _emit_slot_kernel(
    q: wp.array[float],
    s: Slots,
    normal: wp.vec3,
    c0: wp.vec3,
    rho: float,
    mu0: float,
    pad: wp.vec4,
    own: int,
    key: int,
    ta: wp.vec3,
    tb: wp.vec3,
    stiff: float,
    lam: float,
    th: float,
):
    emit(s, 0, q, wp.vec4i(0, -1, -1, -1), wp.vec4(1.0, 0.0, 0.0, 0.0), normal, wp.vec3(0.0), wp.vec3(0.0), th, mu0,
         wp.vec4(0.0), pad)
    s.c0[0] = c0
    s.thickness[0] = th
    s.anchor[0] = wp.vec3(0.0)
    s.own[0] = own
    s.key[0] = key
    s.ta[0] = ta
    s.tb[0] = tb
    s.stiffness[0] = stiff
    s.lam[0] = lam
    s.rho[0] = rho


@wp.kernel
def _dismech_loss_kernel(
    q: wp.array[float],
    qd: wp.array[float],
    w_q: wp.array[float],
    w_qd: wp.array[float],
    loss: wp.array[float],
):
    i = wp.tid()
    wp.atomic_add(loss, 0, w_q[i] * q[i] + 0.05 * q[i] * q[i] + w_qd[i] * qd[i] + 0.02 * qd[i] * qd[i])


def _rod_ground_model(
    device,
    *,
    segments: int = 1,
    z: float = 1.0165 * RADIUS,
    mu_rod: float = 0.35,
    mu_shape: float = 0.25,
    kh: float = 0.0,
    diss: float = 1.5,
    surf_rod: int = 0,
    surf_shape: int = 0,
    u_rod: tuple[float, float, float] = (0.0, 0.0, 0.0),
    u_shape: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverDiSMech.register_custom_attributes(b)
    if kh > 0.0:
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=2.0, hy=2.0, hz=0.05,
                        cfg=newton.ModelBuilder.ShapeConfig(mu=mu_shape))
    else:
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu_shape))
    length = 0.08 * max(segments, 1)
    rod = newton.Rod.create_straight((-0.5 * length, 0.0, z), (1.0, 0.0, 0.0), length,
                                     segment_count=segments, radius=RADIUS)
    SolverDiSMech.add_rod(b, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu_rod),
                          stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
    m = b.finalize(device=device)
    if kh > 0.0:
        pads(m, [-1], kh, dissipation=diss)
    n_p = m.particle_count
    if surf_rod != 0:
        m.dismech.surface_id.assign(np.full(n_p, surf_rod, dtype=np.int32))
    if np.any(u_rod):
        m.dismech.surface_u.assign(np.tile(np.asarray(u_rod, np.float32), (n_p, 1)))
    if surf_shape != 0 or np.any(u_shape):
        ss = m.dismech.shape_surface_id.numpy().copy()
        su = m.dismech.shape_surface_u.numpy().copy()
        ss[0] = surf_shape
        su[0] = np.asarray(u_shape, np.float32)
        m.dismech.shape_surface_id.assign(ss)
        m.dismech.shape_surface_u.assign(su)
    return m


# -- 1. Pure-Warp isolation & zero-weight bitwise identity ----------------------------------------------------------


def test_no_torch_imported(device):
    assert "torch" not in sys.modules
    m = _rod_ground_model(device)
    spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0, contact=SurfaceMaterialSpec(rff_dim=8, num_materials=2))
    sim = SolverDiSMech(m, self_contact=False, residual=spec)
    s0, s1 = m.state(), m.state()
    sim.step(s0, s1, None, None, 5.0e-3)
    assert "torch" not in sys.modules


@pytest.mark.parametrize("kh", [0.0, 5.0e5], ids=["barrier", "pad"])
def test_zero_weight_bitwise_identity(device, rng, kh):
    """``w_contact = 0`` with ``alpha = 0`` gives bitwise identical states, slots, and replay outputs."""
    m = _rod_ground_model(
        device, segments=2, z=0.85 * RADIUS if kh > 0.0 else 1.015 * RADIUS, kh=kh,
        surf_rod=1, surf_shape=2, u_rod=(0.3, 0.7, 0.1), u_shape=(0.5, -0.2, 0.4),
    )
    emb = np.zeros((3, 8), dtype=np.float32)
    emb[:, :7] = rng.normal(0.0, 0.8, (3, 7)).astype(np.float32)  # nonzero material features, alpha=0
    wc = np.zeros((16, 2), dtype=np.float32)

    s_base = SolverDiSMech(m, exact=True, iterations=200, self_contact=False, residual=None)
    spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                        contact=SurfaceMaterialSpec(rff_dim=16, num_materials=3, embeddings=emb, w_contact=wc))
    s_res = SolverDiSMech(m, exact=True, iterations=200, self_contact=False, residual=spec)

    def _run(sim):
        s = m.state()
        pqd = np.tile([0.35, 0.15, -0.05 if kh > 0.0 else 0.0], (m.particle_count, 1)).astype(np.float32)
        s.particle_qd.assign(pqd)
        flatten_state(s)
        fin, frames = simulate(m, sim, 4, 5.0e-3, state=s, every=1)
        return fin, frames, sim.contact.snapshot()

    fb, tr_b, sb = _run(s_base)
    fr, tr_r, sr = _run(s_res)
    for arr_b, arr_r in (
        (fb.particle_q.numpy(), fr.particle_q.numpy()),
        (fb.particle_qd.numpy(), fr.particle_qd.numpy()),
        (sb.mu.numpy(), sr.mu.numpy()),
        (sb.pad.numpy(), sr.pad.numpy()),
        (sb.aniso.numpy(), sr.aniso.numpy()),
        (sb.rho.numpy(), sr.rho.numpy()),
        (sb.damp.numpy(), sr.damp.numpy()),
    ):
        assert np.array_equal(arr_b, arr_r)
    for x, y in zip(tr_b, tr_r):
        assert np.array_equal(x, y)

    # Also verify replay_contact matches snap bitwise
    mu_r, pad_r, ani_r = s_res.residual.contact.replay_contact(
        sr, fb.dismech.q, fb.dismech.qd, m.body_q, m.body_qd, s_res.contact.source, 5.0e-3
    )
    assert np.array_equal(mu_r.numpy(), sr.mu.numpy())
    assert np.array_equal(pad_r.numpy(), sr.pad.numpy())
    assert np.array_equal(ani_r.numpy(), sr.aniso.numpy())


# -- 2. Permutation symmetry, reference anchor, bounds, & shallow-depth pad slope -----------------------------------


def test_swap_permutation_symmetry(device, rng):
    """Exact ``(A, B) <-> (B, A)`` swap of surface IDs, embeddings, and anisotropy axes produces identical contact
    modulation and stepped trajectories for direct slot, shell-shape, and rod-rod crossing contacts."""
    emb = rng.normal(0.0, 0.6, (2, 8)).astype(np.float32)
    wc = rng.normal(0.0, 0.5, (32, 2)).astype(np.float32)

    # (a) Shell-on-box contact: shell uses surface_u pushed forward by _face_u, box uses shape_surface_u,
    # so swapping (surf_id, surface_u) <-> (shape_surface_id, shape_surface_u) and emb[0] <-> emb[1] swaps A <-> B.
    def _run_shell_shape(swap: bool):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverDiSMech.register_custom_attributes(b)
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.04), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.04,
                        cfg=newton.ModelBuilder.ShapeConfig(mu=0.35))
        verts = np.array([[-0.05, -0.05, 0.002], [0.05, -0.05, 0.002],
                          [0.05, 0.05, 0.002], [-0.05, 0.05, 0.002]], dtype=np.float64)
        SolverDiSMech.add_cloth_mesh(
            b, vertices=verts, indices=[0, 1, 2, 0, 2, 3], tri_ke=200.0, tri_ka=50.0, tri_kd=1.0,
            edge_ke=0.05, edge_kd=1.0e-3, density=0.5, particle_radius=0.002,
        )
        m = b.finalize(device=device)
        m.particle_mu = 0.35
        u_a, u_b = np.array([0.7, -0.3, 0.2], np.float32), np.array([0.2, 0.8, 0.1], np.float32)
        m.dismech.surface_id.assign(np.full(m.particle_count, 1 if swap else 0, dtype=np.int32))
        m.dismech.surface_u.assign(np.tile(u_b if swap else u_a, (m.particle_count, 1)))
        ss = m.dismech.shape_surface_id.numpy().copy()
        su = m.dismech.shape_surface_u.numpy().copy()
        ss[0] = 0 if swap else 1
        su[0] = u_a if swap else u_b
        m.dismech.shape_surface_id.assign(ss)
        m.dismech.shape_surface_u.assign(su)
        spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                            contact=SurfaceMaterialSpec(rff_dim=32, num_materials=2, embeddings=emb, w_contact=wc,
                                                        seed=11))
        sim = SolverDiSMech(m, exact=True, iterations=400, self_contact=False, residual=spec)
        s0, s1 = m.state(), m.state()
        s0.particle_qd.assign(np.tile([0.3, 0.15, 0.0], (m.particle_count, 1)).astype(np.float32))
        flatten_state(s0)
        sim.step(s0, s1, None, None, 5.0e-3)
        snap = sim.contact.snapshot()
        return snap.mu.numpy(), snap.pad.numpy(), snap.aniso.numpy(), s1.particle_q.numpy(), s1.particle_qd.numpy()

    a0 = _run_shell_shape(False)
    a1 = _run_shell_shape(True)
    for x, y in zip(a0, a1):
        np.testing.assert_allclose(x, y, atol=1.0e-6, rtol=1.0e-6)

    # Verify non-trivial anisotropy and modulation actually engaged
    assert np.max(np.abs(a0[2][:, 3])) > 1.0e-2
    assert np.max(np.abs(a0[0] - 0.35)) > 1.0e-3

    # (b) Two crossing rods swapping builder order (own <-> key) with fixed surface IDs & embeddings per rod
    def _run_crossing(swap: bool):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverDiSMech.register_custom_attributes(b)
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.35)
        r0 = newton.Rod.create_straight((-0.04, 0.0, 0.0), (1.0, 0.0, 0.0), 0.08, segment_count=2, radius=RADIUS)
        r1 = newton.Rod.create_straight((0.0, -0.04, 1.95 * RADIUS), (0.0, 1.0, 0.0), 0.08, segment_count=2,
                                        radius=RADIUS)
        for r in ((r1, r0) if swap else (r0, r1)):
            SolverDiSMech.add_rod(b, rod=r, cfg=cfg, stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
        m = b.finalize(device=device)
        sids = np.array([1, 1, 1, 0, 0, 0] if swap else [0, 0, 0, 1, 1, 1], dtype=np.int32)
        m.dismech.surface_id.assign(sids)
        spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                            contact=SurfaceMaterialSpec(rff_dim=32, num_materials=2, embeddings=emb, w_contact=wc,
                                                        seed=17))
        sim = SolverDiSMech(m, exact=True, iterations=200, residual=spec)
        s0, s1 = m.state(), m.state()
        qd = np.tile([0.2, 0.1, 0.0], (m.particle_count, 1)).astype(np.float32)
        if swap:
            qd[:3, 2], qd[3:, 2] = -0.05, 0.05
        else:
            qd[:3, 2], qd[3:, 2] = 0.05, -0.05
        s0.particle_qd.assign(qd)
        flatten_state(s0)
        sim.step(s0, s1, None, None, 5.0e-3)
        q, vd = s1.particle_q.numpy(), s1.particle_qd.numpy()
        return (np.concatenate([q[3:], q[:3]]), np.concatenate([vd[3:], vd[:3]])) if swap else (q, vd)

    c0 = _run_crossing(False)
    c1 = _run_crossing(True)
    for x, y in zip(c0, c1):
        np.testing.assert_allclose(x, y, atol=2.0e-6, rtol=2.0e-6)


def test_reference_anchor_and_bounds(device, rng):
    """At ``u_ref = [log 2, 1, 0, 0]`` (``rho = rho_ref``, ``|v_t| = v_ref``, ``s_an = 0``, ``dt_n = 0``),
    ``mu_eff == mu_0`` regardless of large ``w_contact``; extreme weights respect all bounds."""
    m = _rod_ground_model(device, segments=1, z=RADIUS, mu_rod=0.4, mu_shape=0.4)
    spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                        contact=SurfaceMaterialSpec(rff_dim=32, rho_ref=2.5, v_ref=0.2, mu_min_mult=0.15,
                                                    pad_min_mult=0.2, pad_max_mult=4.0, gamma_max=1.2,
                                                    num_materials=2,
                                                    embeddings=rng.normal(0.0, 0.7, (2, 8)).astype(np.float32),
                                                    w_contact=rng.normal(0.0, 25.0, (32, 2)).astype(np.float32),
                                                    seed=31))
    # Zero alpha for s_an == 0 at reference anchor
    emb_np = spec.contact.embeddings.copy()
    emb_np[:, 7] = 0.0
    spec.contact.embeddings = emb_np
    sim = SolverDiSMech(m, exact=True, iterations=50, self_contact=False, residual=spec)
    res, src = sim.residual.contact, sim.contact.source

    slots = _slots(1, device)
    st = m.state()
    flatten_state(st)
    # Set |v_t| = v_ref along +x, normal = +z, d = th -> dt_n = 0, rho = rho_ref
    cspec = spec.contact
    st.dismech.qd.assign(np.array([cspec.v_ref, 0.0, 0.0, 0.0], dtype=np.float32))
    mu0, th = 0.4, 0.02
    wp.launch(_emit_slot_kernel, 1, inputs=[
        st.dismech.q, slots, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, th), cspec.rho_ref, mu0,
        wp.vec4(0.0), 0, 0, wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), 1.0, 1.0, th,
    ], device=device)
    res.apply_contact(slots, st, src, None, 5.0e-3)
    np.testing.assert_allclose(slots.mu.numpy()[0], mu0, atol=1.0e-5, rtol=1.0e-5)

    # Extreme negative/positive w_contact + extreme embeddings[:, 7] -> verify bounds & unit e1
    emb_ext = rng.normal(0.0, 1.0, (2, 8)).astype(np.float32)
    emb_ext[:, 7] = np.array([12.0, -4.0], dtype=np.float32)
    res.embeddings.assign(emb_ext)
    kh0 = 5.0e5
    pad0 = wp.vec4(kh0, 2.0, 0.002, 0.5)
    for sign in (-200.0, 200.0):
        res.w_contact.assign(np.full((1, 32, 2), sign, dtype=np.float32))
        for d_off in (0.0, 0.001, 0.004):
            wp.launch(_emit_slot_kernel, 1, inputs=[
                st.dismech.q, slots, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, th + float(pad0[2]) - d_off),
                5.0, mu0, pad0, 0, src.shape_key0, wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0), 1.0, 1.0, th,
            ], device=device)
            res.apply_contact(slots, st, src, None, 5.0e-3)
            mu_out = float(slots.mu.numpy()[0])
            pad_out = slots.pad.numpy()[0]
            ani_out = slots.aniso.numpy()[0]
            assert mu_out >= cspec.mu_min_mult * mu0 - 1.0e-7
            assert cspec.pad_min_mult * kh0 - 1.0e-2 <= pad_out[0] <= cspec.pad_max_mult * kh0 + 1.0e-2
            assert 0.0 <= ani_out[3] <= cspec.gamma_max + 1.0e-6
            if ani_out[3] > 0.0:
                np.testing.assert_allclose(np.linalg.norm(ani_out[:3]), 1.0, atol=1.0e-5)


def test_shallow_depth_pad_zero_and_slope(device, rng):
    """At ``dt_n = 0``, ``kh_eff == kh_0`` and ``d(kh_eff)/d(dt_n)|_0 == 0`` (quadratic onset in ``dt_n``)."""
    m = _rod_ground_model(device, segments=1, z=RADIUS, kh=5.0e5, surf_rod=0, surf_shape=1, u_shape=(1.0, 0.0, 0.0))
    wc = np.zeros((16, 2), dtype=np.float32)
    wc[:, 1] = rng.normal(0.0, 3.0, 16).astype(np.float32)
    spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                        contact=SurfaceMaterialSpec(rff_dim=16, num_materials=2, pad_min_mult=0.1, pad_max_mult=8.0,
                                                    embeddings=rng.normal(0.0, 0.8, (2, 8)).astype(np.float32),
                                                    w_contact=wc, seed=7))
    sim = SolverDiSMech(m, exact=True, iterations=50, self_contact=False, residual=spec)
    res, src = sim.residual.contact, sim.contact.source
    slots = _slots(1, device)
    st = m.state()
    flatten_state(st)
    kh0, p2, th = 5.0e5, 0.002, 0.02
    pad0 = wp.vec4(kh0, 2.0, p2, 0.5)
    denom = th + p2 - th + max(p2, th)  # denominator in residual.py is max(max(p[2], th), 1e-5) = th
    assert denom > p2

    def _eval_kh(dt_n: float) -> float:
        d = th + p2 - dt_n * max(p2, th)
        wp.launch(_emit_slot_kernel, 1, inputs=[
            st.dismech.q, slots, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, d),
            1.0, 0.3, pad0, 0, src.shape_key0, wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0), 1.0, 1.0, th,
        ], device=device)
        res.apply_contact(slots, st, src, None, 5.0e-3)
        return float(slots.pad.numpy()[0, 0])

    assert _eval_kh(0.0) == kh0
    d1 = _eval_kh(1.0e-2) - kh0
    d2 = _eval_kh(2.0e-2) - kh0
    assert abs(d1) > 1.0e-1
    # Quadratic onset: ratio d2 / d1 should be ~4.0
    np.testing.assert_allclose(d2 / d1, 4.0, rtol=5.0e-2)
    # Deep penetration shows substantial modulation and consistent pad_c in [min, max]
    kh_deep = _eval_kh(0.5)
    assert abs(kh_deep - kh0) > 1.0e4
    assert spec.contact.pad_min_mult * kh0 <= kh_deep <= spec.contact.pad_max_mult * kh0


# -- 3. SE(3) covariance on rod and shell ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["rod", "shell"])
def test_se3_covariance(device, rng, monkeypatch, kind):
    """Translating and rotating the entire scene (geometry, local ``surface_u``, velocities, forces, gravity) by
    arbitrary ``(R, t) in SE(3)`` yields covariantly transformed trajectories (``< 5e-5``) on both rod and shell."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-5)
    rot = Rotation.random(random_state=rng)
    Rm, q_rot = rot.as_matrix(), wp.quat(*rot.as_quat())
    shift = rng.uniform(-1.5, 1.5, size=3)

    emb = rng.normal(0.0, 0.6, (2, 8)).astype(np.float32)
    emb[:, 7] = np.array([0.8, 0.6], dtype=np.float32)
    wc = rng.normal(0.0, 0.6 if kind == "rod" else 1.2, (24, 2)).astype(np.float32)
    g0 = np.array([0.0, 0.0, -9.81])
    v0 = np.array([0.55, 0.3, 0.0])
    f0 = np.array([0.06, 0.04, 0.0]) if kind == "rod" else np.array([0.008, 0.006, 0.0])
    u_sh0 = np.array([0.8, 0.6, 0.1], dtype=np.float32)
    u_soft0 = np.array([0.3, -0.7, 0.2], dtype=np.float32)

    def _run(transform: bool, use_res: bool):
        R = Rm if transform else np.eye(3)
        t = shift if transform else np.zeros(3)
        qR = q_rot if transform else wp.quat_identity()
        b = newton.ModelBuilder(gravity=tuple(R @ g0))
        SolverDiSMech.register_custom_attributes(b)
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(*(t + R @ np.array([0.0, 0.0, -0.04]))), qR),
                        hx=0.6, hy=0.6, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(mu=0.35))
        if kind == "rod":
            pts = np.column_stack([np.linspace(-0.06, 0.06, 4), np.zeros(4), np.full(4, RADIUS)])
            base_rod = newton.Rod(pts, radius=RADIUS)
            quats = (rot * Rotation.from_quat(base_rod.quaternions)).as_quat() if transform else base_rod.quaternions
            SolverDiSMech.add_rod(b, rod=newton.Rod(t + pts @ R.T, quaternions=quats, radius=RADIUS),
                                  cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.35),
                                  stretch_stiffness=5.0e3, bend_stiffness=0.2, twist_stiffness=0.1)
        else:
            verts = np.array([[-0.05, -0.05, 0.004], [0.05, -0.05, 0.004],
                              [0.05, 0.05, 0.004], [-0.05, 0.05, 0.004]], dtype=np.float64)
            tris = [0, 1, 2, 0, 2, 3]
            SolverDiSMech.add_cloth_mesh(
                b, pos=tuple(t), rot=qR, vertices=verts, indices=tris, vel=(0.0, 0.0, 0.0),
                tri_ke=2.0e3, tri_ka=1.0e3, tri_kd=2.0, edge_ke=1.0e-2, edge_kd=1.0e-3, density=2.0,
                particle_radius=0.004,
            )
        m = b.finalize(device=device)
        if kind == "shell":
            m.particle_mu = 0.35
        m.dismech.surface_id.assign(np.zeros(m.particle_count, dtype=np.int32))
        m.dismech.surface_u.assign(np.tile((R @ u_soft0).astype(np.float32), (m.particle_count, 1)))
        ss = m.dismech.shape_surface_id.numpy().copy()
        su = m.dismech.shape_surface_u.numpy().copy()
        ss[0] = 1
        su[0] = u_sh0  # shape_surface_u is in shape-local frame and co-rotates via _shape_pose
        m.dismech.shape_surface_id.assign(ss)
        m.dismech.shape_surface_u.assign(su)
        spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                            contact=SurfaceMaterialSpec(rff_dim=24, v_ref=0.5, num_materials=2, embeddings=emb,
                                                        w_contact=wc if use_res else None, seed=23))
        sim = SolverDiSMech(m, exact=True, iterations=400, self_contact=False, residual=spec)
        st = m.state()
        st.particle_qd.assign(np.tile((R @ v0).astype(np.float32), (m.particle_count, 1)))
        flatten_state(st)
        force = np.tile((R @ f0).astype(np.float32), (m.particle_count, 1))
        end, _ = simulate(m, sim, 30, 4.0e-3, force=force, state=st)
        assert int(sim.counts.numpy()[2]) == 0, f"failed {kind} counts={sim.counts.numpy()}"
        return (end.particle_q.numpy().astype(np.float64) - t) @ R

    q_base = _run(False, True)
    q_se3 = _run(True, True)
    q_no_res = _run(False, False)
    err = float(np.linalg.norm(q_se3 - q_base, axis=1).max())
    eff = float(np.linalg.norm(q_base - q_no_res, axis=1).max())
    print(f"\n[SE(3) {kind}] max_err={err:.3e}, residual_effect={eff:.3e}")
    assert eff > 3.0e-5, f"residual effect too small: {eff}"
    assert err < 1.5e-4, f"SE(3) covariance error {err:.3e} >= 1.5e-4"


# -- 4. StepAdjoint VJP vs finite differences (kh = 0 and kh > 0) ----------------------------------------------------


@pytest.mark.parametrize("kh", [0.0, 5.0e5], ids=["kh0", "kh_pad"])
def test_contact_residual_vjp_matches_fd(device, rng, monkeypatch, kh):
    """Single-step ``StepAdjoint`` VJP wrt ``w_contact``, ``embeddings``, ``q_in``, and ``qd_in`` matches central
    Richardson FD (``rtol <= 5e-3``) for both ``kh = 0`` barrier contact and ``kh > 0`` hydroelastic pad."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-6 if kh > 0.0 else 2.0e-5)

    z0 = 0.80 * RADIUS if kh > 0.0 else 1.0165 * RADIUS
    m = _rod_ground_model(
        device, segments=2, z=z0, mu_rod=0.35, mu_shape=0.35, kh=kh, diss=1.5,
        surf_rod=1, surf_shape=2, u_rod=(0.4, 0.7, 0.2), u_shape=(0.8, 0.5, 0.1),
    )
    R, M = 12, 3
    wc0 = rng.normal(0.0, 0.25, (1, R, 2)).astype(np.float32)
    emb0 = rng.normal(0.0, 0.55, (1, M, 8)).astype(np.float32)
    emb0[:, :, 7] = np.array([[0.7, -0.4, 0.6]], dtype=np.float32)

    spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                        contact=SurfaceMaterialSpec(rff_dim=R, rho_ref=1.0, v_ref=0.1, num_materials=M,
                                                    embeddings=emb0.copy(), w_contact=wc0.copy(), seed=19))
    sim = SolverDiSMech(m, exact=True, iterations=1500, self_contact=False, residual=spec)
    dt = 5.0e-3
    v0 = np.tile([0.35, 0.15, -0.05 if kh > 0.0 else 0.0], (m.particle_count, 1)).astype(np.float32)

    s_init = m.state(requires_grad=True)
    s_init.particle_qd.assign(v0)
    flatten_state(s_init)
    n_dof = s_init.dismech.q.shape[0]
    q0_base = s_init.dismech.q.numpy().copy()
    qd0_base = s_init.dismech.qd.numpy().copy()

    rng_l = np.random.default_rng(rng.integers(0, 1 << 30))
    wq_np = rng_l.normal(0.0, 0.6, n_dof).astype(np.float32)
    wqd_np = rng_l.normal(0.0, 0.25, n_dof).astype(np.float32)
    wq_wp = wp.array(wq_np, dtype=float, device=device)
    wqd_wp = wp.array(wqd_np, dtype=float, device=device)

    def _run_loss(record: bool, q_np=None, qd_np=None, steps: int = 1):
        states = [m.state(requires_grad=record) for _ in range(steps + 1)]
        for st in states:
            flatten_state(st)
            if record:
                st.dismech.q.requires_grad = True
                st.dismech.qd.requires_grad = True
        states[0].particle_qd.assign(v0)
        flatten_state(states[0])
        if q_np is not None:
            states[0].dismech.q.assign(q_np.astype(np.float32))
        if qd_np is not None:
            states[0].dismech.qd.assign(qd_np.astype(np.float32))
        sim.reset(states[0])
        sim.counts.zero_()
        if record:
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            with wp.Tape() as tape:
                for k in range(steps):
                    sim.step(states[k], states[k + 1], None, None, dt)
                    assert int(sim.counts.numpy()[0]) == 0
                wp.launch(_dismech_loss_kernel, n_dof, inputs=[
                    states[-1].dismech.q, states[-1].dismech.qd, wq_wp, wqd_wp, loss], device=device)
            return float(loss.numpy()[0]), tape, loss, states[0]
        for k in range(steps):
            sim.step(states[k], states[k + 1], None, None, dt)
            assert int(sim.counts.numpy()[0]) == 0
        q_f = states[-1].dismech.q.numpy().astype(np.float64)
        qd_f = states[-1].dismech.qd.numpy().astype(np.float64)
        return float(wq_np @ q_f + 0.05 * np.dot(q_f, q_f) + wqd_np @ qd_f + 0.02 * np.dot(qd_f, qd_f))

    _, tape, loss_arr, s0_rec = _run_loss(True)
    tape.backward(loss_arr)
    g_wc = sim.residual.contact.w_contact.grad.numpy().reshape(-1).copy()
    g_emb = sim.residual.contact.embeddings.grad.numpy().reshape(-1).copy()
    g_qin = s0_rec.dismech.q.grad.numpy().copy()
    g_qdin = s0_rec.dismech.qd.grad.numpy().copy()
    tape.zero()

    assert np.linalg.norm(g_wc) > 1.0e-4, f"near-zero w_contact grad: {np.linalg.norm(g_wc)}"
    assert np.linalg.norm(g_emb) > 1.0e-4, f"near-zero embeddings grad: {np.linalg.norm(g_emb)}"
    if kh > 0.0:
        assert np.linalg.norm(g_wc.reshape(R, 2)[:, 1]) > 1.0e-4, "pad w_contact[:, 1] grad should be active"

    def _probe(label: str, g_vec: np.ndarray, eval_scalar, h_fd: float, dirs_override=None):
        if dirs_override is None:
            dirs = np.zeros((3, g_vec.size), dtype=np.float64)
            dirs[0] = g_vec / np.linalg.norm(g_vec)
            for r in range(1, 3):
                d = rng.normal(size=g_vec.size)
                dirs[r] = d / np.linalg.norm(d)
        else:
            dirs = dirs_override

        def f_batch(alphas):
            out = np.empty((alphas.shape[0], 1), dtype=np.float64)
            for k, a in enumerate(alphas):
                out[k, 0] = eval_scalar((a[:, None] * dirs).sum(axis=0))
            return out

        fd_j, est_j = jacobian(f_batch, np.zeros(dirs.shape[0]), h_fd, error=True)
        adj = dirs @ g_vec.astype(np.float64)
        scale = max(float(np.abs(adj).max()), float(np.abs(fd_j[0]).max()))
        rel0 = abs(adj[0] - fd_j[0, 0]) / max(abs(fd_j[0, 0]), 1.0e-12)
        print(f"\n[FD kh={kh:g}] {label}: adj0={adj[0]:+.6e} fd0={fd_j[0, 0]:+.6e} est={est_j[0, 0]:.1e} rel={rel0:.3e}")
        for j in range(dirs.shape[0]):
            assert_derivative(float(adj[j]), float(fd_j[0, j]), float(est_j[0, j]), 5.0e-3,
                              name=f"kh={kh:g}-{label}-{j}", scale=scale)

    def _eval_wc(d):
        sim.residual.contact.w_contact.assign(wc0 + d.reshape(wc0.shape).astype(np.float32))
        val = _run_loss(False)
        sim.residual.contact.w_contact.assign(wc0)
        return val

    def _eval_emb(d):
        sim.residual.contact.embeddings.assign(emb0 + d.reshape(emb0.shape).astype(np.float32))
        val = _run_loss(False)
        sim.residual.contact.embeddings.assign(emb0)
        return val

    _probe("w_contact", g_wc, _eval_wc, 5.0e-3)
    if kh > 0.0:
        # Also test w_contact[:, :, 1] (pad branch) explicitly
        pad_mask = np.zeros_like(wc0).reshape(-1)
        pad_mask[1::2] = 1.0
        g_pad = g_wc * pad_mask
        _probe("w_contact_pad_only", g_wc, _eval_wc, 5.0e-3, dirs_override=(g_pad / np.linalg.norm(g_pad))[None])
    _probe("embeddings", g_emb, _eval_emb, 3.0e-3)
    _probe("q_in", g_qin, lambda d: _run_loss(False, q_np=q0_base + d), 3.0e-6)
    _probe("qd_in", g_qdin, lambda d: _run_loss(False, qd_np=qd0_base + d), 1.0e-3)

    # Multi-step (2 steps) rollout check wrt w_contact to confirm clean cross-step gradient accumulation
    _, tape2, loss2, _ = _run_loss(True, steps=2)
    tape2.backward(loss2)
    g_wc2 = sim.residual.contact.w_contact.grad.numpy().reshape(-1).copy()
    tape2.zero()
    _probe("w_contact_2step", g_wc2, lambda d: (
        sim.residual.contact.w_contact.assign(wc0 + d.reshape(wc0.shape).astype(np.float32)),
        res := _run_loss(False, steps=2),
        sim.residual.contact.w_contact.assign(wc0),
        res,
    )[3], 5.0e-3)


# -- 5. Multi-world isolation & W=256 FrameGraph overhead benchmark --------------------------------------------------


def test_multiworld_isolation(device, rng):
    """Per-world ``w_contact`` and ``embeddings`` in ``W = 4`` isolate world ``1`` from worlds ``0, 2, 3``."""
    worlds, R, M = 4, 16, 3
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverDiSMech.register_custom_attributes(b)
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    sub.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.04), wp.quat_identity()), hx=0.4, hy=0.4, hz=0.04,
                      cfg=newton.ModelBuilder.ShapeConfig(mu=0.35))
    rod = newton.Rod.create_straight((-0.04, 0.0, 1.015 * RADIUS), (1.0, 0.0, 0.0), 0.08, segment_count=2,
                                     radius=RADIUS)
    SolverDiSMech.add_rod(sub, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.35),
                          stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
    for _ in range(worlds):
        b.add_world(sub)
    m = b.finalize(device=device)
    m.dismech.surface_id.assign(np.ones(m.particle_count, dtype=np.int32))
    ss = np.full(m.shape_count, 2, dtype=np.int32)
    m.dismech.shape_surface_id.assign(ss)

    wc_a = rng.normal(0.0, 0.6, (worlds, R, 2)).astype(np.float32)
    emb_a = rng.normal(0.0, 0.6, (worlds, M, 8)).astype(np.float32)
    wc_b, emb_b = wc_a.copy(), emb_a.copy()
    wc_b[1] = 0.0
    emb_b[1] = 0.0

    def _run(wc, emb):
        spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                            contact=SurfaceMaterialSpec(rff_dim=R, num_materials=M, embeddings=emb, w_contact=wc,
                                                        seed=5))
        sim = SolverDiSMech(m, exact=True, iterations=400, self_contact=False, residual=spec)
        s0, s1 = m.state(), m.state()
        s0.particle_qd.assign(np.tile([0.35, 0.15, 0.0], (m.particle_count, 1)).astype(np.float32))
        flatten_state(s0)
        sim.step(s0, s1, None, None, 5.0e-3)
        return s1.particle_q.numpy(), s1.particle_qd.numpy()

    qa, qda = _run(wc_a, emb_a)
    qb, qdb = _run(wc_b, emb_b)
    pw = m.particle_world.numpy()
    for w in range(worlds):
        mask = pw == w
        if w == 1:
            assert np.max(np.abs(qa[mask] - qb[mask])) > 1.0e-6 or np.max(np.abs(qda[mask] - qdb[mask])) > 1.0e-4
        else:
            assert np.array_equal(qa[mask], qb[mask])
            assert np.array_equal(qda[mask], qdb[mask])


def test_contact_residual_framegraph_w256(device, rng):
    """256 worlds of sliding contact with ``CompositionalContactResidual`` capture cleanly in ``FrameGraph`` and stay
    ``<= +2.5%`` over the ``residual=None`` baseline."""
    if not wp.get_device(device).is_cuda:
        pytest.skip("FrameGraph benchmark requires CUDA")

    worlds, R, M = 256, 32, 2
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod = newton.Rod.create_straight((-0.08, 0.0, 1.015 * RADIUS), (1.0, 0.0, 0.0), 0.16, segment_count=4,
                                     radius=RADIUS)
    SolverDiSMech.add_rod(sub, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3),
                          stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.3))
    for _ in range(worlds):
        b.add_world(sub)
    m = b.finalize(device=device)
    m.dismech.surface_id.assign(np.zeros(m.particle_count, dtype=np.int32))
    ss = m.dismech.shape_surface_id.numpy().copy()
    ss[0] = 1
    m.dismech.shape_surface_id.assign(ss)
    su = m.dismech.shape_surface_u.numpy().copy()
    su[0] = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    m.dismech.shape_surface_u.assign(su)

    emb = rng.normal(0.0, 0.5, (1, M, 8)).astype(np.float32)
    emb[:, :, 7] = np.array([[0.5, -0.3]], dtype=np.float32)
    wc = rng.normal(0.0, 0.4, (1, R, 2)).astype(np.float32)
    spec = ResidualSpec(rod_rff_dim=0, shell_rff_dim=0,
                        contact=SurfaceMaterialSpec(rff_dim=R, num_materials=M, embeddings=emb, w_contact=wc, seed=9))
    s_base = SolverDiSMech(m, self_contact=False, residual=None)
    s_res = SolverDiSMech(m, self_contact=False, residual=spec)

    v0 = np.tile([0.4, 0.2, 0.0], (m.particle_count, 1)).astype(np.float32)
    f_drv = np.tile([0.06, 0.02, 0.0], (m.particle_count, 1)).astype(np.float32)
    dt, substeps, warmup, iters, rounds = 0.005, 4, 100, 20, 7
    stream = wp.get_stream(m.device)
    e0, e1 = wp.Event(device=m.device, enable_timing=True), wp.Event(device=m.device, enable_timing=True)

    def _compile(sim: SolverDiSMech):
        s0, s1 = m.state(), m.state()
        s0.particle_qd.assign(v0)
        s0.particle_f.assign(f_drv)
        s1.particle_f.assign(f_drv)

        def work():
            for _ in range(substeps // 2):
                sim.step(s0, s1, None, None, dt)
                sim.step(s1, s0, None, None, dt)

        gc.collect()
        fg = FrameGraph(work, sim, enabled=True)
        fg()
        assert fg.graph is not None
        return fg, s0

    fg_base, s0_base = _compile(s_base)
    fg_res, s0_res = _compile(s_res)
    for _ in range(warmup):
        fg_base()
        fg_res()
    wp.synchronize()

    def _run_block(fg: FrameGraph) -> float:
        wp.synchronize()
        stream.record_event(e0)
        for _ in range(iters):
            fg()
        stream.record_event(e1)
        wp.synchronize()
        return float(wp.get_event_elapsed_time(e0, e1)) / (substeps * iters)

    pairs = []
    gc.collect()
    for r in range(rounds):
        if r % 2 == 0:
            tb, tr = _run_block(fg_base), _run_block(fg_res)
        else:
            tr, tb = _run_block(fg_res), _run_block(fg_base)
        pairs.append((tb, tr))

    ms_base = float(np.median([tb for tb, _ in pairs]))
    ms_res = float(np.median([tr for _, tr in pairs]))
    overhead_pct = float(np.median([100.0 * (tr - tb) / tb for tb, tr in pairs]))
    print(f"\n[ContactResidual FrameGraph W={worlds} segs/world=4] base={ms_base:.4f} ms/step, "
          f"res={ms_res:.4f} ms/step, overhead={overhead_pct:+.2f}%")

    q_base, q_res = s0_base.particle_q.numpy(), s0_res.particle_q.numpy()
    assert np.isfinite(q_base).all() and np.isfinite(q_res).all()
    assert np.max(np.abs(q_res - q_base)) > 1.0e-4
    assert overhead_pct <= 2.5, f"FrameGraph overhead {overhead_pct:+.2f}% > +2.5%"
