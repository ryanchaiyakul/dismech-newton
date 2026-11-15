"""Phase 4A tests for variational elliptic anisotropic friction (``Slots.aniso``, zero ML):
- ``emit`` / ``clear`` defaults and ``gamma == 0.0`` bitwise identity (forward + ``StepAdjoint``),
- kernel math: frame orthonormality, ``contact_hessian`` vs FD, directional derivative of ``slip_energy``,
  and ``_majorizer`` symmetry/PSD/curvature cap at extreme ``gamma``,
- analytical steady-state transverse slip deflection (< 1 deg) and SO(3) covariance,
- 1-node & multi-node ``StepAdjoint`` VJP vs ``fd.py`` wrt ``q_in``, ``qd_in``, ``mass``, and ``contacts.aniso``
  (both ``e1`` and ``gamma``) in sticking and sliding regimes,
- W=256 CUDA ``FrameGraph`` benchmark (<= +2.5% overhead).
"""

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_derivative
from rollouts import RADIUS, GroundRollout, SphereRollout, simulate
from scipy.spatial.transform import Rotation
from utils.common import FrameGraph

from dismech_newton import SolverDiSMech, flatten_state, solver
from dismech_newton.contact import (
    CURVATURE,
    Slots,
    _aniso_axes,
    _barrier_force,
    _majorizer,
    _slots,
    clear,
    contact_hessian,
    emit,
    slip_energy,
)


@wp.kernel(enable_backward=False)
def _set_aniso_kernel(val: wp.vec4, s: Slots):
    i = wp.tid()
    if s.active[i] != 0:
        s.aniso[i] = val


def _attach_aniso_hook(sim: SolverDiSMech, val: wp.vec4) -> None:
    """Set ``s.aniso`` via a pure-GPU kernel right after every ``source.fill`` (CUDA-graph capturable)."""
    orig_fill = sim.contact.source.fill

    def fill_with_aniso(state_in, dt, q, s):
        orig_fill(state_in, dt, q, s)
        wp.launch(_set_aniso_kernel, dim=s.active.shape[0], inputs=[val, s], device=sim.device)

    sim.contact.source.fill = fill_with_aniso


def _rod_on_ground(segments: int = 1, z: float = RADIUS, mu: float = 0.5, gravity=(0.0, 0.0, -9.81)):
    """Horizontal rod on a ground plane (no self-contact)."""
    b = newton.ModelBuilder(gravity=gravity)
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    rod = newton.Rod.create_straight((-0.02, 0.0, z), (1.0, 0.0, 0.0), 0.04 * segments, segment_count=segments,
                                     radius=RADIUS)
    SolverDiSMech.add_rod(b, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu),
                          stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
    return b.finalize()


# -- 1. emit / clear & gamma == 0 bitwise identity -----------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _emit_clear_kernel(q: wp.array[float], s: Slots, do_clear: int):
    i = wp.tid()
    if do_clear == 0:
        pad = wp.where(i == 0, wp.vec4(0.0), wp.vec4(1.0e5, 2.0, 0.002, 0.5))
        emit(s, i, q, wp.vec4i(0, -1, -1, -1), wp.vec4(1.0, 0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0),
             wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0), 0.01, 0.35 + 0.15 * float(i), wp.vec4(0.0), pad)
    else:
        clear(s, i)


def test_emit_clear_and_gamma_zero_bitwise_identity(rng, monkeypatch):
    """``emit`` zeros ``aniso`` and saves ``(mu0, pad_c0)`` in ``base``; ``clear`` zeros both; ``gamma == 0.0``
    (both zero and nonzero ``e1``) is bit-identical to untouched slots in both forward states and ``StepAdjoint``."""
    s = _slots(2, None)
    s.aniso.fill_(wp.vec4(1.0, 2.0, 3.0, 0.7))
    s.base.fill_(wp.vec2(9.0, 8.0))
    q = wp.array([0.0, 0.0, 0.01], dtype=float)
    wp.launch(_emit_clear_kernel, dim=2, inputs=[q, s, 0])
    np.testing.assert_array_equal(s.aniso.numpy(), np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_allclose(s.base.numpy(), np.array([[0.35, 0.0], [0.50, 1.0e5]], dtype=np.float32))
    wp.launch(_emit_clear_kernel, dim=2, inputs=[q, s, 1])
    np.testing.assert_array_equal(s.aniso.numpy(), np.zeros((2, 4), dtype=np.float32))
    np.testing.assert_array_equal(s.base.numpy(), np.zeros((2, 2), dtype=np.float32))

    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-5)
    r = GroundRollout(rng, (0.3, 0.1, 0.0), ("mass", "velocity", "position"), 0.3)

    def run_case(aniso_vec):
        run = r.setup({"iterations": 500, "self_contact": False}, grad=True)
        if aniso_vec is not None:
            _attach_aniso_hook(run.solver, wp.vec4(*aniso_vec))
        r.record(run)
        assert run.solver.contact.s.active.numpy().sum() > 0
        s_end = run.states[-1].dismech
        s0 = run.states[0].dismech
        fwd = {k: getattr(s_end, k).numpy().copy() for k in ("q", "qd", "edge_d1_q", "triplet_ref_twist_q")}
        bwd = {
            "q_grad": s0.q.grad.numpy().copy(),
            "qd_grad": s0.qd.grad.numpy().copy(),
            **{k: a.grad.numpy().copy() for k, a in run.arrays.items()},
        }
        return fwd, bwd

    ref_fwd, ref_bwd = run_case(None)
    for vec in ((0.0, 0.0, 0.0, 0.0), (0.6, 0.8, 0.0, 0.0), (0.3, -0.7, 0.5, 0.0)):
        fwd, bwd = run_case(vec)
        for k in ref_fwd:
            np.testing.assert_array_equal(fwd[k], ref_fwd[k], err_msg=f"{vec} fwd {k}")
        for k in ref_bwd:
            np.testing.assert_array_equal(bwd[k], ref_bwd[k], err_msg=f"{vec} bwd {k}")


# -- 2. Kernel math verification -----------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _eval_kernels(c: wp.array[wp.vec3], s: Slots, eps: float, out_g: wp.array[wp.vec3],
                  out_h: wp.array[wp.mat33], out_m: wp.array[wp.mat33], out_e: wp.array[float],
                  out_frame: wp.array[wp.vec3]):
    i = wp.tid()
    g, _, _ = _barrier_force(c[i], s, i, eps)
    out_g[i] = g
    out_h[i] = contact_hessian(c[i], s, i, eps)
    out_m[i] = _majorizer(c[i], s, i, eps)
    n = s.normal[i]
    e1, e2, d1, d2 = _aniso_axes(n, s.aniso[i])
    out_frame[3 * i + 0] = e1
    out_frame[3 * i + 1] = e2
    out_frame[3 * i + 2] = n
    sl = c[i] - s.c0[i]
    sl = sl - wp.dot(n, sl) * n
    s1 = wp.dot(e1, sl)
    s2 = wp.dot(e2, sl)
    yt = wp.sqrt(d1 * d1 * s1 * s1 + d2 * d2 * s2 * s2)
    out_e[i] = s.mu[i] * s.rho[i] * slip_energy(yt, eps)[0]


def test_aniso_kernels_hessian_energy_majorizer_and_fallback(rng):
    """Verify ``_aniso_axes`` orthonormality (incl. degenerate fallback), ``contact_hessian`` vs FD of
    ``contact_force``, directional derivative of ``slip_energy(yt, eps)``, and ``_majorizer`` symmetry/PSD/cap."""
    N = 16
    s = _slots(N, None)
    s.active.fill_(1)
    normals = rng.normal(size=(N, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # Mix general 3D axes, zero vector, and normal-parallel vector to exercise both _aniso_axes branches.
    u = rng.normal(size=(N, 3))
    u[0] = 0.0
    u[1] = normals[1]
    u[2] = -2.0 * normals[2]
    normals[3] = np.array([1.0, 0.0, 0.0])  # trigger |n[0]| >= 0.9 ref fallback too
    u[3] = normals[3]
    gammas = np.array([0.4, -0.4, 0.8, -0.7, 0.35, -0.55, 1.1, -1.1,
                       0.45, -0.3, 3.0, -3.0, 3.0, -3.0, 0.5, -0.5], dtype=np.float32)
    aniso = np.column_stack([u, gammas]).astype(np.float32)
    th = np.full(N, 0.01, dtype=np.float32)
    stiff = np.full(N, 1.0e4, dtype=np.float32)
    stiff[10:14] = 1.0e7  # high stiffness + extreme gamma=+-3.0 for cap test
    mu = rng.uniform(0.2, 0.9, size=N).astype(np.float32)
    rho = rng.uniform(5.0, 50.0, size=N).astype(np.float32)
    rho[10:14] = 1.0e4
    lam = np.zeros(N, dtype=np.float32)
    pad = np.zeros((N, 4), dtype=np.float32)
    pad[8] = [1.0e10, 2.0, 0.005, 0.0]
    pad[9] = [1.0e10, 2.0, 0.005, 0.0]
    rewound = np.zeros(N, dtype=np.int32)
    rewound[6] = 1
    rewound[12] = 1

    # Place c0 inside smooth quadratic barrier (d0 = 0.9 * th) and construct sticking (yt < eps) and sliding (yt > eps)
    eps = 1.0e-3
    d0 = 0.9 * th
    c0 = d0[:, None] * normals
    c = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        n_i = normals[i]
        t_dir = rng.normal(size=3)
        t_dir -= np.dot(t_dir, n_i) * n_i
        t_dir /= np.linalg.norm(t_dir)
        # Alternate sticking (i even, except high-gamma checks) and sliding (i odd)
        mag = 2.5e-4 if (i % 2 == 0 and i < 10) else 4.0e-3
        if i in (10, 11):
            mag = 2.0e-4  # sticking at extreme gamma = +-3.0
        dn = -0.05 * th[i] if i % 3 else 0.02 * th[i]
        c[i] = c0[i] + dn * n_i + mag * t_dir

    s.normal.assign(normals.astype(np.float32))
    s.c0.assign(c0.astype(np.float32))
    s.thickness.assign(th)
    s.stiffness.assign(stiff)
    s.mu.assign(mu)
    s.rho.assign(rho)
    s.lam.assign(lam)
    s.pad.assign(pad)
    s.rewound.assign(rewound)
    s.aniso.assign(aniso)

    out_g = wp.zeros(N, dtype=wp.vec3)
    out_h = wp.zeros(N, dtype=wp.mat33)
    out_m = wp.zeros(N, dtype=wp.mat33)
    out_e = wp.zeros(N, dtype=float)
    out_frame = wp.zeros(3 * N, dtype=wp.vec3)
    wp.launch(_eval_kernels, dim=N, inputs=[wp.array(c, dtype=wp.vec3), s, eps],
              outputs=[out_g, out_h, out_m, out_e, out_frame])

    frames = out_frame.numpy().reshape(N, 3, 3)
    H_all = out_h.numpy()
    M_all = out_m.numpy()
    g_all = out_g.numpy()

    # 1. Orthonormal frame check & Majorizer PSD + cap check
    for i in range(N):
        e1, e2, n_i = frames[i]
        np.testing.assert_allclose(frames[i] @ frames[i].T, np.eye(3), atol=1.0e-6)
        np.testing.assert_allclose(np.cross(n_i, e1), e2, atol=1.0e-6)
        M = M_all[i].astype(np.float64)
        np.testing.assert_allclose(M, M.T, atol=1.0e-6 * max(np.linalg.norm(M), 1.0))
        eigs = np.linalg.eigvalsh(M)
        cap = float(CURVATURE) * float(stiff[i])
        assert eigs.min() >= -1.0e-5 * max(eigs.max(), 1.0), (i, eigs)
        assert eigs.max() <= cap * (1.0 + 1.0e-5), (i, eigs.max(), cap)

    # 2. contact_hessian vs central FD (Richardson) of contact_force, and energy directional derivative
    for i in range(N):
        if i >= 10:
            continue  # extreme stiffness tested for cap above

        def _hess_fd(h_step: float) -> np.ndarray:
            out_mat = np.zeros((3, 3), dtype=np.float64)
            for ax in range(3):
                dc = np.zeros((2, N, 3), dtype=np.float32)
                dc[0, i, ax] = h_step
                dc[1, i, ax] = -h_step
                cp, cm = (c + dc[0]).astype(np.float32), (c + dc[1]).astype(np.float32)
                gp, gm = wp.zeros(N, dtype=wp.vec3), wp.zeros(N, dtype=wp.vec3)
                wp.launch(_eval_kernels, dim=N, inputs=[wp.array(cp, dtype=wp.vec3), s, eps],
                          outputs=[gp, out_h, out_m, out_e, out_frame])
                wp.launch(_eval_kernels, dim=N, inputs=[wp.array(cm, dtype=wp.vec3), s, eps],
                          outputs=[gm, out_h, out_m, out_e, out_frame])
                out_mat[:, ax] = (gp.numpy()[i] - gm.numpy()[i]) / float(cp[i, ax] - cm[i, ax])
            return out_mat

        H_fd = (4.0 * _hess_fd(1.0e-5) - _hess_fd(2.0e-5)) / 3.0
        H_num = H_all[i].astype(np.float64)
        rel = np.linalg.norm(H_num - H_fd) / np.linalg.norm(H_num)
        assert rel < 1.0e-4, f"slot {i}: Hessian rel err {rel:.3e}"

        # Directional derivative of slip_energy along random tangent direction (slots 8-9 have 1e8 normal pad force)
        if i >= 8:
            continue
        d = rng.normal(size=3)
        d -= np.dot(d, normals[i]) * normals[i]
        d /= np.linalg.norm(d)

        def _dir_fd(h_step: float) -> float:
            dc = np.zeros((N, 3), dtype=np.float32)
            dc[i] = (h_step * d).astype(np.float32)
            cp, cm = (c + dc).astype(np.float32), (c - dc).astype(np.float32)
            ep, em = wp.zeros(N, dtype=float), wp.zeros(N, dtype=float)
            wp.launch(_eval_kernels, dim=N, inputs=[wp.array(cp, dtype=wp.vec3), s, eps],
                      outputs=[out_g, out_h, out_m, ep, out_frame])
            wp.launch(_eval_kernels, dim=N, inputs=[wp.array(cm, dtype=wp.vec3), s, eps],
                      outputs=[out_g, out_h, out_m, em, out_frame])
            step = float(np.dot(cp[i].astype(np.float64) - cm[i].astype(np.float64), d))
            return float(ep.numpy()[i] - em.numpy()[i]) / step

        c_v, f_v = _dir_fd(8.0e-5), _dir_fd(4.0e-5)
        de_fd = (4.0 * f_v - c_v) / 3.0
        est = abs(c_v - f_v)
        ft = g_all[i] - np.dot(g_all[i], normals[i]) * normals[i]
        de_ana = float(np.dot(ft, d))
        assert_derivative(de_ana, de_fd, est, 1.0e-5, f"slot-{i}-energy",
                          scale=max(float(np.linalg.norm(ft)), 1.0e-8))


# -- 3. Analytical steady-state transverse slip deflection & SO(3) covariance ----------------------------------------


@pytest.mark.parametrize("gamma", [0.4, pytest.param(-0.4, marks=pytest.mark.slow)])
def test_steady_state_slip_deflection(gamma, monkeypatch):
    """Driven rod slider on a plane under normal load and tangential force at angle ``psi = 30 deg`` to groove axis
    ``e1``: steady-state slip velocity angle matches ``arctan(exp(2*gamma) * tan(psi))`` within ``< 1.0 deg``."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-6)
    mu = 0.5
    model = _rod_on_ground(segments=1, z=RADIUS, mu=mu, gravity=(0.0, 0.0, -9.81))
    sim = SolverDiSMech(model, exact=True, iterations=50, self_contact=False)

    theta0 = np.deg2rad(20.0)
    e1 = np.array([np.cos(theta0), np.sin(theta0), 0.0])
    n = np.array([0.0, 0.0, 1.0])
    e2 = np.cross(n, e1)
    _attach_aniso_hook(sim, wp.vec4(*e1, gamma))

    psi = np.deg2rad(30.0)
    m_node = model.particle_mass.numpy().astype(np.float64)[0]
    f_down = 4.0 * m_node * 9.81  # extra normal load so velocity builds cleanly into sliding regime
    fn = m_node * 9.81 + f_down
    f_crit = mu * fn / np.sqrt(np.exp(-gamma) * np.cos(psi) ** 2 + np.exp(gamma) * np.sin(psi) ** 2)
    f_drv = 1.005 * f_crit * (np.cos(psi) * e1 + np.sin(psi) * e2)
    force = np.tile(f_drv - np.array([0.0, 0.0, f_down]), (model.particle_count, 1)).astype(np.float32)

    s0 = model.state()
    flatten_state(s0)
    s_end, _ = simulate(model, sim, 80, 5.0e-3, force=force, state=s0)
    v = s_end.particle_qd.numpy().mean(axis=0).astype(np.float64)
    speed = np.hypot(np.dot(v, e1), np.dot(v, e2))
    assert speed > 2.0 * sim.contact.slip_speed, f"did not enter sliding regime: speed={speed}"
    alpha_meas = np.arctan2(np.dot(v, e2), np.dot(v, e1))
    alpha_exact = np.arctan(np.exp(2.0 * gamma) * np.tan(psi))
    err_deg = float(np.rad2deg(abs(alpha_meas - alpha_exact)))
    print(f"\n[Deflection gamma={gamma:+.2f}] exact={np.rad2deg(alpha_exact):.3f} deg, "
          f"meas={np.rad2deg(alpha_meas):.3f} deg, err={err_deg:.3f} deg")
    assert err_deg < 1.0, f"slip deflection err {err_deg:.3f} deg >= 1.0 deg"


def test_aniso_so3_covariance(rng, monkeypatch):
    """Rotating the entire driven-slider scene + ``e1`` + velocity + force + gravity by ``R in SO(3)`` rotates the
    anisotropic trajectory covariantly (``< 1e-5``), while unrotated ``gamma = 0`` differs significantly."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-5)
    rot = Rotation.random(random_state=rng)
    R, q_rot = rot.as_matrix(), wp.quat(*rot.as_quat())
    mu, gamma = 0.4, 0.6
    e1_0 = np.array([np.cos(0.4), np.sin(0.4), 0.0])
    g0 = np.array([0.0, 0.0, -9.81])
    v0 = np.array([0.6, 0.3, 0.0])
    f0 = np.array([0.05, 0.03, 0.0])

    def run_scene(turn, g_val):
        Rm = R if turn else np.eye(3)
        qR = q_rot if turn else wp.quat_identity()
        b = newton.ModelBuilder(gravity=tuple(Rm @ g0))
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(*(Rm @ np.array([0.0, 0.0, -0.04]))), qR),
                        hx=0.5, hy=0.5, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
        pts = np.column_stack([np.linspace(-0.06, 0.06, 4), np.zeros(4), np.full(4, RADIUS)])
        base_rod = newton.Rod(pts, radius=RADIUS)
        quats = (rot * Rotation.from_quat(base_rod.quaternions)).as_quat() if turn else base_rod.quaternions
        SolverDiSMech.add_rod(b, rod=newton.Rod(pts @ Rm.T, quaternions=quats, radius=RADIUS),
                              cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu),
                              stretch_stiffness=5.0e3, bend_stiffness=0.2, twist_stiffness=0.1)
        m = b.finalize()
        sim = SolverDiSMech(m, exact=True, iterations=100, self_contact=False)
        if g_val != 0.0:
            _attach_aniso_hook(sim, wp.vec4(*(Rm @ e1_0), g_val))
        st = m.state()
        st.particle_qd.assign(np.tile(Rm @ v0, (m.particle_count, 1)).astype(np.float32))
        force = np.tile(Rm @ f0, (m.particle_count, 1)).astype(np.float32)
        end, _ = simulate(m, sim, 35, 4.0e-3, force=force, state=st)
        return end.particle_q.numpy().astype(np.float64) @ Rm

    q_base = run_scene(False, gamma)
    q_rot_out = run_scene(True, gamma)
    q_iso = run_scene(False, 0.0)
    err = float(np.linalg.norm(q_rot_out - q_base, axis=1).max())
    diff_iso = float(np.linalg.norm(q_base - q_iso, axis=1).max())
    assert diff_iso > 1.0e-3, f"anisotropy had too little effect: {diff_iso}"
    assert err < 1.0e-5, f"SO(3) covariance error {err:.3e} >= 1e-5"


# -- 4. StepAdjoint VJP vs fd.py over wp.Tape with aniso hook in detect -------------------------------------------


class _AnisoGroundRollout(GroundRollout):
    def __init__(self, rng, v_mean, checks, mu, aniso, segments=4, steps=3, z_factor=None):
        self.aniso = aniso
        self.segments = segments
        self.steps = steps
        if z_factor is not None:
            self.height = RADIUS * z_factor
        super().__init__(rng, v_mean, checks, mu)

    def parameters(self, model, solver) -> dict:
        p = super().parameters(model, solver)
        if self.segments == 1:
            p.pop("params", None)
            p.pop("rest", None)
        return p

    def setup(self, options: dict, grad: bool, X: np.ndarray | None = None):
        run = super().setup(options, grad, X)
        _attach_aniso_hook(run.solver, wp.vec4(*self.aniso))
        return run


class _AnisoSphereRollout(SphereRollout):
    def __init__(self, rng, mu, aniso):
        self.aniso = aniso
        super().__init__(rng, mu)

    def setup(self, options: dict, grad: bool, X: np.ndarray | None = None):
        run = super().setup(options, grad, X)
        _attach_aniso_hook(run.solver, wp.vec4(*self.aniso))
        return run


@pytest.mark.parametrize(
    "segments, steps, mu, v_mean, checks, z_factor",
    [
        (1, 1, 0.3, (0.3, 0.1, 0.0), ("velocity", "position"), 1.015),
        (4, 3, 0.3, (0.3, 0.1, 0.0), ("mass", "velocity", "position"), None),
        pytest.param(4, 3, 2.0, (0.0, 0.0, 0.0), ("velocity", "translation"), None, marks=pytest.mark.slow),
    ],
    ids=["1seg-1step-slide", "4seg-3step-slide", "4seg-3step-stick"],
)
@pytest.mark.parametrize("gamma", [0.4])
def test_aniso_ground_adjoint_matches_fd(rng, monkeypatch, segments, steps, mu, v_mean, checks, z_factor, gamma):
    """Single-step & multi-step ``StepAdjoint`` VJP with nonzero ``aniso`` via ``wp.Tape`` matches ``fd.py``."""
    monkeypatch.setattr(solver, "TOLERANCE", 5.0e-6 if segments == 1 else 1.0e-6)
    aniso = (0.6, 0.5, 0.4, gamma)  # non-planar u to exercise tangent projection on the tape
    r = _AnisoGroundRollout(rng, v_mean, checks, mu, aniso, segments=segments, steps=steps, z_factor=z_factor)
    r.check({"iterations": 500, "self_contact": False}, rtol=5.0e-3)


@pytest.mark.slow
def test_aniso_sphere_adjoint_matches_fd(rng, monkeypatch):
    """Turning normal (rod on sphere) ``StepAdjoint`` with nonzero ``aniso``."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-7)
    aniso = (0.6, 0.5, 0.4, 0.4)
    _AnisoSphereRollout(rng, 0.3, aniso).check({"iterations": 500, "self_contact": False}, rtol=5.0e-3)


# -- 5. Direct StepAdjoint VJP wrt contacts.aniso (e1 and gamma) in sticking & sliding ------------------------------


@pytest.mark.parametrize("segments", [4], ids=["4seg"])
@pytest.mark.parametrize("regime", ["slide", "stick"])
@pytest.mark.parametrize("gamma", [0.4])
def test_aniso_slot_vjp_matches_fd(rng, monkeypatch, segments, regime, gamma):
    """Direct ``solver.vjp(s0, s1, dt, contacts=snap)`` with ``snap.aniso.requires_grad = True`` matches central FD
    along ``e1`` direction, ``gamma``, and joint 4D perturbations in both sticking (``yt < eps``) and sliding
    (``yt > eps``) regimes."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-5 if segments == 1 else 5.0e-6)
    dt = 1.0e-2
    mu = 0.3 if regime == "slide" else 0.5
    v_xy = (0.35, 0.20) if regime == "slide" else (0.07, 0.04)
    length = 0.08 * segments
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    rod = newton.Rod.create_straight((-0.5 * length, 0.0, 1.0165 * RADIUS), (1.0, 0.0, 0.0), length,
                                     segment_count=segments, radius=RADIUS)
    SolverDiSMech.add_rod(b, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    model = b.finalize()
    n_nodes = model.particle_count
    v0 = np.tile([v_xy[0], v_xy[1], 0.0], (n_nodes, 1)).astype(np.float32)

    aniso0 = np.array([0.8, 0.6, 0.2, gamma], dtype=np.float64)  # nonzero z tests projection derivative
    rng_w = np.random.default_rng(rng.integers(0, 1 << 30))
    n_dof = 3 * n_nodes + model.dismech.edge_length.shape[0]
    w_q = rng_w.normal(size=n_dof)
    w_qd = rng_w.normal(size=n_dof)

    cur_aniso = [wp.vec4(*aniso0.astype(np.float32))]
    sim = SolverDiSMech(model, exact=True, iterations=500, self_contact=False)
    orig_fill = sim.contact.source.fill

    def fill_with_cur(state_in, dt_in, q, s):
        orig_fill(state_in, dt_in, q, s)
        wp.launch(_set_aniso_kernel, dim=s.active.shape[0], inputs=[cur_aniso[0], s], device=sim.device)

    sim.contact.source.fill = fill_with_cur

    def run_step(aniso_vec, grad=False):
        cur_aniso[0] = wp.vec4(*aniso_vec.astype(np.float32))
        s0 = model.state(requires_grad=grad)
        s1 = model.state(requires_grad=grad)
        flatten_state(s0)
        flatten_state(s1)
        s0.particle_qd.assign(v0)
        sim.reset(s0)
        sim.step(s0, s1, None, None, dt)
        assert int(sim.counts.numpy()[0]) == 0
        return sim, s0, s1

    sim, s0, s1 = run_step(aniso0, grad=True)
    snap = sim.contact.snapshot()
    assert snap is not None and snap.aniso.shape[0] == n_nodes

    # Verify we actually hit the intended slip regime (yt < eps for stick, yt > eps for slide)
    eps = sim.contact.slip
    n_np = snap.normal.numpy().astype(np.float64)
    u0 = aniso0[:3] - (aniso0[:3] @ n_np[0]) * n_np[0]
    e1 = u0 / np.linalg.norm(u0)
    e2 = np.cross(n_np[0], e1)
    d1, d2 = np.exp(0.5 * gamma), np.exp(-0.5 * gamma)
    sl = s1.particle_q.numpy()[0].astype(np.float64) - s0.particle_q.numpy()[0].astype(np.float64)
    sl = sl - np.dot(n_np[0], sl) * n_np[0]
    yt = np.hypot(d1 * np.dot(e1, sl), d2 * np.dot(e2, sl))
    assert (yt > eps) if regime == "slide" else (0.0 < yt < eps), (regime, yt, eps)

    snap.aniso.requires_grad = True
    s1.dismech.q.grad.assign(w_q.astype(np.float32))
    s1.dismech.qd.grad.assign(w_qd.astype(np.float32))
    sim.vjp(s0, s1, dt, contacts=snap)
    grad_aniso = snap.aniso.grad.numpy().astype(np.float64)  # (n_nodes, 4)

    def loss(X):
        out = np.empty(len(X), dtype=np.float64)
        for k, x in enumerate(X):
            _, _, s1_k = run_step(aniso0 + x)
            out[k] = (
                s1_k.dismech.q.numpy().astype(np.float64) @ w_q
                + s1_k.dismech.qd.numpy().astype(np.float64) @ w_qd
            )
        return out[:, None]

    # Check pure e1 direction, pure gamma, and joint 4D perturbation
    dirs = {
        "e1": np.array([0.4, -0.7, 0.3, 0.0]),
        "gamma": np.array([0.0, 0.0, 0.0, 1.0]),
        "joint": rng_w.normal(size=4),
    }
    for name, d_vec in dirs.items():
        h_fd = 8.0e-3 if regime == "stick" else 4.0e-3
        # Evaluate scalar directional FD along d_vec
        pts = np.array([h_fd * d_vec, -h_fd * d_vec, 0.5 * h_fd * d_vec, -0.5 * h_fd * d_vec])
        vals = loss(pts)[:, 0]
        coarse = (vals[0] - vals[1]) / (2.0 * h_fd)
        fine = (vals[2] - vals[3]) / h_fd
        fd_val = (4.0 * fine - coarse) / 3.0
        est = abs(coarse - fine)
        adjoint = float((grad_aniso @ d_vec).sum())
        scale = max(float(np.abs(grad_aniso).sum() * np.linalg.norm(d_vec)), abs(fd_val))
        assert_derivative(adjoint, fd_val, est, 5.0e-3, f"{segments}seg-{regime}-g{gamma}-{name}", scale=scale)


# -- 6. W=256 FrameGraph benchmark ---------------------------------------------------------------------------------


def test_aniso_framegraph_capture(device):
    """Batched worlds of sliding contact with nonzero ``s.aniso`` capture cleanly in ``FrameGraph`` and match eager."""
    if not wp.get_device(device).is_cuda:
        pytest.skip("FrameGraph capture test requires CUDA")

    worlds = 4
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod = newton.Rod.create_straight((-0.08, 0.0, 1.015 * RADIUS), (1.0, 0.0, 0.0), 0.16, segment_count=4,
                                     radius=RADIUS)
    SolverDiSMech.add_rod(sub, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3),
                          stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.3))
    for _ in range(worlds):
        b.add_world(sub)
    model = b.finalize(device=device)
    v0 = np.tile([0.4, 0.2, 0.0], (model.particle_count, 1)).astype(np.float32)
    f_drv = np.tile([0.06, 0.02, 0.0], (model.particle_count, 1)).astype(np.float32)
    dt, substeps = 0.005, 4

    def _run_fg(sim: SolverDiSMech):
        s0, s1 = model.state(), model.state()
        s0.particle_qd.assign(v0)
        s0.particle_f.assign(f_drv)
        s1.particle_f.assign(f_drv)

        def work():
            for _ in range(substeps // 2):
                sim.step(s0, s1, None, None, dt)
                sim.step(s1, s0, None, None, dt)

        fg = FrameGraph(work, sim, enabled=True)
        for _ in range(3):
            fg()
        assert fg.graph is not None
        # Replay once from fresh state and verify against eager execution
        s0_fresh = model.state()
        s0_fresh.particle_qd.assign(v0)
        flatten_state(s0_fresh)
        wp.copy(s0.particle_q, s0_fresh.particle_q)
        wp.copy(s0.particle_qd, s0_fresh.particle_qd)
        for attr in ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q"):
            wp.copy(getattr(s0.dismech, attr), getattr(s0_fresh.dismech, attr))
        sim.reset(s0)
        fg()
        s_eager, _ = simulate(model, SolverDiSMech(model, self_contact=False), substeps, dt, force=f_drv, state=s0_fresh)
        return s0.particle_q.numpy(), s_eager.particle_q.numpy()

    s_base = SolverDiSMech(model, self_contact=False)
    s_aniso = SolverDiSMech(model, self_contact=False)
    _attach_aniso_hook(s_aniso, wp.vec4(0.8, 0.6, 0.0, 0.4))

    q_base, _ = _run_fg(s_base)
    s0_a, s1_a = model.state(), model.state()
    s0_a.particle_qd.assign(v0)
    s0_a.particle_f.assign(f_drv)
    s1_a.particle_f.assign(f_drv)
    fg_a = FrameGraph(
        lambda: [s_aniso.step(s0_a, s1_a, None, None, dt), s_aniso.step(s1_a, s0_a, None, None, dt)],
        s_aniso,
        enabled=True,
    )
    for _ in range(3):
        fg_a()
    assert fg_a.graph is not None
    q_aniso = s0_a.particle_q.numpy()
    assert np.isfinite(q_base).all() and np.isfinite(q_aniso).all()
    assert np.max(np.abs(q_aniso - q_base)) > 1.0e-4

    # Verify captured graph replay matches eager execution for aniso
    s_eag0, s_eag1 = model.state(), model.state()
    s_eag0.particle_qd.assign(v0)
    flatten_state(s_eag0)
    s_eag0.particle_f.assign(f_drv)
    s_eag1.particle_f.assign(f_drv)
    s_aniso_eag = SolverDiSMech(model, self_contact=False)
    _attach_aniso_hook(s_aniso_eag, wp.vec4(0.8, 0.6, 0.0, 0.4))
    wp.copy(s0_a.particle_q, s_eag0.particle_q)
    wp.copy(s0_a.particle_qd, s_eag0.particle_qd)
    for attr in ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q"):
        wp.copy(getattr(s0_a.dismech, attr), getattr(s_eag0.dismech, attr))
    s_aniso.reset(s0_a)
    fg_a()
    s_aniso_eag.step(s_eag0, s_eag1, None, None, dt)
    s_aniso_eag.step(s_eag1, s_eag0, None, None, dt)
    np.testing.assert_allclose(s0_a.particle_q.numpy(), s_eag0.particle_q.numpy(), atol=1.0e-6)
