"""Physical validation of two-way DER/BAC <-> ImplicitMPM granular coupling via SolverCoupledProxy."""

import newton
import numpy as np
import warp as wp
from newton._src.solvers.coupled.solver_coupled import SolverCoupled
from newton._src.solvers.coupled.solver_coupled_proxy import SolverCoupledProxy
from newton.solvers import SolverImplicitMPM

from dismech_newton import SolverDiSMech, flatten_state
from dismech_newton.coupling import mpm_collider_inertia

GRAVITY = 9.81


def _sand_box(builder: newton.ModelBuilder, lx: float, ly: float, lz: float, *, vs: float = 0.014,
              ppc: float = 2.0, rho_s: float = 1600.0, mu_s: float = 0.65, youngs: float = 5.0e5,
              walls: bool = True, z0: float = 0.0) -> tuple[list[int], float]:
    """Emit dry Drucker-Prager sand particles into ``builder`` (first!) and optional containment walls."""
    sp = vs / ppc
    xs = np.arange(-0.5 * lx + 0.5 * sp, 0.5 * lx, sp)
    ys = np.arange(-0.5 * ly + 0.5 * sp, 0.5 * ly, sp)
    zs = np.arange(z0 + 0.5 * sp, z0 + lz, sp)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n_p = len(pos)
    p0 = builder.particle_count
    p_mass = float(sp**3 * rho_s)
    builder.add_particles(
        pos=pos.tolist(),
        vel=[[0.0, 0.0, 0.0]] * n_p,
        mass=[p_mass] * n_p,
        radius=[0.4 * vs] * n_p,
        custom_attributes={
            "mpm:friction": [mu_s] * n_p,
            "mpm:young_modulus": [youngs] * n_p,
            "mpm:poisson_ratio": [0.3] * n_p,
        },
    )
    if walls:
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
        wcfg = newton.ModelBuilder.ShapeConfig(mu=0.4)
        for px, py, sx, sy in (
            (0.5 * lx + 0.02, 0.0, 0.02, 0.5 * ly + 0.02),
            (-0.5 * lx - 0.02, 0.0, 0.02, 0.5 * ly + 0.02),
            (0.0, 0.5 * ly + 0.02, 0.5 * lx + 0.02, 0.02),
            (0.0, -0.5 * ly - 0.02, 0.5 * lx + 0.02, 0.02),
        ):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(px, py, lz), wp.quat_identity()),
                hx=sx,
                hy=sy,
                hz=lz,
                cfg=wcfg,
            )
    return list(range(p0, builder.particle_count)), p_mass


def _run_rod_drag(H: float, *, ei: float, bend_damping: float = 0.0, vs: float = 0.014,
                  lz: float = 0.14, vx: float = 0.10, steps: int = 120) -> tuple[float, np.ndarray, np.ndarray]:
    """Drag a top-clamped vertical DER rod of diameter D=12 mm immersed to depth ``H`` in dry sand.

    Returns ``(mean_drag_Fx, z_nodes_rest, steady_x_displacement)``.
    """
    b = newton.ModelBuilder()
    SolverDiSMech.register_custom_attributes(b)
    SolverImplicitMPM.register_custom_attributes(b)
    rho_s = 1600.0
    mpm_p, _ = _sand_box(b, lx=0.32, ly=0.18, lz=lz, vs=vs, ppc=2.0, rho_s=rho_s, mu_s=0.65, walls=True)

    d_rod, n_seg = 0.012, 10
    z_top = lz + 0.03
    z_bot = lz - H
    zs_r = np.linspace(z_top, z_bot, n_seg + 1)
    rpos = np.column_stack([np.full(n_seg + 1, -0.07), np.zeros(n_seg + 1), zs_r])
    p0 = b.particle_count
    h_seg = (z_top - z_bot) / n_seg
    rod = newton.Rod(points=rpos.tolist(), radius=0.5 * d_rod)
    rod_b, _ = SolverDiSMech.add_rod(
        b,
        rod=rod,
        stretch_stiffness=1.0e4,
        bend_stiffness=ei / h_seg,
        twist_stiffness=ei / h_seg,
        bend_damping=bend_damping,
    )
    rod_p = list(range(p0, b.particle_count))
    SolverDiSMech.fix_segment(b, 0)
    SolverDiSMech.fix_segment(b, 1)
    m = b.finalize()

    cfg = SolverImplicitMPM.Config()
    cfg.voxel_size = vs
    cfg.max_iterations = 25
    coupled = SolverCoupledProxy(
        m,
        entries=[
            SolverCoupled.Entry(name="soft", solver=lambda v: SolverDiSMech(v, refactor=False), particles=rod_p,
                                bodies=rod_b),
            SolverCoupled.Entry(name="mpm", solver=lambda v: SolverImplicitMPM(v, config=cfg), particles=mpm_p,
                                in_place=True),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="soft", destination="mpm", bodies=rod_b, mode="lagged",
                                              collision_pipeline=lambda _m: None)],
            iterations=1,
        ),
    )
    m_mpm = coupled.view("mpm")
    bm, binv = mpm_collider_inertia(m, coupled._entries["mpm"].body_local_to_global)
    body_ids = list(range(m_mpm.body_count))
    coupled._entries["mpm"].solver.setup_collider(
        collider_body_ids=body_ids + [-1],
        collider_margins=[0.002] * (len(body_ids) + 1),
        model=m_mpm,
        body_mass=bm,
        body_inv_inertia=binv,
    )

    s0, s1 = m.state(), m.state()
    flatten_state(s0)
    flatten_state(s1)
    dt = 1.0 / 180.0
    root_x0 = rpos[:3, 0].copy()
    fs, defls = [], []
    settle = 18
    for k in range(steps):
        if k >= settle:
            q = s0.dismech.q.numpy()
            dx = vx * dt * (k - settle + 1)
            q[3 * p0 : 3 * (p0 + 3) : 3] = root_x0 + dx
            s0.dismech.q.assign(q)
        coupled.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        w = coupled._proxy_mappings[0].coupling_forces.numpy()
        fs.append(float(-w[rod_b, 0].sum()))
        q_cur = s0.dismech.q.numpy()[3 * p0 : 3 * (p0 + n_seg + 1)].reshape(-1, 3)
        defls.append(q_cur[0, 0] - q_cur[:, 0])
    steady = slice(max(settle + 30, steps // 2), steps)
    return float(np.mean(fs[steady])), zs_r, np.mean(defls[steady], axis=0)


def _beam_deflection(zs: np.ndarray, *, lz: float, kh: float, ei: float, d_rod: float = 0.012,
                     rho_s: float = 1600.0) -> np.ndarray:
    """Exact Euler-Bernoulli deflection ``w(s)`` for ``EI w''''(s) = K_h rho_s g D max(s - s_surf, 0)``."""
    z_clamp = 0.5 * (zs[1] + zs[2])  # continuum clamp sits between fixed segment 1's nodes (zs[1], zs[2])
    s = np.maximum(z_clamp - zs, 0.0)
    s_surf = max(z_clamp - lz, 0.0)
    L = float(z_clamp - zs[-1])
    k = kh * rho_s * GRAVITY * d_rod
    # Exact closed-form integral of M(eta) = int_{max(eta, s_surf)}^L (xi - eta) * k * (xi - s_surf) dxi
    xi = np.linspace(0.0, L, 401)
    q_load = k * np.maximum(xi - s_surf, 0.0)
    shear = np.array([np.trapezoid(q_load[i:], xi[i:]) for i in range(len(xi))])
    moment = np.array([np.trapezoid(shear[i:], xi[i:]) for i in range(len(xi))])
    slope = np.array([np.trapezoid(moment[: i + 1] / ei, xi[: i + 1]) for i in range(len(xi))])
    w_xi = np.array([np.trapezoid(slope[: i + 1], xi[: i + 1]) for i in range(len(xi))])
    return np.interp(s, xi, w_xi)


def test_free_rod_sand_conservation():
    """Zero-gravity free DER rod impacting a free sand block conserves 6-DOF linear and angular momentum."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverDiSMech.register_custom_attributes(b)
    SolverImplicitMPM.register_custom_attributes(b)
    vs = 0.012
    mpm_p, _ = _sand_box(b, lx=0.14, ly=0.14, lz=0.12, vs=vs, ppc=2.0, rho_s=1200.0, mu_s=0.5, walls=False, z0=0.0)

    rpos = np.array([[-0.03, -0.02, 0.16], [0.02, 0.01, 0.155], [0.07, 0.04, 0.15]])
    p0 = b.particle_count
    rod = newton.Rod(points=rpos.tolist(), radius=0.01)
    rod_b, _ = SolverDiSMech.add_rod(b, rod=rod, stretch_stiffness=5.0e3, bend_stiffness=5.0)
    rod_p = list(range(p0, b.particle_count))
    m = b.finalize()

    cfg = SolverImplicitMPM.Config()
    cfg.voxel_size = vs
    cfg.max_iterations = 50
    cfg.tolerance = 1.0e-7
    coupled = SolverCoupledProxy(
        m,
        entries=[
            SolverCoupled.Entry(name="soft", solver=lambda v: SolverDiSMech(v, refactor=False), particles=rod_p,
                                bodies=rod_b),
            SolverCoupled.Entry(name="mpm", solver=lambda v: SolverImplicitMPM(v, config=cfg), particles=mpm_p,
                                in_place=True),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="soft", destination="mpm", bodies=rod_b, mode="lagged",
                                              mass_scale=1.0, collision_pipeline=lambda _m: None)],
            iterations=1,
        ),
    )
    m_mpm = coupled.view("mpm")
    bm, binv = mpm_collider_inertia(m, coupled._entries["mpm"].body_local_to_global)
    coupled._entries["mpm"].solver.setup_collider(
        collider_body_ids=list(range(m_mpm.body_count)), model=m_mpm, body_mass=bm, body_inv_inertia=binv
    )

    s0, s1 = m.state(), m.state()
    flatten_state(s0)
    flatten_state(s1)
    qd = s0.particle_qd.numpy()
    qd[rod_p] = [0.25, -0.1, -0.9]
    s0.particle_qd.assign(qd)

    pm = m.particle_mass.numpy()
    q_init = s0.particle_q.numpy().astype(np.float64)
    com0 = (pm[:, None] * q_init).sum(0) / pm.sum()

    def momenta(state):
        q = state.particle_q.numpy().astype(np.float64) - com0
        v = state.particle_qd.numpy().astype(np.float64)
        P_mpm = (pm[mpm_p, None] * v[mpm_p]).sum(0)
        P_rod = (pm[rod_p, None] * v[rod_p]).sum(0)
        L_mpm = np.cross(q[mpm_p], pm[mpm_p, None] * v[mpm_p]).sum(0)
        L_rod = np.cross(q[rod_p], pm[rod_p, None] * v[rod_p]).sum(0)
        return P_mpm + P_rod, L_mpm + L_rod, P_rod, L_rod

    P0, L0, Prod0, Lrod0 = momenta(s0)
    dt = 1.0 / 120.0
    imp_f = np.zeros(3)
    max_dP, max_P_imp = 0.0, 0.0
    for _ in range(15):
        coupled.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        w = coupled._proxy_mappings[0].coupling_forces.numpy()
        imp_f += w[rod_b, :3].sum(0) * dt
        Pt, _, Prod, _ = momenta(s0)
        max_dP = max(max_dP, float(np.linalg.norm(Pt - P0)))
        max_P_imp = max(max_P_imp, float(np.linalg.norm(Prod - Prod0)))

    Pt, Lt, Prod, Lrod = momenta(s0)
    wrench_err = float(np.linalg.norm((Prod - Prod0) - imp_f) / np.linalg.norm(Prod - Prod0))
    dL_rel = float(np.linalg.norm(Lt - L0) / max(np.linalg.norm(Lrod - Lrod0), 1.0e-9))
    assert max_P_imp > 0.02, f"rod did not collide noticeably with sand block (dP={max_P_imp:.4e})"
    assert max_dP / max_P_imp < 1.0e-3, f"linear momentum drift {max_dP / max_P_imp:.4e}"
    assert wrench_err < 1.0e-3, f"harvested wrench impulse mismatch {wrench_err:.4e}"
    assert dL_rel < 0.05, f"angular momentum drift {dL_rel:.4e}"


def test_granular_rft_depth_scaling():
    """Steady stiff-rod drag in dry Drucker-Prager sand scales quadratically as F_x ~ 0.5 K_h rho g D H^2."""
    depths = np.array([0.05, 0.07, 0.09])
    drags = np.array([_run_rod_drag(float(h), ei=1.0)[0] for h in depths])
    assert np.all(np.diff(drags) > 0.0), f"drag not strictly increasing with depth: {drags}"
    rho_s, d_rod = 1600.0, 0.012
    kh = drags / (0.5 * rho_s * GRAVITY * d_rod * depths**2)
    slope = float(np.polyfit(np.log(depths), np.log(drags), 1)[0])
    assert 1.5 < slope < 2.5, f"RFT depth scaling exponent {slope:.2f} outside [1.5, 2.5], drags={drags}"
    cv = float(kh.std() / kh.mean())
    assert kh.mean() > 1.0 and cv < 0.25, f"inconsistent Kh across depths: Kh={kh}, CV={cv:.3f}"


def test_elastogranular_beam_deflection():
    """Flexible DER rod deflection in sand matches Euler-Bernoulli EI w''''(s) = K_h rho g D (H - s)_+."""
    h_imm = 0.08
    lz = 0.14
    f_stiff, zs, _ = _run_rod_drag(h_imm, ei=1.0, lz=lz, vx=0.08, steps=120)
    kh = f_stiff / (0.5 * 1600.0 * GRAVITY * 0.012 * h_imm**2)

    ei_flex = 0.02
    _, _, w_sim = _run_rod_drag(h_imm, ei=ei_flex, bend_damping=0.02, lz=lz, vx=0.08, steps=130)
    w_theory = _beam_deflection(zs, lz=lz, kh=kh, ei=ei_flex)

    assert w_sim[-1] > 4.0e-3, f"tip deflection too small to validate flexure: {w_sim[-1]:.4e} m"
    rel_tip = abs(w_sim[-1] - w_theory[-1]) / w_theory[-1]
    rel_l2 = float(np.linalg.norm(w_sim - w_theory) / np.linalg.norm(w_theory))
    assert rel_tip < 0.20, f"tip deflection sim={w_sim[-1]*1e3:.2f} mm vs theory={w_theory[-1]*1e3:.2f} mm"
    assert rel_l2 < 0.20, f"profile L2 error {rel_l2:.3f}"
