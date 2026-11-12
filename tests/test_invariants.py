"""First-principles invariants: conservation of linear/angular momentum and midpoint Hamiltonian energy, SE(3) frame
invariance, mesh index/winding permutation invariance, multi-world isolation, Newton's third law on body reactions,
CPU/CUDA parity, and solver reset/restart/model-update lifecycles."""

import newton
import numpy as np
import pytest
import warp as wp
from rollouts import ClampedRollout, cloth, energies, grid, simulate
from scipy.spatial.transform import Rotation

from dismech_newton import SolverDiSMech, flatten_state
from dismech_newton import solver as solver_module
from dismech_newton.state import dof_constants


def _free(kind: str):
    """A straight rod or a flat sheet, no gravity, damping or contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if kind == "rod":
        rod = newton.Rod.create_straight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5, segment_count=10, radius=0.01)
        SolverDiSMech.add_rod(builder, rod=rod, stretch_stiffness=1.0e3, bend_stiffness=0.05, twist_stiffness=0.05)
    else:
        v, f = grid(5, 4, 0.3, 0.2)
        cloth(builder, v, f, density=0.5, tri_ke=2.0e3, tri_ka=1.0e3, edge_ke=1.0e-2, damping=0.0)
    return builder.finalize()


@pytest.mark.parametrize(
    "kind, options",
    [("rod", {}), ("rod", {"exact": True, "iterations": 30}), ("sheet", {}),
     ("sheet", {"exact": True, "iterations": 30})],
    ids=["rod-admm", "rod-exact", "sheet-admm", "sheet-exact"],
)
def test_free_body_keeps_momentum_and_loses_energy(rng, kind, options):
    """Thrown at random from rest shape (no potential energy) for 100 steps: internal forces cancel so linear momentum
    stays to float32 round-off, total mechanical energy ``K + V_el`` never exceeds the start's, and a free sheet's
    angular momentum about the origin stays conserved to backward-Euler discretization order."""
    model = _free(kind)
    solver = SolverDiSMech(model, self_contact=False, **options)
    mass = dof_constants(model)[0].numpy().astype(np.float64)
    pm = model.particle_mass.numpy().astype(np.float64)[:, None]
    nd = 3 * model.particle_count
    s0 = model.state()
    flatten_state(s0)
    qd0 = 0.3 * rng.normal(size=mass.size)
    s0.dismech.qd.assign(qd0)
    x0 = s0.particle_q.numpy().astype(np.float64)
    p0 = (mass[:nd] * qd0[:nd]).reshape(-1, 3).sum(0)
    L0 = (pm * np.cross(x0, qd0[:nd].reshape(-1, 3))).sum(0)
    L_scale = float((pm * np.linalg.norm(x0, axis=1, keepdims=True)
                     * np.linalg.norm(qd0[:nd].reshape(-1, 3), axis=1, keepdims=True)).sum())
    e0 = sum(energies(model, solver, s0))

    def check(_, s):
        qd = s.dismech.qd.numpy().astype(np.float64)
        p = (mass[:nd] * qd[:nd]).reshape(-1, 3).sum(0)
        assert np.linalg.norm(p - p0) < 1.0e-3 * np.linalg.norm(p0), (p, p0)
        k_e, el_e, _ = energies(model, solver, s)
        assert k_e <= 1.002 * e0 and k_e + el_e <= 1.002 * e0
        if kind == "sheet":
            x = s.particle_q.numpy().astype(np.float64)
            L = (pm * np.cross(x, qd[:nd].reshape(-1, 3))).sum(0)
            tol = 3.0e-3 if options.get("exact") else 6.0e-2
            assert np.linalg.norm(L - L0) < tol * L_scale, (L, L0)

    simulate(model, solver, 100, 0.01, state=s0, on_step=check)


def test_midpoint_conserves_total_energy():
    """An undamped 3D helical cantilever swinging nonlinearly under gravity with ``theta = 0.5, exact = True``: kinetic
     and potential energy trade over a full swing while total Hamiltonian ``K + V_el + V_g`` stays constant to <0.5% of
    the swing."""
    t = np.linspace(0.0, np.pi, 21)
    pts = np.column_stack([0.2 * np.sin(t), 0.2 * (1.0 - np.cos(t)), 0.05 * t])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    bodies, _ = SolverDiSMech.add_rod(builder, rod=newton.Rod(pts, radius=0.008), stretch_stiffness=4.0e4,
                                      bend_stiffness=5.0, twist_stiffness=4.0)
    SolverDiSMech.fix_segment(builder, bodies[0])
    model = builder.finalize()
    solver = SolverDiSMech(model, exact=True, theta=0.5, iterations=30, self_contact=False)
    s0 = model.state()
    e0 = sum(energies(model, solver, s0))
    kin, drift = [], []

    def check(_, s):
        k, el, ug = energies(model, solver, s)
        kin.append(k)
        drift.append(abs((k + el + ug) - e0))

    simulate(model, solver, 200, 2.0e-3, state=s0, on_step=check)
    swing = max(kin)
    assert swing > 5.0e-3 and min(kin[len(kin) // 3:]) < 0.25 * swing
    assert max(drift) < 5.0e-3 * swing, (max(drift), swing)


@pytest.mark.parametrize("scene", ["rod_junction", "woven_shell", "sliding_contact", "proxy_pin"])
def test_se3_frame_invariance(rng, monkeypatch, scene):
    """Rotating and translating an entire scene by ``(R, offset) in SE(3)``—including 3D rod junctions, anisotropic
    shells, frictional shape contact, and ball-pinned 6-DOF proxy bodies—rotates and translates the trajectory."""
    monkeypatch.setattr(solver_module, "TOLERANCE", 1.0e-6)
    rot = Rotation.random(random_state=rng)
    R, q_rot = rot.as_matrix(), wp.quat(*rot.as_quat())
    offset = rng.uniform(-0.35, 0.35, size=3)
    g0 = np.array([1.5, -2.5, -9.81])
    v_base, f_base = grid(6, 5, 0.24, 0.18, rng, jitter=0.2)
    uv = v_base[:, :2].copy()
    out = []
    for turn in (False, True):
        Rm = R if turn else np.eye(3)
        sh = offset if turn else np.zeros(3)
        qR = q_rot if turn else wp.quat_identity()
        b = newton.ModelBuilder(gravity=tuple(Rm @ g0))
        if scene == "rod_junction":
            dirs = np.array([[1.0, 0.0, 0.0], [-0.5, 0.75**0.5, 0.2], [-0.5, -(0.75**0.5), -0.15]])
            pts = np.vstack([[0.0, 0.0, 0.2], *[k * 0.06 * d + [0, 0, 0.2] for d in dirs for k in range(1, 5)]])
            edges = [(0 if k == 0 else 1 + 4 * br + k - 1, 1 + 4 * br + k) for br in range(3) for k in range(4)]
            base_rod = newton.Rod(pts, edges=edges, radius=0.006)
            quats = (rot * Rotation.from_quat(base_rod.quaternions)).as_quat() if turn else base_rod.quaternions
            SolverDiSMech.add_rod(b, rod=newton.Rod(pts @ Rm.T + sh, edges=edges, quaternions=quats, radius=0.006),
                                  stretch_stiffness=1.0e4, bend_stiffness=0.4, twist_stiffness=0.3)
            SolverDiSMech.fix_segment(b, edge=0)
        elif scene == "woven_shell":
            ids = cloth(b, v_base @ Rm.T + sh, f_base, density=0.3, aniso=(1.5e3, 4.0e2, 1.0e2),
                        bend_aniso=(2.0e-3, 6.0e-4, 2.0e-4), uv=uv, damping=0.02)
            for k in np.flatnonzero(v_base[:, 0] < 1.0e-9):
                b.particle_flags[ids[k]] &= ~newton.ParticleFlags.ACTIVE
        elif scene == "sliding_contact":
            b.add_shape_box(-1, xform=wp.transform(wp.vec3(*(Rm @ [0.0, 0.0, -0.04] + sh)), qR), hx=0.5, hy=0.5,
                            hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(mu=0.4))
            pts = np.column_stack([np.linspace(-0.1, 0.1, 7), np.zeros(7), np.full(7, 0.008)])
            SolverDiSMech.add_rod(b, rod=newton.Rod(pts @ Rm.T + sh, radius=0.008), stretch_stiffness=5.0e3,
                                  bend_stiffness=0.05, twist_stiffness=0.05,
                                  cfg=newton.ModelBuilder.ShapeConfig(density=800.0, mu=0.4))
        else:
            body = b.add_body(xform=wp.transform(wp.vec3(*sh), qR))
            b.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03,
                            cfg=newton.ModelBuilder.ShapeConfig(density=400.0, mu=0.3))
            pts = np.column_stack([np.linspace(0.06, 0.30, 5), np.zeros(5), np.zeros(5)])
            caps, _ = SolverDiSMech.add_rod(b, rod=newton.Rod(pts @ Rm.T + sh, radius=0.006), stretch_stiffness=5.0e3,
                                            bend_stiffness=0.2, twist_stiffness=0.15)
            b.add_joint_ball(parent=body, child=caps[0], parent_xform=wp.transform(wp.vec3(0.06, 0.0, 0.0),
                             wp.quat_identity()), child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.03),
                             wp.quat_identity()))
        model = b.finalize()
        if scene == "proxy_pin":
            flags = model.body_flags.numpy()
            flags[0] |= int(newton.BodyFlags.PROXY)
            model.body_flags.assign(flags)
        s0 = model.state()
        if scene == "sliding_contact":
            s0.particle_qd.assign(np.tile(Rm @ [0.4, 0.15, 0.0], (model.particle_count, 1)).astype(np.float32))
        s_end, _ = simulate(model, SolverDiSMech(model, iterations=25), 35, 5.0e-3, state=s0)
        out.append((s_end.particle_q.numpy().astype(np.float64) - sh) @ Rm)
    motion = np.abs(out[0] - (model.particle_q.numpy().astype(np.float64) - offset) @ R).max()
    assert np.linalg.norm(out[1] - out[0], axis=1).max() < 3.0e-4 * max(motion, 0.05)


def test_shell_index_and_winding_invariance(rng, monkeypatch):
    """Permuting vertex indices, shuffling triangle order, and cyclically shifting each triangle's 3 vertices
    ``(i, j, k) -> (j, k, i)`` leaves the physical midedge shell trajectory unchanged to float32 round-off."""
    monkeypatch.setattr(solver_module, "TOLERANCE", 1.0e-6)
    v, f = grid(7, 6, 0.25, 0.2, rng, jitter=0.25)
    v[:, 2] = 0.05 * np.sin(5.0 * v[:, 0])
    clamped = v[:, 0] < 1.0e-9
    perm = rng.permutation(len(v))
    inv = np.argsort(perm)
    f_shuf = inv[f][rng.permutation(len(f))]
    f_shuf = np.array([np.roll(tri, int(s)) for tri, s in zip(f_shuf, rng.integers(0, 3, size=len(f)))])
    runs = []
    for verts, faces, mask in ((v, f, clamped), (v[perm], f_shuf, clamped[perm])):
        b = newton.ModelBuilder(gravity=(0.0, 1.5, -9.81))
        ids = cloth(b, verts, faces, density=0.4, tri_ke=1.5e3, tri_ka=8.0e2, bend_stiffness=1.5e-3, damping=0.02)
        for k in np.flatnonzero(mask):
            b.particle_flags[ids[k]] &= ~newton.ParticleFlags.ACTIVE
        m = b.finalize()
        runs.append(simulate(m, SolverDiSMech(m, exact=True, iterations=25), 40, 0.01)[0].particle_q.numpy())
    sag = np.abs(runs[0] - v).max()
    assert sag > 0.02
    assert np.abs(runs[1][inv] - runs[0]).max() < 5.0e-5 * sag


def test_multiworld_isolation():
    """Heterogeneous worlds (a rod on a box in world 0, a shell on a sphere in world 1) stepped together in one batched
    model match their standalone single-world trajectories."""
    def make_world(w_idx: int):
        w = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        if w_idx == 0:
            w.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.02), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.02,
                            cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
            pts = np.column_stack([np.linspace(-0.12, 0.12, 9), np.zeros(9), np.full(9, 0.015)])
            SolverDiSMech.add_rod(w, rod=newton.Rod(pts, radius=0.006), stretch_stiffness=5.0e3, bend_stiffness=0.04,
                                  twist_stiffness=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=700.0, mu=0.7))
        else:
            w.add_shape_sphere(-1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), radius=0.06,
                               cfg=newton.ModelBuilder.ShapeConfig(mu=0.2))
            v, f = grid(6, 6, 0.18, 0.18)
            v += [-0.09, -0.09, 0.025]
            cloth(w, v, f, density=0.3, tri_ke=1.0e3, tri_ka=5.0e2, edge_ke=2.0e-2, particle_radius=0.004, damping=0.05)
        return w

    solo = []
    for idx in (0, 1):
        m = make_world(idx).finalize()
        m.particle_mu = 0.2
        solo.append(simulate(m, SolverDiSMech(m, iterations=20), 50, 4.0e-3)[0].particle_q.numpy())
    comb = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    comb.add_world(make_world(0))
    comb.add_world(make_world(1))
    mc = comb.finalize()
    mc.particle_mu = 0.2
    xc = simulate(mc, SolverDiSMech(mc, iterations=20), 50, 4.0e-3)[0].particle_q.numpy()
    pw = mc.particle_world.numpy()
    for idx in (0, 1):
        assert np.abs(xc[pw == idx] - solo[idx]).max() < 2.0e-4


@pytest.mark.parametrize("kind", ["rod", "shell"])
def test_body_reaction_matches_momentum_change(kind, monkeypatch):
    """Newton's third law on ``solver.body_reaction``: a rod or shell impacting and sliding on a tilted kinematic box
    exerts a world force on the body equal and opposite to ``dp_soft / dt - M_soft g`` at every step."""
    monkeypatch.setattr(solver_module, "TOLERANCE", 1.0e-4)
    g = np.array([0.0, 0.0, -9.81])
    b = newton.ModelBuilder(gravity=tuple(g))
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.25)
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.04), tilt), is_kinematic=True)
    b.add_shape_box(body, hx=0.4, hy=0.4, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(mu=0.4))
    if kind == "rod":
        xs = np.linspace(-0.08, 0.08, 9)
        pts = np.column_stack([xs, np.zeros(9), 0.015 - xs * np.tan(0.25)])
        SolverDiSMech.add_rod(b, rod=newton.Rod(pts, radius=0.006), stretch_stiffness=2.0e3, bend_stiffness=0.05,
                              twist_stiffness=0.05, bend_damping=1.0e-3, twist_damping=1.0e-3,
                              cfg=newton.ModelBuilder.ShapeConfig(density=800.0, mu=0.4))
    else:
        v, f = grid(5, 5, 0.12, 0.12)
        v += [-0.06, -0.06, 0.025]
        cloth(b, v, f, density=0.4, tri_ke=1.0e3, tri_ka=5.0e2, edge_ke=1.0e-2, particle_radius=0.004, damping=0.02)
    model = b.finalize()
    model.particle_mu = 0.4
    solver = SolverDiSMech(model, exact=True, iterations=200)
    pm = model.particle_mass.numpy().astype(np.float64)[:, None]
    weight = float(pm.sum() * 9.81)
    dt = 2.0e-3
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    peak = 0.0
    for _ in range(60):
        v_in = s0.particle_qd.numpy().astype(np.float64).copy()
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        assert int(solver.counts.numpy()[0]) == 0
        v_out = s0.particle_qd.numpy().astype(np.float64)
        f_body = solver.body_reaction.numpy()[body, :3].astype(np.float64)
        f_expected = pm.sum() * g - (pm * (v_out - v_in) / dt).sum(0)
        peak = max(peak, np.linalg.norm(f_body))
        assert np.linalg.norm(f_body - f_expected) < 5.0e-3 * max(weight, np.linalg.norm(f_expected))
    assert peak > 1.2 * weight


@pytest.mark.usefixtures("needs_cuda")
def test_devices_agree(rng):
    """A clamped rod's rollout (Gauss-Newton to convergence) on the CPU and on CUDA: the same to float32 round-off
    (1.8e-7 on 2026-10-27), so a device's kernels cannot go wrong alone."""
    out = []
    seed = rng.integers(1 << 31)
    for device in ("cpu", "cuda"):
        with wp.ScopedDevice(device):
            r = ClampedRollout(np.random.default_rng(seed))
            run = r.setup({"exact": True, "iterations": 30, "self_contact": False}, grad=False)
            r.simulate(run)
            out.append(np.concatenate([s.dismech.q.numpy() for s in run.states[1:]]))
    assert np.abs(out[0] - out[1]).max() < 1.0e-5 * np.abs(out[1]).max()


def _helices():
    """Two worlds, a damped helical rod each (different stiffness), dropped 1 cm onto the ground."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    t = np.linspace(0.0, 3.0 * np.pi, 25)
    pts = np.column_stack([0.05 * np.cos(t), 0.05 * np.sin(t), 0.012 + 0.002 * t])
    for ei in (2.0e-3, 5.0e-3):
        w = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverDiSMech.add_rod(w, rod=newton.Rod(pts, radius=0.004), stretch_stiffness=1.0e3, bend_stiffness=ei,
                              twist_stiffness=ei, bend_damping=0.1 * ei, stretch_damping=1.0,
                              cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
        b.add_world(w)
    b.add_ground_plane()
    return b.finalize()


def _rollout(model, steps, reset_at=None, worlds=()):
    solver = SolverDiSMech(model, iterations=30)
    s0 = model.state()
    world = model.particle_world.numpy()
    sel = np.isin(world, worlds)
    mask = wp.array([w in worlds for w in range(model.world_count + 1)], dtype=wp.bool)
    s1 = model.state()
    flatten_state(s0)
    flatten_state(s1)
    for i in range(steps):
        if i == reset_at:
            x, v = s0.particle_q.numpy(), s0.particle_qd.numpy()
            x[sel], v[sel] = model.particle_q.numpy()[sel], 0.0
            s0.particle_q.assign(x)
            s0.particle_qd.assign(v)
            solver.reset(s0, mask)
        solver.step(s0, s1, None, None, 0.005)
        s0, s1 = s1, s0
    return s0.dismech.q.numpy(), world


def test_masked_reset_replays_a_fresh_start():
    """Helices dropped on the ground for 40 steps (world 1 moves 1.9 cm, its twist 0.08 rad), then reset by
    ``world_mask``, then 40 more steps. Both worlds reset: the same as 40 steps from the start. World 1 alone: it
    replays the fresh start and world 0 goes on unaffected."""
    model = _helices()
    fresh, world = _rollout(model, 40)
    plain, _ = _rollout(model, 80)
    both, _ = _rollout(model, 80, reset_at=40, worlds=(0, 1))
    nodes = np.repeat(world, 3)
    nd = nodes.size
    assert np.abs(both - fresh)[:nd].max() < 2.0e-5
    assert np.abs(both - fresh)[nd:].max() < 1.0e-3
    one, _ = _rollout(model, 80, reset_at=40, worlds=(1,))
    assert np.abs(one[:nd][nodes == 1] - fresh[:nd][nodes == 1]).max() < 2.0e-4
    assert np.abs(one[:nd][nodes == 0] - plain[:nd][nodes == 0]).max() < 2.0e-4


def test_friction_follows_the_model():
    """Two worlds of a straight rod sliding sideways on the ground at 1 m/s, world 1's rod ``mu`` lowered after the
    solver is built (``notify_model_changed``): each stops where Coulomb's ``v0^2 / (2 mu g)`` says."""
    g, v0, dt, mu_ground = 9.81, 1.0, 0.002, 0.5
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    for _ in range(2):
        w = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
        pts = np.column_stack([np.linspace(-0.1, 0.1, 11), np.zeros(11), np.full(11, 0.005)])
        SolverDiSMech.add_rod(w, rod=newton.Rod(pts, radius=0.005), stretch_stiffness=1e3, bend_stiffness=1e-2,
                              twist_stiffness=1e-2, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
        b.add_world(w)
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu_ground))
    model = b.finalize()
    solver = SolverDiSMech(model)
    sb, body_world = model.shape_body.numpy(), model.body_world.numpy()
    mu = model.shape_material_mu.numpy()
    mu[(sb >= 0) & (body_world[np.maximum(sb, 0)] == 1)] = 0.1
    model.shape_material_mu.assign(mu)
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    for i in range(20 + int(0.5 / dt)):
        if i == 20:
            v = np.zeros((model.particle_count, 3))
            v[:, 1] = v0
            s0.particle_qd.assign(v)
            y0 = s0.particle_q.numpy()[:, 1].copy()
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    slid = s0.particle_q.numpy()[:, 1] - y0
    world = model.particle_world.numpy()
    for w, mu_rod in ((0, 0.5), (1, 0.1)):
        exact = v0**2 / (2.0 * 0.5 * (mu_rod + mu_ground) * g)
        assert abs(slid[world == w].mean() / exact - 1.0) < 0.03


def test_restart_and_mass_refresh_and_hinge_barrier():
    """1) ``solver.restart(state, prev)`` reconstructs frames, reference twists, stored strains, and Dahl stresses from
    a fresh state's DOFs; 2) ``notify_model_changed(MODEL_PROPERTIES)`` refreshes ``particle_mass`` so doubling density
    doubles static gravity sag; 3) shell line search rejects past-barrier ``|psi + s theta| >= pi/2`` folds cleanly."""
    # 1. restart() reconstructs carried DER + Dahl state from (prev, state) DOFs alone
    b = newton.ModelBuilder(gravity=(0.0, 2.0, -9.81))
    rod = newton.Rod.create_straight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.3, segment_count=8, radius=0.008)
    bodies, _ = SolverDiSMech.add_rod(b, rod=rod, stretch_stiffness=5.0e3, bend_stiffness=0.5, twist_stiffness=0.4,
                                      bend_damping=0.02, dahl_eps_max=0.08, dahl_tau=0.04)
    SolverDiSMech.fix_segment(b, bodies[0])
    m = b.finalize()
    sim = SolverDiSMech(m, exact=True, iterations=25, self_contact=False)
    s_prev, _ = simulate(m, sim, 15, 0.01)
    s_end = m.state()
    flatten_state(s_end)
    sim.step(s_prev, s_end, None, None, 0.01)
    fresh = m.state()
    flatten_state(fresh)
    assert np.isnan(fresh.dismech.triplet_strain_q.numpy()).all()
    wp.copy(fresh.dismech.q, s_end.dismech.q)
    wp.copy(fresh.dismech.qd, s_end.dismech.qd)
    sim.restart(fresh, s_prev)
    for field in ("edge_d1_q", "triplet_ref_twist_q", "triplet_strain_q", "triplet_dahl_q"):
        np.testing.assert_allclose(getattr(fresh.dismech, field).numpy(), getattr(s_end.dismech, field).numpy(),
                                   atol=1.0e-6)
    cont_a, _ = simulate(m, sim, 5, 0.01, state=fresh)
    sim.reset(s_end)
    cont_b, _ = simulate(m, sim, 5, 0.01, state=s_end)
    np.testing.assert_allclose(cont_a.dismech.q.numpy(), cont_b.dismech.q.numpy(), atol=1.0e-5)

    # 2. notify_model_changed refreshes mass -> static cantilever sag doubles when particle_mass doubles
    b2 = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    rod2 = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.4, segment_count=16, radius=0.005)
    bs2, _ = SolverDiSMech.add_rod(b2, rod=rod2, stretch_stiffness=1.0e5, bend_stiffness=8.0, twist_stiffness=8.0,
                                   bend_damping=0.4)
    SolverDiSMech.fix_segment(b2, bs2[0])
    m2 = b2.finalize()
    sim2 = SolverDiSMech(m2, self_contact=False)
    sag1 = 1.0 - simulate(m2, sim2, 120, 0.02)[0].particle_q.numpy()[-1, 2]
    m2.particle_mass.assign(2.0 * m2.particle_mass.numpy())
    sim2.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    sim2.reset(m2.state())
    sag2 = 1.0 - simulate(m2, sim2, 120, 0.02)[0].particle_q.numpy()[-1, 2]
    assert abs(sag2 / sag1 - 2.0) < 0.03, (sag1, sag2)

    # 3. shell hinge barrier keeps face strains finite under an extreme folding kick
    b3 = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    verts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]])
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    ids = cloth(b3, verts, faces, density=0.5, tri_ke=1.0e3, tri_ka=5.0e2, bend_stiffness=1.0e-2, damping=0.0)
    for k in ids[:3]:
        b3.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
    m3 = b3.finalize()
    sim3 = SolverDiSMech(m3, self_contact=False)
    s3 = m3.state()
    qd3 = np.zeros((4, 3), dtype=np.float32)
    qd3[3] = [-40.0, -40.0, 10.0]  # tries to whip flap vertex 3 past 180 deg across the diagonal hinge in 1 step
    s3.particle_qd.assign(qd3)
    out3, _ = simulate(m3, sim3, 5, 0.02, state=s3)
    assert np.isfinite(out3.dismech.q.numpy()).all()
    assert np.isfinite(out3.dismech.face_strain_q.numpy()).all()
