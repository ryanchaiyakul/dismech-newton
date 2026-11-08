"""What holds whatever the elements: a free body's momentum and energy, and the same rollout on both devices."""

import newton
import numpy as np
import pytest
import warp as wp
from rollouts import ClampedRollout, cloth, grid

from dismech_newton import SolverDiSMech, flatten_state
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
    [("rod", {}), ("rod", {"exact": True, "iterations": 30, "theta": 0.5}), ("sheet", {}),
     ("sheet", {"exact": True, "iterations": 30})],
    ids=["rod-admm", "rod-midpoint", "sheet-admm", "sheet-exact"],
)
def test_free_body_keeps_momentum_and_loses_energy(rng, kind, options):
    """Thrown at random from rest shape (no potential energy) for 100 steps: the internal forces cancel, so the linear
    momentum stays to float32 round-off (5e-5 to 3e-4 of it), and the kinetic energy never exceeds the start's. A
    sheet on midpoint is left out: it gains energy (1.7x here, more when thrown harder), the midpoint rule's
    nonlinear instability on a membrane."""
    model = _free(kind)
    solver = SolverDiSMech(model, self_contact=False, **options)
    mass = dof_constants(model)[0].numpy().astype(np.float64)
    nd = 3 * model.particle_count
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    qd0 = 0.3 * rng.normal(size=mass.size)
    s0.dismech.qd.assign(qd0)
    p0, e0 = (mass[:nd] * qd0[:nd]).reshape(-1, 3).sum(0), 0.5 * mass @ qd0**2
    for _ in range(100):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        qd = s0.dismech.qd.numpy().astype(np.float64)
        p = (mass[:nd] * qd[:nd]).reshape(-1, 3).sum(0)
        assert np.linalg.norm(p - p0) < 1.0e-3 * np.linalg.norm(p0), (p, p0)
        assert 0.5 * mass @ qd**2 <= 1.001 * e0


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
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    world = model.particle_world.numpy()
    sel = np.isin(world, worlds)
    mask = wp.array([w in worlds for w in range(model.world_count + 1)], dtype=wp.bool)
    for i in range(steps):
        if i == reset_at:  # the worlds' nodes back to the model's, at rest; the solver resets the rest
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
    ``world_mask``, then 40 more steps. Both worlds reset: the same as 40 steps from the start, as two fresh runs
    differ (the Newton-CG's run-to-run spread: 8e-6 m, twists 2e-4 rad at 4 mm radius). World 1 alone: it replays
    the fresh start and world 0 goes on unaffected, both within a few of the solve's tolerance (0.01 radii: 4e-5 m;
    5e-5 and 1.5e-5 on 2026-11-03): contacts are still found and the matrix refactorized for all worlds at once, so
    another world's iterates reach a world's."""
    model = _helices()
    fresh, world = _rollout(model, 40)
    plain, _ = _rollout(model, 80)
    both, _ = _rollout(model, 80, reset_at=40, worlds=(0, 1))
    nodes = np.repeat(world, 3)
    nd = nodes.size
    assert np.abs(both - fresh)[:nd].max() < 2.0e-5  # 8.3e-6 at the worst of 18 runs
    assert np.abs(both - fresh)[nd:].max() < 1.0e-3  # twists: 2.1e-4
    one, _ = _rollout(model, 80, reset_at=40, worlds=(1,))
    assert np.abs(one[:nd][nodes == 1] - fresh[:nd][nodes == 1]).max() < 2.0e-4
    assert np.abs(one[:nd][nodes == 0] - plain[:nd][nodes == 0]).max() < 2.0e-4


def test_friction_follows_the_model():
    """Two worlds of a straight rod sliding sideways on the ground at 1 m/s, world 1's rod ``mu`` lowered after the
    solver is built (``notify_model_changed``): each stops where Coulomb's ``v0^2 / (2 mu g)`` says, ``mu`` the mean
    of the rod's and the ground's (+0.9%, +0.6% on 2026-11-04): per-world friction, randomizable at a reset."""
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
    for i in range(20 + int(0.5 / dt)):  # settle onto the ground, then slide
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
