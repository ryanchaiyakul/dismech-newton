"""Two-way coupling with a rigid solver's bodies (MuJoCo): the contact reactions on the bodies against momentum and
statics."""

import newton
import numpy as np
import pytest
import warp as wp

from dismech_newton import DiSMechSolver, flatten_state
from dismech_newton.coupling import CoupledSolver

pytest.importorskip("mujoco_warp")
from newton.solvers import SolverMuJoCo  # noqa: E402


def _scene(builder, points, radius, mu, soft_options=None):
    DiSMechSolver.add_rod(builder, newton.Rod(points, radius=radius),
                          cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu), stretch_stiffness=1.0e4,
                          bend_stiffness=2.0e-3, bend_damping=1.0e-4)
    model = builder.finalize()
    solver = CoupledSolver(model, lambda m: SolverMuJoCo(m, use_mujoco_contacts=True), **(soft_options or {}))
    soft = solver.soft
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    return model, soft, solver, s0, s1


@pytest.mark.usefixtures("needs_cuda")
def test_reaction_balance(rng):
    """No gravity: a rod thrown at a free box (turned at random, struck off centre), 300 steps through the impact.
    The box gains what the rod loses: their linear momentum, plus the last step's reaction impulse (applied to the box
    a step late), stays to the soft solve's tolerance (here 1e-3 radii: 1e-3 of it, 2e-2 at the default 1e-2)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.1, float(rng.uniform(0.0, 0.04)), 0.0), turn))
    builder.add_shape_box(body, hx=0.04, hy=0.06, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3))
    n = 21
    points = np.column_stack([np.zeros(n), np.linspace(-0.2, 0.2, n), np.zeros(n)])
    model, soft, solver, s0, s1 = _scene(builder, points, 0.005, 0.3)
    soft._tol = 1.0e-3 * soft.cap
    qd = s0.particle_qd.numpy()
    qd[:, 0] = 1.0
    s0.particle_qd.assign(qd)
    soft.update_proxies(s0)
    m, mb, dt = model.particle_mass.numpy()[:, None], float(model.body_mass.numpy()[body]), 1.0e-3

    def momentum(s):
        return (m * s.particle_qd.numpy()).sum(0) + mb * s.body_qd.numpy()[body, :3]

    p0, moved, worst = momentum(s0), 0.0, 0.0
    for _ in range(300):
        s0.clear_forces()
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        moved = max(moved, mb * np.linalg.norm(s0.body_qd.numpy()[body, :3]))
        worst = max(worst, np.linalg.norm(momentum(s0) - p0) / np.linalg.norm(p0))
    print(f"WORST {worst:.3e} moved {moved / np.linalg.norm(p0):.3f}")
    assert worst < 4.0e-3
    assert moved > 0.3 * np.linalg.norm(p0), "the box was not struck"


@pytest.mark.usefixtures("needs_cuda")
def test_static_load():
    """A rod resting across a free box on the ground, off its centre by ``a``: the rod's reaction on the box is its
    weight ``m g`` (to 1e-3), with the torque ``a m g`` about the box's centre (to 1e-3), and the box stays put (the
    ground carries both)."""
    builder = newton.ModelBuilder()
    hx, a, r, n = 0.05, 0.02, 0.004, 21
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, hx), wp.quat_identity()))
    builder.add_shape_box(body, hx=hx, hy=hx, hz=hx, cfg=newton.ModelBuilder.ShapeConfig(density=50.0, mu=0.5))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    points = np.column_stack([np.full(n, a), np.linspace(-0.04, 0.04, n), np.full(n, 2 * hx + r + 0.002)])
    model, soft, solver, s0, s1 = _scene(builder, points, r, 0.5)
    soft.update_proxies(s0)
    weight = float(model.particle_mass.numpy().sum() * 9.81)
    z0, f = s0.body_q.numpy()[body, 2], []
    for k in range(600):
        s0.clear_forces()
        solver.step(s0, s1, None, None, 2.0e-3)
        s0, s1 = s1, s0
        if k >= 500:
            f.append(soft.body_reaction.numpy()[body])
    f = np.mean(f, 0)
    assert np.allclose(f[:3], [0.0, 0.0, -weight], atol=1.0e-3 * weight), (f, weight)
    assert np.allclose(f[3:], [0.0, a * weight, 0.0], atol=1.0e-3 * a * weight), (f, a * weight)
    assert abs(s0.body_q.numpy()[body, 2] - z0) < 1.0e-4
    assert np.abs(s0.body_qd.numpy()[body]).max() < 1.0e-4


@pytest.mark.usefixtures("needs_cuda")
def test_admm_entry(rng):
    """The solver as an entry of Newton's ``SolverCoupledADMM`` with a MuJoCo box: no gravity, a rod thrown at the
    free box (turned at random, struck off centre), 300 steps through the impact. Linear momentum stays to 1e-4
    (measured 1e-5: the coupler's forces are equal and opposite, and the rod takes its own exactly; without its
    twists and frames carried between the coupler's steps the rod stalls and drops them), the box takes most of it,
    and no node sphere sinks into the box past 0.6 radii (measured 0.44 at most over 7 seeds)."""
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM  # noqa: PLC0415

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.1, float(rng.uniform(0.0, 0.04)), 0.0), turn))
    half = np.array([0.04, 0.06, 0.04])
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3))
    n, r = 41, 0.005
    DiSMechSolver.add_rod(builder, newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.2, 0.2, n), np.zeros(n)]),
                                              radius=r),
                          cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3), stretch_stiffness=1.0e4,
                          bend_stiffness=2.0e-3, bend_damping=1.0e-4)
    model = builder.finalize()
    child = model.joint_child.numpy()
    box_joints = [j for j in range(model.joint_count) if child[j] == body]
    solver = SolverCoupledADMM(model=model, entries=[
        SolverCoupled.Entry(name="rigid", solver=lambda v: SolverMuJoCo(v, use_mujoco_contacts=False), bodies=[body],
                            joints=box_joints),
        SolverCoupled.Entry(name="soft", solver=DiSMechSolver, particles=list(range(model.particle_count)),
                            bodies=list(range(body + 1, model.body_count)),
                            joints=[j for j in range(model.joint_count) if j not in box_joints]),
    ], coupling=SolverCoupledADMM.Config(rho=200.0, gamma=0.0, contact_pairs=[SolverCoupledADMM.ContactPair("rigid", "soft")]))
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver.prepare_contacts(contacts)
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    qd = s0.particle_qd.numpy()
    qd[:, 0] = 1.0
    s0.particle_qd.assign(qd)
    m, mb = model.particle_mass.numpy()[:, None], float(model.body_mass.numpy()[body])

    def momentum(s):
        return (m * s.particle_qd.numpy()).sum(0) + mb * s.body_qd.numpy()[body, :3]

    p0, worst, moved, deepest = momentum(s0), 0.0, 0.0, -np.inf
    for _ in range(300):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, 1.0e-3)
        s0, s1 = s1, s0
        worst = max(worst, np.linalg.norm(momentum(s0) - p0) / np.linalg.norm(p0))
        moved = max(moved, mb * np.linalg.norm(s0.body_qd.numpy()[body, :3]))
        bq = s0.body_q.numpy()[body]
        inv = wp.transform_inverse(wp.transform(wp.vec3(*bq[:3]), wp.quat(*bq[3:])))
        local = np.array([wp.transform_point(inv, wp.vec3(*x)) for x in s0.particle_q.numpy()])
        deepest = max(deepest, (half + r - np.abs(local)).min(1).max())
    assert worst < 1.0e-4, worst
    assert moved > 0.5 * np.linalg.norm(p0), "the box was not struck"
    assert deepest < 0.6 * r, deepest


@pytest.mark.slow
@pytest.mark.usefixtures("needs_cuda")
@pytest.mark.parametrize("segments", [19, 40])
def test_franka_pick_place(segments):
    """Newton's Franka cable pick-and-place through :class:`CoupledSolver`, whole (330 frames), at Newton's 19
    segments and refined (the same cable): placed within 1 cm (Newton's test), each finger's grip its servo force
    (2%) and the mean lift the cable's weight (20%: it swings) while carried."""
    from test_examples import _load  # noqa: PLC0415

    example = _load("newton/franka_cable_ik_pick_place").Example
    example.segments = segments
    ex = example(None, None)
    for _ in range(330):
        ex.step()
    ex.test_final()
