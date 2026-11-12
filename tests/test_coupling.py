"""Two-way coupling with rigid bodies: standalone 6-DOF ``ProxyArm`` + ``ProxyPins`` on CPU/CUDA, and coupled solves
with MuJoCo through ``SolverCoupledADMM`` and ``SolverCoupledProxy``."""

import newton
import numpy as np
import pytest
import warp as wp
from newton import BodyFlags

from dismech_newton import SolverDiSMech, flatten_state


@pytest.fixture
def mujoco():
    return pytest.importorskip("mujoco_warp")


def test_standalone_proxy_arm_and_pin_conserve_momenta(rng):
    """``SolverDiSMech``'s built-in 6-DOF :class:`~dismech_newton.coupling.ProxyArm` and
    :class:`~dismech_newton.coupling.ProxyPins` (no external rigid solver, runs on CPU and CUDA): a rod pinned by a ball
    joint to a free ``BodyFlags.PROXY`` box and a second free rod thrown into the box in zero gravity conserve total
    linear momentum (<1e-4) and angular momentum about the origin (<5e-3) while holding the pin (<5e-4 radii)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.25, 0.25)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0), turn))
    half = np.array([0.04, 0.05, 0.04])
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=350.0, mu=0.3))
    r = 0.005
    pin = wp.vec3(0.0, float(half[1] + 3 * r), 0.0)
    start = np.array(wp.quat_rotate(turn, pin))
    along = np.array(wp.quat_rotate(turn, wp.vec3(0.0, 1.0, 0.0)))
    rod_pinned = newton.Rod(start + np.linspace(0.0, 0.25, 11)[:, None] * along, radius=r)
    caps, _ = SolverDiSMech.add_rod(builder, rod=rod_pinned,
                                    cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3),
                                    stretch_stiffness=1.0e4, bend_stiffness=1.0e-2, bend_damping=1.0e-4)
    seg_half = 0.5 * float(np.linalg.norm(np.diff(rod_pinned.points[:2], axis=0)))
    builder.add_joint_ball(parent=body, child=caps[0], parent_xform=wp.transform(pin, wp.quat_identity()),
                           child_xform=wp.transform(wp.vec3(0.0, 0.0, -seg_half), wp.quat_identity()))
    striker = newton.Rod(np.column_stack([np.full(9, -0.07), np.linspace(-0.08, 0.08, 9), np.zeros(9)]), radius=r)
    SolverDiSMech.add_rod(builder, rod=striker, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3),
                          stretch_stiffness=1.0e4, bend_stiffness=1.0e-2, bend_damping=1.0e-4)
    model = builder.finalize()
    flags = model.body_flags.numpy()
    flags[body] |= int(BodyFlags.PROXY)
    model.body_flags.assign(flags)
    solver = SolverDiSMech(model, iterations=25)
    assert solver.pins.count == 1
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    qd = s0.particle_qd.numpy()
    qd[:11] = [0.2, 0.0, 0.3]
    qd[11:] = [0.8, 0.1, 0.0]
    s0.particle_qd.assign(qd)
    m = model.particle_mass.numpy().astype(np.float64)[:, None]
    mb = float(model.body_mass.numpy()[body])
    inertia = model.body_inertia.numpy()[body].astype(np.float64)
    com = model.body_com.numpy()[body].astype(np.float64)

    def momenta(s):
        x, v = s.particle_q.numpy().astype(np.float64), s.particle_qd.numpy().astype(np.float64)
        bq, bqd = s.body_q.numpy()[body].astype(np.float64), s.body_qd.numpy()[body].astype(np.float64)
        X = wp.transform(wp.vec3(*bq[:3]), wp.quat(*bq[3:]))
        c = np.array(wp.transform_point(X, wp.vec3(*com)), dtype=np.float64)
        R = np.array(wp.quat_to_matrix(wp.quat(*bq[3:])), dtype=np.float64).reshape(3, 3)
        p = (m * v).sum(0) + mb * bqd[:3]
        L = (m * np.cross(x, v)).sum(0) + mb * np.cross(c, bqd[:3]) + R @ inertia @ R.T @ bqd[3:]
        gap = np.linalg.norm(x[0] - np.array(wp.transform_point(X, pin), dtype=np.float64))
        return p, L, gap

    p0, L0, _ = momenta(s0)
    dp = dL = max_gap = 0.0
    for k in range(120):
        solver.step(s0, s1, None, None, 1.0e-3)
        s0, s1 = s1, s0
        p, L, g = momenta(s0)
        dp = max(dp, np.linalg.norm(p - p0) / np.linalg.norm(p0))
        dL = max(dL, np.linalg.norm(L - L0) / np.linalg.norm(L0))
        if k >= 5:
            max_gap = max(max_gap, g / r)
    assert dp < 1.0e-4, dp
    assert dL < 5.0e-3, dL
    assert max_gap < 5.0e-4, max_gap
    assert mb * np.linalg.norm(s0.body_qd.numpy()[body, :3]) > 0.25 * np.linalg.norm(p0)


@pytest.mark.usefixtures("needs_cuda", "mujoco")
def test_admm_entry(rng):
    """The solver as an entry of Newton's ``SolverCoupledADMM`` with a MuJoCo box: no gravity, a rod thrown at the
    free box (turned at random, struck off centre), 300 steps through the impact."""
    from newton.solvers import SolverMuJoCo  # noqa: PLC0415
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM  # noqa: PLC0415

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.1, float(rng.uniform(0.0, 0.04)), 0.0), turn))
    half = np.array([0.04, 0.06, 0.04])
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3))
    n, r = 41, 0.005
    SolverDiSMech.add_rod(builder,
                          rod=newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.2, 0.2, n), np.zeros(n)]),
                                         radius=r),
                          cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3), stretch_stiffness=1.0e4,
                          bend_stiffness=2.0e-3, bend_damping=1.0e-4)
    model = builder.finalize()
    child = model.joint_child.numpy()
    box_joints = [j for j in range(model.joint_count) if child[j] == body]
    solver = SolverCoupledADMM(model=model, entries=[
        SolverCoupled.Entry(name="rigid", solver=lambda v: SolverMuJoCo(v, use_mujoco_contacts=False), bodies=[body],
                            joints=box_joints),
        SolverCoupled.Entry(name="soft", solver=SolverDiSMech, particles=list(range(model.particle_count)),
                            bodies=list(range(body + 1, model.body_count)),
                            joints=[j for j in range(model.joint_count) if j not in box_joints]),
    ], coupling=SolverCoupledADMM.Config(rho=200.0, gamma=0.0,
                                         contact_pairs=[SolverCoupledADMM.ContactPair("rigid", "soft")]))
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


def _proxy_scene(builder, body, rod, mu, rigid_contacts=True, pin=None, fixed=False):
    """The solver as Newton's ``SolverCoupledProxy`` destination: ``body`` MuJoCo's, a proxy in our solve."""
    from newton.solvers import SolverMuJoCo  # noqa: PLC0415
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy  # noqa: PLC0415

    capsules, _ = SolverDiSMech.add_rod(builder, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu),
                                        stretch_stiffness=1.0e4, bend_stiffness=2.0e-3, bend_damping=1.0e-4)
    for c in capsules if fixed else ():
        SolverDiSMech.fix_segment(builder, c)
    if pin is not None:
        half = 0.5 * float(np.linalg.norm(np.diff(rod.points[:2], axis=0)))
        builder.add_joint_ball(parent=body, child=capsules[0], parent_xform=wp.transform(pin, wp.quat_identity()),
                               child_xform=wp.transform(wp.vec3(0.0, 0.0, -half), wp.quat_identity()))
    model = builder.finalize()
    child = model.joint_child.numpy()
    box_joints = [j for j in range(model.joint_count) if child[j] == body]
    solver = SolverCoupledProxy(model, entries=[
        SolverCoupled.Entry(name="rigid", solver=lambda v: SolverMuJoCo(v, use_mujoco_contacts=rigid_contacts),
                            bodies=[body], joints=box_joints),
        SolverCoupled.Entry(name="soft", solver=SolverDiSMech, particles=list(range(model.particle_count)),
                            bodies=list(range(body + 1, model.body_count)),
                            joints=[j for j in range(model.joint_count) if j not in box_joints]),
    ], coupling=SolverCoupledProxy.Config(proxies=[
        SolverCoupledProxy.Proxy(source="rigid", destination="soft", bodies=[body])]))
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    pipeline = newton.CollisionPipeline(model)
    return model, solver, pipeline, s0, s1


@pytest.mark.usefixtures("needs_cuda", "mujoco")
def test_proxy_momentum(rng):
    """The solver as Newton's ``SolverCoupledProxy`` destination, a free MuJoCo box its proxy: linear momentum stays to
    1e-4, the box takes most of it, and no node sphere sinks into the box past 0.4 radii."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.1, float(rng.uniform(0.0, 0.04)), 0.0), turn))
    half = np.array([0.04, 0.06, 0.04])
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3))
    n, r = 41, 0.005
    rod = newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.2, 0.2, n), np.zeros(n)]), radius=r)
    model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.3, rigid_contacts=False)
    contacts = pipeline.contacts()
    qd = s0.particle_qd.numpy()
    qd[:, 0] = 1.0
    s0.particle_qd.assign(qd)
    m, mb, dt = model.particle_mass.numpy()[:, None], float(model.body_mass.numpy()[body]), 1.0e-3

    def momentum(s):
        return (m * s.particle_qd.numpy()).sum(0) + mb * s.body_qd.numpy()[body, :3]

    p0, worst, moved, deepest = momentum(s0), 0.0, 0.0, -np.inf
    for _ in range(300):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
        worst = max(worst, np.linalg.norm(momentum(s0) - p0) / np.linalg.norm(p0))
        moved = max(moved, mb * np.linalg.norm(s0.body_qd.numpy()[body, :3]))
        bq = s0.body_q.numpy()[body]
        inv = wp.transform_inverse(wp.transform(wp.vec3(*bq[:3]), wp.quat(*bq[3:])))
        local = np.array([wp.transform_point(inv, wp.vec3(*x)) for x in s0.particle_q.numpy()])
        deepest = max(deepest, (half + r - np.abs(local)).min(1).max())
    assert worst < 1.0e-4, worst
    assert moved > 0.8 * np.linalg.norm(p0), "the box was not struck"
    assert deepest < 0.4 * r, deepest


@pytest.mark.usefixtures("needs_cuda", "mujoco")
def test_proxy_pin(rng):
    """A rod pinned by a Newton ball joint to a free MuJoCo box (:class:`~dismech_newton.coupling.ProxyPins`), no
    gravity, thrown sideways: linear and angular momentum conserved, pin holds to 4e-4 radii."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0), turn))
    half = np.array([0.04, 0.06, 0.04])
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3))
    n, r = 41, 0.005
    pin = wp.vec3(0.0, float(half[1] + 2 * r), 0.0)
    start = np.array(wp.quat_rotate(turn, pin))
    along = np.array(wp.quat_rotate(turn, wp.vec3(0.0, 1.0, 0.0)))
    rod = newton.Rod(start + np.linspace(0.0, 0.4, n)[:, None] * along, radius=r)
    model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.3, rigid_contacts=False, pin=pin)
    assert solver.solver("soft").pins.count == 1
    contacts = pipeline.contacts()
    a = float(rng.uniform(-0.5, 0.5))
    qd = s0.particle_qd.numpy()
    qd[:] = np.array(wp.quat_rotate(turn, wp.vec3(np.cos(a), 0.0, np.sin(a))))
    s0.particle_qd.assign(qd)
    m, dt = model.particle_mass.numpy()[:, None], 1.0e-3
    mb, inertia, com = (float(model.body_mass.numpy()[body]), model.body_inertia.numpy()[body],
                        model.body_com.numpy()[body])

    def momenta(s):
        x, v = s.particle_q.numpy(), s.particle_qd.numpy()
        bq, bqd = s.body_q.numpy()[body], s.body_qd.numpy()[body]
        X = wp.transform(wp.vec3(*bq[:3]), wp.quat(*bq[3:]))
        c = np.array(wp.transform_point(X, wp.vec3(*com)))
        R = np.array(wp.quat_to_matrix(wp.quat(*bq[3:]))).reshape(3, 3)
        p = (m * v).sum(0) + mb * bqd[:3]
        L = (m * np.cross(x, v)).sum(0) + mb * np.cross(c, bqd[:3]) + R @ inertia @ R.T @ bqd[3:]
        return p, L, np.linalg.norm(x[0] - np.array(wp.transform_point(X, pin)))

    p0, L0, _ = momenta(s0)
    dp = dL = gap = 0.0
    for k in range(300):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, dt)
        s0, s1 = s1, s0
        p, L, g = momenta(s0)
        dp = max(dp, np.linalg.norm(p - p0) / np.linalg.norm(p0))
        dL = max(dL, np.linalg.norm(L - L0) / np.linalg.norm(L0))
        if k >= 10:
            gap = max(gap, g / r)
    assert dp < 1.0e-4, dp
    assert dL < 4.0e-3, dL
    assert gap < 4.0e-4, gap
    assert mb * np.linalg.norm(s0.body_qd.numpy()[body, :3]) > 0.3 * np.linalg.norm(p0), "the rod did not drag the box"


@pytest.mark.usefixtures("needs_cuda", "mujoco")
def test_proxy_impact():
    """A free MuJoCo box dropped 1 cm onto a rod clamped along its length through ``SolverCoupledProxy``."""
    builder = newton.ModelBuilder()
    hx, r, n, drop = 0.02, 0.004, 21, 0.01
    rest = 0.1 + r + hx
    rod = newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.05, 0.05, n), np.full(n, 0.1)]), radius=r)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, rest + drop), wp.quat_identity()))
    builder.add_shape_box(body, hx=hx, hy=hx, hz=hx, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
    model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.5, fixed=True)
    contacts = pipeline.contacts()
    feedback = solver._proxy_mappings[0].coupling_forces
    weight = float(model.body_mass.numpy()[body] * 9.81)
    z, vz, f = [], [], []
    for k in range(600):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, 2.0e-3)
        s0, s1 = s1, s0
        z.append(s0.body_q.numpy()[body, 2])
        vz.append(s0.body_qd.numpy()[body, 2])
        if k >= 500:
            f.append(feedback.numpy()[body, 2])
    impact = -min(vz)
    assert max(vz[int(np.argmin(vz)):]) < 0.05 * impact, (max(vz), impact)
    assert abs(z[-1] - rest) < 0.25 * r, (z[-1], rest)
    assert min(z) > rest - r, (min(z), rest)
    assert abs(np.mean(f) - weight) < 1.0e-2 * weight, (np.mean(f), weight)


@pytest.mark.usefixtures("needs_cuda", "mujoco")
def test_proxy_static_load():
    """A rod resting across a free MuJoCo box on MuJoCo's ground, off its centre by ``a``, through
    ``SolverCoupledProxy``: feedback equals the rod's weight with torque ``a m g``."""
    builder = newton.ModelBuilder()
    hx, a, r, n = 0.05, 0.02, 0.004, 21
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, hx), wp.quat_identity()))
    builder.add_shape_box(body, hx=hx, hy=hx, hz=hx, cfg=newton.ModelBuilder.ShapeConfig(density=50.0, mu=0.5))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    rod = newton.Rod(np.column_stack([np.full(n, a), np.linspace(-0.04, 0.04, n), np.full(n, 2 * hx + r + 0.002)]),
                     radius=r)
    model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.5)
    contacts = pipeline.contacts()
    feedback = solver._proxy_mappings[0].coupling_forces
    weight = float(model.particle_mass.numpy().sum() * 9.81)
    z0, f = s0.body_q.numpy()[body, 2], []
    for k in range(600):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, 2.0e-3)
        s0, s1 = s1, s0
        if k >= 500:
            f.append(feedback.numpy()[body])
    f = np.mean(f, 0)
    assert np.allclose(f[:3], [0.0, 0.0, -weight], atol=1.0e-3 * weight), (f, weight)
    assert np.allclose(f[3:], [0.0, a * weight, 0.0], atol=2.0e-3 * a * weight), (f, a * weight)
    assert abs(s0.body_q.numpy()[body, 2] - z0) < 1.0e-4
    assert np.abs(s0.body_qd.numpy()[body]).max() < 1.0e-4


@pytest.mark.slow
@pytest.mark.usefixtures("needs_cuda", "mujoco")
def test_newton_franka_swap():
    """Plug and play: Newton's own ``example_franka_cable_ik_pick_place`` with only its VBD cable made our rod and the
    VBD entry ``SolverDiSMech``. Newton's test passes and each finger's grip is its servo force (2%) while carried."""
    import newton.examples.multiphysics.example_franka_cable_ik_pick_place as M  # noqa: PLC0415
    import newton.viewer  # noqa: PLC0415
    from newton.solvers import SolverMuJoCo  # noqa: PLC0415
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy  # noqa: PLC0415

    class Example(M.Example):
        def _emit_template(self, builder):
            def add_rod(rod, cfg, stretch_stiffness, stretch_damping, bend_stiffness, bend_damping, **_):
                return SolverDiSMech.add_rod(builder, rod=rod, cfg=cfg, stretch_stiffness=stretch_stiffness,
                                             stretch_damping=stretch_damping, bend_stiffness=bend_stiffness,
                                             bend_damping=bend_damping)

            builder.add_rod = add_rod
            super()._emit_template(builder)
            del builder.add_rod

        def _build_solvers(self, args):
            self.solver = SolverCoupledProxy(self.model, entries=[
                SolverCoupled.Entry(name="mjc", solver=lambda v: SolverMuJoCo(
                    v, solver="newton", integrator="implicitfast", cone="elliptic", iterations=args.mujoco_iterations,
                    ls_iterations=args.mujoco_ls_iterations, use_mujoco_contacts=False, njmax=256, nconmax=64),
                    bodies=self.franka_bodies, joints=self.franka_joints),
                SolverCoupled.Entry(name="soft", solver=SolverDiSMech, particles=list(range(self.model.particle_count)),
                                    bodies=self.payload_bodies, joints=self.payload_joints),
            ], coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(
                source="mjc", destination="soft", bodies=self.gripper_bodies, mass_scale=args.mass_scale,
                mode=args.coupling_mode, collide_interval=1,
                collision_pipeline=lambda m: newton.examples.create_collision_pipeline(m, broad_phase="explicit"))],
                iterations=args.proxy_iterations))

    args = M.Example.create_parser().parse_args(["--viewer", "null"])
    ex = Example(newton.viewer.ViewerNull(), args)
    assert ex.graph is not None
    soft = ex.solver.solver("soft")
    fingers = [b for b, label in enumerate(ex.solver.view("soft").body_label) if "finger" in label]
    grips = []
    for f in range(330):
        ex.step()
        if 3.25 < (f + 1) / 60 < 4.75:
            grips.append(np.abs(soft.body_reaction.numpy()[fingers, 1]))
    ex.test_final()
    servo = M.GRIP_STIFFNESS * (args.payload_radius - M.GRIP_HOLD)
    assert np.allclose(grips, servo, rtol=0.02), (np.min(grips), np.max(grips), servo)
