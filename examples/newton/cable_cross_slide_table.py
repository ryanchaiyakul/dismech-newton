"""A cable-driven XY table (Newton's ``example_cable_cross_slide_table``): the same base, carriages, pulleys, cable
route and drive. A green carriage slides in x, a beige one rides it in y, and one cable loop, wrapped around five
passive pulleys on the carriages and two driven blue inputs, has both ends fixed to the beige carriage: the inputs'
rotations alone trace a rectangle with it. The carriages and the passive pulleys are MuJoCo's, proxies in our solve
through Newton's ``SolverCoupledProxy``; the cable is a DER rod in place of Newton's VBD cable, its ends Newton's
ball joints to the beige carriage (:class:`~dismech_newton.coupling.ProxyPins`); the inputs are kinematic obstacles.
The cable runs on the pulleys by surface friction (it rolls on them, its centreline at ``w (R + r)``), pretensioned
as Newton's runs (its contact margins hold it 1.4% stretched; ours none: a slack side sags and loses motion).

Shows: a closed force loop through two-way coupling: input friction pulls the cable, the cable pulls the carriages
through the pins and the passive pulleys. Newton's test over the whole rectangle: the table within 5 mm of its path
(2.5 mm RMS), the cable in its pulleys' plane, the carriages within their travel.

    uv run --extra mujoco examples/newton/cable_cross_slide_table.py
    uv run --extra mujoco examples/newton/cable_cross_slide_table.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.examples.cable.example_cable_cross_slide_table import (
    CABLE_XY_ABS_BOUND,
    JOINT_LIMIT_TOLERANCE,
    START_RAMP_DURATION,
    TABLE_RECT_HALF_X,
    TABLE_RECT_HALF_Y,
    TABLE_RECT_PERIOD,
    TABLE_TRACKING_MAX_ERROR_TOLERANCE,
    TABLE_TRACKING_RMS_ERROR_TOLERANCE,
    add_pulley,
    add_visual_bar,
    create_xy_table_cable_points,
)
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import SolverDiSMech

BLUE, GREEN, BEIGE = (0.12, 0.34, 0.76), (0.12, 0.58, 0.28), (0.74, 0.63, 0.45)


@wp.func
def table_target(t: float) -> wp.vec2:
    """Newton's path: the rectangle, eased in over the start ramp."""
    ramp = wp.clamp(t / START_RAMP_DURATION, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    side = 4.0 * (t - wp.floor(t / TABLE_RECT_PERIOD) * TABLE_RECT_PERIOD) / TABLE_RECT_PERIOD
    x, y = -TABLE_RECT_HALF_X, -TABLE_RECT_HALF_Y
    if side < 1.0:
        x = -TABLE_RECT_HALF_X + 2.0 * TABLE_RECT_HALF_X * side
    elif side < 2.0:
        x = TABLE_RECT_HALF_X
        y = -TABLE_RECT_HALF_Y + 2.0 * TABLE_RECT_HALF_Y * (side - 1.0)
    elif side < 3.0:
        x = TABLE_RECT_HALF_X - 2.0 * TABLE_RECT_HALF_X * (side - 2.0)
        y = TABLE_RECT_HALF_Y
    else:
        y = TABLE_RECT_HALF_Y - 2.0 * TABLE_RECT_HALF_Y * (side - 3.0)
    return ramp * wp.vec2(x, y)


@wp.func
def input_angle(t: float, side: int, radius: float) -> float:
    """Newton's inverse of the cable drive: world ``(x, y) = (command_y, -command_x)``."""
    p = table_target(t)
    cx, cy = -p[1], p[0]
    if side == 0:
        return (cx + cy) / radius
    return (cy - cx) / radius


@wp.kernel
def _drive_kernel(time: wp.array[float], bodies: wp.array[wp.int32], base: wp.array[wp.transform], radius: float,
                  dt: float, target: wp.array[wp.vec2], q0: wp.array[wp.transform], q1: wp.array[wp.transform],
                  qd0: wp.array[wp.spatial_vector], qd1: wp.array[wp.spatial_vector]):
    """The inputs at the substep's start and their spin over it (their friction drives the cable)."""
    i = wp.tid()
    t = time[0]
    a = input_angle(t, i, radius)
    w = (input_angle(t + dt, i, radius) - a) / dt
    x = wp.transform(wp.transform_get_translation(base[i]),
                     wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), a) * wp.transform_get_rotation(base[i]))
    b = bodies[i]
    q0[b] = x
    q1[b] = x
    v = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0, 0.0, w))
    qd0[b] = v
    qd1[b] = v
    if i == 0:
        target[0] = table_target(t)


@wp.kernel
def _advance_kernel(time: wp.array[float], dt: float):
    time[0] = time[0] + dt


class Example(CableExample):
    substeps = 10  # Newton's: 60 fps x 10
    segment = 0.005  # [m]: Newton's 1.5 cm chords sag 0.5 r into the sheaves (inside a barrier's thickness)
    pretension = 0.01  # the rest length short by this fraction

    def __init__(self, viewer, args=None):
        r = 0.003  # Newton's dimensions
        self.input_radius, green_radius, beige_radius = 0.025, 0.015, 0.025
        clearance = 1.1 * r
        self.drive_radius = self.input_radius + clearance
        self.slide_z, self.table_z, self.pulley_z = 0.014, 0.022, 0.046
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(builder)
        slide_origin, table_origin = wp.vec3(0.0, 0.0, self.slide_z), wp.vec3(0.0, 0.0, self.table_z)
        add_visual_bar(builder, body=-1, center=wp.vec3(0.0, 0.0, 0.006), half_extents=(0.205, 0.025, 0.006),
                       color=BLUE, label="fixed_blue_base")
        self.slide = builder.add_link(xform=wp.transform(slide_origin, wp.quat_identity()), label="green_x_slide")
        self.table = builder.add_link(xform=wp.transform(table_origin, wp.quat_identity()), label="beige_y_table")
        rigid_joints = [
            builder.add_joint_prismatic(parent=-1, child=self.slide, axis=wp.vec3(1.0, 0.0, 0.0),
                                        parent_xform=wp.transform(slide_origin, wp.quat_identity()),
                                        limit_lower=-TABLE_RECT_HALF_X, limit_upper=TABLE_RECT_HALF_X,
                                        limit_ke=2.0e3, limit_kd=1.0e-4),
            builder.add_joint_prismatic(parent=self.slide, child=self.table, axis=wp.vec3(0.0, 1.0, 0.0),
                                        parent_xform=wp.transform(table_origin - slide_origin, wp.quat_identity()),
                                        limit_lower=-TABLE_RECT_HALF_Y, limit_upper=TABLE_RECT_HALF_Y,
                                        limit_ke=2.0e3, limit_kd=1.0e-4),
        ]
        add_visual_bar(builder, body=self.slide, center=wp.vec3(0.0), half_extents=(0.085, 0.052, 0.006),
                       color=GREEN, label="green_horizontal_carriage")
        add_visual_bar(builder, body=self.table, center=wp.vec3(0.0), half_extents=(0.013, 0.215, 0.006),
                       color=BEIGE, label="beige_vertical_carriage")
        z = self.pulley_z
        specs = [  # Newton's route order: (parent, color, centre, radius, mu); None: a driven input
            (self.slide, GREEN, (-0.045, -0.045), green_radius, 0.35),
            (None, BLUE, (-0.19, 0.0), self.input_radius, 1.0),
            (self.slide, GREEN, (-0.045, 0.045), green_radius, 0.35),
            (self.table, BEIGE, (0.0, 0.19), beige_radius, 0.35),
            (self.slide, GREEN, (0.045, 0.045), green_radius, 0.35),
            (None, BLUE, (0.19, 0.0), self.input_radius, 1.0),
            (self.slide, GREEN, (0.045, -0.045), green_radius, 0.35),
        ]
        self.passive, self.inputs = [], []
        for i, (parent, color, (x, y), radius, mu) in enumerate(specs):
            body, joint = add_pulley(builder, wp.vec3(x, y, z), radius, r, color, parent=parent, sheave_mu=mu,
                                     label=f"xy_table_{i + 1}")
            if joint is None:
                self.inputs.append(body)
            else:  # VBD ignores axle friction and armature: MuJoCo's would hold these pulleys
                d = builder.joint_qd_start[joint]
                builder.joint_friction[d] = 0.0
                builder.joint_armature[d] = 0.0
                self.passive.append(body)
                rigid_joints.append(joint)
        builder.add_articulation(rigid_joints, label="xy_table")
        self.anchors = [wp.vec3(-0.028, -0.21, z - self.table_z), wp.vec3(0.028, -0.21, z - self.table_z)]
        route = {"start": table_origin + self.anchors[0], "pulley_centers": [wp.vec3(*s[2], z) for s in specs],
                 "pulley_radii": [s[3] for s in specs], "end": table_origin + self.anchors[1],
                 "wrap_clearance": clearance}
        newton_length = create_xy_table_cable_points(segment_length=0.015, **route)[1]
        points, length = create_xy_table_cable_points(segment_length=self.segment, **route)
        s = newton_length / length  # Newton's per-joint stiffnesses for its segments: the same cable (EA, EI) at
        # ours; its damping is per joint, not length-normalized (Newton's add_rod), so as given
        cfg = newton.ModelBuilder.ShapeConfig(density=200.0, mu=1.0)
        self.cable, _ = SolverDiSMech.add_rod(builder, rod=newton.Rod(np.array(points), radius=r), cfg=cfg,
                                              stretch_stiffness=1.0e5 * s, stretch_damping=1.0e-4,
                                              bend_stiffness=1.0e-2 * s, bend_damping=1.0e-2,
                                              label="xy_table_cable")
        self.pins = [  # Newton's loop closure: both ends ball-jointed to the beige carriage
            builder.add_joint_ball(parent=self.table, child=self.cable[k],
                                   parent_xform=wp.transform(self.anchors[k], wp.quat_identity()),
                                   child_xform=wp.transform(wp.vec3(0.0, 0.0, end * 0.5 * length), wp.quat_identity()))
            for k, end in ((0, -1.0), (-1, 1.0))
        ]
        model = builder.finalize()
        model.dismech.edge_length.assign(model.dismech.edge_length.numpy() * (1.0 - self.pretension))
        solver = self._proxy(model, rigid_joints)
        self.soft = solver.solver("soft")
        rods = self.soft.elements[0]  # Newton's cable rests straight (built straight, then laid along the route)
        rest = rods.rest.numpy()
        rest[:, 2:] = 0.0
        rods.rest.assign(rest)
        self.start(viewer, model, solver)
        dev = model.device
        q = model.body_q.numpy()
        self.input_body = wp.array(self.inputs, dtype=wp.int32, device=dev)
        self.input_base = wp.array([wp.transform(*q[b]) for b in self.inputs], dtype=wp.transform, device=dev)
        self.time = wp.zeros(1, dtype=float, device=dev)
        self.target = wp.zeros(1, dtype=wp.vec2, device=dev)
        self.errors, self.worst = [], {}
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.0, 0.0, 0.8), pitch=-90.0, yaw=90.0)

    def _proxy(self, model, rigid_joints):
        """MuJoCo's carriages and passive pulleys (the beige carriage and the pulleys proxies in our solve); ours the
        cable, its pins and the driven inputs."""
        rigid = [self.slide, self.table, *self.passive]
        soft_joints = [j for j in range(model.joint_count) if j not in rigid_joints]
        return SolverCoupledProxy(model, entries=[
            SolverCoupled.Entry(name="mjc", solver=SolverMuJoCo, bodies=rigid, joints=rigid_joints),
            SolverCoupled.Entry(name="soft", solver=SolverDiSMech, particles=list(range(model.particle_count)),
                                bodies=[b for b in range(model.body_count) if b not in rigid], joints=soft_joints),
        ], coupling=SolverCoupledProxy.Config(proxies=[
            SolverCoupledProxy.Proxy(source="mjc", destination="soft", bodies=[self.table, *self.passive])]))

    def simulate(self):
        for _ in range(self.substeps):
            wp.launch(_drive_kernel, dim=2, inputs=[self.time, self.input_body, self.input_base, self.drive_radius,
                      self.sim_dt], outputs=[self.target, self.state_0.body_q, self.state_1.body_q,
                                             self.state_0.body_qd, self.state_1.body_qd], device=self.model.device)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(_advance_kernel, dim=1, inputs=[self.time, self.sim_dt], device=self.model.device)

    def step(self):
        super().step()
        q = self.state_0.body_q.numpy()
        err = float(np.linalg.norm(q[self.table, :2] - self.target.numpy()[0]))
        self.errors.append(err)
        x = self.state_0.particle_q.numpy()
        for name, v in (("cable z", np.abs(x[:, 2] - self.pulley_z).max()), ("cable xy", np.abs(x[:, :2]).max()),
                        ("slide x", abs(q[self.slide, 0])), ("table y", abs(q[self.table, 1])),
                        ("slide z", abs(q[self.slide, 2] - self.slide_z)),
                        ("table z", abs(q[self.table, 2] - self.table_z)), ("error", err)):
            self.worst[name] = max(self.worst.get(name, 0.0), float(v))

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite cable"
        w, tol = self.worst, JOINT_LIMIT_TOLERANCE
        assert w["cable z"] <= tol, f"cable {w['cable z'] * 1e3:.2f} mm out of the pulleys' plane"
        assert w["cable xy"] <= CABLE_XY_ABS_BOUND, f"cable {w['cable xy']:.3f} m out of the workspace"
        assert w["slide x"] <= TABLE_RECT_HALF_X + tol and w["table y"] <= TABLE_RECT_HALF_Y + tol, w
        assert w["slide z"] <= tol and w["table z"] <= tol, w
        rms = float(np.sqrt(np.mean(np.square(self.errors))))
        assert w["error"] <= TABLE_TRACKING_MAX_ERROR_TOLERANCE, f"table {w['error'] * 1e3:.2f} mm off its path"
        assert rms <= TABLE_TRACKING_RMS_ERROR_TOLERANCE, f"table {rms * 1e3:.2f} mm RMS off its path"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(TABLE_RECT_PERIOD * 60))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
