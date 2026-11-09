"""A Franka folds a napkin in half over a table's edge: the napkin (10 x 30 cm, cotton) lies half on a thin
tabletop, half hanging over its edge. The hand, pointing along the edge, slides its open fingers onto the hanging
hem's corner from the side and pinches it, then turns the flap over like a page, round the edge, turning with it,
and lets go above the napkin's other half.

A parallel gripper cannot pick a flat sheet off a table: no edge to get around (Newton's ``cloth_franka`` has
kinematic fingertips that pass through cloth and table; two-way coupled ones rest on the cloth and close over it).
A hanging flap has two free sides, and a thin tabletop leaves room under it for the inner finger. A downward hand
cannot straddle it (the cloth above runs into the palm), so the hand comes from the side, the hem's corner between
its fingertips. Turning the hand with the page keeps the cloth leaving the pinch toward the fold: at the end the
pads stand upright, the cloth hangs out of their bottom, and opening them drops it (a hand held level carries the
cloth draped over a finger, and lifts it back off when it opens).

The arm is MuJoCo's (IsaacLab's gains, gravity compensated: the cable pick's), its joint targets from Newton's IK on
the keyframed point between the fingertips; the napkin a midedge shell, resting flat, starting draped; the gripper
bodies proxies in our solve through Newton's ``SolverCoupledProxy``: the fingers squeeze the napkin with their servo
force and friction carries it. The tabletop's -y side is flush with the napkin's: the palm, upright at the end,
passes beside it.

Shows: a two-way pinch grasp of a sheet. Tests: while pinched, each finger's grip is its servo force at the
napkin's contact thickness (10%; measured 4% over 4 runs); folded in half: the hem's corner within 2 cm of the
napkin's far corner (measured 0.8 cm), the flap flat on the other half (within 2 cm of the table); no crossings.

    uv run --extra mujoco examples/newton/cloth_franka_fold.py
    uv run --extra mujoco examples/newton/cloth_franka_fold.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from cloth_twist import Crossings
from franka_cable_ik_pick_place import TCP, _set_targets, add_franka
from newton import ik
from newton.examples.multiphysics.example_franka_cable_ik_pick_place import GRIP_HOLD, GRIP_OPEN, GRIP_STIFFNESS
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample, grid, pads

from dismech_newton import SolverDiSMech

# the hand pointing along +y (approach), its fingers closing along x (across the flap): hand (x, y, z) -> (z, x, y)
GRIPPER_ALONG_EDGE = (-0.5, -0.5, -0.5, 0.5)  # (qx, qy, qz, qw)


class Example(CableExample):
    substeps = 10
    top, slab, edge = 0.30, 0.03, 0.45  # tabletop height, thickness, its edge's x [m]
    width, on_table, length, cell = 0.10, 0.15, 0.30, 0.01  # the napkin along the edge (y), across it (x) [m]
    y0, bend = 0.22, 0.015  # the napkin's -y side (clear of the start pose); the drape's radius round the edge [m]
    density, bend_stiffness, damping = 0.2, 1.0e-5, 1.0e-3  # cotton: kg/m^2, N m (bending length 1.7 cm), s
    radius, mu = 0.002, 1.0  # contact thickness [m], friction (fingers, table, itself)

    def __init__(self, viewer, args=None):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(builder)
        builder.rigid_gap = 0.0
        add_franka(builder, 0.0)
        builder.joint_target_ke[:7] = [400.0] * 7
        builder.joint_target_kd[:7] = [80.0] * 7
        builder.joint_target_ke[7:9] = [GRIP_STIFFNESS] * 2
        builder.joint_target_kd[7:9] = [100.0] * 2
        builder.joint_effort_limit[:4] = [87.0] * 4
        builder.joint_effort_limit[4:7] = [12.0] * 3
        builder.joint_effort_limit[7:9] = [1500.0] * 2
        builder.joint_armature[:7] = [1.0e-3] * 7
        builder.joint_armature[7:9] = [0.0] * 2
        arm = builder.body_count
        builder.custom_attributes["mujoco:gravcomp"].values = dict.fromkeys(range(arm), 1.0)
        # a thin tabletop overhanging its (unmodelled) legs, its -y side flush with the napkin's (the turned palm
        # passes beside it)
        y_lo, y_hi = self.y0, self.y0 + self.width + 0.25
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(self.edge + 0.25, 0.5 * (y_lo + y_hi),
                                                             self.top - 0.5 * self.slab), wp.quat_identity()),
                              hx=0.25, hy=0.5 * (y_hi - y_lo), hz=0.5 * self.slab)
        nx, ny = round(self.length / self.cell) + 1, round(self.width / self.cell) + 1
        flat, faces = grid(nx, ny, self.length, self.width)
        ids = SolverDiSMech.add_cloth_mesh(  # rests flat
            builder, vertices=[wp.vec3(v) for v in flat], indices=faces.ravel(), particle_radius=self.radius,
            density=self.density, bend_stiffness=self.bend_stiffness, damping=self.damping,
            tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0, edge_ke=0.0, edge_kd=0.0,
        )
        draped = self.drape(flat)
        for k, p in zip(ids, draped):
            builder.particle_q[k] = wp.vec3(*p)
        self.ids, self.nx, self.ny = ids, nx, ny
        builder.add_ground_plane()
        model = builder.finalize()
        pads(model, [b for b, label in enumerate(model.body_label) if "finger" in label])  # grip pads
        model.particle_mu = self.mu
        model.shape_material_mu.fill_(self.mu)
        self.control = model.control()
        links = list(range(arm))
        gripper = [b for b in links if "hand" in model.body_label[b] or "finger" in model.body_label[b]]

        def mujoco(m):
            return SolverMuJoCo(m, solver="newton", integrator="implicitfast", cone="elliptic", iterations=100,
                                ls_iterations=20, use_mujoco_contacts=True, njmax=256, nconmax=64)

        solver = SolverCoupledProxy(model, entries=[
            SolverCoupled.Entry(name="mjc", solver=mujoco, bodies=links, joints=list(range(model.joint_count))),
            SolverCoupled.Entry(name="soft", solver=lambda m: SolverDiSMech(m, refactor=True),
                                particles=list(range(model.particle_count))),
        ], coupling=SolverCoupledProxy.Config(proxies=[
            SolverCoupledProxy.Proxy(source="mjc", destination="soft", bodies=gripper)]))
        self.soft = solver.solver("soft")
        self.fingers = [b for b, label in enumerate(solver.view("soft").body_label) if "finger" in label]
        self.hand = next(b for b in links if model.body_label[b].endswith("fr3_hand"))
        self.start(viewer, model, solver)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_1)
        self._build_ik(model)
        self._keyframes()
        self.weight = 9.81 * float(model.particle_mass.numpy().sum())
        self.servo = GRIP_STIFFNESS * (self.radius - GRIP_HOLD)  # each finger's force, stopped on the napkin
        self.history = []  # (time, grip per finger)
        self.crossings, self.cross = Crossings(_edges(faces), faces, model.device), 0
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(-0.25, -0.75, 0.75), pitch=-28.0, yaw=60.0)

    def drape(self, flat):
        """The flat napkin over the edge: ``on_table`` of it on the top, a quarter circle of radius ``bend`` round
        the edge, the rest hanging; arc length along x from the far side, so it starts unstretched."""
        s, y = flat[:, 0], self.y0 + self.width - flat[:, 1]  # s from the far side (the hem at s = length); y, x
        # both mirrored: the flat grid turned half round, faces up
        a, r, z = self.on_table, self.bend, self.top + self.radius
        phi = np.clip((s - a) / r, 0.0, 0.5 * np.pi)
        hang = np.maximum(s - a - 0.5 * np.pi * r, 0.0)
        x = np.where(s < a, self.edge + a - s, self.edge - r * np.sin(phi))
        z = np.where(s < a, z, z - r + r * np.cos(phi) - hang)
        return np.column_stack([x, y, z])

    def _build_ik(self, model):
        """Newton's IK (the cable pick's) on a Franka-only model, the hand along the edge."""
        arm = newton.ModelBuilder()
        add_franka(arm, 0.0)
        ik_model = arm.finalize(device=model.device)
        n = self.n_coords = ik_model.joint_coord_count
        self.ik_q = wp.clone(model.joint_q.reshape((1, -1))[:, :n])
        self.target_q = self.control.joint_target_q.reshape((1, -1))
        hand = next(i for i, label in enumerate(ik_model.body_label) if label.endswith("fr3_hand"))
        self.ik_pos = wp.zeros(1, dtype=wp.vec3, device=model.device)
        rot = self.ik_rot = wp.array([wp.vec4(*GRIPPER_ALONG_EDGE)], dtype=wp.vec4, device=model.device)
        lower = wp.clone(model.joint_limit_lower.reshape((1, -1))[:, :n]).flatten()
        upper = wp.clone(model.joint_limit_upper.reshape((1, -1))[:, :n]).flatten()
        self.ik = ik.IKSolver(
            model=ik_model, n_problems=1,
            objectives=[
                ik.IKObjectivePosition(link_index=hand, link_offset=wp.vec3(0.0, 0.0, TCP), target_positions=self.ik_pos),
                ik.IKObjectiveRotation(link_index=hand, link_offset_rotation=wp.quat_identity(), target_rotations=rot),
                ik.IKObjectiveJointLimit(joint_limit_lower=lower, joint_limit_upper=upper, weight=10.0),
            ],
            lambda_initial=0.05, jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.finger = wp.full(1, GRIP_OPEN, dtype=float, device=model.device)

    def _keyframes(self):
        """``(duration, x, y, z, finger, turn)`` of the point between the fingertips (``turn``: the hand's turn about
        its own axis, +y), linearly interpolated: beside the hem's corner, slid onto it (the pads over its last 2 cm,
        from both edges), pinched; then turned over like a page round the table's edge (the flap stays spread), the
        hand turning with it so the cloth leaves the pinch toward the fold (to half a turn: the pads upright, the cloth
        hanging out of their bottom), raised over the last quarter to clear the palm off the napkin; released (it
        drops out), up and away."""
        hem = self.drape(np.array([[self.length, 0.0, 0.0]]))[0]
        gx, gy, gz = hem[0], self.y0 + 0.03, hem[2] + 0.012
        up, lift, n = self.top + 0.15, 0.035, 18
        corner = np.array([self.edge, self.top])  # (x, z) the page's hinge
        rho = 0.92 * np.hypot(gx - corner[0], gz - corner[1])  # slack: the fold takes some cloth
        keys = [
            (1.5, gx - 0.13, gy - 0.15, up, GRIP_OPEN, 0.0),  # clear of the table, beside the napkin
            (1.0, gx, gy - 0.13, gz, GRIP_OPEN, 0.0),  # down beside the hem's corner
            (1.0, gx, gy, gz, GRIP_OPEN, 0.0),  # slide the fingers onto it
            (1.0, gx, gy, gz, GRIP_HOLD, 0.0),  # pinch
        ]
        start = float(sum(k[0] for k in keys))
        self.hold = (start + 0.3, start + 0.25 * n)  # pinched, turning over
        for phi in np.linspace(1.5 * np.pi / n, 1.5 * np.pi, n):  # from hanging (0) out, up and over (270 degrees)
            x, z = corner + rho * np.array([-np.sin(phi), -np.cos(phi)])
            z += lift * np.clip((phi - np.pi) / (0.5 * np.pi), 0.0, 1.0)
            keys.append((0.25, x, gy, z, GRIP_HOLD, min(phi, np.pi)))
        self.target = np.array([self.edge + self.on_table, self.y0])  # the corner's, folded in half: the far corner
        keys += [
            (0.5, x, gy, z, GRIP_HOLD, np.pi),  # settle
            (1.0, x, gy, z, GRIP_OPEN, np.pi),  # release
            (1.0, x, gy, up, GRIP_OPEN, np.pi),  # up
            (1.0, x, gy - 0.12, up, GRIP_OPEN, np.pi),  # away
        ]
        self.key_times = np.cumsum([k[0] for k in keys])
        self.keys = np.array([k[1:] for k in keys])
        self.release = float(self.key_times[-4])

    def drive(self, t0, t1):
        """The keyframes at the frame's start (the cable pick's interpolation, with the turn)."""
        t = min(t0, float(self.key_times[-1]) - 1.0e-6)
        i = int(np.searchsorted(self.key_times, t))
        start = self.key_times[i - 1] if i > 0 else 0.0
        a = float(np.clip((t - start) / max(self.key_times[i] - start, 1.0e-6), 0.0, 1.0))
        x, y, z, finger, turn = (1.0 - a) * self.keys[max(i - 1, 0)] + a * self.keys[i]
        self.ik_pos.fill_(wp.vec3(x, y, z))
        self.finger.fill_(float(finger))
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(turn)) * wp.quat(*GRIPPER_ALONG_EDGE)
        self.ik_rot.fill_(wp.vec4(q[0], q[1], q[2], q[3]))

    def simulate(self):
        self.ik.step(self.ik_q, self.ik_q, iterations=24)
        wp.launch(_set_targets, dim=self.n_coords, inputs=[self.ik_q, self.finger], outputs=[self.target_q])
        super().simulate()

    def step(self):
        super().step()
        f = self.soft.body_reaction.numpy()[self.fingers, :3]  # the last substep's, its pads' normal component
        q = self.state_0.body_q.numpy()[self.hand, 3:]  # the hand's (x, y, z, w): its y the fingers' axis
        closing = np.array(wp.quat_rotate(wp.quat(*q), wp.vec3(0.0, 1.0, 0.0)))
        qf = self.state_0.joint_q.numpy()[7:9]  # the fingers' openings: the servo's k (q - target) at them
        self.history.append((self.sim_time, float(np.mean(np.abs(f @ closing))),
                             GRIP_STIFFNESS * (float(qf.mean()) - GRIP_HOLD)))
        self.cross = max(self.cross, self.crossings(self.state_0.particle_q))

    def corner(self) -> np.ndarray:
        """The grasped corner's position (the hem's, on the -y side)."""
        return self.state_0.particle_q.numpy()[self.ids[self.nx * self.ny - 1]]

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        assert self.cross == 0, f"the napkin passed through itself ({self.cross} crossings)"
        over = x[:, 0] > self.edge + 0.01
        assert (x[over, 2] > self.top).all(), "the napkin went into the table"
        h = np.array(self.history)
        held = (h[:, 0] > self.hold[0]) & (h[:, 0] < self.hold[1])
        grip = h[held, 1]
        servo = h[held, 2]  # at the fingers' openings (a compliant pad: the finger sinks, the servo eases)
        assert np.allclose(grip, servo, rtol=0.1), (
            f"grip {grip.min():.3f}-{grip.max():.3f} N, servo {servo.min():.3f}-{servo.max():.3f} N")
        if self.sim_time > self.release + 1.0:  # folded in half: the hem on the far edge, the flap flat on the rest
            err = float(np.linalg.norm(self.corner()[:2] - self.target))
            assert err < 0.02, f"corner laid {err * 100:.1f} cm from the napkin's far corner"
            flap = x[np.asarray(self.ids).reshape(self.nx, self.ny)[round(self.on_table / self.cell) + 3:].ravel()]
            lo = np.array([self.edge, self.y0 - 0.01, self.top])
            hi = np.array([self.target[0] + 0.02, self.y0 + self.width + 0.01, self.top + 0.02])
            assert ((flap > lo) & (flap < hi)).all(), f"the flap is not flat on the napkin: {flap.min(0)} {flap.max(0)}"


def _edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=720)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
