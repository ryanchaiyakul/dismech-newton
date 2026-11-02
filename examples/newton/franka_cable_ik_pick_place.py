"""A Franka FR3 picks a cable off a table at its middle, carries it and puts it down (Newton's
``example_franka_cable_ik_pick_place``): the same arm, gains, gravity compensation, cable, keyframes and test. The
arm is MuJoCo's, its joint targets from Newton's IK on the keyframed hand pose; the cable is a DER rod in place of
Newton's VBD cable, coupled two-way through Newton's ``SolverCoupledProxy`` with Newton's config (the gripper bodies
proxies in our solve, lagged, one pass): the fingers squeeze it with their servo force and friction lifts it.

Shows: a pinch through two-way coupling with an articulated MuJoCo arm. Newton's test (the cable's middle placed
within 1 cm of the target) and our own: while held, each finger's grip is its servo force and the lift the cable's
weight.

    uv run --extra mujoco examples/newton/franka_cable_ik_pick_place.py
    uv run --extra mujoco examples/newton/franka_cable_ik_pick_place.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import newton.utils
import numpy as np
import warp as wp
from newton import ik
from newton.examples.multiphysics.example_franka_cable_ik_pick_place import (
    CABLE_CENTER,
    CABLE_LENGTH,
    FRANKA_Q,
    GRIP_CLOSE,
    GRIP_HOLD,
    GRIP_OPEN,
    GRIP_STIFFNESS,
    GRIPPER_DOWN,
)
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import SolverDiSMech

TCP = 0.107  # hand origin to the point between the fingertips [m]


def add_franka(builder: newton.ModelBuilder, base_z: float) -> None:
    builder.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
                     xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()), floating=False,
                     enable_self_collisions=False, parse_visuals_as_colliders=False,
                     joint_damping_mode="drive")  # Newton 1.6's mapping (the URDF damping drives the joints)
    n = len(FRANKA_Q)
    builder.joint_q[-n:] = FRANKA_Q
    builder.joint_target_q[-n:] = FRANKA_Q


class Example(CableExample):
    substeps = 10  # Newton's: 60 fps x 10
    radius, segments = 0.005, 19

    def __init__(self, viewer, args=None):
        self.surface = float(CABLE_CENTER[2]) - self.radius
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(builder)
        builder.rigid_gap = 0.0
        add_franka(builder, self.surface)
        # Newton's (IsaacLab's) gains: arm PD with full gravity compensation, stiff fingers.
        builder.joint_target_ke[:7] = [400.0] * 7
        builder.joint_target_kd[:7] = [80.0] * 7
        builder.joint_target_ke[7:9] = [GRIP_STIFFNESS] * 2
        builder.joint_target_kd[7:9] = [100.0] * 2
        builder.joint_effort_limit[:4] = [87.0] * 4
        builder.joint_effort_limit[4:7] = [12.0] * 3
        builder.joint_effort_limit[7:9] = [1500.0] * 2
        builder.joint_armature[:7] = [1.0e-3] * 7
        builder.joint_armature[7:9] = [0.0] * 2
        builder.custom_attributes["mujoco:gravcomp"].values = dict.fromkeys(range(builder.body_count), 1.0)
        n = self.segments + 1
        x = float(CABLE_CENTER[0]) + (np.linspace(0.0, 1.0, n) - 0.5) * CABLE_LENGTH
        points = np.column_stack((x, np.full(n, float(CABLE_CENTER[1])), np.full(n, float(CABLE_CENTER[2]))))
        cfg = newton.ModelBuilder.ShapeConfig(density=100.0, mu=1.0)
        r = self.segments / 19  # Newton's per-joint values are for its 19 segments: the same cable (EA, EI) at others
        SolverDiSMech.add_rod(builder, rod=newton.Rod(points, radius=self.radius), cfg=cfg, stretch_stiffness=1.0e2 * r,
                              stretch_damping=1.0e-1 * r, bend_stiffness=4.0e-4 * r, bend_damping=8.0e-7 * r,
                              collide=True)
        builder.add_ground_plane(height=self.surface, cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
        model = builder.finalize()
        self.control = model.control()
        def mujoco(m):
            return SolverMuJoCo(m, solver="newton", integrator="implicitfast", cone="elliptic", iterations=100,
                                ls_iterations=20, use_mujoco_contacts=True, njmax=256, nconmax=64)

        solver = self._proxy(model, mujoco)
        self.soft = solver.solver("soft")
        self.fingers = [b for b, label in enumerate(solver.view("soft").body_label) if "finger" in label]
        self.start(viewer, model, solver)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_1)
        self._build_ik(model)
        self._keyframes()
        mass = float(model.particle_mass.numpy().sum())
        self.weight = 9.81 * mass
        self.history = []  # (time, grip per finger, lift)
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.9, -1.4, 0.9), pitch=-22.0, yaw=120.0)

    @staticmethod
    def _proxy(model, mujoco):
        """Newton's SolverCoupledProxy: MuJoCo's arm, our rod (its capsule bodies and their joints), the gripper's
        bodies proxies in our solve."""
        capsules = sorted({int(b) for b in model.dismech.edge_body.numpy() if b >= 0})
        franka = [b for b in range(model.body_count) if b not in capsules]
        child = model.joint_child.numpy()
        joints = [j for j in range(model.joint_count) if child[j] in franka]
        gripper = [b for b in franka if "hand" in model.body_label[b] or "finger" in model.body_label[b]]
        return SolverCoupledProxy(model, entries=[
            SolverCoupled.Entry(name="mjc", solver=mujoco, bodies=franka, joints=joints),
            SolverCoupled.Entry(name="soft", solver=SolverDiSMech, particles=list(range(model.particle_count)),
                                bodies=capsules, joints=[j for j in range(model.joint_count) if j not in joints]),
        ], coupling=SolverCoupledProxy.Config(proxies=[
            SolverCoupledProxy.Proxy(source="mjc", destination="soft", bodies=gripper)]))

    def _build_ik(self, model):
        """Newton's IK on a Franka-only model; the Franka's coordinates lead the model's."""
        arm = newton.ModelBuilder()
        add_franka(arm, self.surface)
        ik_model = arm.finalize(device=model.device)
        n = self.n_coords = ik_model.joint_coord_count
        self.ik_q = wp.clone(model.joint_q.reshape((1, -1))[:, :n])
        self.target_q = self.control.joint_target_q.reshape((1, -1))
        hand = next(i for i, label in enumerate(ik_model.body_label) if label.endswith("fr3_hand"))
        self.ik_pos = wp.zeros(1, dtype=wp.vec3, device=model.device)
        rot = wp.array([wp.vec4(*GRIPPER_DOWN)], dtype=wp.vec4, device=model.device)
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
        """Newton's ``(duration, x, y, z, finger)`` of the point between the fingertips, linearly interpolated."""
        cx, cy, cz = (float(c) for c in CABLE_CENTER)
        up, (tx, ty) = cz + 0.20, (0.4, 0.25)
        self.place = (tx, ty)
        keys = [
            (1.0, cx, cy, up, GRIP_OPEN),  # approach above the cable
            (0.5, cx, cy, cz, GRIP_OPEN),  # descend to it
            (1.0, cx, cy, cz, GRIP_CLOSE),  # grasp
            (1.0, cx, cy, up, GRIP_HOLD),  # lift
            (1.0, tx, ty, up, GRIP_HOLD),  # carry
            (0.5, tx, ty, cz, GRIP_HOLD),  # lower
            (0.5, tx, ty, cz, GRIP_HOLD),  # settle
            (1.0, tx, ty, cz, GRIP_OPEN),  # release
            (0.5, tx, ty, up, GRIP_OPEN),  # retract
        ]
        self.key_times = np.cumsum([k[0] for k in keys])
        self.keys = np.array([k[1:] for k in keys])

    def drive(self, t0, t1):
        """Newton's interpolation, at the frame's start."""
        t = min(t0, float(self.key_times[-1]) - 1.0e-6)
        i = int(np.searchsorted(self.key_times, t))
        start = self.key_times[i - 1] if i > 0 else 0.0
        a = float(np.clip((t - start) / max(self.key_times[i] - start, 1.0e-6), 0.0, 1.0))
        x, y, z, finger = (1.0 - a) * self.keys[max(i - 1, 0)] + a * self.keys[i]
        self.ik_pos.fill_(wp.vec3(x, y, z))
        self.finger.fill_(float(finger))

    def simulate(self):
        self.ik.step(self.ik_q, self.ik_q, iterations=24)
        wp.launch(_set_targets, dim=self.n_coords, inputs=[self.ik_q, self.finger], outputs=[self.target_q])
        super().simulate()

    def step(self):
        super().step()
        f = self.soft.body_reaction.numpy()[self.fingers, :3]  # the last substep's, (force, torque) rows; our view's
        self.history.append((self.sim_time, float(np.mean(np.abs(f[:, 1]))), float(-f[:, 2].sum())))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        assert x[:, 2].min() > self.surface, "the cable penetrated the table"
        mid = 0.5 * (x[self.segments // 2] + x[self.segments // 2 + 1])  # Newton's segment count // 2, at its centre
        err = float(np.linalg.norm(mid[:2] - np.array(self.place)))
        assert err < 0.01, f"cable placed {err * 100:.1f} cm from the target (Newton's test: < 1 cm)"
        h = np.array(self.history)
        held = (h[:, 0] > 3.25) & (h[:, 0] < 4.75)  # lifted clear of the table, carried
        servo = GRIP_STIFFNESS * (self.radius - GRIP_HOLD)
        assert np.allclose(h[held, 1], servo, rtol=0.02), f"grip {h[held, 1].min():.3f}-{h[held, 1].max():.3f} N, "                                                           f"servo {servo:.3f} N"
        lift = h[held, 2].mean()  # the cable swings: its mean
        assert abs(lift - self.weight) < 0.2 * self.weight, f"lift {lift:.4f} N, weight {self.weight:.4f} N"


@wp.kernel
def _set_targets(ik_q: wp.array2d[float], finger: wp.array[float], target: wp.array2d[float]):
    """The IK's arm joints, the drive's finger opening."""
    k = wp.tid()
    if k < ik_q.shape[1] - 2:
        target[0, k] = ik_q[0, k]
    else:
        target[0, k] = finger[0]


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=330)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
