"""A Franka FR3 picks up a DER rod from the floor, carries it and puts it down elsewhere.

The arm (URDF from Newton's assets) is MuJoCo's, driven by joint position servos whose targets come
from Newton's GPU IK on a keyframed hand pose; the rod is the ADMM DER solver's. They are coupled
two-way by :class:`~dismech_newton.coupling.MuJoCoCoupledSolver`: the fingers stop on the rod and
squeeze it with their servo force, and friction lifts it. The arm's effective inverse mass, which
changes as it moves, is refreshed every frame.

    uv run examples/franka_rod.py
    uv run examples/franka_rod.py --world-count 4
    uv run examples/franka_rod.py --viewer null --test
"""

import newton
import newton.examples
import newton.utils
import numpy as np
import warp as wp
from common import CableExample
from newton import ik
from newton.solvers import SolverMuJoCo

from dismech_newton import ADMMDiSMechSolver
from dismech_newton.coupling import MuJoCoCoupledSolver

# Arm (7) and finger (2) joints: a ready pose, hand above the workspace.
FRANKA_Q = [0.0, 0.024, 0.0, -2.368, 0.0, 2.392, 0.785, 0.04, 0.04]
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0  # finger joint targets [m]
DOWN = (1.0, 0.0, 0.0, 0.0)  # hand quaternion (x, y, z, w): its z axis to -z, fingers closing along y
TCP = 0.107  # hand origin to the point between the fingertips [m]

RADIUS, LENGTH, SEGMENTS = 0.005, 0.38, 30
PICK, PLACE = (0.5, 0.0), (0.4, 0.3)
ABOVE = 0.2


def add_franka(builder: newton.ModelBuilder) -> None:
    builder.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
                     floating=False, enable_self_collisions=False)
    n = len(FRANKA_Q)
    builder.joint_q[-n:] = FRANKA_Q
    builder.joint_target_q[-n:] = FRANKA_Q


class Example(CableExample):
    substeps = 8

    def __init__(self, viewer, args=None):
        self.worlds = getattr(args, "world_count", None) or 1
        template = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template)
        add_franka(template)
        # Stiff arm servos with gravity compensation; the finger servos set the grip force.
        template.joint_target_ke[:7] = [4.0e3] * 7
        template.joint_target_kd[:7] = [2.0e2] * 7
        template.joint_target_ke[7:9] = [1.0e3] * 2  # 5 mm past the rod: 5 N per finger
        template.joint_target_kd[7:9] = [50.0] * 2
        template.joint_armature[:7] = [0.1] * 7
        gravcomp = template.custom_attributes["mujoco:gravcomp"]
        gravcomp.values = dict.fromkeys(range(template.body_count), 1.0)
        points = np.column_stack((np.linspace(-0.5 * LENGTH, 0.5 * LENGTH, SEGMENTS + 1) + PICK[0],
                                  np.full(SEGMENTS + 1, PICK[1]), np.full(SEGMENTS + 1, RADIUS)))
        ADMMDiSMechSolver.add_rod(template, newton.Rod(points, radius=RADIUS), stretch_stiffness=1.0e5,
                                  bend_stiffness=0.1, bend_damping=0.005)
        self.rod_nodes = SEGMENTS + 1

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.rigid_gap = 0.0
        builder.replicate(template, self.worlds)
        builder.add_ground_plane()
        model = builder.finalize()
        solver = MuJoCoCoupledSolver(model, rod_options={"friction": 1.0},
                                     mujoco_options={"cone": "elliptic", "nconmax": 128, "njmax": 512})
        self.control = model.control()
        self.start(viewer, model, solver, RADIUS)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        self._build_ik(model)
        self._keyframes()
        viewer.set_world_offsets((1.2, 1.2, 0.0))
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(1.45, -0.75, 0.62), pitch=-16.0, yaw=140.0)

    def _build_ik(self, model):
        """Newton's IK on a Franka-only model; the Franka's coordinates lead each world's."""
        arm = newton.ModelBuilder()
        add_franka(arm)
        self.ik_model = arm.finalize(device=model.device)
        n = self.ik_model.joint_coord_count
        self.n_coords = n
        q = model.joint_q.reshape((self.worlds, -1))
        self.ik_q = wp.clone(q[:, :n])
        self.target_q = self.control.joint_target_q.reshape((self.worlds, -1))
        hand = next(i for i, label in enumerate(self.ik_model.body_label) if label.endswith("fr3_hand"))
        self.ik_pos = wp.zeros(self.worlds, dtype=wp.vec3, device=model.device)
        self.ik_rot = wp.array([wp.vec4(*DOWN)] * self.worlds, dtype=wp.vec4, device=model.device)
        lower = wp.clone(model.joint_limit_lower.reshape((self.worlds, -1))[:, :n]).flatten()
        upper = wp.clone(model.joint_limit_upper.reshape((self.worlds, -1))[:, :n]).flatten()
        self.ik = ik.IKSolver(
            model=self.ik_model, n_problems=self.worlds,
            objectives=[
                ik.IKObjectivePosition(link_index=hand, link_offset=wp.vec3(0.0, 0.0, TCP),
                                       target_positions=self.ik_pos),
                ik.IKObjectiveRotation(link_index=hand, link_offset_rotation=wp.quat_identity(),
                                       target_rotations=self.ik_rot),
                ik.IKObjectiveJointLimit(joint_limit_lower=lower, joint_limit_upper=upper, weight=10.0),
            ],
            lambda_initial=0.05, jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.finger = wp.full(1, GRIP_OPEN, dtype=float, device=model.device)

    def _keyframes(self):
        """``(duration, x, y, z, finger)`` of the point between the fingertips."""
        grasp = RADIUS + 0.003  # fingertips just clear of the floor
        keys = [
            (1.0, *PICK, ABOVE, GRIP_OPEN),  # above the rod
            (0.8, *PICK, grasp, GRIP_OPEN),  # down around it
            (0.8, *PICK, grasp, GRIP_CLOSED),  # grip
            (1.0, *PICK, ABOVE, GRIP_CLOSED),  # lift
            (1.2, *PLACE, ABOVE, GRIP_CLOSED),  # carry
            (1.0, *PLACE, grasp + 0.002, GRIP_CLOSED),  # lower
            (0.6, *PLACE, grasp + 0.002, GRIP_OPEN),  # release
            (0.8, *PLACE, ABOVE, GRIP_OPEN),  # retract
        ]
        self.key_times = np.cumsum([k[0] for k in keys])
        self.keys = np.array([k[1:] for k in keys])
        self.duration = float(self.key_times[-1])

    def drive(self, t0, t1):
        t = min(t1, self.duration - 1.0e-6)
        i = int(np.searchsorted(self.key_times, t))
        start = self.key_times[i - 1] if i else 0.0
        s = (t - start) / (self.key_times[i] - start)
        s = s * s * (3.0 - 2.0 * s)
        prev = self.keys[i - 1] if i else self.keys[0]
        x, y, z, finger = (1.0 - s) * prev + s * self.keys[i]
        self.ik_pos.fill_(wp.vec3(x, y, z))
        self.finger.fill_(float(finger))
        self.solver.update_effective_mass()

    def simulate(self):
        self.ik.step(self.ik_q, self.ik_q, iterations=24)
        wp.launch(_set_targets, dim=(self.worlds, self.n_coords), inputs=[self.ik_q, self.finger, self.n_coords],
                  outputs=[self.target_q], device=self.model.device)
        for _ in range(self.substeps):
            self.pipeline.collide(self.state_0, self.contacts, dt=2.0 * self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def test_final(self):
        x = self.state_0.particle_q.numpy().reshape(self.worlds, self.rod_nodes, 3)
        assert np.isfinite(x).all(), "non-finite positions"
        assert x[..., 2].min() > 0.0, "a rod penetrated the ground"
        if self.sim_time >= self.duration:
            mid = x[:, self.rod_nodes // 2, :2]
            err = np.linalg.norm(mid - np.array(PLACE), axis=1)
            assert np.all(err < 0.02), f"rods placed {err * 100} cm from the target"


@wp.kernel
def _set_targets(ik_q: wp.array2d[float], finger: wp.array[float], n: int, target: wp.array2d[float]):
    world, k = wp.tid()
    if k < n - 2:
        target[world, k] = ik_q[world, k]
    else:
        target[world, k] = finger[0]


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    newton.examples.add_world_count_arg(parser)
    parser.set_defaults(num_frames=480, world_count=1)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
