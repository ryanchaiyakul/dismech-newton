"""A Franka FR3 picks up a rod from the floor, carries it and puts it down elsewhere.

Shows: two-way coupling with MuJoCo bodies (:class:`~dismech_newton.experimental.mujoco.MuJoCoCoupledSolver`).
The arm's servo targets come from Newton's IK on a keyframed hand pose; the fingers squeeze the rod with
their servo force and friction lifts it.

    uv run --extra mujoco examples/experimental/franka_rod.py
    uv run --extra mujoco examples/experimental/franka_rod.py --world-count 4
    uv run --extra mujoco examples/experimental/franka_rod.py --viewer null --test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # examples/, for utils

import newton
import newton.examples
import newton.utils
import numpy as np
import warp as wp
from newton import ik
from newton.solvers import SolverMuJoCo
from utils.common import SIM, THEORY, CableExample, inset, inset_scale, smoothstep

from dismech_newton import ADMMDiSMechSolver
from dismech_newton.experimental.mujoco import MuJoCoCoupledSolver

FRANKA_Q = [0.0, 0.024, 0.0, -2.368, 0.0, 2.392, 0.785, 0.04, 0.04]  # arm (7), fingers (2): hand above the floor
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0  # finger joint targets [m]
DOWN = (1.0, 0.0, 0.0, 0.0)  # hand quaternion (x, y, z, w): its z axis to -z, fingers closing along y
TCP = 0.107  # hand origin to the point between the fingertips [m]


def add_franka(builder: newton.ModelBuilder) -> None:
    builder.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
                     floating=False, enable_self_collisions=False)
    n = len(FRANKA_Q)
    builder.joint_q[-n:] = FRANKA_Q
    builder.joint_target_q[-n:] = FRANKA_Q


class Example(CableExample):
    substeps = 8
    plot_every, plot_name = 10, "grip"
    radius, length, segments = 0.005, 0.38, 30
    pick, place, above = (0.5, 0.0), (0.4, 0.3), 0.2

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
        template.custom_attributes["mujoco:gravcomp"].values = dict.fromkeys(range(template.body_count), 1.0)
        n = self.segments + 1
        points = np.column_stack((np.linspace(-0.5, 0.5, n) * self.length + self.pick[0],
                                  np.full(n, self.pick[1]), np.full(n, self.radius)))
        ADMMDiSMechSolver.add_rod(template, newton.Rod(points, radius=self.radius), stretch_stiffness=1.0e5,
                                  bend_stiffness=0.1, bend_damping=0.005)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.rigid_gap = 0.0
        builder.replicate(template, self.worlds)
        builder.add_ground_plane()
        model = builder.finalize()
        solver = MuJoCoCoupledSolver(model, rod_options={"friction": 1.0},
                                     mujoco_options={"cone": "elliptic", "nconmax": 128, "njmax": 512})
        self.control = model.control()
        self.start(viewer, model, solver, self.radius)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        self._build_ik(model)
        self._keyframes()
        # The rod's contact reactions on world 0's fingers, read after every frame: the grip and the lift.
        self.fingers = [i for i, label in enumerate(model.body_label[: model.body_count // self.worlds])
                        if "finger" in label]
        self._reaction = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        mass = solver.rod.mass.numpy()[: 3 * model.particle_count].sum() / 3.0 / self.worlds
        self.weight = mass * float(np.linalg.norm(model.gravity.numpy()[0]))  # [N]
        # A finger servo held at the rod's surface, its radius short of the closed target.
        self.grip_target = float(template.joint_target_ke[7]) * (self.radius - GRIP_CLOSED)  # [N] per finger
        self.history = []  # (time, grip per finger, lift)
        if getattr(viewer, "camera", None) is not None:
            viewer.set_world_offsets((1.2, 1.2, 0.0))
            viewer.set_camera(pos=wp.vec3(1.45, -0.75, 0.62), pitch=-16.0, yaw=140.0)

    def _build_ik(self, model):
        """Newton's IK on a Franka-only model; the Franka's coordinates lead each world's."""
        arm = newton.ModelBuilder()
        add_franka(arm)
        ik_model = arm.finalize(device=model.device)
        n = self.n_coords = ik_model.joint_coord_count
        self.ik_q = wp.clone(model.joint_q.reshape((self.worlds, -1))[:, :n])
        self.target_q = self.control.joint_target_q.reshape((self.worlds, -1))
        hand = next(i for i, label in enumerate(ik_model.body_label) if label.endswith("fr3_hand"))
        self.ik_pos = wp.zeros(self.worlds, dtype=wp.vec3, device=model.device)
        rot = wp.array([wp.vec4(*DOWN)] * self.worlds, dtype=wp.vec4, device=model.device)
        lower = wp.clone(model.joint_limit_lower.reshape((self.worlds, -1))[:, :n]).flatten()
        upper = wp.clone(model.joint_limit_upper.reshape((self.worlds, -1))[:, :n]).flatten()
        self.ik = ik.IKSolver(
            model=ik_model, n_problems=self.worlds,
            objectives=[
                ik.IKObjectivePosition(link_index=hand, link_offset=wp.vec3(0.0, 0.0, TCP), target_positions=self.ik_pos),
                ik.IKObjectiveRotation(link_index=hand, link_offset_rotation=wp.quat_identity(), target_rotations=rot),
                ik.IKObjectiveJointLimit(joint_limit_lower=lower, joint_limit_upper=upper, weight=10.0),
            ],
            lambda_initial=0.05, jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.finger = wp.full(1, GRIP_OPEN, dtype=float, device=model.device)

    def _keyframes(self):
        """``(duration, x, y, z, finger)`` of the point between the fingertips."""
        grasp, place = self.radius + 0.003, self.radius + 0.005  # fingertips just clear of the floor
        keys = [
            (1.0, *self.pick, self.above, GRIP_OPEN),  # above the rod
            (0.8, *self.pick, grasp, GRIP_OPEN),  # down around it
            (0.8, *self.pick, grasp, GRIP_CLOSED),  # grip
            (1.0, *self.pick, self.above, GRIP_CLOSED),  # lift
            (1.2, *self.place, self.above, GRIP_CLOSED),  # carry
            (1.0, *self.place, place, GRIP_CLOSED),  # lower
            (0.6, *self.place, place, GRIP_OPEN),  # release
            (0.8, *self.place, self.above, GRIP_OPEN),  # retract
        ]
        self.key_times = np.concatenate(([0.0], np.cumsum([k[0] for k in keys])))
        self.keys = np.array([keys[0][1:], *(k[1:] for k in keys)])
        self.duration = float(self.key_times[-1])

    def drive(self, t0, t1):
        i = int(np.clip(np.searchsorted(self.key_times, t1), 1, len(self.keys) - 1))
        s = smoothstep(t1, self.key_times[i - 1], self.key_times[i])
        x, y, z, finger = self.keys[i - 1] + s * (self.keys[i] - self.keys[i - 1])
        self.ik_pos.fill_(wp.vec3(x, y, z))
        self.finger.fill_(float(finger))

    def simulate(self):
        self.ik.step(self.ik_q, self.ik_q, iterations=24)
        wp.launch(_set_targets, dim=(self.worlds, self.n_coords), inputs=[self.ik_q, self.finger],
                  outputs=[self.target_q])
        super().simulate()

    def post_substep(self):
        self.solver.rod.add_contact_reactions(self._reaction)  # summed over the frame's substeps

    def step(self):
        super().step()
        # The frame's mean force on each finger (its impulse over the frame): (force, torque) rows.
        f = self._reaction.numpy()[self.fingers, :3] / self.substeps
        self._reaction.zero_()
        self.history.append((self.sim_time, float(np.mean(np.abs(f[:, 1]))), float(-f[:, 2].sum())))

    def image(self, size: tuple[int, int] = (800, 400)) -> np.ndarray:
        """The rod's reactions on the fingers over time beside what they should be: each finger's grip
        (normal, along y) against its servo force, and their lift against the rod's weight."""
        t, grip, lift = np.array(self.history).T if self.history else np.zeros((3, 0))
        held = (self.key_times[3], self.key_times[6])  # gripped, from the lift to the release

        def draw(axes):
            for ax, y, ref, label, ylabel in (
                (axes[0], grip, self.grip_target, "finger servo", r"grip per finger $F_n$ (N)"),
                (axes[1], lift, self.weight, r"rod weight $mg$", r"lift $F_z$ (N)"),
            ):
                ax.axvspan(*held, color="white", alpha=0.06, lw=0)
                ax.axhline(ref, color=THEORY, ls="--", lw=1.2 * inset_scale(size, 2), label=label)
                ax.plot(t, y, color=SIM, lw=1.4 * inset_scale(size, 2), label="simulation")
                ax.set(xlim=(0.0, self.duration), ylim=(0.0, 2.0 * ref))
                ax.set_xlabel(r"time $t$ (s)")
                ax.set_ylabel(ylabel)
                ax.locator_params(nbins=4)
                ax.legend(loc="upper left")

        return inset(draw, size, ncols=2)

    def test_final(self):
        x = self.state_0.particle_q.numpy().reshape(self.worlds, self.segments + 1, 3)
        assert np.isfinite(x).all(), "non-finite positions"
        assert x[..., 2].min() > 0.0, "a rod penetrated the ground"
        if self.sim_time >= self.duration:
            err = np.linalg.norm(x[:, self.segments // 2, :2] - np.array(self.place), axis=1)
            assert np.all(err < 0.02), f"rods placed {err * 100} cm from the target"


@wp.kernel
def _set_targets(ik_q: wp.array2d[float], finger: wp.array[float], target: wp.array2d[float]):
    """The IK's arm joints, the drive's finger opening."""
    world, k = wp.tid()
    if k < ik_q.shape[1] - 2:
        target[world, k] = ik_q[world, k]
    else:
        target[world, k] = finger[0]


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    newton.examples.add_world_count_arg(parser)
    parser.set_defaults(num_frames=480, world_count=1)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
