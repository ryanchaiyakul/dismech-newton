"""The MIDAS hand (MuJoCo, github.com/midas-hand-org/midas_hand_mujoco, MIT) grips a rod clamped at one end and
pushes it along its axis until the grip slides: two-way coupling through Newton's ``SolverCoupledProxy``, the hand's
contact bodies proxies in our solve. The fingers wrap the rod against the palm with their servos (13 position
actuators, +-1.03 N m); a stiff prismatic servo pushes the hand toward the clamp at 2 cm/s.

Theory: sliding, the push is Coulomb's mu sum(N), the contacts' normal forces (``rho``, mu the mean of the rod's 0.5
and the hand's 1.0). Two-way: the push measured on MuJoCo's side (the slide servo) equals the rod's axial force on
ours (EA x strain between clamp and hand). ``--gravity``: the same, horizontal under gravity.

The grip's friction converges linearly in the step (majorized), so the step takes 30 iterations; its force balance
is the stopping test.

    uv run --extra mujoco examples/midas_axial.py
    uv run --extra mujoco examples/midas_axial.py --viewer null --test
"""

import io
import urllib.request
import zipfile

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from utils.common import CACHE, SIM, THEORY, CableExample, inset

from dismech_newton import SolverDiSMech

COMMIT = "6c7d17f4a8cb5414e81653aa2795acc9610e1730"
FINGERS = ("index", "middle", "ring")


def hand_xml():
    """The MIDAS hand's MJCF at ``COMMIT``, downloaded once into the examples' cache (52 MB of meshes)."""
    root = CACHE / f"midas_hand_mujoco-{COMMIT}"
    if not root.exists():
        url = f"https://github.com/midas-hand-org/midas_hand_mujoco/archive/{COMMIT}.zip"
        with urllib.request.urlopen(url) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(CACHE)
    return root / "assets" / "midas_description" / "midas_whole_hand.xml"


@wp.kernel
def _carriage(target: wp.array[float], i: int, speed: wp.array[float], dt: float):
    """The slide's target ramps each substep."""
    target[i] = target[i] + speed[0] * dt


@wp.kernel
def _capacity(rho: wp.array[float], mu: wp.array[float], out: wp.array[float]):
    """sum(mu rho), summed over the frame's substeps."""
    i = wp.tid()
    wp.atomic_add(out, 0, mu[i] * rho[i])


class Example(CableExample):
    substeps = 8  # 2 ms (the MJCF's 1 ms: the same within the tests' tolerances; 4 substeps are not)
    radius, segments, youngs, mu = 0.012, 60, 3.0e8, 0.5  # a stiff rod: Euler's load ~kN, it never buckles
    clamp, free, y, z = 0.25, -0.07, 0.085, -0.0315  # along x under the palm, in the fingers' wrap; clamped at +x
    grip, close, push, speed = -1.2, 0.8, 1.2, 0.02  # MCP and PIP targets (rad), closing time, push start, speed
    slide_ke = 2.0e4

    def __init__(self, viewer, args=None):
        gravity = 9.81 if getattr(args, "gravity", False) else 0.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
        SolverMuJoCo.register_custom_attributes(builder)
        slide = newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0), target_ke=self.slide_ke,
                                                   target_kd=2.0 * np.sqrt(self.slide_ke))
        builder.add_mjcf(str(hand_xml()), base_joint={"joint_type": newton.JointType.PRISMATIC,
                                                      "linear_axes": [slide]})
        labels = [label.split("/")[-1] for label in builder.joint_label]
        self.slide = builder.joint_q_start[0]
        self.drives_q = [builder.joint_q_start[labels.index(f + j)] for f in FINGERS
                         for j in ("_mcp_pitch_joint", "_pip_joint")]
        n = self.segments + 1
        points = np.column_stack((np.linspace(self.clamp, self.free, n), np.full(n, self.y), np.full(n, self.z)))
        rod = newton.Rod(points, radius=self.radius, youngs_modulus=self.youngs, poissons_ratio=0.45)
        bodies, _ = SolverDiSMech.add_rod(builder, rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=1100.0,
                                                                                                  mu=self.mu))
        SolverDiSMech.fix_segment(builder, bodies[0])
        model = builder.finalize()
        self.ea = self.youngs * np.pi * self.radius**2
        self.h = (self.clamp - self.free) / self.segments
        self.control = model.control()
        solver = self._proxy(model)
        self.soft = solver.solver("soft")
        self.mjc = solver.solver("mjc")
        self.speed_dev = wp.zeros(1, dtype=float, device=model.device)
        self.cap = wp.zeros(1, dtype=float, device=model.device)
        self.start(viewer, model, solver)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_1)
        self.target = self.control.joint_target_q.numpy()
        self.x0 = self.state_0.particle_q.numpy()[:, 0].copy()
        self.history = []  # (time, hand travel, push (MuJoCo), rod axial force (ours), mu sum N, rod shift)
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.05, -0.35, 0.05), pitch=-10.0, yaw=90.0)

    @staticmethod
    def _proxy(model):
        """MuJoCo's hand and carriage, our rod; the hand's bodies with contact shapes proxies in our solve, two
        passes (one throws the closing fingers off the rod)."""
        capsules = sorted({int(b) for b in model.dismech.edge_body.numpy() if b >= 0})
        hand = [b for b in range(model.body_count) if b not in capsules]
        child = model.joint_child.numpy()
        joints = [j for j in range(model.joint_count) if child[j] in hand]
        shaped = set(model.shape_body.numpy().tolist())

        def mujoco(view):
            return SolverMuJoCo(view, solver="newton", integrator="implicitfast", cone="elliptic", iterations=100,
                                ls_iterations=20, use_mujoco_contacts=True, njmax=512, nconmax=270)

        return SolverCoupledProxy(model, entries=[
            SolverCoupled.Entry(name="mjc", solver=mujoco, bodies=hand, joints=joints),
            SolverCoupled.Entry(name="soft", solver=lambda view: SolverDiSMech(view, iterations=30),
                                particles=list(range(model.particle_count)), bodies=capsules,
                                joints=[j for j in range(model.joint_count) if j not in joints]),
        ], coupling=SolverCoupledProxy.Config(iterations=2, proxies=[SolverCoupledProxy.Proxy(
            source="mjc", destination="soft", bodies=[b for b in hand if b in shaped])]))

    def drive(self, t0, t1):
        """The fingers close over ``close`` s; from ``push`` the slide's target ramps (on the device, per substep)."""
        for i in self.drives_q:
            self.target[i] = min(t0 / self.close, 1.0) * self.grip
        self.target[self.slide] = self.speed * max(t0 - self.push, 0.0)
        self.control.joint_target_q.assign(self.target)
        self.speed_dev.fill_(self.speed if t0 >= self.push else 0.0)
        self.cap.zero_()

    def post_substep(self):
        wp.launch(_carriage, dim=1, inputs=[self.control.joint_target_q, self.slide, self.speed_dev, self.sim_dt])
        s = self.soft.contact.s
        wp.launch(_capacity, dim=s.rho.shape[0], inputs=[s.rho, s.mu, self.cap])

    def step(self):
        super().step()
        x = self.state_0.particle_q.numpy().astype(np.float64)
        seg = np.linalg.norm(np.diff(x, axis=0), axis=1)
        span = (x[:-1, 0] > 0.06) & (x[:-1, 0] < self.clamp - 2 * self.h)  # between the clamp and the hand
        rod = -self.ea * float((seg[span] / self.h - 1.0).mean())  # compression
        push = float(self.mjc.mjw_data.qfrc_actuator.numpy()[0][0])
        travel = float(self.state_0.joint_q.numpy()[self.slide])
        shift = float(np.abs(x[:, 0] - self.x0).max())
        self.history.append((self.sim_time, travel, push, rod, float(self.cap.numpy()[0]) / self.substeps, shift))

    def sliding(self, h):
        return h[:, 1] > 0.005  # past the grip's elastic take-up

    def image(self, size: tuple[int, int] = (800, 400)) -> np.ndarray:
        """The push against the hand's travel: MuJoCo's slide servo and the rod's axial force (ours) against
        mu sum(N), as an RGBA image."""
        h = np.array(self.history) if self.history else np.zeros((1, 6))
        go = h[:, 0] >= self.push

        def draw(axes):
            ax = axes
            mm = 1e3 * h[go, 1]
            ax.plot(mm, h[go, 4], color=THEORY, ls="--", lw=1.0, label=r"$\mu \sum N$")
            ax.plot(mm, h[go, 2], color=SIM, label="hand (MuJoCo)")
            ax.plot(mm, h[go, 3], color=SIM, ls=":", label=r"rod $EA\,\varepsilon$")
            ax.set_xlabel(r"travel (mm)")
            ax.set_ylabel(r"$P$ (N)")
            ax.set_title("Slip")
            ax.legend(loc="lower right")

        return inset(draw, size)

    def test_final(self):
        h = np.array(self.history)
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        assert h[-1, 1] > 0.02, f"the hand travelled {1e3 * h[-1, 1]:.1f} mm"
        assert h[-1, 5] < 1.0e-3, f"the clamped rod moved {1e3 * h[-1, 5]:.2f} mm: the grip did not slide"
        s = self.sliding(h)
        r = h[s, 2] / h[s, 4]  # 0.990 mean, 0.974-1.040 (gravity: 0.991, 0.970-1.044)
        assert abs(r.mean() - 1.0) < 0.02 and np.abs(r - 1.0).max() < 0.06,             f"push / mu sum N: mean {r.mean():.3f}, {r.min():.3f}-{r.max():.3f}"
        t = h[s, 3] / h[s, 2]  # 0.992-1.000 (gravity: 0.993-1.008)
        assert np.abs(t - 1.0).max() < 0.015, f"rod's axial force / the hand's push: {t.min():.4f}-{t.max():.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--gravity", action="store_true", help="horizontal under gravity")
    parser.set_defaults(num_frames=162)  # 2.7 s: closed by 0.8 s, 1.5 s of push (3 cm)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
