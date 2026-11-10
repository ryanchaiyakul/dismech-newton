"""The MIDAS hand (MuJoCo, github.com/midas-hand-org/midas_hand_mujoco, MIT) with PaXini tactile fingertips
(:mod:`utils.tactile`) takes a soft rod clamped at one end, changes its grip, then pushes it along its axis until the
grip slides: two-way coupling through Newton's ``SolverCoupledProxy``, the hand's contact bodies proxies in our solve.

The rod sags 5 cm under gravity with the hand open; the hand comes down to it, its fingers reach down in front of it
and curl under it, their sensor pads pressing it into the palm, and lifts it past the clamp's height (bending it
back); the ring finger lets go and takes it again, then the index; from ``push`` a stiff prismatic servo pushes the
hand toward the clamp at 2 cm/s. The sensors read at their 83.3 Hz (a frame per sample, one CUDA graph); the viewer
shows the taxels' normal forces and each sensor's sum.

Theory: before the grasp, the sag is Euler-Bernoulli's cantilever ``w = q s^2 (6 L^2 - 4 L s + s^2) / 24 EI``.
Sliding, the push is Coulomb's mu sum(N), the contacts' normal forces (``rho``, mu the mean of the rod's 0.5 and the
hand's 1.0); the push measured on MuJoCo's side (the slide servo) equals the rod's axial force on ours (EA x strain
between clamp and hand). The taxels' forces sum to their sensor's contact force.

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
from utils import tactile
from utils.common import CACHE, MUTED, SIM, THEORY, CableExample, inset, pads

from dismech_newton import SolverDiSMech

COMMIT = "6c7d17f4a8cb5414e81653aa2795acc9610e1730"
FINGERS = ("index", "middle", "ring")
THUMB = ("thumb_cmc_roll_joint", "thumb_cmc_side_joint", "thumb_mcp_joint", "thumb_dip_joint")


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
    fps, substeps = 1000.0 / 12.0, 6  # a frame per sensor sample (83.3 Hz), 2 ms steps
    plot_every, plot_name = 8, "tactile"
    radius, segments, youngs, density, mu = 0.012, 75, 2.0e7, 1100.0, 0.5  # sags 5 cm at its tip
    clamp, free, y, z = 0.25, -0.15, 0.082, -0.034  # along x under the palm; clamped at +x
    tau = 0.1  # bending damping time (s): the sag settles critically (its first mode 18 rad/s)
    sag_end = 0.6  # the hand starts down
    # The grip (FK-chosen, the DIPs on their four-bar): the fingers down in front of the rod, then curled under it,
    # their sensor pads flat on its underside (normals (0, 0.23, 0.97), perpendicular to it) pressing it into the
    # palm: each finger opposed in line. The PIP targets past the rod (the squeeze). The thumb parked off the rod.
    open_, down, grip = (0.0, 0.0), (-1.55, 0.0), (-1.6, -1.1)  # finger MCP, PIP (rad)
    park = (0.0, 0.56, 0.52, 0.0)  # thumb roll, side, MCP, DIP: every thumb shape >= 20 mm off the rod
    # keyframes (s, value): the hand's height in units of the rod's sag at the grip (down to it, up past the clamp's
    # height), the fingers'; ring then index let go and take it again
    lift = ((0.6, 0.0), (1.0, -1.0), (1.8, -1.0), (2.3, 0.5))
    middle = ((1.0, open_), (1.4, down), (1.8, grip))
    ring = (*middle, (2.5, grip), (2.7, down), (2.9, down), (3.2, grip))
    index = (*middle, (3.3, grip), (3.5, down), (3.7, down), (4.0, grip))
    thumb = ((0.0, park),)
    push, speed, slide_ke = 4.3, 0.02, 2.0e4

    def __init__(self, viewer, args=None):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(builder)
        slides = [newton.ModelBuilder.JointDofConfig(axis=axis, target_ke=self.slide_ke,
                                                     target_kd=2.0 * np.sqrt(self.slide_ke))
                  for axis in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))]  # the push, the height
        builder.add_mjcf(str(hand_xml()), base_joint={"joint_type": newton.JointType.D6, "linear_axes": slides})
        sensors = tactile.mount(builder)
        labels = [label.split("/")[-1] for label in builder.joint_label]
        self.slide = builder.joint_q_start[0]
        self.servo = {f: [builder.joint_q_start[labels.index(f + j)] for j in ("_mcp_pitch_joint", "_pip_joint")]
                      for f in FINGERS}
        self.servo["thumb"] = [builder.joint_q_start[labels.index(j)] for j in THUMB]
        for i, v in zip(self.servo["thumb"], self.park):  # parked from the start (at rest it crosses the rod)
            builder.joint_q[i] = v
        n = self.segments + 1
        points = np.column_stack((np.linspace(self.clamp, self.free, n), np.full(n, self.y), np.full(n, self.z)))
        rod = newton.Rod(points, radius=self.radius, youngs_modulus=self.youngs, poissons_ratio=0.45)
        self.h = (self.clamp - self.free) / self.segments
        self.ei = self.youngs * np.pi * self.radius**4 / 4.0
        self.ea = self.youngs * np.pi * self.radius**2
        self.q = self.density * np.pi * self.radius**2 * 9.81  # weight per length
        s, w = self.exact_sag()
        self.drop = float(np.interp(self.clamp - self.h - (-0.0025), s, w))  # under the middle finger
        bodies, _ = SolverDiSMech.add_rod(builder, rod=rod, bend_damping=self.tau * self.ei / self.h,
                                          cfg=newton.ModelBuilder.ShapeConfig(density=self.density, mu=self.mu))
        SolverDiSMech.fix_segment(builder, bodies[0])
        model = builder.finalize()
        pads(model, set(range(model.body_count)) - set(model.dismech.edge_body.numpy().tolist()))  # the hand: pads
        self.control = model.control()
        solver = self._proxy(model)
        self.soft = solver.solver("soft")
        self.mjc = solver.solver("mjc")
        self.tactile = tactile.Tactile(model, self.soft, sensors)
        self.tactile.hook()
        self.taxels = [self.tactile.pos.numpy()[a:b] * 1e3 for a, b in zip(self.tactile.start, self.tactile.start[1:])]
        self.speed_dev = wp.zeros(1, dtype=float, device=model.device)
        self.cap = wp.zeros(1, dtype=float, device=model.device)
        self.start(viewer, model, solver)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_1)
        self.target = self.control.joint_target_q.numpy()
        self.x0 = self.state_0.particle_q.numpy().astype(np.float64)
        self.sag = None  # the rod's deflection at ``sag_end``
        self.history = []  # (time, hand travel, push (MuJoCo), rod axial force (ours), mu sum N, slip: the hand's travel
        # less the rod's under the middle finger)
        self.touch = []  # per frame, each sensor's sum of taxel normal forces (read, quantized) [N]
        self.balance = 0.0  # the worst |taxels' sum - sensor's contact force| [N]
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.05, -0.4, 0.0), pitch=-5.0, yaw=90.0)

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
        """The servos' keyframes (held before the first, after the last); from ``push`` the slide's target ramps
        (on the device, per substep)."""
        self.target[self.slide + 1] = self.drop * np.interp(t0, *np.array(self.lift).T)
        for name in (*FINGERS, "thumb"):
            keys = getattr(self, name)
            times, values = np.array([k[0] for k in keys]), np.array([k[1] for k in keys], dtype=float)
            values = values.reshape(len(keys), -1)
            for j, i in enumerate(self.servo[name]):
                self.target[i] = np.interp(t0, times, values[:, j % values.shape[1]])
        self.target[self.slide] = self.speed * max(t0 - self.push, 0.0)
        self.control.joint_target_q.assign(self.target)
        self.speed_dev.fill_(self.speed if t0 >= self.push else 0.0)
        self.cap.zero_()

    def post_substep(self):
        wp.launch(_carriage, dim=1, inputs=[self.control.joint_target_q, self.slide, self.speed_dev, self.sim_dt])
        s = self.soft.contact.s
        wp.launch(_capacity, dim=s.rho.shape[0], inputs=[s.rho, s.mu, self.cap])
        self.tactile.commit(self.sim_dt)

    def simulate(self):
        super().simulate()
        self.tactile.sample()

    def step(self):
        super().step()
        x = self.state_0.particle_q.numpy().astype(np.float64)
        seg = np.linalg.norm(np.diff(x, axis=0), axis=1)
        span = (x[:-1, 0] > 0.06) & (x[:-1, 0] < self.clamp - 2 * self.h)  # between the clamp and the hand
        rod = -self.ea * float((seg[span] / self.h - 1.0).mean())  # compression
        push = float(self.mjc.mjw_data.qfrc_actuator.numpy()[0][0])
        travel = float(self.state_0.joint_q.numpy()[self.slide])
        if self.sim_time <= self.push:
            self.xpush = x
        under = int(np.argmin(np.abs(self.xpush[:, 0] + 0.0025)))
        slip = travel - float(x[under, 0] - self.xpush[under, 0])  # (the push from 0 travel)
        self.history.append((self.sim_time, travel, push, rod, float(self.cap.numpy()[0]) / self.substeps, slip))
        if self.sag is None and self.sim_time >= self.sag_end:
            self.sag = self.x0[:, 2] - x[:, 2]
        t = self.tactile
        force, raw, total = t.force.numpy(), t.raw.numpy(), t.total.numpy()
        back = np.einsum("tji,tj->ti", t.frame.numpy(), force * [1.0, 1.0, -1.0])  # taxel frames -> sensor's
        ranges = list(zip(t.start, t.start[1:]))
        sums = np.array([back[a:b].sum(0) for a, b in ranges])
        self.balance = max(self.balance, float(np.abs(sums - total).max()))
        self.touch.append([float(tactile.LSB) * raw[a:b, 2].sum() for a, b in ranges])
        self.reading = float(tactile.LSB) * raw[:, 2]

    def exact_sag(self):
        """Euler-Bernoulli's cantilever from the clamped segment's end: the nodes' arc length and deflection."""
        s = np.arange(self.segments) * (self.clamp - self.free) / self.segments  # nodes 1..n
        length = s[-1]
        return s, self.q * s**2 * (6 * length**2 - 4 * length * s + s**2) / (24.0 * self.ei)

    def image(self, size: tuple[int, int] = (1600, 400)) -> np.ndarray:
        """Four panels: the sag against Euler-Bernoulli; the taxels' normal forces now (index, middle, ring,
        thumb); each sensor's sum over time; the push against mu sum(N) while sliding. RGBA."""
        h = np.array(self.history) if self.history else np.zeros((1, 6))
        touch = np.array(self.touch) if self.touch else np.zeros((1, 4))
        reading = getattr(self, "reading", np.zeros(self.tactile.count))
        go = h[:, 0] >= self.push
        names = ("index", "middle", "ring", "thumb")

        def draw(axes):
            ax = axes[0]
            s, w = self.exact_sag()
            ax.plot(1e3 * s, 1e3 * w, color=THEORY, ls="--", lw=1.0, label="EB")
            if self.sag is not None:
                ax.plot(1e3 * s, 1e3 * self.sag[1:], color=SIM, label="sim")
            ax.set_xlabel(r"$s$ (mm)")
            ax.set_ylabel(r"$w$ (mm)")
            ax.set_title("Sag")
            ax.legend(loc="upper left")

            ax = axes[1]  # the fingers' maps in a row (tips up), the thumb's under them
            for k, (p, dx, dy) in enumerate(zip(self.taxels, (-20.0, 0.0, 20.0, 0.0), (36.0, 36.0, 36.0, 0.0))):
                a, b = self.tactile.start[k], self.tactile.start[k + 1]
                sc = ax.scatter(p[:, 0] + dx, p[:, 1] + dy, c=reading[a:b], s=22, cmap="Blues_r", vmin=0.0,
                                vmax=6.0, edgecolors=tactile.COLORS[self.tactile.links[k]], linewidths=0.8)
                ax.text(*((dx, 33.0) if dy else (-16.0, 10.0)), names[k], ha="center" if dy else "right",
                        va="top" if dy else "center", fontsize="small", color=THEORY)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title("Taxels")
            ax.figure.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label=r"$F_z$ (N)")

            ax = axes[2]
            for k in range(touch.shape[1]):
                ax.plot(h[:, 0], touch[:, k], color=tactile.COLORS[self.tactile.links[k]], lw=1.5, label=names[k])
            ax.set_xlabel(r"$t$ (s)")
            ax.set_ylabel(r"$\sum F_z$ (N)")
            ax.set_title("Touch")
            ax.legend(loc="upper left")

            ax = axes[3]
            mm = 1e3 * h[go, 1]
            ax.plot(mm, h[go, 4], color=THEORY, ls="--", lw=1.0, label=r"$\mu \sum N$")
            ax.plot(mm, h[go, 2], color=SIM, label="hand")
            ax.plot(mm, h[go, 3], color=MUTED, ls=":", label=r"rod $EA\,\varepsilon$")
            ax.set_xlabel(r"travel (mm)")
            ax.set_ylabel(r"$P$ (N)")
            ax.set_title("Slip")
            ax.legend(loc="lower right")

        return inset(draw, size, ncols=4)

    def sliding(self, h):
        return h[:, 5] > 0.005  # past the grip's elastic take-up

    def test_final(self):
        h = np.array(self.history)
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        s, w = self.exact_sag()
        err = np.abs(self.sag[1:] - w).max() / w[-1]
        assert err < 0.025, f"sag vs Euler-Bernoulli: worst {100 * err:.1f}% of the tip's"  # 1.6%
        assert self.balance < 1.0e-4, f"taxels' sum vs sensor's force: {self.balance:.2e} N"  # 3.6e-6
        assert h[-1, 1] > 0.02, f"the hand travelled {1e3 * h[-1, 1]:.1f} mm"
        assert h[-1, 5] > 0.015, f"the grip slid {1e3 * h[-1, 5]:.1f} mm of the hand's {1e3 * h[-1, 1]:.1f}"  # 23
        sl = self.sliding(h)
        r = h[sl, 2] / h[sl, 4]  # 1.002 mean, 0.995-1.016
        assert abs(r.mean() - 1.0) < 0.02 and np.abs(r - 1.0).max() < 0.06, \
            f"push / mu sum N: mean {r.mean():.3f}, {r.min():.3f}-{r.max():.3f}"
        t = h[sl, 3] / h[sl, 2]  # 0.992-1.005
        assert np.abs(t - 1.0).max() < 0.015, f"rod's axial force / the hand's push: {t.min():.4f}-{t.max():.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=475)  # 5.7 s: regripped by 4 s, 1.4 s of push (2.8 cm)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
