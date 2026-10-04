"""Helical flagella spin in a viscous fluid and bundle (Tong, Choi et al., arXiv:2205.10309, Sec. 4.1).

Shows: external forces on the nodes (``state.particle_f``, here the fluid drag of :mod:`utils.stokes`),
clamped segments spun through their twist, and contact between rods. Parameters are the paper's
(``PAPER``); its contact is frictionless, ``--mu`` adds friction.

    uv run examples/flagella.py --flagella 3
    uv run examples/flagella.py --viewer null --test
"""

import math

import newton
import newton.examples
import numpy as np
import warp as wp
from utils.common import SIM, inset, segment_distance, segment_dofs
from utils.stokes import Stokeslets

from dismech_newton import ADMMDiSMechSolver, flatten_state

PAPER = {
    "youngs_modulus": 3.0e6,  # [Pa]
    "poissons_ratio": 0.5,
    "density": 1000.0,  # [kg / m^3]
    "radius": 1.0e-3,  # h [m]
    "viscosity": 0.1,  # [Pa s]
    "epsilon": 1.02e-3,  # RSS regularization, 1.02 h
    "helix_radius": 0.01,  # a [m]
    "helix_pitch": 0.05,  # lambda [m]
    "axial_length": 0.2,  # z0 [m]
    "edge_length": 5.0e-3,  # target edge length of the helix (68 nodes)
    "spacing": 0.03,  # side of the clamp polygon [m]
    "omega": 15.0,  # [rad / s], the paper's text (its code, world::updateBoundary, reads 15 as rpm)
    "dt": 1.0e-3,  # [s]
    "total_time": 250.0,  # [s]
    "friction": 0.0,  # Coulomb mu between flagella (the paper's contact is frictionless)
}

TOP = 0.3  # height of the clamps [m]


def flagellum(p: dict, offset: tuple[float, float]) -> np.ndarray:
    """Nodes of one flagellum, clamp at the top, as the paper's code builds it (``world::rodGeometry``, its x
    axis turned to -z): a straight clamped edge on the helix axis, a 45-degree lead-in, then 4 helical turns."""
    a, b = p["helix_radius"], p["helix_pitch"] / (2.0 * math.pi)
    turns = p["axial_length"] / b
    length = turns * math.hypot(a, b)
    ne = int(length / p["edge_length"])
    dl = length / ne
    lead = int(a * math.sqrt(2.0) / dl)  # nodes on the 45-degree lead-in
    x = [(-dl - a, 0.0, 0.0)]
    x += [(-a + k * a / lead, k * a / lead, 0.0) for k in range(lead)]
    x += [(b * s, a * math.cos(s), a * math.sin(s)) for s in np.arange(ne + 1) * turns / ne]
    x = np.array(x)
    # (x, y, z) of the reference to (z, y, -x): a proper rotation, so the helix stays right-handed.
    return np.column_stack((x[:, 2] + offset[1], x[:, 1] + offset[0], TOP - (x[:, 0] + dl + a)))


def _rod(points: np.ndarray, p: dict) -> newton.Rod:
    # Newton's Rod takes nu < 0.5 only; the paper's incompressible nu = 0.5 is G = E / 3.
    g = p["youngs_modulus"] / (2.0 * (1.0 + p["poissons_ratio"]))
    return newton.Rod(points, radius=p["radius"], youngs_modulus=p["youngs_modulus"], shear_modulus=g)


def clamp_offsets(count: int, spacing: float) -> list[tuple[float, float]]:
    """Clamps on a regular ``count``-gon of side ``spacing`` (``world::setRodStepper``)."""
    theta = (count - 2) * math.pi / (2 * count)
    r = spacing / 2.0 / math.cos(theta)
    return [(r - r * math.cos(2 * math.pi * i / count), r * math.sin(2 * math.pi * i / count)) for i in range(count)]


@wp.kernel
def _spin_twist_kernel(dofs: wp.array[wp.int32], rest: wp.array[float], omega: float, dt: float,
                       step: wp.array[wp.int32], q: wp.array[float]):
    """Clamped twist at the end of step ``step``, right-handed about the edge tangent (down): ``+omega`` about +z."""
    i = wp.tid()
    q[dofs[i]] = rest[i] - omega * dt * float(step[0] + 1)


@wp.kernel
def _advance_step(step: wp.array[wp.int32]):
    step[0] = step[0] + 1


class DERFlagella:
    """Our DER, ADMM with contact. One step is: drag at the step's start (eager: cuSOLVER does not
    capture), then drive, collide and solve (a CUDA graph, one per state parity)."""

    name = "DER ADMM"

    def __init__(self, count: int, p: dict = PAPER, *, implicit_drag: bool = True, **solver_options):
        self.p = dict(p)
        self.count, self.dt = count, p["dt"]
        self.points = [flagellum(p, o) for o in clamp_offsets(count, p["spacing"])]
        self.nv = len(self.points[0])
        self.stokes = Stokeslets(count, self.nv, p["viscosity"], p["epsilon"], wp.get_device())
        self._step = wp.zeros(1, dtype=wp.int32)
        self._graphs = [None, None]
        self._parity = 0
        self.steps = 0
        h = p["radius"]
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        self.rods = []
        for points in self.points:
            rod = _rod(points, p)
            bodies = ADMMDiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=p["density"]))
            ADMMDiSMechSolver.fix_segment(builder, bodies[0])  # nodes 0, 1 and the first twist
            self.rods.append(bodies)
        self.model = model = builder.finalize()
        self.solver = ADMMDiSMechSolver(model, friction=p["friction"], **solver_options)
        if implicit_drag:  # the paper's explicit drag diverges at the tips of a tight bundle (stokes.py)
            self.stokes.implicit(self.solver.mass.numpy()[: 3 * count * self.nv], self.dt)
        self.pipeline = newton.CollisionPipeline(model, soft_contact_max=0, verify_buffers=False,
                                                 speculative_contact_gap_max=2.0 * h, contact_matching="latest")
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = model.state(), model.state()
        for s in (self.state_0, self.state_1):
            flatten_state(s)
            s.particle_f = self.stokes.force  # the drag, shared by both states
        dofs = [segment_dofs(model, b[0], twist_only=True)[0] for b in self.rods]
        self._spin_dofs = wp.array(dofs, dtype=wp.int32)
        self._spin_rest = wp.array(self.state_0.dismech.q.numpy()[dofs], dtype=float)

    def step(self) -> None:
        self.stokes.compute(self.state_0.particle_q, self.state_0.particle_qd)
        graph = self._graphs[self._parity]
        if graph is not None:
            wp.capture_launch(graph)
        else:
            self._simulate()
            if self.steps >= 1 and self.solver.graph_capturable:  # the solver sets itself up on its first step
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self._graphs[self._parity] = capture.graph
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._parity ^= 1
        self.steps += 1

    def _simulate(self) -> None:
        wp.launch(_spin_twist_kernel, dim=self.count,
                  inputs=[self._spin_dofs, self._spin_rest, self.p["omega"], self.dt, self._step],
                  outputs=[self.state_0.dismech.q])
        self.pipeline.collide(self.state_0, self.contacts, dt=2.0 * self.dt)
        self.solver.step(self.state_0, self.state_1, None, self.contacts, self.dt)
        wp.launch(_advance_step, dim=1, inputs=[self._step])

    @property
    def time(self) -> float:
        return self.steps * self.dt

    def nodes(self) -> np.ndarray:
        """``(count, nv, 3)`` node positions."""
        return self.state_0.particle_q.numpy().reshape(self.count, self.nv, 3)

    def body_q(self) -> np.ndarray:
        """Capsule proxy poses, for replay in the viewer."""
        self.solver.update_proxies(self.state_0)
        return self.state_0.body_q.numpy()

    def gaps(self) -> tuple[float, float]:
        """Closest approach between flagella, and within one (edges > 2 apart), in diameters."""
        x = self.nodes()
        a, b = x[:, :-1].reshape(-1, 3), x[:, 1:].reshape(-1, 3)
        rod = np.repeat(np.arange(self.count), self.nv - 1)
        idx = np.tile(np.arange(self.nv - 1), self.count)
        i, j = np.triu_indices(len(a), k=1)
        other = rod[i] != rod[j]
        own = ~other & (np.abs(idx[i] - idx[j]) > 2)
        d = segment_distance(a[i], b[i], a[j], b[j]) / (2.0 * self.p["radius"])
        return float(d[other].min()) if other.any() else np.inf, float(d[own].min())

    def tip_spread(self) -> float:
        """Mean distance between the free ends [m], the reference code's bundling measure."""
        tips = self.nodes()[:, -1]
        i, j = np.triu_indices(self.count, k=1)
        return float(np.linalg.norm(tips[i] - tips[j], axis=1).mean())

    def max_stretch(self) -> float:
        x = self.nodes()
        l = np.linalg.norm(np.diff(x, axis=1), axis=2)
        l0 = np.linalg.norm(np.diff(np.array(self.points), axis=1), axis=2)
        return float(np.abs(l / l0 - 1.0).max())


class Example:
    fps = 60
    plot_every = 30  # frames between plot updates
    plot_time = PAPER["total_time"]  # [s] the time axis of the plot

    def __init__(self, viewer, args=None):
        count = getattr(args, "flagella", 3) if args is not None else 3
        mu = getattr(args, "mu", PAPER["friction"]) if args is not None else PAPER["friction"]
        self.sim = DERFlagella(count, {**PAPER, "friction": mu})
        self.viewer = viewer
        self.frame_dt = 1.0 / self.fps
        self.steps_per_frame = max(1, round(self.frame_dt / self.sim.dt))
        self.min_gap = np.inf
        self.frame = 0
        self.history = [(0.0, self.sim.tip_spread())]  # (time, mean tip distance), every plot update
        viewer.set_model(self.sim.model)
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.015, -0.55, 0.2), pitch=0.0, yaw=90.0)

    def step(self):
        for _ in range(self.steps_per_frame):
            self.sim.step()
        self.frame += 1
        if self.frame % self.plot_every == 0:
            self.history.append((self.sim.time, self.sim.tip_spread()))

    def image(self, size: tuple[int, int] = (400, 400)) -> np.ndarray:
        """The mean distance between the free ends over time (the reference code's bundling measure),
        as an RGBA image (see :func:`utils.common.inset`)."""
        t, d = np.array(self.history).T

        def draw(ax):
            ax.plot(t, 1e3 * d, color=SIM)
            ax.set(xlim=(0.0, max(self.plot_time, self.sim.time)), ylim=(0.0, 1e3 * max(1.6 * d[0], 1.1 * d.max())))
            ax.set_xlabel(r"time $t$ (s)")
            ax.set_ylabel(r"tip spread $\bar{d}$ (mm)")
            ax.locator_params(nbins=5)

        return inset(draw, size)

    def render(self):
        if self.frame % self.plot_every == 0:
            self.viewer.log_image("tip spread", self.image())
        self.viewer.begin_frame(self.sim.time)
        self.viewer.log_state(self.sim.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        self.min_gap = min(self.min_gap, *self.sim.gaps())

    def test_final(self):
        x = self.sim.nodes()
        assert np.isfinite(x).all(), "non-finite positions"
        assert self.min_gap > 0.8, f"flagella passed into each other (closest {self.min_gap:.2f} diameters)"
        assert self.sim.max_stretch() < 0.05, "flagellum stretched by more than 5%"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--flagella", type=int, default=3, help="number of flagella M")
    parser.add_argument("--mu", type=float, default=PAPER["friction"], help="friction between flagella")
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
