"""Clamped rods released straight under gravity: the DER solvers beside Newton's VBD cable.

Shows: both solvers (``ADMMDiSMechSolver``, ``DiSMechSolver``) and their integrators (``theta``: 1 implicit
Euler, 1/2 implicit midpoint), each rod in its own model, over the Euler-Bernoulli curve. Bending damping
is off, so what decays is the solvers' numerical damping.

    uv run examples/cantilever.py
    uv run examples/cantilever.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from utils.common import THEORY, FrameGraph, frame_box, inset, inset_scale

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, flatten_state

GRAVITY = 9.81


@wp.kernel
def _place_bodies(src: wp.array[wp.transform], offset: wp.vec3, start: int, dst: wp.array[wp.transform]):
    i = wp.tid()
    t = src[i]
    dst[start + i] = wp.transform(wp.transform_get_translation(t) + offset, wp.transform_get_rotation(t))


@wp.kernel
def _place_points(src: wp.array[wp.vec3], offset: wp.vec3, start: int, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[start + i] = src[i] + offset


# name: (solver, its options, color)
DER_SOLVERS = {
    "DER ADMM, implicit Euler": (ADMMDiSMechSolver, {}, (0.3, 0.8, 0.6)),
    "DER Newton-Raphson, implicit Euler": (DiSMechSolver, {}, (0.6, 0.45, 0.9)),
    # Undamped, ADMM's leftover residual pumps energy into the stiff modes unless it is solved tighter.
    "DER ADMM, implicit midpoint": (ADMMDiSMechSolver, {"theta": 0.5, "tol": 1.0e-5}, (0.3, 0.6, 0.95)),
    "DER Newton-Raphson, implicit midpoint": (DiSMechSolver, {"theta": 0.5}, (0.3, 0.75, 0.8)),
}
VBD_COLORS = ((0.95, 0.55, 0.2), (0.85, 0.25, 0.25), (0.95, 0.75, 0.25), (0.7, 0.3, 0.7))


class Example:
    fps = 60
    plot_time = 4.0  # [s] the time axis of the plot (the default run), longer if the run is
    length, segments, radius = 1.0, 40, 0.01
    bend_stiffness = 140.0  # per joint [N m / rad]: EI = K h
    bend_damping = 0.0  # per joint [N m s / rad]: off, so what decays is numerical damping
    height, gap, row_gap, columns = 1.0, 0.3, 1.2, 2  # bottom row's clamp height; spacing of the grid

    def __init__(self, viewer, args=None, *, der: tuple[str, ...] = ("DER ADMM, implicit midpoint", "DER ADMM, implicit Euler"),
                 vbd: tuple[tuple[int, int], ...] = ((10, 5), (100, 50)), der_substeps: int = 8):
        self.viewer = viewer
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # (name, color, add the rod to a builder at a clamp, make its solver, substeps), DER first.
        rods = []
        for name in der:
            solver, options, color = DER_SOLVERS[name]
            rods.append((name, color, self._der_adder(solver, color), lambda m, s=solver, o=options: s(m, **o),
                         der_substeps))
        for k, (substeps, it) in enumerate(vbd):
            color = VBD_COLORS[k % len(VBD_COLORS)]
            rods.append((f"VBD, {it} iterations", color, self._vbd_adder(color),
                         lambda m, it=it: newton.solvers.SolverVBD(m, iterations=it, rigid_compliant_alm=True),
                         substeps))
        rows = -(-len(rods) // self.columns)
        clamps = [((k % self.columns) * (self.length + self.gap), self.height + (rows - 1 - k // self.columns) * self.row_gap)
                  for k in range(len(rods))]

        # One model per rod, clamped at the origin, and a scene with every rod in place, only rendered:
        # positions are float32, and a metre from the origin a substep's motion falls below their resolution.
        self.sims = {}
        scene = newton.ModelBuilder()
        self.colors = {}  # each rod's color, for the plot's legend
        for (name, color, add, make_solver, substeps), clamp in zip(rods, clamps):
            self.colors[name] = color
            builder = newton.ModelBuilder()
            bodies = add(builder, (0.0, 0.0))
            builder.color()
            model = builder.finalize()
            self.sims[name] = _Sim(model, make_solver(model), bodies, substeps, offset=(clamp[0], 0.0, clamp[1]))
            add(scene, clamp)
        self.model = scene.finalize()
        self.state_0 = self.model.state()
        self._gather()

        # Euler-Bernoulli: uniform load W. A joint's stiffness K = EI / h spreads the curvature over the half
        # segments on either side, so the continuum clamp sits midway between the two fixed nodes, at h / 2.
        h = self.length / self.segments
        self.w = scene.default_shape_cfg.density * np.pi * self.radius**2 * GRAVITY  # load per length
        self.ei = self.bend_stiffness * h
        self.clamp = 0.5 * h
        self.span = self.length - self.clamp
        x = np.linspace(0.0, self.length, 101)
        starts, ends = [], []
        for x0, z0 in clamps:
            p = np.column_stack((x0 + x, np.zeros_like(x), z0 - self.beam(x)))
            starts.append(p[:-1])
            ends.append(p[1:])
        self.beam_starts = wp.array(np.concatenate(starts), dtype=wp.vec3)
        self.beam_ends = wp.array(np.concatenate(ends), dtype=wp.vec3)
        viewer.set_model(self.model)
        # The grid, with room for a rod that droops nearly its length and swings back past its clamp (VBD at
        # few iterations) on both sides, and an equal margin around.
        margin, swing, droop = 0.15, 0.61 * self.length, 0.944 * self.length
        grid = min(len(rods), self.columns) * (self.length + self.gap) - self.gap + self.radius
        frame_box(viewer, -swing - margin, grid + swing + margin, self.height - droop - margin,
                  self.height + (rows - 1) * self.row_gap + margin)

    def _rod(self, clamp) -> newton.Rod:
        return newton.Rod.create_straight((clamp[0], 0.0, clamp[1]), (1.0, 0.0, 0.0), self.length,
                                          segment_count=self.segments, radius=self.radius)

    def _der_adder(self, solver, color):
        def add(builder, clamp) -> list[int]:
            bodies = solver.add_rod(builder, self._rod(clamp), stretch_stiffness=1.0e6, color=color,
                                    bend_stiffness=self.bend_stiffness, twist_stiffness=self.bend_stiffness,
                                    bend_damping=self.bend_damping)
            solver.fix_segment(builder, bodies[0])
            return bodies

        return add

    def _vbd_adder(self, color):
        def add(builder, clamp) -> list[int]:
            h, r = self.length / self.segments, self.radius
            # VBD's capsules count their end caps in their mass: scale the density down to DER's mass per length.
            cfg = newton.ModelBuilder.ShapeConfig(density=builder.default_shape_cfg.density * h / (h + 4.0 * r / 3.0))
            bodies, _ = builder.add_rod(rod=self._rod(clamp), cfg=cfg, stretch_stiffness=1.0e6,
                                        bend_stiffness=self.bend_stiffness,
                                        twist_stiffness=self.bend_stiffness, bend_damping=self.bend_damping,
                                        twist_damping=self.bend_damping,
                                        body_frame_origin="com", color=color)
            b = bodies[0]  # clamp the first segment
            builder.body_mass[b], builder.body_inv_mass[b] = 0.0, 0.0
            builder.body_inertia[b], builder.body_inv_inertia[b] = wp.mat33(0.0), wp.mat33(0.0)
            return list(bodies)

        return add

    def beam(self, x: np.ndarray) -> np.ndarray:
        """Euler-Bernoulli sag (positive down) at ``x``."""
        s = np.maximum(x - self.clamp, 0.0)
        return self.w * s**2 * (6.0 * self.span**2 - 4.0 * self.span * s + s**2) / (24.0 * self.ei)

    def beam_frequency(self) -> float:
        """First bending mode of the clamped span [Hz]."""
        mu = self.w / GRAVITY
        return 1.8751**2 / (2.0 * np.pi) * np.sqrt(self.ei / (mu * self.span**4))

    def _gather(self):
        """Copy every rod's bodies (and the DER nodes) into the rendered state, moved to its place."""
        bodies = particles = 0
        for sim in self.sims.values():
            wp.launch(_place_bodies, dim=len(sim.bodies), inputs=[sim.state_0.body_q, sim.offset, bodies],
                      outputs=[self.state_0.body_q])
            bodies += len(sim.bodies)
            if sim.model.particle_count:
                wp.launch(_place_points, dim=sim.model.particle_count,
                          inputs=[sim.state_0.particle_q, sim.offset, particles], outputs=[self.state_0.particle_q])
                particles += sim.model.particle_count

    def step(self):
        for sim in self.sims.values():
            sim.run(self.frame_dt)
        self.sim_time += self.frame_dt
        self._gather()

    def render(self):
        if round(self.sim_time * self.fps) % 10 == 0:
            self.viewer.log_image("tip sag", self.image())
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("euler_bernoulli", self.beam_starts, self.beam_ends, (1.0, 1.0, 1.0))
        self.viewer.end_frame()

    def image(self, size: tuple[int, int] = (400, 400)) -> np.ndarray:
        """Each rod's tip sag over time against the Euler-Bernoulli beam's, as an RGBA image (see
        :func:`utils.common.inset`): released straight and undamped, its first mode swings between 0 and twice
        the static sag, ``1 - cos(2 pi f t)`` at :meth:`beam_frequency`."""
        exact = float(self.beam(np.array([self.length]))[0])

        def draw(ax):
            t = np.linspace(0.0, self.sim_time, max(2, round(200 * self.sim_time)))  # grows with the rods'
            ax.plot(t, 1.0 - np.cos(2.0 * np.pi * self.beam_frequency() * t), color=THEORY, ls="--",
                    lw=1.2 * inset_scale(size), zorder=3, label="Euler-Bernoulli")
            for name, sim in self.sims.items():
                every = max(1, len(sim.t) // 3000)
                label = name.removeprefix("DER ").replace("iterations", "iter.")
                ax.plot(sim.t[::every], np.array(sim.sag[::every]) / exact, color=self.colors[name],
                        lw=1.4 * inset_scale(size), label=label)
            ax.set(xlim=(0.0, max(self.plot_time, self.sim_time)), ylim=(0.0, 3.6))
            ax.set_xlabel(r"time $t$ (s)")
            ax.set_ylabel(r"tip sag $w / w_\mathrm{EB}$")
            ax.locator_params(axis="x", nbins=4)
            ax.set_yticks([0, 1, 2, 3])
            ax.legend(loc="upper right", fontsize=14 * inset_scale(size))

        return inset(draw, size)

    def summary(self) -> dict[str, dict[str, float]]:
        """Tip sag against the beam (averaged over the last period), frequency, amplitude decay, and
        iterations per step (the DER solvers' adaptive count, VBD's fixed one)."""
        exact, freq = float(self.beam(np.array([self.length]))[0]), self.beam_frequency()
        out = {}
        for name, sim in self.sims.items():
            t, z = np.array(sim.t), np.array(sim.sag)
            osc = z - exact
            up = t[1:][(osc[1:] > 0) & (osc[:-1] <= 0)]  # upward crossings of the beam sag
            period = int(round(1.0 / freq / sim.dt))
            out[name] = {
                "sag / beam": z[-period:].mean() / exact,
                "frequency / beam": 1.0 / (up[1] - up[0]) / freq if len(up) > 1 else float("nan"),
                "amplitude after 1 period": (np.ptp(osc[period:2 * period]) / np.ptp(osc[:period])
                                             if len(osc) >= 2 * period else float("nan")),
                "iterations / step": np.mean(sim.iterations) if sim.iterations else sim.solver.iterations,
                "dt": sim.dt,
            }
        return out

    def test_final(self):
        for name, row in self.summary().items():
            assert np.isfinite(row["sag / beam"]), f"{name}: non-finite tip"
            if name.startswith("DER"):
                assert abs(row["sag / beam"] - 1.0) < 0.05, f"{name}: sag {row['sag / beam']:.3f} of the beam's"


@wp.kernel
def _record_tip(body_q: wp.array[wp.transform], body: int, half: float, z0: float,
                iterations: wp.array[wp.int32], count: wp.array[wp.int32],
                # outputs
                sag: wp.array[float], iters: wp.array[wp.int32]):
    """Append the tip's sag (its capsule's far end, below its start) and the step's iterations at ``count``."""
    k = count[0]
    if k < sag.shape[0]:
        X = body_q[body]
        tip = wp.transform_get_translation(X) + half * wp.quat_rotate(wp.transform_get_rotation(X), wp.vec3(0.0, 0.0, 1.0))
        sag[k] = z0 - tip[2]
        if iterations.shape[0] > 0:
            iters[k] = iterations[0]
    count[0] = k + 1


class _Sim:
    """One rod, its solver and its tip-sag history (sampled every substep).

    The history is recorded on the device, so a frame's substeps are one CUDA graph (:class:`FrameGraph`); reading
    ``t``, ``sag`` or ``iterations`` copies it to the host (a sync), as does a full device buffer (every
    ``buffer_frames`` frames)."""

    buffer_frames = 600

    def __init__(self, model, solver, bodies: list[int], substeps: int, offset):
        self.model, self.solver, self.bodies, self.substeps = model, solver, bodies, substeps
        self.offset = wp.vec3(*offset)  # where the rod is drawn
        self.state_0, self.state_1 = model.state(), model.state()
        if model.particle_count:  # a DER rod
            flatten_state(self.state_0)
            flatten_state(self.state_1)
        shape = list(model.shape_body.numpy()).index(bodies[-1])
        self.tip, self.half = bodies[-1], float(model.shape_scale.numpy()[shape, 1])  # the capsule's half length
        self.z0 = float(model.body_q.numpy()[bodies[-1], 2])
        n, dev = substeps * self.buffer_frames, model.device
        self._iterations = getattr(solver, "iteration_count", wp.zeros(0, dtype=wp.int32, device=dev))
        self._count = wp.zeros(1, dtype=wp.int32, device=dev)
        self._sag, self._iters = wp.zeros(n, dtype=float, device=dev), wp.zeros(n, dtype=wp.int32, device=dev)
        self._pending = 0  # substeps recorded on the device since the last read
        self._t, self._z, self._it = [], [], []
        self.dt = None
        self.frame = FrameGraph(self._substeps, solver, enabled=substeps % 2 == 0)  # even: state_0 stays in place

    def _substeps(self):
        for _ in range(self.substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(_record_tip, dim=1,
                      inputs=[self.state_0.body_q, self.tip, self.half, self.z0, self._iterations, self._count],
                      outputs=[self._sag, self._iters], device=self.model.device)

    def run(self, frame_dt: float):
        self.dt = frame_dt / self.substeps
        if self._pending + self.substeps > self._sag.shape[0]:
            self._flush()
        self.frame()
        self._pending += self.substeps

    def _flush(self) -> None:
        """Move the device's records to the host."""
        k = self._pending
        if not k:
            return
        self._t.extend((len(self._t) + np.arange(1, k + 1)) * self.dt)
        self._z.extend(self._sag.numpy()[:k].tolist())
        if self._iterations.shape[0]:
            self._it.extend(self._iters.numpy()[:k].tolist())
        self._count.zero_()
        self._pending = 0

    @property
    def t(self) -> list[float]:
        self._flush()
        return self._t

    @property
    def sag(self) -> list[float]:
        self._flush()
        return self._z

    @property
    def iterations(self) -> list[int]:
        self._flush()
        return self._it


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
