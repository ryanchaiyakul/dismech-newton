"""A clamped cantilever released straight under gravity: the DER solvers next to Newton's VBD cable.

Each rod is its own model, stepped by its own solver; all are drawn in one scene, in rows of two:
DER (by default ADMM with implicit midpoint and with implicit Euler), then VBD at
each (substeps, iterations): by default 10 x 5, and 100 x 50, where refining either no longer
changes the result. Each rod has the Euler-Bernoulli small-deflection curve it should settle
around. Bending damping is off, so what decays is the solvers' numerical damping.

The first segment is clamped. A joint's stiffness ``K = EI / h`` spreads the curvature over the
half segments on either side, so the continuum clamp sits midway between the two fixed nodes,
``x = h / 2``, for DER's triplets and VBD's joints alike. VBD's capsule bodies count their end caps
in their mass, so their density is scaled down to give both rods the same mass per length.

Every rod is simulated clamped at the origin and only drawn at its place in the grid: positions are
float32, and at a metre from the origin a small substep's motion falls below their resolution.

    uv run examples/cantilever.py
    uv run examples/cantilever.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from common import capsules

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
    length, segments, radius = 1.0, 40, 0.01
    bend_stiffness = 140.0  # per joint [N m / rad]: EI = K h
    bend_damping = 0.0  # per joint [N m s / rad]: off, so what decays is numerical damping
    height, gap, row_gap, columns = 1.0, 0.3, 1.2, 2  # bottom row's clamp height; spacing of the grid

    def __init__(self, viewer, args=None, *, der: tuple[str, ...] = ("DER ADMM, implicit midpoint", "DER ADMM, implicit Euler"),
                 vbd: tuple[tuple[int, int], ...] = ((10, 5), (100, 50)), der_substeps: int = 4):
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

        # One model per rod, clamped at the origin, and a scene with every rod in place, only rendered.
        self.sims = {}
        scene = newton.ModelBuilder()
        self.labels = []  # (text, world point just above the clamp, color), for annotating renders
        for (name, color, add, make_solver, substeps), clamp in zip(rods, clamps):
            dt_ms = 1e3 * self.frame_dt / substeps
            self.labels.append((f"{name}\ndt = {dt_ms:.3g} ms", (clamp[0], 0.0, clamp[1] + 0.03), color))
            builder = newton.ModelBuilder()
            bodies = add(builder, (0.0, 0.0))
            builder.color()
            model = builder.finalize()
            self.sims[name] = _Sim(model, make_solver(model), bodies, substeps, offset=(clamp[0], 0.0, clamp[1]))
            add(scene, clamp)
        self.model = scene.finalize()
        self.state_0 = self.model.state()
        self._gather()

        # Euler-Bernoulli: uniform load W, clamp at h / 2.
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
        camera = getattr(viewer, "camera", None)
        if camera is not None:
            # Centre the grid, with room for a rod that droops nearly its length and swings back past its
            # clamp (VBD at few iterations) on both sides, the labels above, and an equal margin around.
            margin, swing, droop, label = 0.15, 0.61 * self.length, 0.944 * self.length, 0.19
            grid = min(len(rods), self.columns) * (self.length + self.gap) - self.gap + self.radius
            left, right = -swing - margin, grid + swing + margin
            top, bottom = self.height + (rows - 1) * self.row_gap + label + margin, self.height - droop - margin
            aspect = camera.width / camera.height if camera.height else 16.0 / 9.0
            half = max(0.5 * (top - bottom), 0.5 * (right - left) / aspect)
            dist = half / np.tan(np.deg2rad(0.5 * camera.fov))
            viewer.set_camera(pos=wp.vec3(0.5 * (left + right), -dist, 0.5 * (top + bottom)), pitch=0.0, yaw=90.0)

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
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("euler_bernoulli", self.beam_starts, self.beam_ends, (1.0, 1.0, 1.0))
        self.viewer.end_frame()

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


class _Sim:
    """One rod, its solver and its tip-sag history (sampled every substep)."""

    def __init__(self, model, solver, bodies: list[int], substeps: int, offset):
        self.model, self.solver, self.bodies, self.substeps = model, solver, bodies, substeps
        self.offset = wp.vec3(*offset)  # where the rod is drawn
        self.state_0, self.state_1 = model.state(), model.state()
        if model.particle_count:  # a DER rod
            flatten_state(self.state_0)
            flatten_state(self.state_1)
        self.z0 = float(model.body_q.numpy()[bodies[-1], 2])
        self.dt, self.t, self.sag, self.iterations = None, [], [], []

    def run(self, frame_dt: float):
        self.dt = frame_dt / self.substeps
        for _ in range(self.substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            tip = capsules(self.model, self.state_0, [self.bodies])[0][1][-1, 2]
            self.t.append((len(self.t) + 1) * self.dt)
            self.sag.append(self.z0 - tip)
            if hasattr(self.solver, "last_iterations"):
                self.iterations.append(self.solver.last_iterations)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
