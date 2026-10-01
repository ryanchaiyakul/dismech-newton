"""Fit a rod's bending stiffness and damping to an observed motion, by gradients through the solver.

The observation is a cantilever released under gravity, simulated with the true parameters. From a
wrong guess, L-BFGS fits the parameters' logs to it. Each gradient is one ``wp.Tape`` over the rollout:
the solver records every step as its implicit-function adjoint (:mod:`dismech_newton.adjoint`), one
linear solve per step, never the ADMM iterations.

The viewer replays the fit: the truth (left, grey) and each L-BFGS iterate in turn (right, green)
swing side by side in slow motion, with a pause before the next iterate. The image window shows the
iterate's parameters and its place on the loss landscape over (stiffness, damping), with the downhill
gradient (arrows) and the optimiser's path so far. The landscape is only the picture: 961 rods, one
per grid point, in one simulation (they never touch), each with its own loss; one backward pass of
their sum gives every rod's own gradient, since each rod's parameters reach only its own loss.

    uv run examples/fit_stiffness.py
    uv run examples/fit_stiffness.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import scipy.ndimage
import warp as wp
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize

from dismech_newton import ADMMDiSMechSolver as Solver
from dismech_newton import add_rod
from dismech_newton.strains import vec10f

STEPS, DT = 60, 1.0 / 120.0  # half a second: the sag and the first bounce
TRUE, GUESS = (10.0, 0.05), (2.5, 0.3)  # (bend stiffness [N m / rad], bend damping [N m s / rad])
RADIUS, SEGMENTS = 0.01, 20
TOL = 1.0e-5  # the fit resolves the motion only as well as the solver does
GRID = 31  # landscape samples per axis
K_RANGE, C_RANGE = (0.625, 20.0), (0.003125, 0.8)  # geometric, with TRUE on the grid


@wp.kernel
def set_bend(log_p: wp.array[float], base: wp.array[vec10f], params: wp.array[vec10f]):
    """Bend stiffness and damping (both curvatures) from their logs; the rest from ``base``."""
    t = wp.tid()
    p = base[t]
    k, c = wp.exp(log_p[0]), wp.exp(log_p[1])
    params[t] = vec10f(p[0], p[1], k, k, p[4], p[5], p[6], c, c, p[9])


@wp.kernel
def mse(x: wp.array[wp.vec3], x_obs: wp.array[wp.vec3], nodes: int, scale: float, loss: wp.array[float]):
    """Per rod (``nodes`` nodes each, all against the one observation), the scaled squared error."""
    i = wp.tid()
    wp.atomic_add(loss, i // nodes, scale * wp.length_sq(x[i] - x_obs[i % nodes]))


def add_cantilever(builder, add=Solver.add_rod, **kwargs):
    """A clamped rod, all of them at the same place (rods never touch here: no contacts are passed)."""
    first_edge = len(builder.custom_attributes["dismech:edge_fixed"].values) if builder.custom_attributes.get(
        "dismech:edge_fixed") else 0
    rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.5, segment_count=SEGMENTS, radius=RADIUS)
    add(builder, rod, stretch_stiffness=1.0e4, twist_stiffness=1.0, **kwargs)
    Solver.fix_segment(builder, edge=first_edge)


class Rollout:
    """``STEPS`` steps of a model from rest, and every rod's error against an observation."""

    def __init__(self, model):
        self.model = model
        self.solver = Solver(model, tol=TOL)
        self.states = [model.state(requires_grad=True) for _ in range(STEPS + 1)]
        self.rods = len(model.dismech.edge_length) // SEGMENTS
        self.nodes = model.particle_count // self.rods
        self.loss = wp.zeros(self.rods, dtype=float, requires_grad=True)

    def run(self, observed=None, before=None):
        self.loss.zero_()
        if before is not None:
            before()
        scale = 1.0 / (STEPS * self.nodes * RADIUS**2)  # mean squared error in rod radii
        for t in range(STEPS):
            self.solver.step(self.states[t], self.states[t + 1], None, None, DT)
            if observed is not None:
                wp.launch(mse, dim=self.model.particle_count,
                          inputs=[self.states[t + 1].particle_q, observed[t], self.nodes, scale], outputs=[self.loss])


def _motion(r: Rollout, **extra) -> dict:
    return dict(nodes=np.stack([s.particle_q.numpy() for s in r.states]),
                bodies=np.stack([s.body_q.numpy() for s in r.states]), **extra)


def fit():
    """Observe, then L-BFGS on the parameters' logs: the model, the observed motion, every iterate's."""
    builder = newton.ModelBuilder()
    add_cantilever(builder, bend_stiffness=1.0, color=(0.3, 0.8, 0.5))
    model = builder.finalize()
    r = Rollout(model)
    params = model.dismech.triplet_params
    params.requires_grad = True
    base = wp.clone(params, requires_grad=False)
    log_p = wp.array(np.log(TRUE), dtype=float, requires_grad=True)

    def set_params():
        wp.launch(set_bend, dim=len(params), inputs=[log_p, base], outputs=[params])

    r.run(before=set_params)
    truth = _motion(r)
    observed = [wp.clone(s.particle_q, requires_grad=False) for s in r.states[1:]]

    def loss_and_grad(x):
        log_p.assign(x)
        tape = wp.Tape()
        with tape:
            r.run(observed, before=set_params)
        tape.backward(r.loss)
        g = log_p.grad.numpy().astype(np.float64)
        tape.zero()
        return float(r.loss.numpy()[0]), g

    iterates = [np.log(GUESS)]
    minimize(loss_and_grad, iterates[0], jac=True, method="L-BFGS-B", callback=lambda x: iterates.append(x.copy()))

    replays = []  # every iterate's motion, for the viewer
    for x in iterates:
        log_p.assign(x)
        r.run(observed, before=set_params)
        replays.append(_motion(r, x=x, loss=float(r.loss.numpy()[0])))
    return model, truth, replays


def landscape(observed: np.ndarray):
    """Loss and its gradient in (log k, log c) on the grid (rows: damping): one batched rollout, one backward."""
    ks, cs = np.geomspace(*K_RANGE, GRID), np.geomspace(*C_RANGE, GRID)
    builder = newton.ModelBuilder()
    for _ in range(GRID**2):
        add_cantilever(builder, add=add_rod, bend_stiffness=1.0, proxies=False)
    model = builder.finalize()
    params = model.dismech.triplet_params
    p = params.numpy().reshape(GRID, GRID, -1, 10)  # [damping, stiffness, triplet]: as set_bend sets them
    p[..., 2:4] = ks[None, :, None, None]
    p[..., 7:9] = cs[:, None, None, None]
    params.assign(p.reshape(-1, 10))
    r = Rollout(model)  # after the parameters: the ADMM penalties scale with them
    params.requires_grad = True
    tape = wp.Tape()
    with tape:
        r.run([wp.array(o, dtype=wp.vec3) for o in observed[1:]])
    tape.backward(grads={r.loss: wp.ones(r.rods, dtype=float)})
    g = params.grad.numpy().reshape(GRID, GRID, -1, 10)
    p = params.numpy().reshape(GRID, GRID, -1, 10)
    grad = np.stack([np.sum((g[..., 2] + g[..., 3]) * p[..., 2], axis=2),  # d loss / d log k
                     np.sum((g[..., 7] + g[..., 8]) * p[..., 7], axis=2)], axis=-1)  # d loss / d log c
    return r.loss.numpy().reshape(GRID, GRID), grad


# -- the picture ----------------------------------------------------------------------------

_VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], dtype=float)
_TEXT = (220, 220, 220)


def _font(size: int):
    for name in ("segoeui.ttf", "Arial.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


class LandscapePlot:
    """The loss landscape as an image: banded log loss, downhill arrows, the true parameters (cross)
    and the optimiser's path (red start, green current)."""

    size, margin = 360, 48

    def __init__(self, loss: np.ndarray, grad: np.ndarray):
        s, m = self.size, self.margin
        z = scipy.ndimage.zoom(np.log10(loss), s / GRID, order=1)[::-1]  # top row: high damping
        band = np.floor(12 * (z - z.min()) / (z.max() - z.min())) / 12  # banded colours read as contours
        rgb = np.stack([np.interp(band, np.linspace(0, 1, 5), _VIRIDIS[:, i]) for i in range(3)], axis=-1)
        rgb[(np.diff(band, axis=0, prepend=band[:1]) != 0) | (np.diff(band, axis=1, prepend=band[:, :1]) != 0)] *= 0.7
        self.base = Image.new("RGB", (s + 2 * m, s + 2 * m), (24, 24, 28))
        self.base.paste(Image.fromarray(rgb.astype(np.uint8)), (m, m))
        draw = ImageDraw.Draw(self.base)
        font = _font(13)
        log_k, log_c = np.log(np.geomspace(*K_RANGE, GRID)), np.log(np.geomspace(*C_RANGE, GRID))
        for j in range(1, GRID, 3):
            for i in range(1, GRID, 3):
                d = -grad[j, i] / max(np.linalg.norm(grad[j, i]), 1e-30)
                p = np.array(self.to_px(log_k[i], log_c[j]))
                self._arrow(draw, p, p + 13 * np.array([d[0], -d[1]]))
        for k in (1, 2, 5, 10):
            draw.text((self.to_px(np.log(k), 0)[0], s + m + 6), f"{k:g}", fill=_TEXT, font=font, anchor="mt")
        for c in (0.01, 0.03, 0.1, 0.3):
            draw.text((m - 6, self.to_px(0, np.log(c))[1]), f"{c:g}", fill=_TEXT, font=font, anchor="rm")
        draw.text((m + s / 2, s + m + 26), "bend stiffness", fill=_TEXT, font=font, anchor="mt")
        draw.text((m + s / 2, m / 2), "loss over (stiffness, damping)", fill=_TEXT, font=_font(15), anchor="mm")
        draw.text((8, m - 14), "damping", fill=_TEXT, font=font, anchor="lm")
        x, y = self.to_px(*np.log(TRUE))
        draw.line([(x - 7, y - 7), (x + 7, y + 7)], fill=(255, 255, 255), width=3)
        draw.line([(x - 7, y + 7), (x + 7, y - 7)], fill=(255, 255, 255), width=3)

    def to_px(self, log_k: float, log_c: float) -> tuple[float, float]:
        s, m = self.size, self.margin
        u = (log_k - np.log(K_RANGE[0])) / (np.log(K_RANGE[1]) - np.log(K_RANGE[0]))
        v = (log_c - np.log(C_RANGE[0])) / (np.log(C_RANGE[1]) - np.log(C_RANGE[0]))
        return m + u * s, m + (1.0 - v) * s

    @staticmethod
    def _arrow(draw, p, q):
        d = (q - p) / np.linalg.norm(q - p)
        o = np.array([-d[1], d[0]])
        draw.line([tuple(p), tuple(q)], fill=(235, 235, 235), width=1)
        draw.polygon([tuple(q), tuple(q - 4 * d + 2.5 * o), tuple(q - 4 * d - 2.5 * o)], fill=(235, 235, 235))

    def image(self, path: list[np.ndarray], banner: list[tuple[str, tuple]] = ()) -> np.ndarray:
        """The landscape with ``path`` (iterates so far), under ``banner``: lines of (text, colour)."""
        plot = self.base.copy()
        draw = ImageDraw.Draw(plot)
        pts = [self.to_px(*x) for x in path]
        if len(pts) > 1:
            draw.line(pts, fill=(255, 140, 60), width=3)
        for (x, y), r, color in [(p, 3, (255, 140, 60)) for p in pts[1:-1]] + [(pts[0], 5, (230, 70, 60))]:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        x, y = pts[-1]
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(80, 220, 130), outline=(255, 255, 255), width=2)
        line = 26
        img = Image.new("RGB", (plot.width, plot.height + len(banner) * line + 16), (24, 24, 28))
        img.paste(plot, (0, img.height - plot.height))
        draw = ImageDraw.Draw(img)
        font = _font(18)
        for i, (text, color) in enumerate(banner):
            draw.text((self.margin, 12 + i * line), text, fill=color, font=font)
        return np.asarray(img)


class Example:
    """The truth (left) and an iterate (right) side by side, in slow motion; a pause on the last
    frame, then the next iterate. The image window: the iterate, and its place on the landscape."""

    slow = 3  # rendered frames per simulated step
    pause = 60  # frames held on the last step before the next iterate
    offset = 0.75  # [m] between the two rods

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        print("fitting ...")
        model, self.truth, self.replays = fit()
        for r in self.replays:
            print(f"loss {r['loss']:.3e}  stiffness {np.exp(r['x'][0]):.4f}  damping {np.exp(r['x'][1]):.5f}")
        print(f"true  stiffness {TRUE[0]:.4f}  damping {TRUE[1]:.5f}")
        print("loss landscape: one rollout of", GRID**2, "rods ...")
        self.plot = LandscapePlot(*landscape(self.truth["nodes"]))

        builder = newton.ModelBuilder()  # two rods to draw: the truth and the iterate
        add_cantilever(builder, bend_stiffness=1.0, color=(0.85, 0.85, 0.85))
        add_cantilever(builder, bend_stiffness=1.0, color=(0.3, 0.8, 0.5))
        self.model = builder.finalize()
        self.state = self.model.state()
        self.shift = np.array([self.offset, 0.0, 0.0], dtype=np.float32)
        self.k, self.frame, self.sim_time = 0, 0, 0.0
        viewer.set_model(self.model)
        viewer.set_camera(pos=wp.vec3(0.62, -0.95, 0.86), pitch=0.0, yaw=90.0)
        if hasattr(viewer, "renderer"):
            viewer.renderer.line_width = 2.5
        self._log_iterate()

    @property
    def t(self) -> int:
        """The simulated step on screen: slow motion, then held during the pause."""
        return min(self.frame // self.slow, STEPS)

    def _log_iterate(self):
        self.viewer.log_image("fit", self.panel())

    def panel(self) -> np.ndarray:
        """The iterate's parameters above its place on the landscape."""
        r = self.replays[self.k]
        k, c = np.exp(r["x"])
        lines = [(f"iterate {self.k} / {len(self.replays) - 1}", (255, 255, 255)),
                 (f"stiffness  {k:7.3f}   (true {TRUE[0]:g})", (120, 220, 150)),
                 (f"damping    {c:7.4f}   (true {TRUE[1]:g})", (120, 220, 150)),
                 (f"loss       {r['loss']:.2e}", (220, 220, 220))]
        return self.plot.image([x["x"] for x in self.replays[: self.k + 1]], lines)

    def step(self):
        self.frame += 1
        if self.frame > STEPS * self.slow + self.pause:
            self.frame = 0
            self.k = (self.k + 1) % len(self.replays)
            self._log_iterate()
        self.sim_time += DT / self.slow

    def render(self):
        t, r = self.t, self.replays[self.k]
        body_q = np.concatenate([self.truth["bodies"][t], r["bodies"][t]])
        body_q[len(body_q) // 2 :, :3] += self.shift
        self.state.body_q.assign(body_q)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self._polyline("truth tip", self.truth["nodes"][: t + 1, -1], (0.7, 0.7, 0.7))
        self._polyline("iterate tip", r["nodes"][: t + 1, -1] + self.shift, (0.3, 0.8, 0.5))
        self.viewer.end_frame()

    def _polyline(self, name: str, x: np.ndarray, color):
        if len(x) < 2:
            x = np.repeat(x, 2, axis=0)
        self.viewer.log_lines(name, wp.array(x[:-1], dtype=wp.vec3), wp.array(x[1:], dtype=wp.vec3), color)

    def test_final(self):
        k, c = np.exp(self.replays[-1]["x"])
        assert abs(k / TRUE[0] - 1.0) < 0.01, f"stiffness {k:.4f}, true {TRUE[0]}"
        assert abs(c / TRUE[1] - 1.0) < 0.02, f"damping {c:.5f}, true {TRUE[1]}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1200)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
