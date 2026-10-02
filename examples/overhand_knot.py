"""Pull a loose overhand knot tight with the ADMM DER solver, and compare its traction with theory.

The rope starts as Shastri's long trefoil with tails turned onto one line, inflated so no two parts
are closer than 1.6 diameters. It first relaxes for ``settle`` seconds with its velocities damped, so
the loose knot does not swing; then both end segments, clamped, are pulled apart along that line.
Self-contact with friction holds the knot as it tightens, and the rope must never pass through itself.

Every frame records the traction ``F`` (from the axial strain of the tails), the end-to-end distance
and ``R``, the radius of curvature where the rope leaves the braid. The viewer plots ``F h^2 / B``
against Audoly, Clauvelin & Neukirch, *Elastic knots*, PRL 99, 164301 (2007), for weak friction:
``F h^2 / B = eps^4 / 2 + mu sigma eps^3`` with ``eps = sqrt(h / R)``, ``sigma = 0.492`` (trefoil),
``h`` the rod radius and ``B = E pi h^4 / 4``. A larger ``--scale`` starts at a smaller ``eps`` (0.22
at the default 0.3, 0.41 at 0.1), where the asymptotics hold better, and takes longer to pull tight.

    uv run examples/overhand_knot.py
    uv run examples/overhand_knot.py --viewer null --test
    uv run examples/overhand_knot.py --viewer null --scale 0.1 --num-frames 480 --plot knot.png
"""

import newton
import newton.examples
import numpy as np
from common import CableExample, Drive, font, segment_distance, segment_dofs, smoothstep
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from dismech_newton import ADMMDiSMechSolver

SIGMA_TREFOIL = 0.492  # Audoly et al. (2007), the trefoil's friction constant
BACKGROUND, TEXT = (24, 24, 28), (220, 220, 220)


class Example(CableExample):
    radius, youngs_modulus = 0.005, 1.0e7
    contact_tol = 1.05  # segments closer than this many diameters touch
    exclude = 6  # segments this close along the rope never count as touching
    plot_every = 10  # frames between plot updates
    settle = 2.5  # seconds the loose knot relaxes before the pull, its velocities damped (about critically)

    def __init__(self, viewer, args=None):
        self.friction = getattr(args, "friction", 0.05)
        self.scale = getattr(args, "scale", 0.3)
        h = self.radius
        points = side_on(inflate(long_trefoil(scale=self.scale, tail=0.25, seg=h), h, 3.2 * h))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        rod = newton.Rod(points, radius=h, youngs_modulus=self.youngs_modulus, poissons_ratio=0.3)
        bodies = ADMMDiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
                                           bend_damping=0.1)
        for body in (bodies[0], bodies[-1]):
            ADMMDiSMechSolver.fix_segment(builder, body)
        model = builder.finalize()
        solver = ADMMDiSMechSolver(model, friction=self.friction)
        solver.triplets.rest.zero_()  # a rope: straight and untwisted at rest, not knotted
        self.start(viewer, model, solver, h, contact_matching_pos_threshold=h)
        self.ends = Drive(model, segment_dofs(model, bodies[0])[:6] + segment_dofs(model, bodies[-1])[:6])
        self.drives = (self.ends,)
        # Pull each end this far (to eps of about 0.7 at scale 0.3), slowly: near equilibrium.
        self.pull, self.pull_time = 1.3 * (1.25 * self.scale - 0.04), 52.0 * self.scale
        axis = (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0])
        self.direction = np.concatenate([-axis, -axis, axis, axis])
        self.rest = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.pairs = np.triu_indices(len(points) - 1, k=self.exclude)
        self.stretch = self.youngs_modulus * np.pi * h**2  # EA
        self.bend = self.youngs_modulus * np.pi * h**4 / 4.0  # B = EI
        self.history = []  # (t, F, e, R, d) per frame, d the end-to-end distance
        self.min_gap = np.inf
        self.frame = 0

    def drive(self, t0, t1):
        s = [smoothstep(t, self.settle, self.settle + self.pull_time) for t in (t0, t1)]
        self.ends.set(*(self.pull * v * self.direction for v in s))

    def step(self):
        super().step()
        # The loose knot relaxes without swinging: damp its velocities by 10% a frame, easing off over
        # the first fifth of the pull (before comparison() reads any frame).
        damping = 0.1 * (1.0 - smoothstep(self.sim_time, self.settle, self.settle + 0.2 * self.pull_time))
        if damping > 0.0:
            self.state_0.dismech.qd.assign((1.0 - damping) * self.state_0.dismech.qd.numpy())
        self.record()

    def render(self):
        if self.frame % self.plot_every == 0:
            self.viewer.log_image("theory", self.image())
        self.frame += 1
        super().render()

    def gaps(self, x: np.ndarray) -> np.ndarray:
        i, j = self.pairs
        return segment_distance(x[i], x[i + 1], x[j], x[j + 1]) / (2.0 * self.radius)

    def record(self):
        x = self.state_0.particle_q.numpy().astype(np.float64)
        edge = np.diff(x, axis=0)
        length = np.linalg.norm(edge, axis=1)
        d = np.linalg.norm(x[-1] - x[0])
        e = length.sum() - d
        i, j = self.pairs
        close = self.gaps(x) < self.contact_tol
        hit = np.flatnonzero(np.bincount(np.r_[i[close], j[close]], minlength=len(length)))
        if len(hit) == 0:  # no knot
            self.history.append((self.sim_time, np.nan, e, np.nan, d))
            return
        first, last = hit[0], hit[-1]
        # R: the curvature peaks where the rope leaves the braid (the braid's helices bend less).
        tangent = edge / length[:, None]
        kappa = np.linalg.norm(np.cross(tangent[:-1], tangent[1:]), axis=1) / (0.5 * (length[:-1] + length[1:]))
        # F: axial strain of the free tails, away from the clamps and the knot.
        tails = np.r_[2:max(first - 4, 2), min(last + 5, len(length) - 2):len(length) - 2]
        force = self.stretch * np.mean(length[tails] / self.rest[tails] - 1.0) if len(tails) else np.nan
        self.history.append((self.sim_time, force, e, 1.0 / kappa[first:last].max(), d))

    def comparison(self) -> dict[str, np.ndarray]:
        """Simulated and theoretical ``F h^2 / B`` while the knot slides: the theory assumes sliding
        friction. A stuck knot loads its tails instead, so ``e`` stalls while the ends part; ``slide``
        is the share of the pull that tightens the knot, ``-de / dd``."""
        if len(self.history) < 2:
            return {key: np.zeros(0) for key in ("eps", "Fbar", "audoly")}
        t, force, e, radius, d = np.array(self.history).T
        slide = -np.gradient(e, t) / np.maximum(np.gradient(d, t), 1e-12)
        t = t - self.settle
        keep = (t > 0.2 * self.pull_time) & (t <= self.pull_time) & np.isfinite(force) & (slide > 0.5)
        eps = np.sqrt(self.radius / radius[keep])
        fbar = force[keep] * self.radius**2 / self.bend
        return {"eps": eps, "Fbar": fbar, "audoly": audoly_force(eps, self.friction)}

    def image(self) -> np.ndarray:
        """``F h^2 / B`` against Audoly et al. (2007), as an image for the viewer."""
        c, mu, s, m = self.comparison(), self.friction, 300, 60
        img = Image.new("RGB", (s + 2 * m, s + 2 * m + 30), BACKGROUND)
        draw = ImageDraw.Draw(img)
        eps_max = max(0.8, 1.05 * c["eps"].max()) if len(c["eps"]) else 0.8  # the default pull reaches 0.72
        eps = np.linspace(0.0, eps_max, 100)
        ax = Axes(draw, (m, m), s, (0.0, eps_max), (0.0, float(audoly_force(eps_max, mu))))
        ax.frame("eps = sqrt(h / R)", "F h^2 / B", f"trefoil: Audoly et al. 2007, mu = {mu:g}")
        ax.curve(eps, audoly_force(eps, 0.0), (130, 130, 130))
        ax.curve(eps, audoly_force(eps, mu), (235, 235, 235))
        ax.points(c["eps"], c["Fbar"], (90, 170, 255))
        x0 = m
        for text, color in (("theory", (235, 235, 235)), ("frictionless", (130, 130, 130)),
                            ("simulation", (90, 170, 255))):
            draw.rectangle([x0, img.height - 22, x0 + 10, img.height - 12], fill=color)
            draw.text((x0 + 16, img.height - 17), text, fill=TEXT, font=font(13), anchor="lm")
            x0 += 30 + int(draw.textlength(text, font=font(13)))
        return np.asarray(img)

    def test_post_step(self):
        self.min_gap = min(self.min_gap, self.gaps(self.state_0.particle_q.numpy()).min())

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        assert self.min_gap > 0.8, f"rope passed into itself (min gap {self.min_gap:.2f} diameters)"
        c = self.comparison()
        assert len(c["Fbar"]) > 50, "the knot did not slide"
        ratio = float(np.median(c["Fbar"] / c["audoly"]))
        assert 0.75 < ratio < 1.33, f"traction {ratio:.2f} times Audoly et al. (2007)"


class Axes:
    """A square plot of side ``size`` at pixel ``origin`` (top left), linear or log-log."""

    def __init__(self, draw, origin, size: int, xlim, ylim, log: bool = False):
        self.draw, self.origin, self.size, self.log = draw, origin, size, log
        self.lim = [np.log10(v) if log else np.asarray(v, dtype=float) for v in (xlim, ylim)]

    def px(self, x, y):
        x, y = (np.log10(np.asarray(v, dtype=float)) if self.log else np.asarray(v, dtype=float) for v in (x, y))
        (x0, x1), (y0, y1) = self.lim
        return self.origin[0] + (x - x0) / (x1 - x0) * self.size, self.origin[1] + (y1 - y) / (y1 - y0) * self.size

    def frame(self, xlabel: str, ylabel: str, title: str):
        (ox, oy), s, d, f = self.origin, self.size, self.draw, font(12)
        d.rectangle([ox, oy, ox + s, oy + s], outline=(110, 110, 110))
        for axis, (lo, hi) in enumerate(self.lim):
            if self.log:  # 1, 2 and 5 of every decade in range
                ticks = [m * 10.0**k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) for m in (1, 2, 5)]
                ticks = [v for v in ticks if lo <= np.log10(v) <= hi]
            else:
                ticks = np.linspace(lo, hi, 5)
            for v in ticks:
                label = f"{v:.2g}"
                if axis == 0:
                    x = float(self.px(v, 1.0 if self.log else 0.0)[0])
                    d.text((x, oy + s + 5), label, fill=TEXT, font=f, anchor="mt")
                else:
                    y = float(self.px(1.0 if self.log else 0.0, v)[1])
                    d.text((ox - 5, y), label, fill=TEXT, font=f, anchor="rm")
        d.text((ox + s / 2, oy + s + 24), xlabel, fill=TEXT, font=font(13), anchor="mt")
        d.text((ox, oy - 8), ylabel, fill=TEXT, font=font(13), anchor="ld")
        d.text((ox + s / 2, oy - 30), title, fill=TEXT, font=font(15), anchor="mm")

    def curve(self, x, y, color):
        (ox, oy), s = self.origin, self.size
        u, v = self.px(x, y)
        ok = np.isfinite(u) & np.isfinite(v) & (u >= ox) & (u <= ox + s) & (v >= oy) & (v <= oy + s)
        for a in range(len(u) - 1):
            if ok[a] and ok[a + 1]:
                self.draw.line([(u[a], v[a]), (u[a + 1], v[a + 1])], fill=color, width=2)

    def points(self, x, y, color, r: float = 2.5):
        (ox, oy), s = self.origin, self.size
        for a, b in zip(*self.px(x, y), strict=True):
            if np.isfinite(a) and np.isfinite(b) and ox <= a <= ox + s and oy <= b <= oy + s:
                self.draw.ellipse([a - r, b - r, a + r, b + r], fill=color)


# -- theory and geometry --------------------------------------------------------------------------


def audoly_force(eps, mu: float, sigma: float = SIGMA_TREFOIL):
    """``F h^2 / B`` of a trefoil at ``eps = sqrt(h / R)`` (Audoly et al. 2007)."""
    eps = np.asarray(eps)
    return 0.5 * eps**4 + mu * sigma * eps**3


def long_trefoil(scale: float, tail: float, seg: float, blend: float = 0.06) -> np.ndarray:
    """``(t^3 - 3t, t^4 - 4t^2, t^5 - 10t)`` (axes rescaled) with tails that turn, over ``blend``, onto
    the line through their ends, resampled to ``seg``."""
    t = np.linspace(-2.4, 2.4, 4000)
    core = np.column_stack((t**3 - 3 * t, t**4 - 4 * t**2, t**5 - 10 * t)) * np.array([1 / 6, 1 / 10, 1 / 30]) * scale
    s = np.linspace(0.0, tail, 400)
    w = np.array([smoothstep(v, 0.0, blend) for v in s])[:, None]

    def grow(p, d, axis):
        d = d / np.linalg.norm(d)
        tangent = (1.0 - w) * d + w * axis
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        return p + np.cumsum(tangent * np.gradient(s)[:, None], axis=0)

    axis = np.array([0.0, 0.0, 1.0])
    for _ in range(4):  # aim both tails along the line through their ends
        t0, t1 = grow(core[0], core[0] - core[1], -axis), grow(core[-1], core[-1] - core[-2], axis)
        axis = (t1[-1] - t0[-1]) / np.linalg.norm(t1[-1] - t0[-1])
    p = np.vstack((t0[::-1], core, t1))
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
    u = np.linspace(0.0, arc[-1], int(round(arc[-1] / seg)) + 1)
    return np.column_stack([np.interp(u, arc, p[:, k]) for k in range(3)])


def side_on(x: np.ndarray) -> np.ndarray:
    """``x`` turned rigidly so its ends lie along ``x`` and its widest extent across them along ``z``
    (the viewer's up): a camera looking along ``y`` sees the knot side on."""
    axis = (x[-1] - x[0]) / np.linalg.norm(x[-1] - x[0])
    off = (x - x.mean(0)) - np.outer((x - x.mean(0)) @ axis, axis)
    wide = np.linalg.svd(off, full_matrices=False)[2][0]
    rot = Rotation.align_vectors([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [axis, wide], weights=[1e6, 1.0])[0]
    return rot.apply(x)


def inflate(x: np.ndarray, seg: float, clearance: float, exclude: int = 6) -> np.ndarray:
    """Push points more than ``exclude`` apart along the curve to ``clearance``, in steps far smaller
    than a segment (nothing passes through), keeping the segment lengths."""
    x = x.copy()
    i, j = np.triu_indices(len(x), k=exclude)
    for _ in range(400):
        d = x[i] - x[j]
        dist = np.linalg.norm(d, axis=1)
        close = dist < clearance
        if not close.any():
            break
        step = (0.25 * np.minimum(clearance - dist[close], 0.2 * seg) / np.maximum(dist[close], 1e-9))[:, None]
        np.add.at(x, i[close], step * d[close])
        np.add.at(x, j[close], -step * d[close])
        x[1:-1] += 0.05 * (x[:-2] + x[2:] - 2.0 * x[1:-1])
        for _ in range(10):
            e = x[1:] - x[:-1]
            c = 0.5 * (1.0 - seg / np.linalg.norm(e, axis=1, keepdims=True)) * e
            x[:-1] += c
            x[1:] -= c
    return x


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--friction", type=float, default=0.05, help="Coulomb coefficient mu")
    parser.add_argument("--scale", type=float, default=0.3, help="size of the loose knot")
    parser.add_argument("--plot", default=None, help="save the comparison with theory to this image")
    parser.set_defaults(num_frames=1140)  # the settle, the pull (52 * scale seconds) and a little
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if args.plot:
        Image.fromarray(example.image()).save(args.plot)
        print(f"wrote {args.plot}")
