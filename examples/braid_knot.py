"""Pull a long overhand knot tight with the ADMM DER solver, and compare it with Jawed et al. (2015).

A long overhand knot is a braid of two strands winding ``n`` full turns around each other (``2n + 1``
crossings, unknotting number ``n``), a nearly planar loop and two tails. Jawed, Dieleman, Audoly &
Reis, *Untangling the mechanics and topology in the frictional response of long overhand elastic
knots*, PRL 115, 118302 (2015), treat it for ``n >> 1``, ``eps = sqrt(h / R) << 1`` and friction
large against the loop's pull ``B / (2 R^2)``, with ``h`` the rod radius, ``R`` the radius of
curvature where the rope leaves the braid for the loop, ``B = E pi h^4 / 4`` and ``e`` the end-to-end
shortening. The braid is a helix of wave number ``k = (12 h R)^(-1/2)`` (Eq. 3), friction carries the
traction, ``F / (B k^2) = mu 2 pi n h k`` (Eq. 4), and the loop's elastica closes the system (Eq. 5).

The rope starts as that shape, loose: strands 1.6 diameters apart, a braid of the pitch Eq. (3) gives
at ``R = h / eps0^2``, and the loop the planar elastica that fits it. Both end segments are clamped and
pulled apart along the braid. The tails are long: as the knot tightens the loop turns about the
braid and the rope screws through it, and a knot that reached a clamp would slip off the end. Every
frame records ``F`` (from the axial strain of the tails), ``e``, ``R`` and the braid length ``l`` (the
contact region on one strand); the viewer plots Figs. 2(c), 2(d) and 4 of the paper while the knot
slides. ``mu = 0.119`` is the paper's fit to its experiments.

    uv run examples/braid_knot.py
    uv run examples/braid_knot.py --viewer null --test   # about 6 minutes
    uv run examples/braid_knot.py --viewer null --turns 2 --plot braid.png
"""

from functools import cache

import newton
import newton.examples
import numpy as np
from common import Drive, font, segment_distance, segment_dofs
from overhand_knot import BACKGROUND, TEXT, Axes, inflate
from overhand_knot import Example as KnotExample
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.special import ellipe, ellipk

from dismech_newton import ADMMDiSMechSolver

SIM, THEORY = (90, 170, 255), (235, 235, 235)


class Example(KnotExample):
    """The trefoil example's frame loop and records (:class:`overhand_knot.Example`) on a long knot."""

    radius, youngs_modulus = 0.005, 1.0e7
    contact_tol = 1.05  # segments closer than this many diameters touch
    exclude = 6  # segments this close along the rope never count as touching
    plot_every = 20  # frames between plot updates
    settle = 0.0  # starts from the theory's shape: no relaxation

    def __init__(self, viewer, args=None):
        self.friction = getattr(args, "friction", 0.119)
        self.turns = getattr(args, "turns", 3)
        self.eps0 = getattr(args, "eps", 0.1)
        h = self.radius
        self.seg = getattr(args, "seg", 2.0) * h
        points = long_knot(self.turns, h, h / self.eps0**2, clearance=3.2 * h, tail=1.0, seg=self.seg)
        points = inflate(points, self.seg, 3.2 * h, exclude=self.exclude)
        # Seen across the braid (off its axes) the straight tails leave the picture: this is the knot's diagram.
        over = [o for _, o in crossings(points, (1.0, 0.13, 0.07))]
        if len(over) != 2 * (2 * self.turns + 1) or any(a == b for a, b in zip(over[:-1], over[1:], strict=True)):
            raise RuntimeError(f"not an alternating diagram of {2 * self.turns + 1} crossings")
        points = points[:, [2, 0, 1]]  # the braid along x, the loop rising in z (the viewer's up)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        rod = newton.Rod(points, radius=h, youngs_modulus=self.youngs_modulus, poissons_ratio=0.3)
        bodies = ADMMDiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
                                           bend_damping=0.1)
        for body in (bodies[0], bodies[-1]):
            ADMMDiSMechSolver.fix_segment(builder, body)
        model = builder.finalize()
        solver = ADMMDiSMechSolver(model, friction=self.friction, iterations=getattr(args, "iterations", 200))
        solver.triplets.rest.zero_()  # a rope: straight and untwisted at rest, not knotted
        self.start(viewer, model, solver, h, contact_matching_pos_threshold=h)
        self.ends = Drive(model, segment_dofs(model, bodies[0])[:6] + segment_dofs(model, bodies[-1])[:6])
        self.drives = (self.ends,)
        length = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.pull, self.pull_time = pull_plan(self.turns, h, self.eps0, getattr(args, "speed", 0.03))
        axis = (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0])
        self.direction = np.concatenate([-axis, -axis, axis, axis])
        self.rest = length
        self.stretch = self.youngs_modulus * np.pi * h**2  # EA
        self.bend = self.youngs_modulus * np.pi * h**4 / 4.0  # B = EI
        self.history = []  # (t, F, e, R, d, l) per frame, d the end-to-end distance
        self.min_gap = np.inf
        self.min_tail = np.inf  # the shortest tail seen, in segments: the knot must not reach a clamp
        self.frame = 0

    def gaps(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment pairs ``(i, j)`` closer than 1.5 diameters, and their gaps in diameters."""
        mid = 0.5 * (x[1:] + x[:-1])
        pairs = cKDTree(mid).query_pairs(self.seg + 3.0 * self.radius, output_type="ndarray")
        i, j = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) >= self.exclude].T
        return i, j, segment_distance(x[i], x[i + 1], x[j], x[j + 1]) / (2.0 * self.radius)

    def record(self):
        """As :meth:`overhand_knot.Example.record`, and ``l``: the contact region on strand X (before
        the loop) and on strand Y (after it), in arc length, averaged."""
        x = self.state_0.particle_q.numpy().astype(np.float64)
        edge = np.diff(x, axis=0)
        length = np.linalg.norm(edge, axis=1)
        d = np.linalg.norm(x[-1] - x[0])
        e = length.sum() - d
        i, j, gap = self.gaps(x)
        self.min_gap = min(self.min_gap, gap.min(initial=np.inf))
        close = gap < self.contact_tol
        hit = np.flatnonzero(np.bincount(np.r_[i[close], j[close]], minlength=len(length)))
        if len(hit) < 2:  # no knot
            self.history.append((self.sim_time, np.nan, e, np.nan, d, np.nan))
            return
        first, last = hit[0], hit[-1]
        self.min_tail = min(self.min_tail, first, len(length) - 1 - last)
        split = int(np.argmax(np.diff(hit)))  # the loop: the longest stretch without contact
        braid = np.mean([length[a:b + 1].sum() for a, b in ((first, hit[split]), (hit[split + 1], last))])
        # R: the curvature peaks where the rope leaves the braid for the loop.
        tangent = edge / length[:, None]
        kappa = np.linalg.norm(np.cross(tangent[:-1], tangent[1:]), axis=1) / (0.5 * (length[:-1] + length[1:]))
        # F: axial strain of the free tails, away from the clamps and the knot.
        tails = np.r_[2:max(first - 4, 2), min(last + 5, len(length) - 2):len(length) - 2]
        force = self.stretch * np.mean(length[tails] / self.rest[tails] - 1.0) if len(tails) else np.nan
        self.history.append((self.sim_time, force, e, 1.0 / kappa[first:last].max(), d, braid))

    def test_post_step(self):
        pass  # record tracks the closest approach

    def comparison(self) -> dict[str, np.ndarray]:
        """The records while the knot slides (see :meth:`overhand_knot.Example.comparison`) in the
        variables of Figs. 2(c), 2(d) and 4 of Jawed et al. (2015), with the theory at each."""
        keys = ("sqrt_hR", "wavelength", "x2d", "y2d", "x4", "y4", "jawed")
        if len(self.history) < 2:
            return {key: np.zeros(0) for key in keys}
        t, force, e, radius, d, l = np.array(self.history).T
        n, h, mu, b = self.turns, self.radius, self.friction, self.bend
        slide = -np.gradient(e, t) / np.maximum(np.gradient(d, t), 1e-12)
        keep = ((t > 0.1 * self.pull_time) & (t <= self.pull_time) & np.isfinite(force) & np.isfinite(l)
                & (slide > 0.5))
        force, e, radius, l = force[keep], e[keep], radius[keep], l[keep]
        k = 2.0 * n * np.pi / l
        x4 = n**2 * h / e
        return {"sqrt_hR": np.sqrt(h * radius), "wavelength": l / n, "x2d": 2.0 * np.pi * n * h * k,
                "y2d": force / (b * k**2), "x4": x4, "y4": n**2 * force * h**2 / b, "jawed": jawed_force(x4, mu)}

    def image(self) -> np.ndarray:
        """Figs. 2(c), 2(d) and 4 of Jawed et al. (2015), as an image for the viewer."""
        c, mu, s, m = self.comparison(), self.friction, 240, 62
        img = Image.new("RGB", (3 * (s + 2 * m), s + 2 * m + 30), BACKGROUND)
        draw = ImageDraw.Draw(img)

        def top(v, floor):
            return max(floor, 1.15 * float(np.nanmax(v))) if len(v) else floor

        # Fig. 2(c): the braid's wavelength l / n against sqrt(h R), Eq. (3).
        xm = top(c["sqrt_hR"], 0.06)
        ax = Axes(draw, (m, m), s, (0.0, xm), (0.0, 2.0 * np.pi * 12**0.25 * xm))
        ax.frame("sqrt(h R)  [m]", "l / n  [m]", "braid wavelength, Eq. (3)")
        ax.curve(np.array([0.0, xm]), 2.0 * np.pi * 12**0.25 * np.array([0.0, xm]), THEORY)
        ax.points(c["sqrt_hR"], c["wavelength"], SIM)
        # Fig. 2(d): F / (B k^2) against 2 pi n h k, slope mu (Eq. 4).
        xm = top(c["x2d"], 0.5)
        ax = Axes(draw, (3 * m + s, m), s, (0.0, xm), (0.0, top(np.r_[c["y2d"], mu * xm], 0.05)))
        ax.frame("2 pi n h k", "F / (B k^2)", f"friction, Eq. (4), mu = {mu:g}")
        ax.curve(np.array([0.0, xm]), mu * np.array([0.0, xm]), THEORY)
        ax.points(c["x2d"], c["y2d"], SIM)
        # Fig. 4: n^2 F h^2 / B against n^2 h / e, Eq. (5).
        ax = Axes(draw, (5 * m + 2 * s, m), s, (1e-3, 1e-1), (1e-5, 1e-1), log=True)
        ax.frame("n^2 h / e", "n^2 F h^2 / B", f"traction, Eq. (5), n = {self.turns}")
        ax.curve(*jawed_curve(mu), THEORY)
        ax.points(c["x4"], c["y4"], SIM)
        x0 = m
        for text, color in (("Jawed et al. 2015", THEORY), ("simulation", SIM)):
            draw.rectangle([x0, img.height - 22, x0 + 10, img.height - 12], fill=color)
            draw.text((x0 + 16, img.height - 17), text, fill=TEXT, font=font(13), anchor="lm")
            x0 += 30 + int(draw.textlength(text, font=font(13)))
        return np.asarray(img)

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        assert self.min_gap > 0.8, f"rope passed into itself (min gap {self.min_gap:.2f} diameters)"
        assert self.min_tail > 20, f"the knot reached a clamp ({self.min_tail} segments away)"
        c = self.comparison()
        assert len(c["y4"]) > 50, "the knot did not slide"
        ratio = float(np.nanmedian(c["y4"] / c["jawed"]))
        assert 0.85 < ratio < 1.18, f"traction {ratio:.2f} times Jawed et al. (2015)"


# -- theory and geometry --------------------------------------------------------------------------


@cache
def loop_table() -> tuple[np.ndarray, np.ndarray]:
    """Arc length ``lambda / R`` of the knot's loop against the braid length ``l / R`` (Jawed et al. 2015).

    The loop is a planar elastica leaving the braid with curvature ``1 / R`` and re-entering it a
    distance ``l`` behind after turning by ``2 pi``: ``kappa^2 R^2 = 1 - m sin^2(theta / 2)`` gives
    ``lambda / R = 4 K(m)`` and ``l / R = 8 (K(m) - E(m)) / m - 4 K(m)`` (``m = 0``: a circle).
    """
    m = 1.0 - np.geomspace(1.0, 1e-12, 400)[1:]
    m = np.concatenate(([1e-9], m[m > 1e-9]))
    k, e = ellipk(m), ellipe(m)
    return 8.0 * (k - e) / m - 4.0 * k, 4.0 * k


def jawed_curve(mu: float) -> tuple[np.ndarray, np.ndarray]:
    """``(n^2 h / e, n^2 F h^2 / B)`` along Eq. (5) of Jawed et al. (2015), by ``x = l / R``:
    ``g(x) = x^2 / (x + lambda / R)``, and ``x`` is the argument of ``g``."""
    x, lam = loop_table()
    x, lam = x[x > 1e-3], lam[x > 1e-3]
    c = np.sqrt(3.0) * np.pi**2
    return x**2 / (x + lam) / (8.0 * c), mu * x**3 / (96.0 * c)


def jawed_force(n2h_e, mu: float):
    """``n^2 F h^2 / B`` at ``n^2 h / e`` from Eq. (5) of Jawed et al. (2015) (nan off the table)."""
    s, f = jawed_curve(mu)
    return np.exp(np.interp(np.log(n2h_e), np.log(s), np.log(f), left=np.nan, right=np.nan))


def elastica_loop(chord: float, radius: float, num: int = 4000) -> np.ndarray:
    """``(z, y)`` of the loop of :func:`loop_table` with end curvature ``1 / radius`` from ``(chord / 2, 0)``
    to ``(-chord / 2, 0)``, leaving and arriving along ``+z`` and bulging to ``+y``."""
    m = brentq(lambda m: 8.0 * (ellipk(m) - ellipe(m)) / m - 4.0 * ellipk(m) - chord / radius, 1e-12, 1.0 - 1e-15)
    theta = np.linspace(0.0, 2.0 * np.pi, num)
    ds = radius / np.sqrt(1.0 - m * np.sin(0.5 * theta) ** 2)
    s = np.concatenate(([0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(theta))))
    w = np.gradient(s)
    z, y = np.cumsum(np.cos(theta) * w), np.cumsum(np.sin(theta) * w)
    z, y = z - z[0], y - y[0]
    z = 0.5 * chord + z * chord / -z[-1]  # remove the quadrature's drift from the exact chord
    return np.column_stack((z, y - np.linspace(0.0, y[-1], num)))


def pull_plan(n: int, h: float, eps: float, speed: float) -> tuple[float, float]:
    """How far to pull each end, and for how long: 60% of the loose knot's shortening ``e`` (Eq. 5's
    loop and braid at ``R = h / eps^2``), at ``speed`` per end."""
    x = 2.0 * n * np.pi * 12**0.25 * eps  # l / R, Eq. (3)
    e = h / eps**2 * (x + np.interp(x, *loop_table()))
    return 0.3 * e, 0.3 * e / speed


def long_knot(n: int, h: float, radius: float, clearance: float, tail: float, seg: float) -> np.ndarray:
    """An overhand knot of unknotting number ``n`` along ``z``, after Eq. (2) of Jawed et al. (2015).

    Strand X runs through the braid, ``(a cos phi, a sin phi, z)`` with ``phi = k (z + l / 2)`` and
    ``a = clearance / 2``, turns into the loop (the elastica of end radius ``radius``, in the plane
    ``yz``), comes back through the braid as strand Y, ``-(a cos phi, a sin phi)``, and leaves. ``k``
    is Eq. (3) at ``radius``, ``l = 2 n pi / k``. Corners are smoothed and the curve resampled to ``seg``.
    """
    a = 0.5 * clearance
    k = 1.0 / np.sqrt(np.sqrt(12.0) * h * radius)
    l = 2.0 * n * np.pi / k
    z = np.linspace(-0.5 * l, 0.5 * l, max(int(40 * n * l / seg), 200))
    phi = k * (z + 0.5 * l)
    helix = np.column_stack((a * np.cos(phi), a * np.sin(phi), z))
    loop = elastica_loop(l, radius)
    u = np.linspace(0.0, 1.0, len(loop))
    loop = np.column_stack((a * (1.0 - 2.0 * u * u * (3.0 - 2.0 * u)), loop[:, 1], loop[:, 0]))
    line = np.linspace(0.0, tail, 400)[1:, None] * np.array([0.0, 0.0, 1.0])
    left = np.array([a, 0.0, -0.5 * l]) - line[::-1]
    right = np.array([-a, 0.0, 0.5 * l]) + line
    p = np.vstack((left, helix, loop[1:-1], -helix * np.array([1.0, 1.0, -1.0]), right))
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
    u = np.linspace(0.0, arc[-1], int(round(arc[-1] / seg)) + 1)
    x = np.column_stack([np.interp(u, arc, p[:, c]) for c in range(3)])
    # Round the corners where the pieces meet (the helix leaves at an angle a k); the tails stay straight.
    smooth = gaussian_filter1d(x, 4.0, axis=0, mode="nearest")
    keep = int(0.5 * tail / seg)
    smooth[:keep], smooth[-keep:] = x[:keep], x[-keep:]
    return smooth


def crossings(x: np.ndarray, view) -> list[tuple[float, bool]]:
    """Crossings of the curve ``x`` seen along ``view``, in order along it: ``(arc parameter, over)``,
    every crossing twice. A reduced alternating diagram with ``c`` crossings has crossing number ``c``."""
    view = np.asarray(view, dtype=float) / np.linalg.norm(view)
    e1 = np.cross(view, [1.0, 0.0, 0.0] if abs(view[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(view, e1)
    p, depth = x @ np.column_stack((e1, e2)), x @ view
    i, j = np.triu_indices(len(x) - 1, k=2)
    a, b, c, d = p[i], p[i + 1], p[j], p[j + 1]
    r, s = b - a, d - c
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    ok = np.abs(den) > 1e-15
    qa = c - a
    t = np.where(ok, (qa[:, 0] * s[:, 1] - qa[:, 1] * s[:, 0]) / np.where(ok, den, 1.0), -1.0)
    v = np.where(ok, (qa[:, 0] * r[:, 1] - qa[:, 1] * r[:, 0]) / np.where(ok, den, 1.0), -1.0)
    hit = ok & (t >= 0.0) & (t < 1.0) & (v >= 0.0) & (v < 1.0)
    out = []
    for i, j, t, v in zip(i[hit], j[hit], t[hit], v[hit], strict=True):
        di = depth[i] + t * (depth[i + 1] - depth[i])
        dj = depth[j] + v * (depth[j + 1] - depth[j])
        out += [(i + t, bool(di < dj)), (j + v, bool(dj < di))]  # nearer the viewer is over
    return sorted(out)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--friction", type=float, default=0.119, help="Coulomb coefficient mu")
    parser.add_argument("--turns", type=int, default=3, help="unknotting number n (2n + 1 crossings)")
    parser.add_argument("--eps", type=float, default=0.1, help="sqrt(h / R) of the loose knot")
    parser.add_argument("--seg", type=float, default=2.0, help="segment length, in radii")
    parser.add_argument("--speed", type=float, default=0.03, help="pull speed of each end [m/s]")
    parser.add_argument("--iterations", type=int, default=200, help="maximum ADMM iterations per step")
    parser.add_argument("--plot", default=None, help="save the comparison with theory to this image")
    parser.set_defaults(num_frames=0)
    known = parser.parse_known_args()[0]
    if known.num_frames <= 0:  # the pull and a little
        pull_time = pull_plan(known.turns, Example.radius, known.eps, known.speed)[1]
        parser.set_defaults(num_frames=int(1.05 * pull_time * Example.fps))
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if args.plot:
        Image.fromarray(example.image()).save(args.plot)
        print(f"wrote {args.plot}")
