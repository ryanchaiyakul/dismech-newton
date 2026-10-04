"""Pull a loose overhand knot tight, and compare its traction with Audoly et al. (2007).

Shows: self-contact with friction, both end segments clamped (``fix_segment``) and pulled apart
(:class:`utils.common.Drive`), and a measured force checked against theory: Audoly, Clauvelin &
Neukirch, PRL 99, 164301 (2007), ``F h^2 / B = eps^4 / 2 + mu sigma eps^3``, ``eps = sqrt(h / R)``.

    uv run examples/overhand_knot.py
    uv run examples/overhand_knot.py --viewer null --test
    uv run examples/overhand_knot.py --viewer null --scale 0.1 --num-frames 480 --plot knot.png
"""

import newton
import newton.examples
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from utils.common import MUTED, SIM, THEORY, CableExample, Drive, close_pairs, inset, segment_dofs, smoothstep

from dismech_newton import ADMMDiSMechSolver

SIGMA_TREFOIL = 0.492  # Audoly et al. (2007), the trefoil's friction constant


class Example(CableExample):
    radius, youngs_modulus = 0.005, 1.0e7
    contact_tol = 1.05  # segments closer than this many diameters touch
    exclude = 6  # segments this close along the rope never count as touching
    plot_every = 10  # frames between plot updates
    settle = 2.5  # seconds the loose knot relaxes before the pull, its velocities damped (about critically)

    def __init__(self, viewer, args=None):
        self.friction = getattr(args, "friction", 0.05)
        self.scale = getattr(args, "scale", 0.3)  # larger: a looser start, closer to the theory's limit
        h = self.seg = self.radius
        # Shastri's long trefoil, its tails turned onto one line, no two parts closer than 1.6 diameters.
        points = side_on(inflate(long_trefoil(scale=self.scale, tail=0.25, seg=h), h, 3.2 * h))
        # Pull each end this far (to eps of about 0.7 at scale 0.3), slowly: near equilibrium.
        self.pull, self.pull_time = 1.3 * (1.25 * self.scale - 0.04), 52.0 * self.scale
        self.build(viewer, points)

    def build(self, viewer, points: np.ndarray, **solver_options):
        """The rope along ``points``, both end segments clamped, to be pulled apart along the line through them."""
        h = self.radius
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        rod = newton.Rod(points, radius=h, youngs_modulus=self.youngs_modulus, poissons_ratio=0.3)
        bodies = ADMMDiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
                                           bend_damping=0.1)
        for body in (bodies[0], bodies[-1]):
            ADMMDiSMechSolver.fix_segment(builder, body)
        model = builder.finalize()
        solver = ADMMDiSMechSolver(model, friction=self.friction, **solver_options)
        solver.triplets.rest.zero_()  # a rope: straight and untwisted at rest, not knotted
        self.start(viewer, model, solver, h, contact_matching_pos_threshold=h)
        # Both clamped segments' nodes (not their twist), moved along the end-to-end line.
        self.ends = Drive(model, segment_dofs(model, bodies[0])[:6] + segment_dofs(model, bodies[-1])[:6])
        self.drives = (self.ends,)
        axis = (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0])
        self.direction = np.concatenate([-axis, -axis, axis, axis])
        self.rest = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.stretch = self.youngs_modulus * np.pi * h**2  # EA
        self.bend = self.youngs_modulus * np.pi * h**4 / 4.0  # B = EI
        self.history = []  # per frame: (t, F, e, R, d), e the shortening, d the end-to-end distance
        self.min_gap = np.inf  # the closest approach, in diameters: below 1 the rope passes into itself

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

    def gaps(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment pairs ``(i, j)`` near each other, and their gaps in diameters."""
        return close_pairs(x, self.radius, self.exclude, self.seg + 3.0 * self.radius)

    def record(self):
        """Every frame: the traction ``F`` (axial strain of the tails), ``e``, ``R`` (where the rope leaves
        the braid) and ``d``."""
        x = self.state_0.particle_q.numpy().astype(np.float64)
        edge = np.diff(x, axis=0)
        length = np.linalg.norm(edge, axis=1)
        d = np.linalg.norm(x[-1] - x[0])
        e = length.sum() - d
        i, j, gap = self.gaps(x)
        self.min_gap = min(self.min_gap, gap.min(initial=np.inf))
        close = gap < self.contact_tol
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

    def image(self, size: tuple[int, int] = (400, 400)) -> np.ndarray:
        """``F h^2 / B`` against Audoly et al. (2007), as an RGBA image (see :func:`utils.common.inset`)."""
        c, mu = self.comparison(), self.friction
        eps_max = max(0.8, 1.05 * c["eps"].max()) if len(c["eps"]) else 0.8  # the default pull reaches 0.72

        def draw(ax):
            eps = np.linspace(0.0, eps_max, 200)
            ax.plot(eps, audoly_force(eps, mu), color=THEORY, label="Audoly et al. (2007)")
            ax.plot(eps, audoly_force(eps, 0.0), color=MUTED, ls="--", label=r"frictionless, $\mu = 0$")
            ax.plot(c["eps"], c["Fbar"], "o", color=SIM, label="simulation")
            ax.set(xlim=(0.0, eps_max), ylim=(0.0, float(audoly_force(eps_max, mu))))
            ax.set_xlabel(r"$\sqrt{h / R}$")
            ax.set_ylabel(r"$F h^2 / B$")
            ax.locator_params(nbins=4)
            ax.legend(loc="upper left")

        return inset(draw, size)

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        assert self.min_gap > 0.8, f"rope passed into itself (min gap {self.min_gap:.2f} diameters)"
        c = self.comparison()
        assert len(c["Fbar"]) > 50, "the knot did not slide"
        ratio = float(np.median(c["Fbar"] / c["audoly"]))
        assert 0.75 < ratio < 1.33, f"traction {ratio:.2f} times Audoly et al. (2007)"


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
