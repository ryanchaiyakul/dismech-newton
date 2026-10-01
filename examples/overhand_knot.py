"""Pull a loose overhand knot tight with the ADMM DER solver, and compare its traction with theory.

The rope starts as Shastri's long trefoil with tails turned onto one line, inflated so no two parts
are closer than 1.6 diameters. Both end segments are clamped and pulled apart along that line;
self-contact with friction holds the knot as it tightens, and the rope must never pass through itself.

Every frame records the traction ``F`` (from the axial strain of the tails), the end-to-end shortening
``e`` and ``R`` (the peak curvature in the knot, where the rope leaves the braid), and compares ``F`` with

* Audoly, Clauvelin & Neukirch, *Elastic knots*, PRL 99, 164301 (2007), for weak friction:
  ``F h^2 / B = eps^4 / 2 + mu sigma eps^3`` with ``eps = sqrt(h / R)`` and ``sigma = 0.492`` (trefoil);
* Jawed, Dieleman, Audoly & Reis, *Untangling the mechanics and topology in the frictional response of
  long overhand elastic knots*, PRL 115, 118302 (2015), Eq. (5), friction-dominated:
  ``n^2 h / e = g([96 sqrt(3) pi^2 / mu * n^2 F h^2 / B]^(1/3)) / (8 sqrt(3) pi^2)``,

with ``h`` the rod radius, ``B = E pi h^4 / 4``, ``R`` the radius of curvature where the rope leaves the
braid for the loop and ``n = 1`` the unknotting number of a trefoil.

    uv run examples/overhand_knot.py
    uv run examples/overhand_knot.py --viewer null --test --friction 0.1
    uv run --with matplotlib examples/overhand_knot.py --viewer null --plot knot.png
    uv run --with matplotlib examples/overhand_knot.py --viewer null --scale 0.3 --num-frames 720 --plot knot.png

``--scale`` sets the loose knot's size; a larger knot starts at a smaller ``eps`` (0.22 at 0.3, against
0.41 at the default 0.1), where the asymptotic theories hold better, and takes longer to pull tight.
"""

from functools import cache

import newton
import newton.examples
import numpy as np
from common import CableExample, Drive, segment_distance, segment_dofs, smoothstep
from scipy.special import ellipe, ellipk

from dismech_newton import ADMMDiSMechSolver

SIGMA_TREFOIL = 0.492  # Audoly et al. (2007), the trefoil's friction constant


# -- theory ------------------------------------------------------------------------------------------------


@cache
def loop_table() -> tuple[np.ndarray, np.ndarray]:
    """Arc length ``lambda / R`` of the knot's loop against the braid length ``l / R`` (Jawed et al. 2015).

    The loop is a planar elastica leaving one end of the straight braid tangentially with curvature
    ``1 / R`` and entering the other end, a distance ``l`` behind, after turning by ``2 pi``; a force
    along the braid acts at its ends. Its first integral, ``kappa^2 R^2 = 1 - m sin^2(theta / 2)``,
    gives ``lambda / R = 4 K(m)`` and ``l / R = 8 (K(m) - E(m)) / m - 4 K(m)``: ``m = 0`` is the circle,
    ``m -> 1`` an infinitely long braid.
    """
    m = 1.0 - np.geomspace(1.0, 1e-12, 400)[1:]
    m = np.concatenate(([1e-9], m[m > 1e-9]))
    k, e = ellipk(m), ellipe(m)
    return 8.0 * (k - e) / m - 4.0 * k, 4.0 * k


def audoly_force(eps, mu: float, sigma: float = SIGMA_TREFOIL):
    """``F h^2 / B`` of a trefoil at ``eps = sqrt(h / R)`` (Audoly et al. 2007)."""
    eps = np.asarray(eps)
    return 0.5 * eps**4 + mu * sigma * eps**3


def jawed_curve(mu: float) -> tuple[np.ndarray, np.ndarray]:
    """``(n^2 h / e, n^2 F h^2 / B)`` along Eq. (5) of Jawed et al. (2015).

    Parametrised by ``x = l / R``: ``g(x) = l^2 / (e R) = x^2 / (x + lambda / R)`` and the argument of
    ``g`` in Eq. (5) is ``x``.
    """
    x, lam = loop_table()
    x, lam = x[x > 1e-3], lam[x > 1e-3]
    c = np.sqrt(3.0) * np.pi**2
    return x**2 / (x + lam) / (8.0 * c), mu * x**3 / (96.0 * c)


def jawed_force(n2h_e, mu: float):
    """``n^2 F h^2 / B`` at ``n^2 h / e`` from Eq. (5) of Jawed et al. (2015) (nan off the table)."""
    s, f = jawed_curve(mu)
    return np.exp(np.interp(np.log(n2h_e), np.log(s), np.log(f), left=np.nan, right=np.nan))


# -- geometry ----------------------------------------------------------------------------------------------


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


class Example(CableExample):
    radius, youngs_modulus = 0.005, 1.0e7
    contact_tol = 1.05  # segments closer than this many diameters touch
    exclude = 6  # segments this close along the rope never count as touching

    def __init__(self, viewer, args=None):
        self.friction = getattr(args, "friction", 0.05)
        scale = getattr(args, "scale", 0.1)
        # Pull each end this far (to e of about 30 h at the default scale, 50 h at 0.3), slowly enough
        # that the knot stays near equilibrium.
        self.pull, self.pull_time = 1.25 * scale - 0.04, 40.0 * scale
        points = inflate(long_trefoil(scale=scale, tail=0.25, seg=self.radius), self.radius, 3.2 * self.radius)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        bodies = ADMMDiSMechSolver.add_rod(
            builder, newton.Rod(points, radius=self.radius, youngs_modulus=self.youngs_modulus, poissons_ratio=0.3),
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0), bend_damping=0.1,
        )
        for body in (bodies[0], bodies[-1]):
            ADMMDiSMechSolver.fix_segment(builder, body)
        model = builder.finalize()
        self.rods = [bodies]
        solver = ADMMDiSMechSolver(model, friction=self.friction)
        solver.triplets.rest.zero_()  # a rope: straight and untwisted at rest, not knotted
        self.start(viewer, model, solver, self.radius, contact_matching_pos_threshold=self.radius)

        # Pull both clamped end segments apart along the line through them.
        axis = (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0])
        self.ends = Drive(model, segment_dofs(model, bodies[0])[:6] + segment_dofs(model, bodies[-1])[:6])
        self.drives = (self.ends,)
        self.direction = np.concatenate([-axis, -axis, axis, axis])
        self.min_gap = np.inf

        h = self.radius
        self.rest = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.stretch = self.youngs_modulus * np.pi * h**2  # EA
        self.bend = self.youngs_modulus * np.pi * h**4 / 4.0  # B = EI
        self.pairs = np.triu_indices(len(points) - 1, k=self.exclude)
        self.history = []  # (t, F, e, R) per frame

    def drive(self, t0, t1):
        def pull(t):
            return self.pull * smoothstep(t, 0.0, self.pull_time) * self.direction

        self.ends.set(pull(t0), pull(t1))

    def step(self):
        super().step()
        self.measure()

    def gaps(self, x: np.ndarray) -> np.ndarray:
        i, j = self.pairs
        return segment_distance(x[i], x[i + 1], x[j], x[j + 1]) / (2.0 * self.radius)

    def measure(self):
        """Record the traction ``F``, the end-to-end shortening ``e`` and the loop radius ``R``."""
        x = self.state_0.particle_q.numpy().astype(np.float64)
        edge = np.diff(x, axis=0)
        length = np.linalg.norm(edge, axis=1)
        e = length.sum() - np.linalg.norm(x[-1] - x[0])
        i, j = self.pairs
        close = self.gaps(x) < self.contact_tol
        hit = np.flatnonzero(np.bincount(np.r_[i[close], j[close]], minlength=len(length)))
        if len(hit) == 0:  # no knot
            self.history.append((self.sim_time, np.nan, e, np.nan))
            return
        first, last = hit[0], hit[-1]
        # R: the curvature peaks where the rope leaves the braid for the loop (the braid's helices
        # bend less, 1 / (sqrt(12) R)).
        t = edge / length[:, None]
        kappa = np.linalg.norm(np.cross(t[:-1], t[1:]), axis=1) / (0.5 * (length[:-1] + length[1:]))
        radius = 1.0 / kappa[first:last].max()
        # F: axial strain of the free tails, away from the clamps and the knot.
        tails = np.r_[2:max(first - 4, 2), min(last + 5, len(length) - 2):len(length) - 2]
        force = self.stretch * np.mean(length[tails] / self.rest[tails] - 1.0) if len(tails) else np.nan
        self.history.append((self.sim_time, force, e, radius))

    def comparison(self) -> dict[str, np.ndarray]:
        """Simulated and theoretical traction ``F h^2 / B`` per frame while the knot slides tight
        (the theories assume sliding friction; it is indeterminate once the ends stop)."""
        t, force, e, radius = np.array(self.history).T
        keep = (t > 0.2 * self.pull_time) & (t <= self.pull_time) & np.isfinite(force)
        t, force, e, radius = t[keep], force[keep], e[keep], radius[keep]
        h, mu = self.radius, self.friction
        eps = np.sqrt(h / radius)
        return {
            "t": t, "e": e, "R": radius, "eps": eps, "n2h_e": h / e, "Fbar": force * h**2 / self.bend,
            "audoly": audoly_force(eps, mu), "jawed": jawed_force(h / e, mu),
        }

    def report(self, plot: str | None = None, every: int = 15):
        c = self.comparison()
        print(f"mu = {self.friction}, h = {self.radius}, B = {self.bend:.3e}, n = 1")
        print(f"{'t':>6} {'e/h':>6} {'R/h':>5} {'eps':>6} {'F h^2/B':>9} {'Audoly07':>9} {'Jawed15':>9}")
        for k in range(0, len(c["t"]), every):
            print(f"{c['t'][k]:6.2f} {c['e'][k] / self.radius:6.1f} {c['R'][k] / self.radius:5.2f} {c['eps'][k]:6.3f} "
                  f"{c['Fbar'][k]:9.2e} {c['audoly'][k]:9.2e} {c['jawed'][k]:9.2e}")
        if plot:
            self.plot(c, plot)

    def plot(self, c: dict[str, np.ndarray], path: str):
        import matplotlib.pyplot as plt  # noqa: PLC0415 (optional: uv run --with matplotlib)

        mu = self.friction
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        eps = np.linspace(0.0, 1.1 * c["eps"].max(), 100)
        ax0.plot(c["eps"], c["Fbar"], ".", ms=3, label="simulation")
        ax0.plot(eps, audoly_force(eps, mu), "--k", label=rf"Audoly et al. 2007, $\mu={mu}$")
        ax0.plot(eps, audoly_force(eps, 0.0), ":k", label=r"frictionless, $\epsilon^4/2$")
        ax0.set(xlabel=r"$\epsilon = \sqrt{h/R}$", ylabel="$F h^2 / B$", title="trefoil")
        s, f = jawed_curve(mu)
        ax1.loglog(c["n2h_e"], c["Fbar"], ".", ms=3, label="simulation")
        ax1.loglog(s, f, "--k", label=rf"Jawed et al. 2015, Eq. (5), $\mu={mu}$")
        ax1.set(xlabel="$n^2 h / e$", ylabel="$n^2 F h^2 / B$", title="overhand knot, $n = 1$")
        for ax in (ax0, ax1):
            ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")

    def test_post_step(self):
        self.min_gap = min(self.min_gap, self.gaps(self.state_0.particle_q.numpy()).min())

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        assert self.min_gap > 0.8, f"rope passed into itself (min gap {self.min_gap:.2f} diameters)"
        c = self.comparison()
        ratio = float(np.median(c["Fbar"] / c["audoly"]))
        assert 0.6 < ratio < 1.5, f"traction {ratio:.2f} times Audoly et al. (2007)"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--friction", type=float, default=0.05, help="Coulomb coefficient mu")
    parser.add_argument("--scale", type=float, default=0.1, help="size of the loose knot")
    parser.add_argument("--plot", default=None, help="save the comparison with theory to this image")
    parser.set_defaults(num_frames=420)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.report(args.plot)
