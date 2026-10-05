"""Twist a slack clamped rod into a plectoneme, and compare it with Clauvelin et al. (2009).

Shows: a rod built from material constants (``newton.Rod``), a clamped end pushed in and turned
(:class:`utils.common.Drive`), frictionless self-contact, and the hard-core ply of Clauvelin, Audoly &
Neukirch, Biophys. J. 96 (2009): the tails' tension and torque against the ply's angle and radius.
The strands must never pass through each other: the linking number has to follow the end's turns.

Frictionless, the ply slides freely along the axis, so the viewer draws a second rod with a little
friction (``--display-friction``) whose ply stays put; the plot and the test are the frictionless rod's.

    uv run examples/plectoneme.py
    uv run examples/plectoneme.py --viewer null --test
"""

from types import SimpleNamespace

import newton
import newton.examples
import numpy as np
from scipy.spatial import cKDTree
from utils.common import (
    SIM,
    THEORY,
    CableExample,
    Drive,
    default_frames,
    frame_box,
    inset,
    inset_scale,
    segment_dofs,
    smoothstep,
)

from dismech_newton import ADMMDiSMechSolver

A, RHO, E, NU = 1.55e-3, 1200.0, 1.3e6, 0.5  # radius [m], density, Young's modulus, Poisson's ratio
B = E * np.pi * A**4 / 4.0  # bending stiffness EI
C = E / (2.0 * (1.0 + NU)) * np.pi * A**4 / 2.0  # twisting stiffness GJ
EA = E * np.pi * A**2


class Example(CableExample):
    substeps = 16  # at 4, the end loop snaps through its neck within a step under load
    length, slack = 0.5, 0.3
    t_push, t_settle = 2.0, 2.0
    measure_every = plot_every = 15  # frames
    settle = t_push + t_settle  # not rendered to the README's WebP
    min_ply = 0.08  # [m] shorter plies (their end loop dominates) are not compared with the theory

    @classmethod
    def duration(cls, turns: float, twist_rate: float) -> float:
        """Push, settle, then twist."""
        return cls.t_push + cls.t_settle + turns / twist_rate

    def __init__(self, viewer, args=None):
        segments = getattr(args, "segments", 500)
        self.turns = getattr(args, "turns", 20.0)
        self.twist_rate = getattr(args, "twist_rate", 0.25)  # [turn/s]
        damping = getattr(args, "damping", 1.0e-5)
        friction = getattr(args, "friction", 0.0)  # Coulomb mu between the strands (the theory has none)
        display_friction = getattr(args, "display_friction", 0.02)
        display = viewer is not None and not getattr(args, "test", False) and display_friction > 0.0 and not friction

        x = np.linspace(0.0, self.length, segments + 1)
        z = 1.0e-3 * np.sin(np.pi * x / self.length) ** 2  # seeds the Euler buckle
        nodes = np.column_stack((x, np.zeros_like(x), 0.5 + z))
        builder = newton.ModelBuilder(gravity=0.0)
        rod = newton.Rod(nodes, radius=A, youngs_modulus=E, shear_modulus=E / (2.0 * (1.0 + NU)))
        bodies = ADMMDiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=RHO),
                                           bend_damping=damping, twist_damping=damping)
        for body in (bodies[0], bodies[-1]):
            ADMMDiSMechSolver.fix_segment(builder, body)
        model = builder.finalize()
        solver = ADMMDiSMechSolver(model, friction=friction, iterations=200)  # the cap before the default became 50
        self.start(None if display else viewer, model, solver, A)
        self.end = Drive(model, segment_dofs(model, bodies[-1]))  # the last segment's nodes and twist
        self.drives = (self.end,)
        self.t_end = self.duration(self.turns, self.twist_rate)

        self.l0 = model.dismech.edge_length.numpy().astype(np.float64)
        conn = self.solver.triplets.conn.numpy()
        self.trip_e, self.trip_f = conn[:, 0], conn[:, 1]
        self.l_dual = 0.5 * (self.l0[self.trip_e] + self.l0[self.trip_f])
        self.rows = []  # (t, turns, F, M, alpha, R, ply length, Lk)
        self.ply_center = None  # [m] along the clamp axis, from measure()
        self.verbose = True
        self.display = None  # the rod the viewer draws, if not this one
        if display:
            self.display = Example(None, SimpleNamespace(segments=segments, turns=self.turns, twist_rate=self.twist_rate,
                                                         damping=damping, friction=display_friction))
            self.display.verbose = False
            self.viewer = viewer
            viewer.set_model(self.display.model)
        self.view_x = 0.5 * (1.0 - self.slack) * self.length
        self._place_camera()

    def _place_camera(self, track: float = 0.0):
        """From slightly above, 30 cm of the clamp axis and 11 cm either side (the ply turns about it),
        ``track`` of the way toward the ply's centre, never past a clamp."""
        span, half_width = (1.0 - self.slack) * self.length, 0.15
        ply_center = (self.display or self).ply_center
        if ply_center is not None:
            target = np.clip(ply_center, half_width - 0.01, span + 0.01 - half_width)
            self.view_x += track * (target - self.view_x)
        frame_box(self.viewer, self.view_x - half_width, self.view_x + half_width, 0.39, 0.61, pitch=-12.0)

    def phi(self, t):
        """End rotation [rad]: none while pushing in and settling, then ``twist_rate``."""
        return 2.0 * np.pi * min(self.turns, self.twist_rate * max(0.0, t - self.t_push - self.t_settle))

    def drive(self, t0, t1):
        def delta(t):
            dx = -self.slack * self.length * smoothstep(t, 0.0, self.t_push)
            return np.array([dx, 0.0, 0.0, dx, 0.0, 0.0, self.phi(t)])

        self.end.set(delta(t0), delta(t1))

    def step(self):
        super().step()
        if self.display is not None:
            self.display.step()
        if self.frame % self.measure_every == 0:
            F, M, alpha, R, ply = self.measure()
            self.rows.append((self.sim_time, self.phi(self.sim_time) / (2.0 * np.pi), F, M, alpha, R, ply,
                              self.linking_number()))
            if self.verbose and np.isfinite(alpha) and self.frame % (4 * self.measure_every) == 0:
                print(f"t {self.sim_time:5.1f} s  {self.rows[-1][1]:5.2f} turns  F {1e3 * F:6.2f} mN  "
                      f"M {1e6 * M:6.1f} uNm  alpha {np.rad2deg(alpha):5.1f} deg  R/a {R / A:4.2f}  "
                      f"ply {100 * ply:4.1f} cm  F/F_th {F / ply_force(alpha, R):5.2f}  "
                      f"M/M_th {M / ply_moment(alpha, R, F):5.2f}  Lk {self.rows[-1][7]:5.2f}", flush=True)

    def render(self):
        self.log_plot()
        self._place_camera(track=0.03)
        shown = self.display or self
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(shown.state_0)
        self.viewer.log_contacts(shown.contacts, shown.state_0)
        self.viewer.end_frame()

    def image(self, size: tuple[int, int] = (400, 400)) -> np.ndarray:
        """Tension against ply angle beside Eq. 9a, with the linking number, as an RGBA image."""
        rows = np.array(self.rows) if self.rows else np.full((0, 8), np.nan)
        t, turns, F, M, alpha, R, ply, lk = rows.T
        ok = np.isfinite(F) & np.isfinite(alpha) & (ply > self.min_ply)

        def draw(ax):
            a = np.linspace(10.0, 26.0, 100)
            ax.plot(a, ply_force(np.deg2rad(a), A), color=THEORY, label="Clauvelin et al. (2009)")
            ax.plot(np.rad2deg(alpha[ok]), F[ok], "o", color=SIM, label=r"simulation, $\mu = 0$")
            if ok.any():
                k = np.flatnonzero(ok)[-1]
                ax.plot(np.rad2deg(alpha[k]), F[k], "o", color="white", ms=9 * inset_scale(size))
            if len(lk):
                ax.text(0.96, 0.05, rf"$\mathrm{{Lk}} = {lk[-1]:.2f}$" "\n" rf"$n = {turns[-1]:.2f}$",
                        transform=ax.transAxes, ha="right", va="bottom", linespacing=1.4)
            ax.set(xlim=(10.0, 26.0), ylim=(0.0, 0.16))
            ax.set_xlabel(r"ply angle $\alpha$ ($^\circ$)")
            ax.set_ylabel(r"tension $F$ (N)")
            ax.locator_params(nbins=4)
            ax.legend(loc="upper left")

        return inset(draw, size)

    def measure(self):
        """``(F, M, alpha, R, ply length)``, NaNs without a ply.

        ``F`` and ``M`` from the straight tails' axial and twist strains; ``alpha`` from the strands'
        tangents at contact, ``t_i . t_j = -cos(2 alpha)``; ``R`` half the distance between them.
        """
        x = self.state_0.particle_q.numpy().astype(np.float64)
        if not np.isfinite(x).all():
            return (np.nan,) * 5
        seg = x[1:] - x[:-1]
        mid = 0.5 * (x[1:] + x[:-1])
        t = seg / np.linalg.norm(seg, axis=1, keepdims=True)
        n, h = len(seg), self.l0.mean()

        # The ply: each segment's closest partner within 1.2 diameters, more than 12 radii along the rod.
        pairs = cKDTree(mid).query_pairs(2.4 * A, output_type="ndarray")
        pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) * h > 12.0 * A] if len(pairs) else pairs
        in_ply = np.zeros(n, bool)
        alpha = R = np.nan
        if len(pairs) >= 20:
            i, j = np.concatenate((pairs, pairs[:, ::-1])).T
            dist = np.linalg.norm(mid[i] - mid[j], axis=1)
            order = np.lexsort((dist, i))
            first = np.r_[True, i[order][1:] != i[order][:-1]]
            u, v, d = i[order][first], j[order][first], dist[order][first]
            in_ply[u] = True
            self.ply_center = float(np.median(mid[u, 0]))
            lo, hi = np.percentile(u, [10, 90])  # the ply's interior: no end loop, no junction
            core = (u > lo) & (u < hi)
            alpha = 0.5 * np.arccos(np.clip(np.median(-np.sum(t[u[core]] * t[v[core]], axis=1)), -1.0, 1.0))
            R = 0.5 * np.median(d[core])

        # The tails: segments within a diameter of the clamp axis and 20 radii (along the rod) from the ply.
        off_axis = np.linalg.norm(mid[:, 1:] - x[0, 1:], axis=1)
        ply_idx = np.flatnonzero(in_ply)
        gap = np.min(np.abs(np.arange(n)[:, None] - ply_idx[None, :]), axis=1) if len(ply_idx) else np.full(n, n)
        tail = (off_axis < 2.0 * A) & (gap * h > 20.0 * A)
        tail[[0, 1, -2, -1]] = False
        strain = np.linalg.norm(seg, axis=1) / self.l0 - 1.0
        F = EA * np.median(strain[tail]) if tail.sum() > 5 else np.nan
        twist = self.state_0.dismech.triplet_strain_q.numpy()[:, 4].astype(np.float64) / self.l_dual
        tail_t = tail[self.trip_e] & tail[self.trip_f]
        M = C * np.median(twist[tail_t]) if tail_t.sum() > 5 else np.nan
        return F, M, alpha, R, in_ply.sum() * h

    def linking_number(self) -> float:
        """``Tw + Wr`` of the rod closed far along the clamp axis: a strand passage loses 2 at once."""
        x = self.state_0.particle_q.numpy().astype(np.float64)
        if not np.isfinite(x).all():
            return np.nan
        twist = self.state_0.dismech.triplet_strain_q.numpy()[:, 4].astype(np.float64).sum() / (2.0 * np.pi)
        return twist + writhe(closure(x))

    def comparison(self) -> dict[str, np.ndarray]:
        """Per measured frame with a long ply: ``F / F_th``, ``M / M_th`` and ``R / a``."""
        t, n, F, M, alpha, R, ply, _ = np.array(self.rows).T
        ok = np.isfinite(F) & np.isfinite(M) & np.isfinite(alpha) & (ply > self.min_ply)
        F, M, alpha, R = F[ok], M[ok], alpha[ok], R[ok]
        return {"F": F / ply_force(alpha, R), "M": M / ply_moment(alpha, R, F), "R": R / A, "turns": n[ok]}

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        t, turns, *_, lk = np.array(self.rows).T
        slip = np.abs(lk - turns)
        assert not (slip > 0.1).any(), (
            f"strands passed through each other: Lk {lk[slip > 0.1][0]:.2f} at {turns[slip > 0.1][0]:.2f} turns "
            f"(t {t[slip > 0.1][0]:.2f} s)")
        c = self.comparison()
        assert len(c["F"]) > 100, f"no plectoneme ({len(c['F'])} frames with a ply)"
        for key, lo, hi in (("R", 0.97, 1.05), ("F", 0.93, 1.03), ("M", 0.97, 1.03)):
            median = float(np.median(c[key]))
            spread = float(np.percentile(np.abs(c[key] / median - 1.0), 90))
            print(f"{key}: median {median:.3f}, 90% within {100 * spread:.1f}%")
            assert lo < median < hi, f"{key} median {median:.3f} outside ({lo}, {hi})"


# -- theory and topology --------------------------------------------------------------------------


def ply_force(alpha, R):
    """Tension of the hard-core ply, Eq. 9a with ``U = 0``: ``F R^2 / B = sin^3 a (cos a tan 2a - sin a / 2)``."""
    return B / R**2 * np.sin(alpha) ** 3 * (np.cos(alpha) * np.tan(2.0 * alpha) - 0.5 * np.sin(alpha))


def ply_moment(alpha, R, F):
    """Twisting moment of the hard-core ply, Eq. 9c: ``M = 2 / sin 2a (B sin^4 a / (2R) + R F)``."""
    return 2.0 / np.sin(2.0 * alpha) * (B * np.sin(alpha) ** 4 / (2.0 * R) + R * F)


def writhe(x: np.ndarray) -> float:
    """Writhe of the closed polygon ``x`` (first point repeated last; Klenin & Langowski 2000, exact)."""
    p1, p2 = x[:-1], x[1:]
    n = len(p1)
    i, j = np.triu_indices(n, 2)
    keep = ~((i == 0) & (j == n - 1))  # the closing pair is adjacent
    i, j = i[keep], j[keep]
    a1, a2, b1, b2 = p1[i], p2[i], p1[j], p2[j]
    r13, r14, r23, r24 = b1 - a1, b2 - a1, b1 - a2, b2 - a2

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)

    def angle(u, v):
        return np.arcsin(np.clip(np.sum(u * v, axis=1), -1.0, 1.0))

    n1, n2, n3, n4 = (unit(np.cross(u, v)) for u, v in ((r13, r14), (r14, r24), (r24, r23), (r23, r13)))
    omega = angle(n1, n2) + angle(n2, n3) + angle(n3, n4) + angle(n4, n1)
    sign = np.sign(np.sum(np.cross(b2 - b1, a2 - a1) * r13, axis=1))
    return float(np.sum(omega * sign) / (2.0 * np.pi))


def closure(x: np.ndarray, far: float = 200.0) -> np.ndarray:
    """``x`` closed far away: out along the clamp axis at both ends, over and back."""
    axis = (x[-1] - x[0]) / np.linalg.norm(x[-1] - x[0])
    up = np.array([0.0, 0.0, 1.0])
    a, b = x[0] - far * axis, x[-1] + far * axis
    return np.vstack([x, b, b + far * up, a + far * up, a, x[:1]])


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--segments", type=int, default=500, help="rod segments")
    parser.add_argument("--turns", type=float, default=20.0, help="end rotation [turns]")
    parser.add_argument("--twist-rate", type=float, default=0.25, help="end rotation rate [turn/s]")
    parser.add_argument("--damping", type=float, default=1.0e-5, help="bend and twist strain-rate damping")
    parser.add_argument("--display-friction", type=float, default=0.02,
                        help="friction of the rod the viewer draws (0: the frictionless rod)")
    default_frames(parser, lambda a: Example.duration(a.turns, a.twist_rate))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
