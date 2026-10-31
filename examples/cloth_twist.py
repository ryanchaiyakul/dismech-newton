"""The cloth twisting stress test: a square of cloth held along two opposite edges, one edge spun about the axis
between them, turn after turn, so the cloth wrings into a rope that coils on itself; then held.

Shows: shell self-contact under sustained load (Offset Geometric Contact, vertex-triangle and edge-edge with
friction, as a barrier energy: ``contact_stiffness``), clamped edges driven through their nodes
(:class:`utils.common.Drive`). The test asks only that nothing goes wrong: no two parts of the sheet pass through
each other (no edge crosses a face it does not touch) and no vertex comes within half a contact thickness of one it
did not start near (the barrier's steep part), at any frame. The membrane must stretch (to ~3x at the free edges, as with Newton's VBD): the
layers cannot pass, and the ends stay apart.

Before the layers touch, the twist is a helicoid: by the clamps' symmetry (a half turn about the vertical through
the middle swaps them) the middle cross-section turns by exactly half the driven angle.

    uv run examples/cloth_twist.py
    uv run examples/cloth_twist.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from utils.common import THEORY, CableExample, Drive, default_frames, grid, inset, inset_scale

from dismech_newton import DiSMechSolver


@wp.kernel
def _boxes_kernel(x: wp.array[wp.vec3], tris: wp.array[wp.vec3i], lo: wp.array[wp.vec3], hi: wp.array[wp.vec3]):
    t = wp.tid()
    a, b, c = x[tris[t][0]], x[tris[t][1]], x[tris[t][2]]
    lo[t] = wp.min(wp.min(a, b), c)
    hi[t] = wp.max(wp.max(a, b), c)


@wp.kernel
def _crossings_kernel(bvh: wp.uint64, x: wp.array[wp.vec3], edges: wp.array[wp.vec2i], tris: wp.array[wp.vec3i],
                      count: wp.array[wp.int32], hits: wp.array[wp.vec2i]):
    """Per edge: the faces it shares no vertex with that it passes through (its ends on either side of the face's
    plane, where it meets the plane inside the face)."""
    e = wp.tid()
    i, j = edges[e][0], edges[e][1]
    a, b = x[i], x[j]
    query = wp.bvh_query_aabb(bvh, wp.min(a, b), wp.max(a, b))
    t = int(0)
    while wp.bvh_query_next(query, t):
        f = tris[t]
        if f[0] == i or f[1] == i or f[2] == i or f[0] == j or f[1] == j or f[2] == j:
            continue
        v0, v1, v2 = x[f[0]], x[f[1]], x[f[2]]
        n = wp.cross(v1 - v0, v2 - v0)
        da, db = wp.dot(a - v0, n), wp.dot(b - v0, n)
        if da * db >= 0.0:
            continue
        y = a + da / (da - db) * (b - a)
        if (wp.dot(wp.cross(v1 - v0, y - v0), n) >= 0.0 and wp.dot(wp.cross(v2 - v1, y - v1), n) >= 0.0
                and wp.dot(wp.cross(v0 - v2, y - v2), n) >= 0.0):
            k = wp.atomic_add(count, 0, 1)
            if k < hits.shape[0]:
                hits[k] = wp.vec2i(e, t)


class Crossings:
    """How many mesh edges pass through a face they share no vertex with, on the device; ``hits``: the first
    (edge, face) pairs found."""

    def __init__(self, edges: np.ndarray, tris: np.ndarray, device):
        self.edges = wp.array(edges.astype(np.int32), dtype=wp.vec2i, device=device)
        self.tris = wp.array(tris.astype(np.int32), dtype=wp.vec3i, device=device)
        self.lo, self.hi = (wp.zeros(len(tris), dtype=wp.vec3, device=device) for _ in range(2))
        self.bvh = None
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.hits = wp.zeros(64, dtype=wp.vec2i, device=device)

    def __call__(self, x: wp.array) -> int:
        wp.launch(_boxes_kernel, dim=self.tris.shape[0], inputs=[x, self.tris, self.lo, self.hi])
        if self.bvh is None:
            self.bvh = wp.Bvh(self.lo, self.hi)
        else:
            self.bvh.refit()
        self.count.zero_()
        wp.launch(_crossings_kernel, dim=self.edges.shape[0],
                  inputs=[self.bvh.id, x, self.edges, self.tris, self.count, self.hits])
        return int(self.count.numpy()[0])


class Example(CableExample):
    duration, substeps = 9.0, 4  # [s]
    plot_every, plot_name = 6, "twist"
    size, spacing = 0.3, 0.006  # [m]: a 51 x 51 node square
    turns, rate, ramp = 4, 0.75, 0.5  # driven turns, at [turn/s], eased in and out over [s]
    pull = 0.4  # the spun edge moves in by this share of the size over the turns, as hands wringing a towel do
    radius = 2.0e-3  # the vertices' contact thickness: layers 4 mm apart
    density, tri_ke, bend_stiffness, damping = 0.2, 1.0e4, 1.0e-4, 0.05  # [kg/m^2], [N/m], D [N m], [s]
    mu, contact_stiffness = 0.3, 1.0e3  # the barrier's, in units of a pair's inertia
    check_every = 10  # frames between the geometry checks (on the host)

    def __init__(self, viewer, args=None):
        builder = newton.ModelBuilder(gravity=0.0)
        self.n = int(round(self.size / self.spacing)) + 1
        v, f = grid(self.n, self.n, self.size, self.size)
        v[:, 1] -= 0.5 * self.size  # the twist axis: the x axis, through the middle
        v[:, 2] += 0.5 * self.size
        self.ids = DiSMechSolver.add_cloth_mesh(builder, vertices=v.tolist(), indices=f.ravel().tolist(),
                                                density=self.density, tri_ke=self.tri_ke, tri_ka=0.0, edge_ke=0.0,
                                                bend_stiffness=self.bend_stiffness, damping=self.damping,
                                                particle_radius=self.radius)
        held, spun = v[:, 0] < 1.5 * self.spacing, v[:, 0] > self.size - 1.5 * self.spacing  # two lattice lines each
        for k in self.ids[held | spun]:
            builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
            builder.particle_mass[k] = 0.0
        model = builder.finalize()
        model.particle_mu = self.mu
        self.start(viewer, model, DiSMechSolver(model, contact_stiffness=self.contact_stiffness))
        self.spun = self.ids[spun]
        self.spin = Drive(model, [3 * k + c for k in self.spun for c in range(3)])
        self.drives = (self.spin,)

        self.tris, self.rest = f, v
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        self.edges = np.unique(e, axis=0)
        self.crossings = Crossings(self.edges, f, model.device)
        self.middle = (self.n // 2) * self.n + np.array([0, self.n - 1])  # the middle cross-section's ends
        self.t, self.angle, self.mid_angle, self.gap, self.cross = [0.0], [0.0], [0.0], [np.inf], [0]
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.5 * self.size, -0.6, 0.5 * self.size), pitch=0.0, yaw=90.0)

    def angle_at(self, t: float) -> float:
        """The spun edge's angle [rad]: ``turns`` turns at ``rate``, eased in and out."""
        def eased(t0):  # the integral of smoothstep(t, t0, t0 + ramp)
            u = np.clip((t - t0) / self.ramp, 0.0, 1.0)
            return self.ramp * (u**3 - 0.5 * u**4) + max(t - t0 - self.ramp, 0.0)

        return 2.0 * np.pi * self.rate * (eased(0.0) - eased(self.turns / self.rate))

    def offsets(self, t: float) -> np.ndarray:
        """The spun edge's displacement from rest at ``t``: turned about the axis, moved in along it."""
        a = self.angle_at(t)
        y, z = self.rest[self.spun, 1], self.rest[self.spun, 2] - 0.5 * self.size
        c, s = np.cos(a), np.sin(a)
        dx = np.full(len(y), -self.pull * self.size * a / (2.0 * np.pi * self.turns))
        return np.column_stack([dx, c * y - s * z - y, s * y + c * z - z]).ravel()

    def drive(self, t0, t1):
        self.spin.set(self.offsets(t0), self.offsets(t1))

    def step(self):
        super().step()
        if self.frame % self.check_every:
            return
        x = self.state_0.particle_q.numpy().astype(np.float64)
        d = x[self.middle[1]] - x[self.middle[0]]
        self.t.append(self.sim_time)
        self.angle.append(self.angle_at(self.sim_time))
        mid = np.arctan2(d[2], d[1])
        self.mid_angle.append(np.unwrap([self.mid_angle[-1], mid])[1])
        self.gap.append(self.min_gap(x))
        self.cross.append(self.crossings(self.state_0.particle_q))

    def min_gap(self, x: np.ndarray) -> float:
        """The closest two vertices that started more than two contact thicknesses apart come, in thicknesses."""
        i, j = cKDTree(x).query_pairs(4.0 * self.radius, output_type="ndarray").T
        apart = np.linalg.norm(self.rest[i] - self.rest[j], axis=1) > 4.0 * self.radius
        return float(np.linalg.norm(x[i[apart]] - x[j[apart]], axis=1).min(initial=np.inf) / (2.0 * self.radius))

    def image(self, size: tuple[int, int] = (800, 400)) -> np.ndarray:
        """Left: the middle cross-section's angle against half the driven one. Right: the closest gap between
        layers, in contact thicknesses."""
        import matplotlib  # noqa: PLC0415

        pt = inset_scale(size, 2)
        color = matplotlib.colormaps["viridis"](0.5)
        turns = np.array(self.angle) / (2.0 * np.pi)

        def draw(axes):
            left, right = axes
            left.plot(self.t, 0.5 * turns, color=THEORY, ls="--", lw=1.2 * pt, label=r"$\phi / 2$")
            left.plot(self.t, np.array(self.mid_angle) / (2.0 * np.pi), color=color, lw=1.6 * pt, label="middle")
            left.set(xlim=(0.0, self.duration))
            left.set_xlabel(r"time $t$ (s)")
            left.set_ylabel(r"turns")
            left.legend(loc="upper left")
            right.axhline(1.0, color=THEORY, ls="--", lw=1.2 * pt, label="contact")
            right.plot(self.t, np.minimum(self.gap, 5.0), color=color, lw=1.6 * pt)
            right.set(xlim=(0.0, self.duration), ylim=(0.0, 5.0))
            right.set_xlabel(r"time $t$ (s)")
            right.set_ylabel(r"closest gap / $2r$")
            right.legend(loc="lower left")

        return inset(draw, size, ncols=2)

    def test_final(self):
        x = self.state_0.particle_q.numpy().astype(np.float64)
        assert np.isfinite(x).all(), "non-finite positions"
        assert max(self.cross) == 0, f"the cloth passed through itself ({max(self.cross)} crossings)"
        assert min(self.gap) > 0.5, f"layers {min(self.gap):.2f} contact thicknesses apart"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    default_frames(parser, lambda args: Example.duration)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
