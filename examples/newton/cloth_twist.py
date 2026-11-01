"""Newton's cloth twist (``python -m newton.examples cloth_twist``) with midedge shells: the same mesh, material,
drive and contact settings, so the two run side by side. A 1.5 x 1 m cloth held along its two short edges, each
spun about the long axis in opposite directions at pi/3 rad/s, wrings into a rope that coils on itself.

Shows: shell self-contact under sustained load (Offset Geometric Contact, vertex-triangle and edge-edge with
friction, as a barrier energy), clamped edges driven through their nodes (:class:`utils.common.Drive`), Newton's
VBD cloth parameters (``tri_ke``, ``tri_ka``, ``tri_kd``, ``edge_ke``, ``edge_kd``) as a midedge shell's
(:func:`~dismech_newton.builder.add_cloth_mesh`). Newton's test (every particle within a box, speeds below
1.5 m/s) and two of its own: no two parts of the sheet pass through each other (no edge crosses a face it does not
touch), and no vertex comes within half a contact thickness of one it did not start near, at any checked frame.
By the drive's symmetry (a half turn about the vertical through the middle swaps the ends) the middle cross-section
does not turn. Many steps of a compressed cloth end on their iteration budget (``solver.counts``): ``A`` is
factorised once a step (``refactor=False``), as accurate there and ~3.5x faster.

    uv run examples/newton/cloth_twist.py
    uv run examples/newton/cloth_twist.py --viewer null --test
"""

import os
import sys
from pathlib import Path

import newton
import newton.examples
import newton.usd
import numpy as np
import warp as wp
import warp.examples
from pxr import Usd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import THEORY, CableExample, Drive, default_frames, inset, inset_scale

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
    duration, substeps = 5.0, 10  # [s]: Newton's 300 frames at 60 fps, 10 substeps
    plot_every, plot_name = 6, "twist"
    cols = 50  # the mesh's vertices along the twist axis (and across it)
    rate, end = np.pi / 3.0, 10.0  # each end's spin [rad/s], until [s]
    # Newton's example: density [kg/m^2], tri_ke, tri_ka [N/m], tri_kd, edge_ke, edge_kd; its self-contact gap
    # (1.5 mm) as the contact thickness, its friction
    material = dict(density=0.2, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=2.0e-4, edge_ke=1.0e-3, edge_kd=1.0e-2)
    radius, mu = 0.75e-3, 0.2
    check_every = 10  # frames between the geometry checks (on the host)

    def mesh(self):
        """Newton's ``square_cloth.usd``: ``cols x cols`` vertices, index ``r cols + c`` (``c`` along its x)."""
        stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/cloth/cloth"))
        return mesh.vertices, mesh.indices

    def __init__(self, viewer, args=None):
        vertices, indices = self.mesh()
        builder = newton.ModelBuilder(gravity=0.0)
        self.ids = DiSMechSolver.add_cloth_mesh(builder, pos=(0.0, 0.0, 0.0), rot=wp.quat_from_axis_angle(
            wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi), scale=0.01, vertices=[wp.vec3(v) for v in vertices],
            indices=indices, particle_radius=self.radius, **self.material)
        n = self.cols
        ends = (self.ids[np.arange(n) * n], self.ids[np.arange(n) * n + n - 1])  # y = -0.75 m and y = +0.75 m
        for k in np.concatenate(ends):
            builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
        model = builder.finalize()
        model.particle_mu = self.mu
        # its compressed steps end on their budget either way: one factorization a step
        self.start(viewer, model, DiSMechSolver(model, refactor=False, iterations=10))
        self.rest = model.particle_q.numpy().astype(np.float64)
        self.ends = np.concatenate(ends)
        self.axis = np.repeat([1.0, -1.0], n)  # about +y at y = -0.75, -y at +0.75 (Newton's rot_axes)
        self.spin = Drive(model, [3 * k + c for k in self.ends for c in range(3)])
        self.drives = (self.spin,)

        f = np.asarray(indices).reshape(-1, 3)
        self.tris = f
        self.edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
        self.crossings = Crossings(self.edges, f, model.device)
        mid = n // 2 - 1, n // 2  # the two columns either side of the middle, top and bottom rows
        self.middle = np.array([[c, (n - 1) * n + c] for c in mid])
        self.t, self.angle, self.mid_angle, self.gap, self.cross = [0.0], [0.0], [0.0], [np.inf], [0]
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0)  # Newton's

    def angle_at(self, t: float) -> float:
        """Each end's angle [rad] about its axis."""
        return self.rate * min(t, self.end)

    def offsets(self, t: float) -> np.ndarray:
        """The ends' displacements from rest at ``t``: each turned about the y axis (its own sense)."""
        a = self.angle_at(t) * self.axis
        x, y, z = self.rest[self.ends].T
        c, s = np.cos(a), np.sin(a)  # about y: x' = c x + s z, z' = -s x + c z
        return np.column_stack([c * x + s * z - x, np.zeros_like(y), -s * x + c * z - z]).ravel()

    def drive(self, t0, t1):
        self.spin.set(self.offsets(t0), self.offsets(t1))

    def step(self):
        super().step()
        if self.frame % self.check_every:
            return
        x = self.state_0.particle_q.numpy().astype(np.float64)
        d = x[self.middle[:, 1]] - x[self.middle[:, 0]]
        self.t.append(self.sim_time)
        self.angle.append(self.angle_at(self.sim_time))
        mid = np.arctan2(d[:, 0], d[:, 2]).mean()  # about y, from the rest direction +z
        self.mid_angle.append(np.unwrap([self.mid_angle[-1], mid])[1])
        self.gap.append(self.min_gap(x))
        self.cross.append(self.crossings(self.state_0.particle_q))

    def min_gap(self, x: np.ndarray) -> float:
        """The closest two vertices that started more than two contact thicknesses apart come, in thicknesses."""
        i, j = cKDTree(x).query_pairs(4.0 * self.radius, output_type="ndarray").T
        apart = np.linalg.norm(self.rest[i] - self.rest[j], axis=1) > 4.0 * self.radius
        return float(np.linalg.norm(x[i[apart]] - x[j[apart]], axis=1).min(initial=np.inf) / (2.0 * self.radius))

    def image(self, size: tuple[int, int] = (800, 400)) -> np.ndarray:
        """Left: the middle cross-section's angle against the ends' (zero by symmetry). Right: the closest gap
        between layers, in contact thicknesses."""
        import matplotlib  # noqa: PLC0415

        pt = inset_scale(size, 2)
        color = matplotlib.colormaps["viridis"](0.5)
        turns = np.array(self.angle) / (2.0 * np.pi)

        def draw(axes):
            left, right = axes
            left.plot(self.t, turns, color="0.6", lw=1.2 * pt, label=r"ends $\pm\phi$")
            left.plot(self.t, -turns, color="0.6", lw=1.2 * pt)
            left.axhline(0.0, color=THEORY, ls="--", lw=1.2 * pt, label="0")
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
        v = self.state_0.particle_qd.numpy().astype(np.float64)
        assert np.isfinite(x).all(), "non-finite positions"
        assert (np.abs(x) <= [0.6, 0.9, 0.6]).all(), "particles outside Newton's box"
        assert np.abs(v).max() < 1.5, f"a particle at {np.abs(v).max():.2f} m/s (Newton's bound: 1.5)"
        assert max(self.cross) == 0, f"the cloth passed through itself ({max(self.cross)} crossings)"
        assert min(self.gap) > 0.5, f"layers {min(self.gap):.2f} contact thicknesses apart"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    default_frames(parser, lambda args: Example.duration)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
