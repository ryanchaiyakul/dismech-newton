"""A 6.4 x 3.2 m cloth held along one short edge swings down from horizontal and piles onto the ground below
(Newton's ``example_cloth_hanging``, its default ``--solver vbd``): the same grid, material and contact settings, as
a midedge shell.

Newton's grid gives every vertex the same ``mass``; a midedge shell takes a ``density`` per area, here
``mass / cell area``, so its boundary vertices are lighter (lumped by area). Its contact thickness is Newton's
``particle_radius``.

Shows: Newton's ``add_cloth_grid`` (its vertices, faces and fixed edge) as a shell, a cloth falling onto itself and
the ground. Newton's test (above the ground, inside its box) and one of its own: no edge passes through a face.

    uv run examples/newton/cloth_hanging.py
    uv run examples/newton/cloth_hanging.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from cloth_twist import Crossings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import DiSMechSolver


def newton_grid(**grid):
    """Vertices (world), faces and fixed vertices of Newton's ``add_cloth_grid``."""
    b = newton.ModelBuilder()
    b.add_cloth_grid(**grid)
    fixed = (np.asarray(b.particle_flags) & int(newton.ParticleFlags.ACTIVE)) == 0
    return np.asarray(b.particle_q), np.asarray(b.tri_indices).reshape(-1, 3), np.flatnonzero(fixed)


class Example(CableExample):
    substeps = 10
    width, height, cell, mass, radius, mu = 64, 32, 0.1, 0.1, 0.05, 1.0

    def __init__(self, viewer, args=None):
        self.width = getattr(args, "width", self.width)
        self.height = getattr(args, "height", self.height)
        vertices, faces, fixed = newton_grid(
            pos=wp.vec3(0.0, 0.0, 4.0), rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            vel=wp.vec3(0.0), dim_x=self.width, dim_y=self.height, cell_x=self.cell, cell_y=self.cell, mass=self.mass,
            fix_left=True,
        )
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        ids = DiSMechSolver.add_cloth_mesh(
            builder, vertices=[wp.vec3(v) for v in vertices], indices=faces.ravel(), particle_radius=self.radius,
            density=self.mass / self.cell**2, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e2, edge_ke=1.0e1, edge_kd=0.0,
        )
        for k in ids[fixed]:
            builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
        model = builder.finalize()
        model.particle_mu = self.mu
        model.shape_material_mu.fill_(self.mu)
        self.start(viewer, model, DiSMechSolver(model))
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0)
        self.crossings = Crossings(edges, faces, model.device)
        self.cross = 0

    def step(self):
        super().step()
        self.cross = max(self.cross, self.crossings(self.state_0.particle_q))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        assert x[:, 2].min() > 0.0, "the cloth went into the ground"
        lower, upper = np.array([-0.11 * self.width, -4.0, -1.8]), np.array([0.1, 7.0, 4.0])
        assert ((x >= lower) & (x <= upper)).all(), "particles outside Newton's box"
        assert self.cross == 0, f"the cloth passed through itself ({self.cross} crossings)"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
