"""A curved sheet (Newton's 12-triangle ``curvedSurface.usd``) dropped from 10 m onto the ground, where its bending
stiffness holds it from lying flat (Newton's ``example_cloth_bending``): the same mesh, pose, material and friction,
as a midedge shell.

The sheet lands on its edge at 12 m/s and does not bounce (the contact is near perfectly inelastic), but its edge
grips (mu 1.5): the fall turns into rotation about it, and the curved sheet rolls ~4 m over and falls flat, at rest
from ~3.3 s; the same at 4 and 8 times the substeps. Newton's VBD ends the landing at rest.

Shows: Newton's VBD cloth parameters on a coarse mesh (``add_cloth_mesh``), ground contact with friction. Newton's
test (the sheet comes to rest within 5 m of where it fell: here its centroid, the cartwheel takes its far corner
past) and two of its own: no edge stretches by 20% (Newton's spring check, here on the triangle edges) and the
sheet does not pass through itself or the ground.

    uv run examples/newton/cloth_bending.py
    uv run examples/newton/cloth_bending.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import newton.usd
import numpy as np
import warp as wp
from cloth_twist import Crossings
from pxr import Usd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import DiSMechSolver


class Example(CableExample):
    substeps = 10
    # Newton's: density [kg/m^2], tri_ke, tri_ka [N/m], tri_kd, edge_ke, edge_kd; half its self-contact gap (0.15 m)
    # as the contact radius; its friction
    material = dict(density=0.02, tri_ke=5.0e1, tri_ka=5.0e1, tri_kd=5.0e0, edge_ke=1.0e1, edge_kd=1.0e1)
    radius, mu = 0.075, 1.5

    def __init__(self, viewer, args=None):
        stage = Usd.Stage.Open(newton.examples.get_asset("curvedSurface.usd"))
        mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/cloth"))
        builder = newton.ModelBuilder()
        DiSMechSolver.add_cloth_mesh(
            builder, pos=wp.vec3(0.0, 0.0, 10.0), rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 6.0),
            vertices=[wp.vec3(v) for v in mesh.vertices], indices=mesh.indices, particle_radius=self.radius,
            **self.material,
        )
        builder.add_ground_plane()
        model = builder.finalize()
        model.particle_mu = self.mu
        model.shape_material_mu.fill_(self.mu)
        self.start(viewer, model, DiSMechSolver(model))
        f = np.asarray(mesh.indices).reshape(-1, 3)
        self.edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
        x0 = model.particle_q.numpy()
        self.rest = np.linalg.norm(x0[self.edges[:, 0]] - x0[self.edges[:, 1]], axis=1)
        self.crossings = Crossings(self.edges, f, model.device)
        self.cross = 0

    def step(self):
        super().step()
        self.cross = max(self.cross, self.crossings(self.state_0.particle_q))

    def test_final(self):
        x, v = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        assert np.abs(v).max() < 0.5, f"not at rest ({np.abs(v).max():.2f} m/s; Newton's bound: 0.5)"
        assert np.abs(x[:, :2].mean(0)).max() < 5.0, "the sheet went away"
        stretch = np.abs(np.linalg.norm(x[self.edges[:, 0]] - x[self.edges[:, 1]], axis=1) / self.rest - 1.0)
        assert stretch.max() < 0.2, f"an edge stretched {stretch.max():.0%}"
        assert x[:, 2].min() > 0.0, "the sheet went into the ground"
        assert self.cross == 0, f"the sheet passed through itself ({self.cross} crossings)"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
