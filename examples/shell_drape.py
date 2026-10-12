"""A sheet falls onto a sphere and drapes over it, its skirt folding onto the ground.

Shows: a midedge shell on Newton's cloth mesh (``add_cloth_mesh``: Newton's cloth parameters mapped to a plate),
Offset Geometric Contact with shapes and with itself, on the unified strain-space ADMM solver.

    uv run examples/shell_drape.py
    uv run examples/shell_drape.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from utils.common import CableExample, default_frames

from dismech_newton.unified import DiSMechSolver


def sheet(n: int, size: float):
    """An ``n x n``-node square in the xy plane centred on the origin, faces counter-clockwise, diagonals
    alternating (no preferred fold direction)."""
    s = np.linspace(-0.5 * size, 0.5 * size, n)
    X, Y = np.meshgrid(s, s, indexing="ij")
    v = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            f += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return v, np.array(f, dtype=np.int32)


class Example(CableExample):
    substeps = 4
    nodes, size, height = 41, 0.8, 0.45  # the sheet: 41 x 41 nodes over 0.8 m (2 cm apart), dropped from 0.45 m
    sphere = 0.15  # radius, resting on the ground
    radius = 4.0e-3  # the vertices' contact thickness
    # A rubber sheet (E = 5 MPa, h = 1 mm, nu = 0.3) in Newton's cloth parameters: the 2D Lame mu = E h / (2 (1 + nu))
    # and lam = E h nu / (1 - nu^2) [N/m]; edge_ke ~ D / (0.9 spacing) for the plate's D = E h^3 / (12 (1 - nu^2))
    # = 4.6e-4 N m (add_cloth_mesh maps D = edge_ke A_e / L_e, and A_e / L_e averages 0.9 spacing on this mesh).
    tri_ke, tri_ka, edge_ke = 1.92e3, 1.65e3, 2.6e-2
    density = 0.3  # [kg/m^2]
    damping = 2.0e-3  # strain-rate damping as a time [s], both forms

    def __init__(self, viewer, args=None):
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        builder.add_ground_plane(cfg=cfg)
        builder.add_shape_sphere(-1, xform=wp.transform((0.0, 0.0, self.sphere), wp.quat_identity()),
                                 radius=self.sphere, cfg=cfg)
        v, f = sheet(self.nodes, self.size)
        # Slightly off-centre and turned, so the folds are not symmetric.
        turn = np.deg2rad(20.0)
        rot = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]])
        v = v @ rot.T + [0.03, -0.02, self.height]
        self.sheet = DiSMechSolver.add_cloth_mesh(
            builder, vertices=v.tolist(), indices=f.ravel().tolist(), density=self.density, tri_ke=self.tri_ke,
            tri_ka=self.tri_ka, edge_ke=self.edge_ke, particle_radius=self.radius,
            damping=self.damping,
        )
        model = builder.finalize()
        model.particle_mu = 0.5
        solver = DiSMechSolver(model, iterations=10)
        self.start(viewer, model, solver, self.radius)
        if viewer is not None and hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(1.0, -1.0, 0.75), pitch=-25.0, yaw=135.0)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        x = self.state_0.particle_q.numpy()[self.sheet]
        assert np.isfinite(x).all(), "non-finite positions"
        centre = np.array([0.0, 0.0, self.sphere])
        assert (np.linalg.norm(x - centre, axis=1) > self.sphere + 0.5 * self.radius).all(), \
            "the sheet went into the sphere"
        assert x[:, 2].min() > 0.5 * self.radius, "the sheet went through the ground"
        top = np.linalg.norm(x[:, :2], axis=1) < 0.04
        assert top.sum() >= 4 and (x[top, 2] < 2.0 * self.sphere + 0.03).all(), "the sheet is not on the sphere"
        assert (x[:, 2] < 2.0 * self.sphere + 0.03).all(), "the sheet did not settle"
        assert (x[:, 2] < 1.5 * self.radius).sum() > 0.05 * len(x), "the skirt does not reach the ground"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    default_frames(parser, lambda args: 3.0)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
