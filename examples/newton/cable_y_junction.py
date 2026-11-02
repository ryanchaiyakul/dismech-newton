"""Three cables joined at one node, a Y held by one tip segment (Newton's ``example_cable_y_junction``).

Shows: a rod graph (``newton.Rod(points, edges=...)``) in one ``add_rod``: the junction node is shared, and triplets
couple its first segment (the held branch's) to the other two, the arms' segments reversed. The Y is stiff
(``EI = bend_stiffness * h = 150 N m^2``): it droops as a frame, a few mm, and the droop is checked against beam
theory (+10% at 20 segments: the discretization's first-order excess, as in ``tests/test_rod.py``'s cantilever).

Two substeps a frame, not Newton's 20: with Newton's bending damping (absolute, ``1e3 N m s``: 0.2 s per
``bend_stiffness``) the Y creeps overdamped, ~1e-7 m per 1/1200 s step, the float32 spacing at ``z = 1.25 m``, so
at Newton's step its positions never change.

    uv run examples/newton/cable_y_junction.py
    uv run examples/newton/cable_y_junction.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import SolverDiSMech


class Example(CableExample):
    substeps = 2
    radius, num_segments, segment_length, z0 = 0.01, 20, 0.03, 1.25
    stretch_stiffness, bend_stiffness, bend_damping = 1.0e7, 5.0e3, 1.0e3
    dirs = np.array([[1.0, 0.0, 0.0], [-0.5, 0.75**0.5, 0.0], [-0.5, -(0.75**0.5), 0.0]])

    def __init__(self, viewer, args=None):
        n, h = self.num_segments, self.segment_length
        junction = np.array([0.0, 0.0, self.z0])
        points, edges = [junction], []
        for d in self.dirs:  # each branch from the junction outward (Newton's edge order)
            prev = 0
            for i in range(1, n + 1):
                points.append(junction + d * i * h)
                edges.append((prev, len(points) - 1))
                prev = len(points) - 1
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.add_ground_plane()
        self.bodies, _ = SolverDiSMech.add_rod(
            builder, rod=newton.Rod(np.array(points), edges=edges, radius=self.radius),
            stretch_stiffness=self.stretch_stiffness, bend_stiffness=self.bend_stiffness,
            bend_damping=self.bend_damping, label="y_graph",
        )
        self.pinned = self.bodies[n - 1]  # the first branch's tip segment
        SolverDiSMech.fix_segment(builder, self.pinned)
        model = builder.finalize()
        model.shape_material_mu.fill_(1.0)
        self.start(viewer, model, SolverDiSMech(model))
        self.x0 = self.state_0.particle_q.numpy().copy()
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(2.10, 0.0, self.z0 - 0.15), pitch=0.0, yaw=-180.0)

    def theory(self):
        """Static droop of the junction and the arm tips: the stem a cantilever from its clamp (the held segment's
        inner node) under its weight ``w``, the arms' weight ``P = 2 w L`` at the junction and their moment
        ``P L / 4`` (their centroid ``L / 4`` beyond it); an arm a cantilever under ``w`` from the junction, pitched
        by half the stem's end slope (its share about the arm's own bending axis)."""
        n, h = self.num_segments, self.segment_length
        w = self.model.particle_mass.numpy()[1:].sum() / (3 * n * h) * 9.81  # weight per length, as lumped
        EI, a, L = self.bend_stiffness * h, (n - 1) * h, n * h
        P, M = 2.0 * w * L, 0.5 * w * L**2
        drop = w * a**4 / (8 * EI) + P * a**3 / (3 * EI) + M * a**2 / (2 * EI)
        slope = w * a**3 / (6 * EI) + P * a**2 / (2 * EI) + M * a / EI
        return drop, drop + 0.5 * L * slope + w * L**4 / (8 * EI)

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "NaN/Inf in cable positions"
        assert np.abs(x[:, :2]).max() < 2.0, "cable drifted too far in X/Y"
        assert -0.2 < x[:, 2].min() and x[:, 2].max() < 3.0, "cable out of Z bounds"
        q = self.state_0.body_q.numpy()[self.pinned]
        assert np.abs(q[:3] - self.model.body_q.numpy()[self.pinned][:3]).max() < 1.0e-4, "pinned tip moved"
        n = self.num_segments
        drop = self.x0[[0, 2 * n, 3 * n], 2] - x[[0, 2 * n, 3 * n], 2]
        junction, tip = self.theory()
        print(f"droop: junction {drop[0]:.3e} (theory {junction:.3e}), arm tips {drop[1]:.3e} {drop[2]:.3e} "
              f"(theory {tip:.3e})")
        assert 1.0 < drop[0] / junction < 1.15, "junction droop off beam theory"
        assert abs(drop[1] - drop[2]) < 0.01 * drop[1], "arms not symmetric"
        assert 1.0 < drop[1] / tip < 1.15, "arm droop off beam theory"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=120)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
