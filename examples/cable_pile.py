"""Newton's ``example_cable_pile`` with the ADMM DER solver.

Ten layers of ten wavy cables, alternating along x and y, drop onto the ground and settle into a pile.

    uv run examples/cable_pile.py
    uv run examples/cable_pile.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
from common import CableExample

from dismech_newton import ADMMDiSMechSolver


class Example(CableExample):
    layers, lanes = 10, 10
    num_elements, segment_length, radius = 40, 0.05, 0.012

    def __init__(self, viewer, args=None):
        length = self.num_elements * self.segment_length
        spacing = max(8.0 * self.radius, 0.15)
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.0
        builder.add_ground_plane()
        s = np.linspace(0.0, 1.0, self.num_elements + 1)
        along = (s - 0.5) * length
        wave = 0.5 * length * 0.05 * np.sin(4.0 * np.pi * s)  # two cycles
        self.rods = []
        for layer in range(self.layers):
            z = 0.3 + layer * 3.0 * self.radius
            for lane in range(self.lanes):
                offset = (lane - (self.lanes - 1) * 0.5) * spacing
                x, y = (along, offset + wave) if layer % 2 == 0 else (offset + wave, along)
                points = np.column_stack((x, y, np.full_like(s, z)))
                self.rods.append(ADMMDiSMechSolver.add_rod(
                    builder, newton.Rod(points, radius=self.radius), stretch_stiffness=5.0e5, bend_stiffness=1.0e2,
                    bend_damping=2.0e1,
                ))
        model = builder.finalize()
        self.start(viewer, model, ADMMDiSMechSolver(model, friction=1.0), self.radius)

    def test_final(self):
        z = self.state_0.particle_q.numpy()[:, 2]
        assert np.isfinite(z).all(), "non-finite positions"
        assert z.min() > -0.5, "cables penetrated the ground"
        assert z.max() < self.layers * 2.0 * self.radius + 0.5, "pile too high"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
