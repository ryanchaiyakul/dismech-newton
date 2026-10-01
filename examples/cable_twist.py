"""Newton's ``example_cable_twist`` with the ADMM DER solver.

Three zigzag cables lie on the ground with isotropic bend/twist stiffness 1e2, 1e3 and 1e4; the first
segment of each spins about its axis and the twist propagates through the 90-degree turns.

    uv run examples/cable_twist.py
    uv run examples/cable_twist.py --viewer null --test
"""

import newton
import newton.examples
import numpy as np
import warp as wp
from common import CableExample, Drive, segment_dofs
from newton.examples.cable.example_cable_twist import Example as NewtonExample

from dismech_newton import ADMMDiSMechSolver


class Example(CableExample):
    num_elements, segment_length, radius = 64, 0.1, 0.02
    stiffness = (1.0e2, 1.0e3, 1.0e4)
    spin_rate = 0.5  # rad/s

    def __init__(self, viewer, args=None):
        length = self.num_elements * self.segment_length
        builder = newton.ModelBuilder()
        self.rods = []
        for i, k in enumerate(self.stiffness):
            y = (i - (len(self.stiffness) - 1) / 2.0) * 3.0
            rod = NewtonExample.create_cable_geometry_with_turns(
                None, pos=wp.vec3(-0.25 * length, y, self.radius), num_elements=self.num_elements,
                length=length, radius=self.radius,
            )
            bodies = ADMMDiSMechSolver.add_rod(
                builder, rod, stretch_stiffness=1.0e6, bend_stiffness=k, twist_stiffness=k,
                bend_damping=1.0e-2 * k, twist_damping=1.0e-2 * k,
            )
            ADMMDiSMechSolver.fix_segment(builder, bodies[0])
            self.rods.append(bodies)
        builder.add_ground_plane()
        model = builder.finalize()
        self.start(viewer, model, ADMMDiSMechSolver(model, friction=0.5), self.radius)
        self.spin = Drive(model, [d for bodies in self.rods for d in segment_dofs(model, bodies[0], True)])
        self.drives = (self.spin,)

    def drive(self, t0, t1):
        self.spin.set(self.spin_rate * t0, self.spin_rate * t1)

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        stretch = np.linalg.norm(np.diff(x.reshape(len(self.rods), -1, 3), axis=1), axis=2) / self.segment_length
        assert np.abs(stretch - 1.0).max() < 0.1, "cable stretched by more than 10%"
        assert x[:, 2].min() > -0.5, "cable penetrated the ground"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
