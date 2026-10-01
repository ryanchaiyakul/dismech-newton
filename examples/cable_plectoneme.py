"""Newton's ``example_cable_plectoneme`` with the ADMM DER solver.

A cable hangs between two clamped ends, which are counter-twisted until it buckles out of plane
and folds into a plectoneme held open by self-contact.

    uv run examples/cable_plectoneme.py
    uv run examples/cable_plectoneme.py --viewer null --test
"""

import math

import newton
import newton.examples
import numpy as np
from common import CableExample, Drive, segment_dofs, smoothstep
from newton.examples.cable.example_cable_plectoneme import Example as NewtonExample

from dismech_newton import ADMMDiSMechSolver

N = NewtonExample  # scene constants and geometry


class Example(CableExample):
    def __init__(self, viewer, args=None):
        nodes = N._hanging_arc_nodes()
        seg = float(np.mean(np.linalg.norm(np.diff(nodes, axis=0), axis=1)))
        mid = N.NUM_ELEMENTS // 2
        nodes[mid : mid + 2, 1] += N.SEED_BODY_OFFSET_Y  # break the symmetry, as Newton does
        radius = 0.42 * seg

        builder = newton.ModelBuilder(gravity=N.GRAVITY)
        bodies = ADMMDiSMechSolver.add_rod(
            builder, newton.Rod(nodes, radius=radius), cfg=newton.ModelBuilder.ShapeConfig(gap=0.6 * seg),
            stretch_stiffness=N.STRETCH_STIFFNESS, bend_stiffness=N.BEND_STIFFNESS,
            twist_stiffness=N.TWIST_STIFFNESS, bend_damping=N.BEND_DAMPING, twist_damping=N.TWIST_DAMPING,
        )
        for body in (bodies[0], bodies[-1]):
            ADMMDiSMechSolver.fix_segment(builder, body)
        model = builder.finalize()
        self.rods = [bodies]
        self.start(viewer, model, ADMMDiSMechSolver(model), radius)
        self.twist = Drive(model, [segment_dofs(model, b, twist_only=True)[0] for b in (bodies[0], bodies[-1])])
        self.drives = (self.twist,)

    def drive(self, t0, t1):
        # Counter-twist, split symmetrically between the ends.
        def angle(t):
            return 2.0 * math.pi * N.TWIST_TURNS * smoothstep(t, N.SETTLE_TIME, N.SETTLE_TIME + N.TWIST_TIME)

        self.twist.set(0.5 * angle(t0) * np.array([-1.0, 1.0]), 0.5 * angle(t1) * np.array([-1.0, 1.0]))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        assert np.abs(x).max() < 10.0, "positions blew up"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(N.FPS * N.TOTAL_TIME))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
