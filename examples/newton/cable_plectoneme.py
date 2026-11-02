"""A hanging cable, its ends turned in opposite directions, coils into a plectoneme (Newton's
``example_cable_plectoneme``): the same arc, stiffness, damping and drive. It settles for 2 s, is twisted 6 turns
over 8 s and held for 3 s.

Shows: both end segments clamped (``fix_segment``) and driven through their twist (:class:`utils.common.Drive`),
self-contact under gravity. Newton's test (finite, bounded positions and speeds) and two of its own: the strands
never pass into each other, and the twist has gone into writhe (the cable touches itself).

    uv run examples/newton/cable_plectoneme.py
    uv run examples/newton/cable_plectoneme.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.examples.cable.example_cable_plectoneme import Example as NewtonExample

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample, Drive, close_pairs, segment_dofs, smoothstep

from dismech_newton import SolverDiSMech

N = NewtonExample


class Example(CableExample):
    def __init__(self, viewer, args=None):
        nodes = N._hanging_arc_nodes()
        self.segment_length = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).mean())
        self.radius = 0.42 * self.segment_length
        builder = newton.ModelBuilder(gravity=N.GRAVITY)
        bodies, _ = SolverDiSMech.add_rod(
            builder, rod=newton.Rod(nodes, radius=self.radius), stretch_stiffness=N.STRETCH_STIFFNESS,
            bend_stiffness=N.BEND_STIFFNESS, bend_damping=N.BEND_DAMPING, twist_stiffness=N.TWIST_STIFFNESS,
            twist_damping=N.TWIST_DAMPING,
        )
        for body in (bodies[0], bodies[-1]):
            SolverDiSMech.fix_segment(builder, body)
        model = builder.finalize()
        model.shape_material_mu.fill_(0.0)  # Newton's default rod-rod contact: no friction set
        self.start(viewer, model, SolverDiSMech(model))
        x = self.state_0.particle_q.numpy()
        x[N.NUM_ELEMENTS // 2, 1] += N.SEED_BODY_OFFSET_Y  # Newton's seed: the middle off the plane
        self.state_0.particle_q.assign(x)
        # Root turned by -angle/2, tip by +angle/2, each about its own axis.
        self.spin = Drive(model, segment_dofs(model, bodies[0], True) + segment_dofs(model, bodies[-1], True))
        self.drives = (self.spin,)
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(pos=wp.vec3(0.0, -4.0, 1.30), pitch=1.4, yaw=90.0)

    @staticmethod
    def angle(t: float) -> float:
        return 2.0 * np.pi * N.TWIST_TURNS * smoothstep(t, N.SETTLE_TIME, N.SETTLE_TIME + N.TWIST_TIME)

    def drive(self, t0, t1):
        self.spin.set(*(0.5 * self.angle(t) * np.array([-1.0, 1.0]) for t in (t0, t1)))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        v = self.state_0.particle_qd.numpy()
        assert np.isfinite(x).all() and np.isfinite(v).all(), "non-finite state"
        assert np.abs(x).max() < 10.0, "positions blew up"
        assert np.abs(v).max() < 1.0e3, "velocities blew up"
        _, _, gap = close_pairs(x, self.radius, exclude=4, reach=4.0 * self.radius)
        assert gap.min() > 0.9, f"strands pass into each other (gap {gap.min():.2f} diameters)"
        assert gap.min() < 1.1, "no plectoneme: the cable does not touch itself"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(N.FPS * N.TOTAL_TIME))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
