"""A bundle of seven cables on the ground, pushed sideways by four upright capsules and released (Newton's
``example_cable_bundle_hysteresis``): the same bundle, stiffness, obstacles and timing. The capsules sweep across in
1 s, hold for 1 s and are taken away.

Each cable has Newton's Dahl friction, a bending and twist hysteresis inside the cable (``eps_max`` 2 rad, ``tau``
0.1 rad: ``add_rod``'s ``dahl_eps_max``, ``dahl_tau``), and Coulomb friction (mu 1) between the cables and on the
ground. ``--no-dahl``: friction alone holds the bend.

Shows: kinematic bodies moved by the example, their pose and velocity written into the state every substep
(:class:`Obstacles`): the solver sweeps a shape over the step with its velocity. Newton's tests: after the release the
cables keep their length (within 20%); with Dahl they keep a bend (end-to-end distance over arc length between 0.5 and
0.9), without it they spring back (above 0.9).

    uv run examples/newton/cable_bundle_hysteresis.py
    uv run examples/newton/cable_bundle_hysteresis.py --viewer null --test
    uv run examples/newton/cable_bundle_hysteresis.py --no-dahl
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.examples.cable.example_cable_bundle_hysteresis import Example as NewtonExample

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import SolverDiSMech


@wp.func
def _sweep_y(t: float, y0: float, period: float, stop: float) -> float:
    """Newton's triangle wave, ``y0`` at ``t = 0``, frozen from ``stop``."""
    c = wp.min(t, stop) / period + 0.5
    frac = c - wp.floor(c)
    return (2.0 * (1.0 - wp.abs(2.0 * frac - 1.0)) - 1.0) * y0


@wp.kernel
def _obstacles_kernel(bodies: wp.array[int], x0: wp.array[wp.vec3], frame: wp.array[float], s: float, h: float,
                      period: float, stop: float, release: float, body_q: wp.array[wp.transform],
                      body_qd: wp.array[wp.spatial_vector]):
    i = wp.tid()
    t1 = frame[0] + s * (frame[1] - frame[0])  # the substep's end
    t0 = t1 - h
    p = x0[i]
    y0, y1 = _sweep_y(t0, p[1], period, stop), _sweep_y(t1, p[1], period, stop)
    lift = wp.where(t0 >= release, 10.0, 0.0)  # released: out of the way (a jump, not a sweep)
    body_q[bodies[i]] = wp.transform(wp.vec3(p[0], y0, p[2] + lift), wp.quat_identity())
    body_qd[bodies[i]] = wp.spatial_vector(wp.vec3(0.0, (y1 - y0) / h, 0.0), wp.vec3(0.0))


class Obstacles:
    """The capsules' pose at the substep's start and their velocity over it, written into ``state`` (a drive)."""

    period = 2.0
    stop, release = 0.5 * period, period

    def __init__(self, example, bodies, x0):
        dev = example.model.device
        self.bodies = wp.array(bodies, dtype=int, device=dev)
        self.x0 = wp.array(x0, dtype=wp.vec3, device=dev)
        self.frame = wp.zeros(2, dtype=float, device=dev)
        self.h = example.sim_dt

    def set(self, t0, t1):
        self.frame.assign(np.array([t0, t1], dtype=np.float32))

    def __call__(self, state, s):
        wp.launch(_obstacles_kernel, dim=self.bodies.shape[0],
                  inputs=[self.bodies, self.x0, self.frame, s, self.h, self.period, self.stop, self.release],
                  outputs=[state.body_q, state.body_qd])


class Example(CableExample):
    substeps = 10
    num_cables, num_elements, length, radius = 7, 40, 4.0, 0.02

    def __init__(self, viewer, args=None):
        self.dahl = not getattr(args, "no_dahl", False)
        dahl = dict(dahl_eps_max=getattr(args, "eps_max", 2.0), dahl_tau=getattr(args, "tau", 0.1)) if self.dahl else {}
        builder = newton.ModelBuilder()
        offsets = NewtonExample.bundle_start_offsets_yz(None, self.num_cables, self.radius, 1.1)
        self.rods = []
        for dy, dz in offsets:
            rod = newton.Rod.create_straight(start=wp.vec3(-0.5, dy, 0.3 + dz), direction=wp.vec3(1.0, 0.0, 0.0),
                                             length=self.length, segment_count=self.num_elements, radius=self.radius)
            self.rods.append(SolverDiSMech.add_rod(builder, rod=rod, bend_stiffness=1.0e2, bend_damping=5.0,
                                                     **dahl)[0])
        obstacles, x0 = [], []
        radius, height = 0.05, 0.8
        for i, x in enumerate(np.linspace(0.5, 2.5, 4)):
            x0.append((x, 0.3 if i % 2 == 0 else -0.3, 0.5 * height - 0.1))
            body = builder.add_body(xform=wp.transform(wp.vec3(*x0[-1]), wp.quat_identity()), is_kinematic=True)
            builder.add_shape_capsule(body, radius=radius, half_height=0.5 * height - radius)
            obstacles.append(body)
        builder.add_ground_plane()
        model = builder.finalize()
        mu = np.ones(model.shape_count, dtype=np.float32)  # cables and ground: Coulomb mu 1
        mu[np.isin(model.shape_body.numpy(), obstacles)] = 0.0  # Newton's frictionless obstacles (a pair: the mean)
        model.shape_material_mu.assign(mu)
        self.start(viewer, model, SolverDiSMech(model))
        self.obstacles = Obstacles(self, obstacles, x0)
        self.drives = (self.obstacles,)
        self.rest = self.state_0.particle_q.numpy().reshape(self.num_cables, -1, 3)

    def drive(self, t0, t1):
        self.obstacles.set(t0, t1)

    def shape(self):
        """Per cable: arc length over rest, end-to-end distance over arc length."""
        x = self.state_0.particle_q.numpy().reshape(self.num_cables, -1, 3)
        arc = np.linalg.norm(np.diff(x, axis=1), axis=2).sum(1)
        rest = np.linalg.norm(np.diff(self.rest, axis=1), axis=2).sum(1)
        return arc / rest, np.linalg.norm(x[:, -1] - x[:, 0], axis=1) / arc

    def test_final(self):
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "non-finite positions"
        if self.sim_time < Obstacles.release + 0.4:
            return
        ratio, straight = self.shape()
        assert 0.8 < ratio.min() and ratio.max() < 1.2, f"cables changed length: {ratio.round(3)}"
        if self.dahl:
            assert 0.5 < straight.min() and straight.max() < 0.9, f"cables lost their bend: {straight.round(3)}"
        else:
            assert straight.min() > 0.9, f"cables did not spring back: {straight.round(3)}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--no-dahl", action="store_true", help="no Dahl hysteresis: friction alone")
    parser.add_argument("--eps-max", type=float, default=2.0, help="Dahl's largest plastic strain [rad]")
    parser.add_argument("--tau", type=float, default=0.1, help="Dahl's memory length [rad]")
    parser.set_defaults(num_frames=int(3.0 * CableExample.fps))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
