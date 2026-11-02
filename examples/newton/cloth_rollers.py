"""A long cloth rolled on one roller is wound onto a second (Newton's ``example_cloth_rollers``): the same rolled
mesh (300 rows of 15 around roller 1, 0.4 between turns, then a straight run to roller 2), material and speeds, as
a midedge shell. The cloth's end row is carried round roller 2; both rollers turn at the same surface speed. No
gravity; Newton's units (its viewer scales them by 0.01).

Newton's rollers are cloth meshes whose fixed vertices it turns each substep; here they are kinematic cylinders
(Newton shapes) whose turning reaches the cloth by friction: each substep the example writes their angular
velocity into ``state.body_qd`` (:class:`Spin`). Roller 1 is 9.8 in radius (Newton's: 9.9), as the barrier
contact must not start inside its contact thickness (0.3, Newton's self-contact gap): the innermost turn sits at 10.
Newton's straight run is aimed at a mirrored point (its rotation flips the target's height), then its end row is
moved down onto roller 2; here it is aimed at roller 2, and the roll is 810 long (Newton's: 800) so the run leaves
the roll's outer turn instead of cutting through the turns below it.

Shows: kinematic shapes in motion against a shell, friction from a moving surface, self-contact in a tight roll.
Newton's test (the cloth's centroid has moved towards roller 2, its extent stays bounded) and one of its own: no
edge passes through a face.

    uv run examples/newton/cloth_rollers.py
    uv run examples/newton/cloth_rollers.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from cloth_twist import Crossings
from newton.examples.cloth.example_cloth_rollers import rolled_cloth_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample, Drive, frame_box, smoothstep

from dismech_newton import SolverDiSMech


class Spin:
    """The rollers' velocities (angular, about +y) written into ``state.body_qd`` each substep (a drive)."""

    def __init__(self, model, rate: np.ndarray):
        self.rate = rate
        self.qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)

    def set(self, scale: float) -> None:
        self.qd.assign(np.array([[0.0, 0.0, 0.0, 0.0, scale * w, 0.0] for w in self.rate], np.float32))

    def __call__(self, state, s):
        wp.copy(state.body_qd, self.qd)


class Example(CableExample):
    substeps = 10
    nv, rows, length, thickness = 15, 300, 810.0, 0.4  # the roll: vertices across, rows, length, between turns
    roller = ((-27.2, 7.4, 9.8), (40.0, 0.0, 14.9))  # (x, z, radius)
    y0, rate, spin_time = 50.0, 2.0 * np.pi, 20.0  # rollers' middle; roller 1's turn rate [rad/s], until [s]
    ramp = 1.0  # [s] to full speed (Newton's start at it)
    material = dict(density=0.02, tri_ke=1.0e5, tri_ka=1.0e5, tri_kd=1.0, edge_ke=1.0e2, edge_kd=1.0e1)
    radius, mu = 0.15, 0.1

    def __init__(self, viewer, args=None):
        (x1, z1, _), (x2, z2, r2) = self.roller
        end_x = x2 - r2 - 1.2 * 0.4  # Newton's: its end row 0.48 off roller 2
        verts, faces, _, ext = rolled_cloth_mesh(length=self.length, nu=self.rows, thickness=self.thickness,
                                                 target_x=end_x - x1, target_y=z1 - z2, extension_segments=20)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        ids = SolverDiSMech.add_cloth_mesh(
            builder, pos=wp.vec3(x1, self.y0, z1), rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi),
            vertices=[wp.vec3(v) for v in verts], indices=faces.ravel(), particle_radius=self.radius, **self.material,
        )
        self.seam = ids[:self.nv]  # the first row (the roll's core), turned with roller 1
        self.end = ids[(self.rows + ext - 1) * self.nv:]  # the last row, carried round roller 2
        for k in [*self.seam, *self.end]:
            builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
        axis_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)  # a cylinder's axis is its z
        for x, z, r in self.roller:
            body = builder.add_body(xform=wp.transform(wp.vec3(x, self.y0, z), axis_y), is_kinematic=True)
            builder.add_shape_cylinder(body, radius=r, half_height=60.0)
        model = builder.finalize()
        model.particle_mu = self.mu
        model.shape_material_mu.fill_(self.mu)
        self.start(viewer, model, SolverDiSMech(model, refactor=True))
        self.rest = model.particle_q.numpy().astype(np.float64)
        speed = self.rate * 9.9  # Newton's surface speed: roller 1's turn rate times its radius
        self.spin = Spin(model, np.array([speed / 9.9, speed / r2]))
        self.carry = Drive(model, [3 * k + c for k in [*self.seam, *self.end] for c in range(3)])
        self.drives = (self.spin, self.carry)
        self.crossings = Crossings(
            np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0),
            faces, model.device)
        self.cross = 0
        frame_box(viewer, -45.0, 60.0, -25.0, 25.0, y=self.y0 - 60.0)

    def carried(self, t: float) -> np.ndarray:
        """The seam's and end row's displacements from rest: turned about their rollers' axes (+y) with them."""
        out = []
        for rows, (cx, cz, _), rate in zip((self.seam, self.end), self.roller, self.spin.rate):
            a = rate * self.turned(t)
            x, z = self.rest[rows, 0] - cx, self.rest[rows, 2] - cz
            c, s = np.cos(a), np.sin(a)
            out.append(np.column_stack([c * x + s * z - x, np.zeros_like(x), -s * x + c * z - z]).ravel())
        return np.concatenate(out)

    def speed(self, t: float) -> float:
        """The rollers' speed over full speed: up a smoothstep over ``ramp``, none after ``spin_time``."""
        return float(smoothstep(t, 0.0, self.ramp)) if t < self.spin_time else 0.0

    def turned(self, t: float) -> float:
        """The integral of :meth:`speed` [s]."""
        t = min(t, self.spin_time)
        s = min(t / self.ramp, 1.0)
        return self.ramp * (s**3 - 0.5 * s**4) + max(t - self.ramp, 0.0)

    def drive(self, t0, t1):
        self.spin.set(self.speed(t0))
        self.carry.set(self.carried(t0), self.carried(t1))

    def step(self):
        super().step()
        self.cross = max(self.cross, self.crossings(self.state_0.particle_q))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        shift = x[:, 0].mean() - self.rest[:, 0].mean()
        assert shift > 5.0, f"the cloth's centroid moved {shift:.1f} towards roller 2 (Newton's: at least 5)"
        extent = np.linalg.norm(x.max(0) - x.min(0))
        assert extent < 150.0, f"the cloth spread to {extent:.0f}"
        assert self.cross == 0, f"the cloth passed through itself ({self.cross} crossings)"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(Example.spin_time * CableExample.fps))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
