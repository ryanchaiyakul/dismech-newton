"""Heavy spinning bodies shot onto a cloth trampoline (Newton's ``vbd_dat_rigid_soft``): the same sheets, bodies,
launch velocities, material and timing. A sheet pinned along its four edges holds a box, a sphere and a lying
capsule; a smaller free sheet falls over them, and a 10 kg sphere and two capsules (6 and 4 kg) are thrown onto the
stack, spinning. The bodies are MuJoCo's (free joints, its body-body contacts), proxies in our solve through
Newton's ``SolverCoupledProxy``; both sheets are midedge shells in place of Newton's VBD cloth (``mass`` per vertex
as ``density`` per area), their contact with the bodies and each other our Offset Geometric Contact, which keeps
them outside without Newton's DAT truncation.

Shows: free rigid bodies caught by cloth through two-way coupling, cloth-cloth contact between two sheets. Newton's
test (no vertex inside a body, by its SDF, every body held above the pinned sheet's 1 m drop) and two of our own: no
point inside a triangle (28 per triangle) inside a body either (a small capsule or a box's edge meets a sheet between
its vertices), and no cloth edge passes through a face.

    uv run --extra mujoco examples/newton/cloth_rigid_trampoline.py
    uv run --extra mujoco examples/newton/cloth_rigid_trampoline.py --viewer null --test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import newton
import newton.examples
import numpy as np
import warp as wp
from cloth_hanging import newton_grid
from cloth_twist import Crossings
from newton.examples.multiphysics.example_vbd_dat_rigid_soft import PARAMS
from newton.examples.multiphysics.example_vbd_dat_rigid_soft import Example as Newton
from newton.solvers import SolverMuJoCo
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import SolverDiSMech


class Example(CableExample):
    fps, substeps = PARAMS["fps"], PARAMS["sim_substeps"]
    iterations, proxy = 2, {}  # proxy passes: one lags the impact a step and throws the bodies back at twice the speed
    _extent_below_center = Newton._extent_below_center
    _cloth_penetration = Newton._cloth_penetration

    def __init__(self, viewer, args=None):
        self.params = p = PARAMS
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        faces = []
        for size, res, mass, z, pinned in ((p["cloth_size"], p["cloth_res"], p["cloth_mass"], p["cloth_z"], True),
                                           (p["top_cloth_size"], p["top_cloth_res"], p["top_cloth_mass"],
                                            p["cloth_z"] + p["top_cloth_height"], False)):
            cell = size / res
            x, tris, fixed, _ = newton_grid(pos=wp.vec3(-0.5 * size, -0.5 * size, z), rot=wp.quat_identity(),
                                            vel=wp.vec3(0.0), dim_x=res, dim_y=res, cell_x=cell, cell_y=cell,
                                            mass=1.0, fix_left=pinned, fix_right=pinned, fix_top=pinned,
                                            fix_bottom=pinned)
            ids = SolverDiSMech.add_cloth_mesh(builder, vertices=[wp.vec3(v) for v in x], indices=tris.ravel(),
                                               particle_radius=p["particle_radius"],
                                               density=mass / (res + 1) ** 2 / cell**2, tri_ke=p["cloth_tri_ke"],
                                               tri_ka=p["cloth_tri_ke"], tri_kd=p["cloth_tri_kd"],
                                               edge_ke=p["cloth_edge_ke"])
            for k in ids[fixed]:
                builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
            faces.append(ids[tris])
        Newton._build_bodies(self, builder)  # Newton's bodies: shapes, masses, inertias, start poses
        model = builder.finalize()
        model.particle_mu = p["soft_contact_mu"]
        model.shape_material_mu.fill_(p["soft_contact_mu"])
        solver = SolverCoupledProxy(model, entries=[
            SolverCoupled.Entry(name="mjc", solver=SolverMuJoCo, bodies=self._bodies,
                                joints=list(range(model.joint_count))),
            SolverCoupled.Entry(name="soft", solver=SolverDiSMech, particles=list(range(model.particle_count))),
        ], coupling=SolverCoupledProxy.Config(iterations=self.iterations, proxies=[
            SolverCoupledProxy.Proxy(source="mjc", destination="soft", bodies=self._bodies, **self.proxy)]))
        self.start(viewer, model, solver)
        qd = self.state_0.body_qd.numpy()  # Newton's launch velocities, as joint velocities too (MuJoCo's)
        for b, spec in zip(self._bodies, p["bodies"], strict=True):
            qd[b] = (*spec[3], *spec[4])
        for s in (self.state_0, self.state_1):
            s.body_qd.assign(qd)
            newton.eval_ik(model, s, s.joint_q, s.joint_qd)
        faces = np.concatenate(faces)
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0)
        self.crossings = Crossings(edges, faces, model.device)
        self.faces = faces
        u = np.linspace(0.0, 1.0, 7)  # barycentric samples of each triangle
        self.samples = np.array([(a, b, 1.0 - a - b) for a in u for b in u if a + b <= 1.0 + 1.0e-9])
        self.cross, self.max_penetration, self.face_penetration = 0, 0.0, 0.0
        if viewer is not None and hasattr(viewer, "set_camera"):
            o = p["camera_offset"]
            viewer.set_camera(wp.vec3(o[0], o[1], p["cloth_z"] + o[2]), p["camera_pitch"], p["camera_yaw"])

    def step(self):
        super().step()
        self.cross = max(self.cross, self.crossings(self.state_0.particle_q))
        Newton.test_post_step(self)  # the deepest vertex inside a body so far
        x = np.einsum("sk,tkc->tsc", self.samples, self.state_0.particle_q.numpy()[self.faces]).reshape(-1, 3)
        points = SimpleNamespace(particle_q=wp.array(x, dtype=wp.vec3), body_q=self.state_0.body_q)
        depth = Newton._cloth_penetration(SimpleNamespace(params=self.params, _bodies=self._bodies, state_0=points))
        self.face_penetration = max(self.face_penetration, depth)

    def test_final(self):
        Newton.test_final(self)
        assert self.face_penetration <= 1.0e-4, f"a triangle {self.face_penetration * 1e3:.2f} mm inside a body"
        assert self.cross == 0, f"the cloth passed through itself ({self.cross} crossings)"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=PARAMS["num_frames"])
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
