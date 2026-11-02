"""A sweatshirt falls onto a standing avatar and drapes over it (Newton's ``example_cloth_style3d``): Style3D's
garment, its panels (UVs) and woven-cloth parameters, as a midedge shell on a static mesh.

Style3D's ``tri_aniso_ke`` is the shell's ``aniso`` (the same small-strain membrane in the panel's axes) and its
``edge_aniso_ke`` times 3 the ``bend_aniso`` (as in ``cloth_hanging.py --solver style3d``). Style3D's rest shape is
the flat panel; here it is the garment as loaded, within about 2% of its panels' edge lengths.

Shows: woven cloth on a garment mesh with face-varying panel coordinates, cloth on a triangle-mesh shape. Newton's
test (inside its box) and two of its own: no edge passes through a cloth or avatar face. The particle radius
(2.5 mm) is below the garment's 3 mm clearance from the avatar as loaded. Downloading the asset needs GitPython.

    uv run --with GitPython examples/newton/cloth_style3d.py
    uv run --with GitPython examples/newton/cloth_style3d.py --viewer null --test
"""

import sys
from pathlib import Path

import newton
import newton.examples
import newton.usd
import newton.utils
import numpy as np
import warp as wp
from cloth_twist import Crossings
from pxr import Usd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/: utils

from utils.common import CableExample

from dismech_newton import SolverDiSMech


def usd_mesh(path, prim, **kwargs):
    stage = Usd.Stage.Open(str(path))  # alive while the prim is read
    return newton.usd.get_mesh(stage.GetPrimAtPath(prim), **kwargs)


class Example(CableExample):
    substeps = 10
    capture = False  # cuDSS allocates while factorising a system this size: it cannot be captured in a graph
    garment = "Women_Sweatshirt"

    def __init__(self, viewer, args=None):
        assets = newton.utils.download_asset("style3d")
        mesh, uv_indices = usd_mesh(assets / "garments" / f"{self.garment}.usd", f"/Root/{self.garment}/Root_Garment",
                                    load_uvs=True, preserve_facevarying_uvs=True, return_uv_indices=True)
        faces = np.asarray(mesh.indices).reshape(-1, 3)
        panel = (np.asarray(mesh.uvs) * 1.0e-3)[np.asarray(uv_indices).reshape(-1, 3)]  # mm, per face corner
        avatar = usd_mesh(assets / "avatars" / "Female.usd", "/Root/Female/Root_SkinnedMesh_Avatar_0_Sub_2")
        up = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)  # the assets are y-up

        builder = newton.ModelBuilder()
        SolverDiSMech.add_cloth_mesh(
            builder, vertices=mesh.vertices.tolist(), indices=faces.ravel(), rot=up, density=0.3,
            particle_radius=2.5e-3, aniso=(1.0e2, 1.0e2, 1.0e1), bend_aniso=3.0 * np.array([2.0e-5, 1.0e-5, 5.0e-6]),
            uv=panel, damping=0.0,
        )
        builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(0.0), up),
                               mesh=newton.Mesh(avatar.vertices, avatar.indices))
        body = np.asarray(avatar.vertices) @ np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # rotated by up
        builder.add_ground_plane()
        model = builder.finalize()
        model.particle_mu = 0.2
        model.shape_material_mu.fill_(0.2)
        self.start(viewer, model, SolverDiSMech(model))
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0)
        self.crossings = Crossings(edges, faces, model.device)
        n = model.particle_count  # and the cloth's edges through the avatar: its triangles after the cloth's nodes
        self.through = Crossings(edges, np.asarray(avatar.indices).reshape(-1, 3) + n, model.device)
        self.x = wp.array(np.concatenate([model.particle_q.numpy(), body]), dtype=wp.vec3, device=model.device)
        self.cross = self.body = 0
        if viewer is not None:
            viewer.set_camera(wp.vec3(0.0, -1.7, 1.4), 0.0, -270.0)

    def step(self):
        super().step()
        self.cross = max(self.cross, self.crossings(self.state_0.particle_q))
        wp.copy(self.x, self.state_0.particle_q, count=self.model.particle_count)
        self.body = max(self.body, self.through(self.x))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), "non-finite positions"
        lower, upper = np.array([-0.5, -0.2, 0.9]), np.array([0.5, 0.2, 1.6])
        assert ((x >= lower) & (x <= upper)).all(), "particles outside Newton's box"
        assert self.cross == 0, f"the cloth passed through itself ({self.cross} crossings)"
        assert self.body == 0, f"the cloth passed through the avatar ({self.body} edges)"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
