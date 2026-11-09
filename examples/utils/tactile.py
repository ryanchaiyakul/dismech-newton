"""PaXini PX-6AX GEN3 tactile fingertips on the MIDAS hand: the sensors' outer surfaces as the fingertips' contact
shapes (hydroelastic pads in our contact, :func:`utils.common.pads`), the MJCF's fingertip capsules moved behind them.

The vendor's data stays local (the repo is public): ``PAXINI`` (default ``scratch/paxini``) holds per sensor its
taxel map (``S2015.json``, ``M2826.json``: positions [mm], frames), its meshed STEP (``*_mesh.npz``, mm) and
``mount/mount.json`` (each link's sensor and its pose, fitted to the mounting holes) and ``mount/dorsal.json`` (the
capsule left on the finger's back)."""

import json
import os
from pathlib import Path

import numpy as np
import warp as wp

PAXINI = Path(os.environ.get("PAXINI", Path(__file__).resolve().parents[2] / "scratch" / "paxini"))
CAPSULES = {"index_dip_link": "index_fingertip_contact", "middle_dip_link": "middle_fingertip_contact",
            "ring_dip_link": "ring_fingertip_contact", "thumb_dip": "thumb_dip_contact"}


def mount(builder, root: Path = PAXINI) -> dict[str, int]:
    """After ``add_mjcf`` of the MIDAS hand: each sensor's surface a contact shape on its link (massless, the
    fingertip capsule's friction), the capsule itself no longer colliding, a smaller one on the finger's back.
    Returns per link its sensor shape."""
    import newton  # noqa: PLC0415
    from newton import ShapeFlags  # noqa: PLC0415

    poses = json.loads((root / "mount" / "mount.json").read_text())
    backs = json.loads((root / "mount" / "dorsal.json").read_text())
    body_of = {label.split("/")[-1]: b for b, label in enumerate(builder.body_label)}
    shape_of = {label.split("/")[-1]: s for s, label in enumerate(builder.shape_label)}
    collide = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    out = {}
    for link, capsule in CAPSULES.items():
        p, body, old = poses[link], body_of[link], shape_of[capsule]
        builder.shape_flags[old] &= ~collide
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=builder.shape_material_mu[old])
        mesh = np.load(root / f"{p['sensor']}_mesh.npz")
        q = wp.quat_from_matrix(wp.mat33(*np.asarray(p["R"], dtype=np.float32).ravel()))
        out[link] = builder.add_shape_mesh(body, xform=wp.transform(wp.vec3(*p["t"]), q), cfg=cfg, label=f"{link}_paxini",
                                           mesh=newton.Mesh(mesh["vertices"] * 1.0e-3, mesh["faces"].ravel()))
        b = backs[link]
        w, x, y, z = b["quat"]  # MJCF (w, x, y, z); the capsule's axis its +Z, as MuJoCo's
        builder.add_shape_capsule(body, xform=wp.transform(wp.vec3(*b["pos"]), wp.quat(x, y, z, w)), radius=b["radius"],
                                  half_height=b["half"], cfg=cfg, label=f"{link}_back")
    return out
