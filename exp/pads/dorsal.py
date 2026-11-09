"""The fingertip capsules (MJCF) moved off the sensor side: shifted against the sensor's normal and shrunk by s until
no capsule surface point on the sensor's side of its base plane lies outside the sensor (the sensor alone takes the
pad side's contacts). Prints each link's capsule (body frame, m). Data: $PAXINI."""
import json, os
import numpy as np, trimesh
D = os.environ.get("PAXINI", "../../../scratch/paxini")
mount = json.load(open(f"{D}/mount/mount.json"))
def quat_mat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
caps = {"finger": dict(r=8.8, half=4.0, pos=[1.6, 21.0, -8.5], quat=[0.7071067812, -0.7071067812, 0, 0]),
        "thumb": dict(r=11.8, half=6.0, pos=[-4.0, 23.0, -17.0], quat=[0.7352302348, 0.6100421737, -0.0533717745, 0.2905795965])}
def surface(c, r, half, axis, n=4000):
    g = np.random.default_rng(0).normal(size=(n, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    s = np.random.default_rng(1).uniform(-half, half, n)
    cyl = g - np.outer(g @ axis, axis); cyl /= np.linalg.norm(cyl, axis=1, keepdims=True)
    pts = np.vstack([c + np.outer(s, axis) + r * cyl, c + half * axis + r * g * (g @ axis > 0)[:, None], c - half * axis + r * g * (g @ axis < 0)[:, None]])
    return pts
out = {}
for link, m in mount.items():
    if link == "note": continue
    kind = "thumb" if link.startswith("thumb") else "finger"
    d = np.load(f"{D}/{m['sensor']}_mesh.npz"); R, t = np.array(m["R"]), np.array(m["t"]) * 1e3
    sens = trimesh.Trimesh(d["vertices"] @ R.T + t, d["faces"])
    nrm = R[:, 2]; base = (d["vertices"][:, 2].min() * R[:, 2] + t) @ nrm  # the sensor's base plane along its normal
    cp = caps[kind]; axis = quat_mat(*cp["quat"]) @ [0, 0, 1.0]
    for s in np.arange(0.0, 10.01, 0.25):
        c = np.array(cp["pos"]) - s * nrm; r = cp["r"] - s
        p = surface(c, r, cp["half"], axis); side = p[p @ nrm > base + 0.2]
        bad = (~sens.contains(side)).sum() if len(side) else 0
        if bad == 0: break
    out[link] = dict(pos=(c * 1e-3).round(6).tolist(), radius=round(r * 1e-3, 6), half=cp["half"] * 1e-3, quat=cp["quat"], shift_mm=float(s))
    print(link, out[link])
json.dump(out, open(f"{D}/mount/dorsal.json", "w"), indent=1)
