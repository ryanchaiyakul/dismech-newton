"""Evaluate fit_ift's drive choices: loss landscape in k and branch agreement for alternative fixed-DOF drives.

* grip_obs: the gripper's noisy 6-DOF pose readout (grip_obs) applied as a rigid edge pose; clamp exactly at rest
* grip_obs_smooth: the same, low-pass filtered in time first
* grip_true: the hidden actual gripper pose (oracle upper bound for the drive)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import fit_ift as fi
import gen_baseline as gb

GRID = np.geomspace(2.0, 60.0, 13)


def pose_schedule(d, poses, rods):
    """Fixed DOF values from per-trial gripper poses (rods, frames, 6) at the 40 Hz frames, clamp at rest."""
    N, S = gb.NODES, gb.SEGMENTS
    t_step = np.arange(gb.STEPS + 1) * gb.DT
    P = np.stack([np.stack([np.interp(t_step, d["t"], poses[r, :, j]) for j in range(6)], -1) for r in range(rods)])
    xr = d["x_rest"]
    a, b = xr[-2], xr[-1]
    c = 0.5 * (a + b)
    R = Rotation.from_rotvec(P[..., 3:].reshape(-1, 3))
    na = (R.apply(a - c).reshape(rods, -1, 3) + c + P[..., :3])
    nb = (R.apply(b - c).reshape(rods, -1, 3) + c + P[..., :3])
    vals = np.zeros((gb.STEPS + 1, rods, 14))
    vals[:, :, 0:3] = xr[0]
    vals[:, :, 3:6] = xr[1]
    vals[:, :, 6:9] = na.transpose(1, 0, 2)
    vals[:, :, 9:12] = nb.transpose(1, 0, 2)
    vals[:, :, 13] = P[..., 3].T  # roll -> the gripper edge's twist
    return vals.reshape(gb.STEPS + 1, -1)


def main(rods=256):
    d = fi.load()
    drives = {
        "grip_obs": d["grip_obs"][:rods].astype(np.float64),
        "grip_obs_smooth": gaussian_filter1d(d["grip_obs"][:rods].astype(np.float64), 2.0, axis=1, mode="nearest"),
        "grip_true": d["grip"][:rods],
    }
    up = d["up_buckle"][:rods]
    for name, poses in drives.items():
        f = fi.Fit(rods, "command", d)  # builds everything; the schedule is replaced below
        f.fix_vals.assign(pose_schedule(d, poses, rods).astype(np.float32))
        pitch_err = np.degrees(poses[:, :, 4] - d["grip"][:rods, :, 4]).std()
        rows = []
        for k in GRID:
            L = f.forward(np.log(k))
            X = f.trajectories()
            u = gb.mid_rel(X[:, gb.K_BUCKLE]) > 0
            rows.append((k, L.mean(), L[u == up].mean() if (u == up).any() else np.nan, np.mean(u == up), u.mean()))
        print(f"\n{name}: pitch error vs actual {pitch_err:.2f} deg")
        for k, L, Lm, m, pu in rows:
            print(f"  k {k:7.2f}  loss {L:8.2f}  loss(matched) {Lm:8.2f}  branch match {m:.3f}  sim P(up) {pu:.3f}",
                  flush=True)
        kbest = rows[int(np.nanargmin([r[1] for r in rows]))][0]
        print(f"  argmin over grid: k {kbest:.2f} (true {gb.BEND})")
        del f


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
