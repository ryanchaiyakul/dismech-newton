"""Adam from fit_ift's guess with the gripper pose readout as the drive (eval_ift_drive.pose_schedule)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import eval_ift_drive as ev
import fit_ift as fi
import gen_baseline as gb

d = fi.load()
rods = 256
f = fi.Fit(rods, "command", d)
f.fix_vals.assign(ev.pose_schedule(d, d["grip_obs"][:rods].astype(np.float64), rods).astype(np.float32))


def fun(x):
    per, g = f.loss_and_grad(float(x[0]))
    return per.mean(), np.array([g])


xs = fi.adam(fun, np.log(fi.K_GUESS))
print("k iterates:", np.round(np.exp(xs), 2).tolist())
L = f.forward(xs[-1])
u = gb.mid_rel(f.trajectories()[:, gb.K_BUCKLE]) > 0
print(f"k_fit {np.exp(xs[-1]):.3f} (true {gb.BEND}); loss {L.mean():.2f}; branch match {np.mean(u == d['up_buckle']):.3f}")
