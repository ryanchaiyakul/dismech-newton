"""IFT gradient near the truth, split into trials on the observed branch vs the wrong one (gripper pose drive)."""
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))
import eval_ift_drive as ev
import fit_ift as fi
import gen_baseline as gb

d = fi.load()
rods = 256
f = fi.Fit(rods, "command", d)
f.fix_vals.assign(ev.pose_schedule(d, d["grip_obs"][:rods].astype(np.float64), rods).astype(np.float32))
up = d["up_buckle"][:rods]


def grad(log_k, w):
    f.log_k.assign(np.array([log_k], dtype=np.float32))
    tape = wp.Tape()
    with tape:
        f.run()
    tape.backward(grads={f.loss: wp.array(w.astype(np.float32), dtype=float, device=f.model.device)})
    g = float(f.log_k.grad.numpy()[0])
    L = f.loss.numpy().astype(np.float64)
    tape.zero()
    tape.reset()
    return g, L


for k in (8.25, 10.0, 12.0):
    lk = np.log(k)
    L = f.forward(lk)
    match = (gb.mid_rel(f.trajectories()[:, gb.K_BUCKLE]) > 0) == up
    g_all, _ = grad(lk, np.full(rods, 1.0 / rods))
    g_m, _ = grad(lk, match / rods)
    g_w, _ = grad(lk, ~match / rods)
    h = 0.05
    fd = (f.forward(lk + h).mean() - f.forward(lk - h).mean()) / (2 * h)
    print(f"k {k:5.2f}: match {match.mean():.3f}, loss matched {L[match].mean():.2f} wrong {L[~match].mean():.0f}; "
          f"dL/dlogk IFT all {g_all:+.1f} = matched {g_m:+.1f} + wrong {g_w:+.1f}; secant FD (h=.05) {fd:+.1f}",
          flush=True)
