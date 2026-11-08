"""Pick-place grip chatter vs phase: per substep both fingers' y reaction (ours), the cable's centre-of-mass
acceleration and the hand's velocity, in windows lift (3.0-3.2), carry (3.6-3.8), settle (5.1-5.3).
Env PASSES as grip_trace; DM_SCHUR etc. as the experiment."""
import os, sys
sys.path.insert(0, os.path.abspath("tests"))
import numpy as np
from newton.solvers.experimental.coupled import SolverCoupledProxy
from test_examples import _load

passes = int(os.environ.get("PASSES", "1"))
init = SolverCoupledProxy.Config.__init__
def config(self, *a, **kw):
    init(self, *a, **kw)
    object.__setattr__(self, "iterations", passes)
SolverCoupledProxy.Config.__init__ = config
M = _load("newton/franka_cable_ik_pick_place")
M.Example.capture = False
ex = M.Example(None, None)
soft, fingers = ex.soft, ex.fingers
hand = [b for b, label in enumerate(ex.model.body_label) if "hand" in label][0]
mass = ex.model.particle_mass.numpy()
W = {"lift": (3.0, 3.2), "carry": (3.6, 3.8), "settle": (5.1, 5.3)}
rec = {k: [] for k in W}
def post():
    for k, (a, b) in W.items():
        if a <= ex.sim_time < b:
            f = soft.body_reaction.numpy()[fingers, 1]
            v = (mass[:, None] * ex.state_0.particle_qd.numpy()).sum(0) / mass.sum()
            rec[k].append((f[0], f[1], *v, *ex.state_0.body_qd.numpy()[hand, :3]))
ex.post_substep = post
for f in range(int(5.3 * 60) + 1):
    ex.step()
dt = 1 / 600
print(f"PASSES {passes} DM_SCHUR={os.environ.get('DM_SCHUR', '')} eps {os.environ.get('DM_SCHUR_EPS', '')}")
for k, r in rec.items():
    r = np.array(r)
    sq, anti = 0.5 * (np.abs(r[:, 0]) + np.abs(r[:, 1])), 0.5 * (np.abs(r[:, 0]) - np.abs(r[:, 1]))
    acc = np.diff(r[:, 2:5], axis=0) / dt
    da = np.diff(sq)
    c = [np.corrcoef(da[1:], acc[1:, j])[0, 1] if acc[:, j].std() > 0 else 0 for j in range(3)]
    alt = np.mean(np.sign(da[1:]) != np.sign(da[:-1]))  # fraction of sign flips: 1 = pure 2-cycle
    print(f"{k:6s} squeeze {sq.mean():.3f} +- {sq.std():.3f} [{sq.min():.3f}, {sq.max():.3f}]  anti +- {anti.std():.3f}"
          f"  flips {alt:.2f}  cable |a| {np.linalg.norm(acc, axis=1).mean():.2f} m/s2  hand |v| "
          f"{np.linalg.norm(r[:, 5:8], axis=1).mean() * 1e3:.0f} mm/s  corr(dsq, a_xyz) {np.round(c, 2)}")
