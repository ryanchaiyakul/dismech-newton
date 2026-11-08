"""Pick-place held at 3.6 s with DM_SCHUR=body: the measured source response vs the view inertia."""
import os, sys
sys.path.insert(0, os.path.abspath("tests"))
import numpy as np
from test_examples import _load
np.set_printoptions(precision=4, suppress=False, linewidth=200)
M = _load("newton/franka_cable_ik_pick_place")
M.Example.capture = False
ex = M.Example(None, None)
arm = ex.soft._proxy_arm
for f in range(int(3.6 * 60)):
    ex.step()
S, view = arm._schur[0], arm._arm.Minv.numpy()[0]
print("labels", [ex.soft.model.body_label[b] for b in arm._body.numpy()], "slots", arm._slot)
print("asym", arm._schur_asym)
print("diag Minv_s ", np.diag(S))
print("diag Minv_view (after reg, M=inv(Minv_s+eps view))", np.diag(view))
w, V = np.linalg.eigh(S)
print("eig Minv_s", w)
for b in range(S.shape[0] // 6):
    sl = slice(6 * b, 6 * b + 6)
    print(f"body {b}: lin block\n", S[sl, sl][:3, :3], "\n eff mass along axes", 1 / np.diag(S[sl, sl])[:3])
print("finger-finger y coupling", S[6 * 1 + 1, 6 * 2 + 1] if S.shape[0] >= 18 else None)
print("probes", __import__("dismech_newton.schur_exp", fromlist=["x"]).STATS)
