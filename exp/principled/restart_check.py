"""Pick-place, PASSES=2: does each pass start MuJoCo from the same state? Per pass at a few held substeps: the
source's joint_q / joint_qd input, MuJoCo's qpos / qvel, and the finger body pose and velocity it ends with."""
import os, sys
sys.path.insert(0, os.path.abspath("tests"))
import numpy as np
from newton.solvers.experimental.coupled import SolverCoupledProxy
from test_examples import _load
init = SolverCoupledProxy.Config.__init__
def config(self, *a, **kw):
    init(self, *a, **kw)
    object.__setattr__(self, "iterations", 2)
SolverCoupledProxy.Config.__init__ = config
M = _load("newton/franka_cable_ik_pick_place")
M.Example.capture = False
ex = M.Example(None, None)
cpl = ex.solver
src = cpl._entries["mjc"]
mj = src.solver
log, on = [], [False]
step_entry = cpl._step_entry
def traced(entry, *a, **kw):
    if on[0] and entry is src:
        d = mj.mjw_data
        pre = (src.state_0.joint_q.numpy()[-2:].copy(), d.qpos.numpy()[0, -2:].copy(), d.qvel.numpy()[0, -2:].copy())
        out = step_entry(entry, *a, **kw)
        fing = [b for b, l in enumerate(src.solver.model.body_label) if "finger" in l] if hasattr(src.solver, "model") else []
        post = (d.qpos.numpy()[0, -2:].copy(), d.qvel.numpy()[0, -2:].copy())
        log.append((pre, post))
        return out
    return step_entry(entry, *a, **kw)
cpl._step_entry = traced
for f in range(int(3.6 * 60)):
    ex.step()
on[0] = True
ex.step()
np.set_printoptions(precision=7, suppress=False, linewidth=200)
for i, ((jq, qp, qv), (qp1, qv1)) in enumerate(log[:8]):
    print(f"pass {i % 2}: state joint_q {jq} | mj qpos in {qp} qvel in {qv} -> qpos out {qp1} qvel out {qv1}")
