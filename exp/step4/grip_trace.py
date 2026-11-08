"""Pick-place grip per soft solve (each proxy pass) and per substep (MuJoCo's finger), PASSES=1|2, while held.
Prints per substep: our contact on each finger per pass, the coupler's feedback, MuJoCo's finger y."""
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
pinit = SolverCoupledProxy.Proxy.__init__
def proxy(self, *a, **kw):
    pinit(self, *a, **kw)
    object.__setattr__(self, "proxy_relaxation", float(os.environ.get("RELAX", "1")))
    object.__setattr__(self, "proxy_relaxation_mode", os.environ.get("RELAX_MODE", "fixed"))
    object.__setattr__(self, "mass_scale", float(os.environ.get("MASS_SCALE", "1")))
SolverCoupledProxy.Proxy.__init__ = proxy
M = _load("newton/franka_cable_ik_pick_place")
M.Example.capture = False
ex = M.Example(None, None)
soft, fingers = ex.soft, ex.fingers
rec, sub = [], []
step = soft.step
glob = [b for b, label in enumerate(ex.model.body_label) if "finger" in label]
fb = ex.solver._proxy_mappings[0].coupling_forces
inp = []
def traced(state_in, *a, **kw):
    if 3.6 <= ex.sim_time < 3.7:  # what this pass gets: the force MuJoCo just took, the synced finger velocity
        q = state_in.body_q.numpy()[fingers, 1]
        inp.append((fb.numpy()[glob, 1].copy(), (q - soft._proxy_arm._last.numpy()[fingers, 1]) * 1e3))
    out = step(state_in, *a, **kw)
    if 3.6 <= ex.sim_time < 3.7:
        rec.append(soft.body_reaction.numpy()[fingers, 1].copy())
    return out
soft.step = traced
def post():
    if 3.6 <= ex.sim_time < 3.7:
        sub.append((len(rec), fb.numpy()[glob, 1].copy(), ex.state_0.body_qd.numpy()[glob, 1].copy()))
ex.post_substep = post
c0 = None
for f in range(int(3.7 * 60) + 1):
    if ex.sim_time >= 3.6 and c0 is None:
        c0 = soft.counts.numpy().copy()
    ex.step()
print("counts (unconverged, stalled, nonfinite) over the window:", soft.counts.numpy() - c0, "of", len(rec), "solves")
print(f"PASSES {passes}: {len(rec)} soft solves, {len(sub)} substeps")
for i, (n, f, y) in enumerate(sub[:40]):
    calls = rec[n - passes:n]
    print(f"sub {i:2d} " + " | ".join(f"in F {inp[n - passes + k][0][0]:+.3f} dc {inp[n - passes + k][1][0] * 1e3:+.2f}um -> ours {c[0]:+.3f}" for k, c in enumerate(calls))
          + f"  || feedback {f[0]:+.3f}  mj v {y[0] * 1e3:+.2f}mm/s")
r = np.array([rec[n - 1] for n, _, _ in sub])
print("last pass |grip| mean", np.abs(r).mean(0), "min", np.abs(r).min(0), "max", np.abs(r).max(0))
