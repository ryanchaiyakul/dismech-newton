"""Trampoline per substep (no graph): capsule 2's pairs and the cloth's depth inside it, frames START..END."""
import os, runpy, sys
import numpy as np
sys.path.insert(0, os.path.abspath("examples/newton")); sys.path.insert(0, os.path.abspath("examples"))
sys.argv = ["cloth_rigid_trampoline.py", "--viewer", "null"] + sys.argv[1:]
g = runpy.run_path("examples/newton/cloth_rigid_trampoline.py", run_name="tramp")
import newton, newton.examples, warp as wp
from scipy.spatial.transform import Rotation
from dismech_newton import SolverDiSMech
START, END = int(os.environ.get("START", 60)), int(os.environ.get("END", 80))
Ex = g["Example"]
HH, RAD = g["PARAMS"]["capsule_half_height"], 0.08
st = {"frame": 0, "sub": 0}
orig = SolverDiSMech.step
def step(self, state_in, state_out, control, contacts, dt, **kw):
    orig(self, state_in, state_out, control, contacts, dt, **kw)
    if not (START <= st["frame"] <= END):
        return
    ex = st["ex"]; b2 = ex._bodies[2]
    a = self._proxy_arm
    bq = a._last.numpy()[b2]
    R = Rotation.from_quat(bq[3:7]).as_matrix()
    x = state_out.particle_q.numpy()
    loc = (x - bq[:3]) @ R
    t = np.clip(loc[:, 2], -HH, HH); qq = loc.copy(); qq[:, 2] -= t
    dep = RAD - np.linalg.norm(qq, axis=1)
    s = self.contact.s
    act = s.active.numpy().astype(bool); key = s.key.numpy(); w = s.w.numpy()
    key0 = self.contact.arm.key0
    shp = (key - key0) // 2
    sb = self.model.shape_body.numpy()
    sel = act & (key >= key0) & (w.min(1) >= 0)
    sel &= sb[np.clip(shp, 0, len(sb) - 1)] == b2
    pr = s.pair.numpy()[sel]
    c = self.counts.numpy()
    deep = np.argsort(-dep)[:3]
    paired = set(pr[:, 0].tolist())
    print(f"S f{st['frame']} s{st['sub']} pairs {sel.sum()} inside {(dep > 0).sum()} deepest {dep.max()*1e3:+.2f}mm "
          f"nodes {deep.tolist()} paired {[int(n) in paired for n in deep]} counts {c} z {bq[2]:.4f}", flush=True)
    st["sub"] += 1
SolverDiSMech.step = step
class T(Ex):
    def __init__(self, *a):
        super().__init__(*a); st["ex"] = self
        self.solver.graph_capturable = False
    def step(self):
        st["sub"] = 0
        super().step(); st["frame"] += 1
        if self.state_0.body_q.numpy()[self._bodies[2], 2] < 0.5 or st["frame"] > END: raise SystemExit("stop")
parser = newton.examples.create_parser()
parser.set_defaults(num_frames=g["PARAMS"]["num_frames"])
viewer, args = newton.examples.init(parser)
newton.examples.run(T(viewer, args), args)
