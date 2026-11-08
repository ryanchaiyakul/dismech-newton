"""Trampoline: per frame, MuJoCo's body z vs our proxy's (ProxyArm._last) and the soft step's counts."""
import os, runpy, sys
sys.path.insert(0, os.path.abspath("examples/newton")); sys.path.insert(0, os.path.abspath("examples"))
sys.argv = ["cloth_rigid_trampoline.py", "--viewer", "null"] + sys.argv[1:]
g = runpy.run_path("examples/newton/cloth_rigid_trampoline.py", run_name="tramp")
import newton, newton.examples
Ex = g["Example"]
class T(Ex):
    def step(self):
        super().step()
        soft = self.solver._entries["soft"].solver
        mj = self.state_0.body_q.numpy()[self._bodies, 2]
        px = soft._proxy_arm._last.numpy()[self._bodies, 2]
        c = soft.counts.numpy() if hasattr(soft, "counts") else None
        f = getattr(self, "_f", 0); self._f = f + 1
        if f % 5 == 0 or abs(mj[2] - px[2]) > 0.02:
            print(f"F {f} t {self.sim_time:.3f} mj {' '.join(f'{z:+.3f}' for z in mj[:3])} proxy {' '.join(f'{z:+.3f}' for z in px[:3])} pen {self.max_penetration:.4f} counts {c}", flush=True)
        if f >= 0:
            import numpy as np
            x = self.state_0.particle_q.numpy()
            bq = self.state_0.body_q.numpy()[self._bodies[2]]
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat(bq[3:7]).as_matrix()
            hh, r = g["PARAMS"]["capsule_half_height"], 0.08
            def depth(P):  # capsule axis: local z? try all and report
                loc = (P - bq[:3]) @ R
                out = []
                for ax in range(3):
                    t = np.clip(loc[:, ax], -hh, hh)
                    q = loc.copy(); q[:, ax] -= t
                    out.append(r - np.linalg.norm(q, axis=1))
                return out
            fa = self.faces
            e = np.unique(np.sort(np.concatenate([fa[:, [0, 1]], fa[:, [1, 2]], fa[:, [2, 0]]]), 1), axis=0)
            u = np.linspace(0, 1, 9)[1:-1]
            E = (x[e[:, 0], None] * (1 - u)[None, :, None] + x[e[:, 1], None] * u[None, :, None]).reshape(-1, 3)
            T = np.einsum("sk,tkc->tsc", self.samples, x[fa]).reshape(-1, 3)
            dv, de, dt_ = depth(x), depth(E), depth(T)
            dz = dv[2]; k = np.argsort(-dz)[:3]
            if dz.max() > 0: print(f"   IN {f} verts {k.tolist()} depth {[f'{dz[i]*1e3:.1f}' for i in k]} border {[bool(i//41 in (0,40) or i%41 in (0,40)) if i < 1681 else 'top' for i in k]}", flush=True)
            if False: print(f"   D {f} vert {[f'{d.max()*1e3:.1f}' for d in dv]} edge {[f'{d.max()*1e3:.1f}' for d in de]} tri {[f'{d.max()*1e3:.1f}' for d in dt_]} (mm inside, axis x/y/z)", flush=True)
        if f % 5 == 0: print(f"  XY {f} body2 {self.state_0.body_q.numpy()[self._bodies[2], :3].round(3).tolist()}", flush=True)
        if mj[2] < -2: raise SystemExit("fell")
parser = newton.examples.create_parser()
parser.set_defaults(num_frames=g["PARAMS"]["num_frames"])
viewer, args = newton.examples.init(parser)
newton.examples.run(T(viewer, args), args)
