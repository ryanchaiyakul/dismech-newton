"""test_proxy_momentum (seed 0), 2 proxy passes: each pass's harvested force on the box; the box takes pass 1's
(pass 2's MuJoCo step applies it; the next step's pass 1 is discarded), the rod pass 2's contact. Momentum drift
(rod + box, no pending term) vs sum h (F_pass1 - F_pass2)."""
import os, sys
sys.path.insert(0, os.path.join(os.environ.get("ROOT", "."), "tests"))
import newton, numpy as np, warp as wp
wp.config.quiet = True
from test_coupling import _proxy_scene
from dismech_newton import SolverDiSMech

rng = np.random.default_rng(0)
builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
body = builder.add_body(xform=wp.transform(wp.vec3(0.1, float(rng.uniform(0.0, 0.04)), 0.0), turn))
builder.add_shape_box(body, hx=0.04, hy=0.06, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3))
n, r = 41, 0.005
rod = newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.2, 0.2, n), np.zeros(n)]), radius=r)
model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.3, rigid_contacts=False)
object.__setattr__(solver._coupling, "iterations", 2)
rec = []
orig = SolverDiSMech.coupling_harvest_proxy_wrenches
def harvest(self, ids, out, **kw):
    before = out.numpy()[body, :3].copy()
    orig(self, ids, out, **kw)
    rec.append(out.numpy()[body, :3].copy())
SolverDiSMech.coupling_harvest_proxy_wrenches = harvest
contacts = pipeline.contacts()
qd = s0.particle_qd.numpy(); qd[:, 0] = 1.0; s0.particle_qd.assign(qd)
m, mb, dt = model.particle_mass.numpy()[:, None], float(model.body_mass.numpy()[body]), 1.0e-3
p = lambda s: (m * s.particle_qd.numpy()).sum(0) + mb * s.body_qd.numpy()[body, :3]
p0 = p(s0); gap = np.zeros(3); worst = worst_pred = 0.0
for k in range(300):
    s0.clear_forces(); pipeline.collide(s0, contacts); solver.step(s0, s1, None, contacts, dt); s0, s1 = s1, s0
    F1, F2 = rec[-2], rec[-1]
    gap += dt * (F1 - F2)
    worst = max(worst, np.linalg.norm(p(s0) - p0) / np.linalg.norm(p0))
    worst_pred = max(worst_pred, np.linalg.norm(gap) / np.linalg.norm(p0))
    if k % 30 == 0 or np.linalg.norm(F1 - F2) > 0.05 * max(np.linalg.norm(F2), 1e-9) and np.linalg.norm(F2) > 0.1:
        print(f"{k:3d} F1 {F1.round(3)} F2 {F2.round(3)}  drift {(p(s0) - p0).round(6)}  sum h dF {gap.round(6)}")
print(f"worst drift {worst:.2e}  worst |sum h (F1 - F2)| {worst_pred:.2e}")
