"""Box dropped on a rod clamped at its ends (stiff), proxy coupling: rebound speed / impact speed, and the most the
box's MuJoCo pose ever went below rest. ROOT: the checkout's tests dir."""
import os, sys
sys.path.insert(0, os.path.join(os.environ.get("ROOT", "."), "tests"))
import newton, numpy as np, warp as wp
import test_coupling
from test_coupling import _proxy_scene
from dismech_newton import SolverDiSMech
builder = newton.ModelBuilder()
hx, r, n, drop = 0.02, 0.004, 21, 0.01
rod = newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.05, 0.05, n), np.full(n, 0.1)]), radius=r)
body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1 + r + hx + drop), wp.quat_identity()))
builder.add_shape_box(body, hx=hx, hy=hx, hz=hx, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
orig = SolverDiSMech.add_rod
def add_rod(b, **kw):
    kw["stretch_stiffness"], kw["bend_stiffness"] = 1.0e6, 10.0
    caps, j = orig(b, **kw)
    for c in (caps[0], caps[-1]):
        SolverDiSMech.fix_segment(b, c)
    return caps, j
test_coupling.SolverDiSMech.add_rod = add_rod
model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.5, rigid_contacts=False)
contacts = pipeline.contacts()
vin, vout, zmin = 0.0, 0.0, 1.0
for k in range(150):
    s0.clear_forces(); pipeline.collide(s0, contacts); solver.step(s0, s1, None, contacts, 2e-3); s0, s1 = s1, s0
    vz = s0.body_qd.numpy()[body, 2]; z = s0.body_q.numpy()[body, 2]
    vin, vout, zmin = min(vin, vz), max(vout, vz), min(zmin, z)
print(f"BOUNCE impact {vin:.3f} rebound {vout:.3f} ratio {vout / -vin:.2f} deepest below rest {(0.1 + r + hx - zmin) * 1e3:.2f} mm")
