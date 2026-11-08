import os, sys
sys.path.insert(0, os.path.abspath("tests"))
import newton, numpy as np, warp as wp
from test_coupling import _proxy_scene
builder = newton.ModelBuilder()
hx, r, n = 0.02, 0.004, 21
rod = newton.Rod(np.column_stack([np.full(n, 1.0), np.linspace(-0.05, 0.05, n), np.full(n, 0.1)]), radius=r)
body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
builder.add_shape_box(body, hx=hx, hy=hx, hz=hx, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.5, rigid_contacts=False)
contacts = pipeline.contacts(); fb = solver._proxy_mappings[0].coupling_forces
dt = 2e-3
for k in range(101):
    s0.clear_forces(); pipeline.collide(s0, contacts); solver.step(s0, s1, None, contacts, dt); s0, s1 = s1, s0
    if k % 20 == 0:
        t = (k + 1) * dt
        print(f"k {k} z {s0.body_q.numpy()[body, 2]:.5f} free-fall {1.0 - 0.5 * 9.81 * t * t:.5f} vz {s0.body_qd.numpy()[body, 2]:+.4f} fb {fb.numpy()[body, :3]}")
