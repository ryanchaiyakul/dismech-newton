"""(1) test_proxy_impact's drop, per step: box z / vz (MuJoCo), the feedback F after our step, the 1-DOF model's
prediction alongside. (2) test_proxy_momentum (seed 0) with PASSES proxy passes: linear momentum error with the
pending feedback counted (the test's) and without. ROOT: the checkout. PASSES env (default 1)."""
import os, sys
sys.path.insert(0, os.path.join(os.environ.get("ROOT", "."), "tests"))
sys.path.insert(0, os.path.dirname(__file__))
import newton, numpy as np, warp as wp
wp.config.quiet = True
from test_coupling import _proxy_scene
import model1d

P = int(os.environ.get("PASSES", "1"))

def impact():
    builder = newton.ModelBuilder()
    hx, r, n, drop = 0.02, 0.004, 21, 0.01
    rest = 0.1 + r + hx
    rod = newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.05, 0.05, n), np.full(n, 0.1)]), radius=r)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, rest + drop), wp.quat_identity()))
    builder.add_shape_box(body, hx=hx, hy=hx, hz=hx, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
    model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.5, fixed=True)
    object.__setattr__(solver._coupling, "iterations", P)
    contacts = pipeline.contacts()
    fb = solver._proxy_mappings[0].coupling_forces
    m, h = float(model.body_mass.numpy()[body]), 2e-3
    rows = []
    for k in range(120):
        s0.clear_forces(); pipeline.collide(s0, contacts); solver.step(s0, s1, None, contacts, h); s0, s1 = s1, s0
        rows.append((s0.body_q.numpy()[body, 2] - rest, s0.body_qd.numpy()[body, 2], fb.numpy()[body, 2]))
    rows = np.array(rows)
    k0 = int(np.argmax(rows[:, 2] > 0)); vin = -rows[k0 - 1, 1]
    print(f"P={P} AHEAD={os.environ.get('DM_AHEAD', '0')} impact {vin:.3f} rebound {rows[k0:, 1].max():.3f} ratio {rows[k0:, 1].max() / vin:.2f}  m {m:.4f} "
          f"mV/h {m * vin / h:.2f} N")
    # the model, started at the same height above the contact onset with the same speed
    first = int(np.argmax(rows[:, 2] > 0))
    ratio, tr = model1d.run("lagged", "hard", V=vin, h=h, m=m)
    mf = int(np.argmax(tr[:, 2] > 0))
    print(f"model lagged hard ratio {ratio:.2f}")
    print(" k  | sim: z-rest[mm] vz    F[N]   | model: x[mm]  v     F[N]")
    for j in range(-2, 8):
        a, b = rows[first + j], tr[mf + j]
        print(f"{j:+3d} | {a[0] * 1e3:+8.3f} {a[1]:+.3f} {a[2]:+7.2f} | {b[0] * 1e3:+8.3f} {b[1]:+.3f} {b[2]:+7.2f}")

def momentum():
    rng = np.random.default_rng(0)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.3, 0.3)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.1, float(rng.uniform(0.0, 0.04)), 0.0), turn))
    half = np.array([0.04, 0.06, 0.04])
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.3))
    n, r = 41, 0.005
    rod = newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.2, 0.2, n), np.zeros(n)]), radius=r)
    model, solver, pipeline, s0, s1 = _proxy_scene(builder, body, rod, 0.3, rigid_contacts=False)
    object.__setattr__(solver._coupling, "iterations", P)
    contacts = pipeline.contacts()
    qd = s0.particle_qd.numpy(); qd[:, 0] = 1.0; s0.particle_qd.assign(qd)
    m, mb, dt = model.particle_mass.numpy()[:, None], float(model.body_mass.numpy()[body]), 1.0e-3
    fb = solver._proxy_mappings[0].coupling_forces
    def p(s, pending):
        return (m * s.particle_qd.numpy()).sum(0) + mb * s.body_qd.numpy()[body, :3] + pending * dt * fb.numpy()[body, :3]
    p0 = p(s0, 1); w1 = w0 = 0.0
    for _ in range(300):
        s0.clear_forces(); pipeline.collide(s0, contacts); solver.step(s0, s1, None, contacts, dt); s0, s1 = s1, s0
        w1 = max(w1, np.linalg.norm(p(s0, 1) - p0) / np.linalg.norm(p0))
        w0 = max(w0, np.linalg.norm(p(s0, 0) - p0) / np.linalg.norm(p0))
    print(f"P={P} momentum error: with pending feedback {w1:.2e}, without {w0:.2e}")

impact()
momentum()
