"""Draft: a free MuJoCo box dropped onto a fully clamped rod through SolverCoupledProxy."""
import sys, os
sys.path.insert(0, os.path.abspath("tests"))
import newton
import numpy as np
import warp as wp
from test_coupling import _proxy_scene
from dismech_newton import SolverDiSMech


def test_proxy_on_clamped_rod():
    """A free MuJoCo box dropped onto a rod clamped along its length (every node fixed: only the box can yield),
    through ``SolverCoupledProxy``: it comes to rest on the rod, a contact thickness (the rod's radius) above it, its
    weight carried by the coupler's feedback."""
    builder = newton.ModelBuilder()
    hx, r, n, drop = 0.02, 0.004, 21, 0.01
    rod = newton.Rod(np.column_stack([np.zeros(n), np.linspace(-0.05, 0.05, n), np.full(n, 0.1)]), radius=r)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1 + r + hx + drop), wp.quat_identity()))
    builder.add_shape_box(body, hx=hx, hy=hx, hz=hx, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.5))
    model, solver, pipeline, s0, s1 = _proxy_scene_clamped(builder, body, rod)
    contacts = pipeline.contacts()
    feedback = solver._proxy_mappings[0].coupling_forces
    weight = float(model.body_mass.numpy()[body] * 9.81)
    zs, f = [], []
    for k in range(600):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, 2.0e-3)
        s0, s1 = s1, s0
        zs.append(s0.body_q.numpy()[body, 2])
        if 8 <= k < 60 or k % 50 == 0:
            soft = solver._entries["soft"].solver
            c = soft.contact
            act = c.s.active.numpy().astype(bool)
            th = c.s.thickness.numpy()[act]; key = c.s.key.numpy()[act]; pr = c.s.pair.numpy()[act]
            print(f"k {k} z {zs[-1]:.5f} proxy {soft._proxy_arm._last.numpy()[body, 2]:.5f} vz {s0.body_qd.numpy()[body, 2]:+.3f} fb {feedback.numpy()[body, 2]:+.3f} pairs {act.sum()} keys {sorted(set(key.tolist()))[:6]} th {th[:2]} on {soft.contact.source.mesh.edge_on.numpy()[:3]} cnt {c.count} counts {soft.counts.numpy()}")
        if k >= 500:
            f.append(feedback.numpy()[body])
    z = zs[-1]
    print("z", z, "rest", 0.1 + r + hx, "min", min(zs), "f", np.mean(f, 0), "weight", weight)
    assert abs(z - (0.1 + r + hx)) < 0.25 * r, (z, 0.1 + r + hx)
    assert min(zs) > 0.1 + hx, "the box went into the rod"
    assert abs(np.mean(f, 0)[2] - weight) < 1.0e-2 * weight


def _proxy_scene_clamped(builder, body, rod):
    import test_coupling
    orig = SolverDiSMech.add_rod

    def add_rod(b, **kw):
        caps, joints = orig(b, **kw)
        for c in caps:
            SolverDiSMech.fix_segment(b, c)
        return caps, joints
    test_coupling.SolverDiSMech.add_rod = add_rod
    try:
        return _proxy_scene(builder, body, rod, 0.5, rigid_contacts=os.environ.get("RC", "1") == "1")
    finally:
        test_coupling.SolverDiSMech.add_rod = orig
