"""The contact adjoint against finite differences of a rollout (ground, rod-rod, a rod edge on a sphere, a sheet, a
proxy pinch),
and its closest points against the collision's. Sliding (mu = 0.3) unless stuck. Contacts start at the barrier's
onset (:data:`rollouts.ONSET`). The steps are converged far below the solver's tolerance: the differences are of the
converged step, which the adjoint is. Every check holds over 16 seeds (``--seeds 16``) at
:meth:`~rollouts.Rollout.check`'s default tolerance."""

from contextlib import nullcontext

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_derivative
from newton import BodyFlags
from newton._src.geometry import kernels as G
from rollouts import CrossingRollout, GroundRollout, PadRollout, SheetContactRollout, SphereRollout

from dismech_newton import SolverDiSMech, flatten_state, solver
from dismech_newton.adjoint import _closest_weights

# Converged: 12 step lengths, as a step from beyond the barrier's onset can need far shorter steps than 4 give.
OPTIONS = {"iterations": 500}


@pytest.fixture(autouse=True)
def converged(monkeypatch):
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-5)


@pytest.mark.parametrize(
    "mu, v_mean, checks",
    [
        (0.3, (0.3, 0.1, 0.0), ("mass", "velocity", "position")),
        (2.0, (0.0, 0.0, 0.0), ("velocity", "translation")),
    ],
    ids=["slide", "stick"],
)
def test_ground_adjoint_matches_finite_differences(rng, mu, v_mean, checks):
    """A rod's nodes on the ground plane (no self-contact: the copies overlap)."""
    GroundRollout(rng, v_mean, checks, mu).check(dict(OPTIONS, self_contact=False))


SLIDE = pytest.mark.parametrize("mu", [0.3], ids=["slide"])


@SLIDE
def test_rod_contact_adjoint_matches_finite_differences(rng, mu, monkeypatch):
    """Rod-rod (edge-edge): the closest points and normal the step takes from ``q_in`` recomputed from it. Half the
    steps, so no node moves half a radius in one (the pairs found again mid-step), and converged further: a two-sided
    pair's common motion converges slowly. Tilt and yaw turn the normal; a random position's stretching
    is too stiff for the step's float32 convergence."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-7)
    r = CrossingRollout(rng, mu)
    r.dt = 5.0e-3
    del r.directions["position"]
    r.check(dict(OPTIONS, iterations=3000))


@SLIDE
def test_shape_contact_adjoint_matches_finite_differences(rng, mu):
    """A rod edge sliding over a sphere: its closest point to the surface and the normal there turn with the rod."""
    SphereRollout(rng, mu).check(OPTIONS)


@SLIDE
def test_hydroelastic_pad_adjoint_matches_finite_differences(rng, mu, monkeypatch):
    """A rod pressing and sliding on a Winkler hydroelastic pad with Hunt-Crossley dissipation."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-7)
    PadRollout(rng, (0.3, 0.1, -0.05), ("mass", "velocity", "position"), mu).check(
        dict(OPTIONS, exact=True, self_contact=False))


@SLIDE
@pytest.mark.parametrize("scene", ["ground", "sphere", "foot", "edge"])
def test_sheet_contact_adjoint_matches_finite_differences(rng, scene, mu, monkeypatch):
    """A sheet's vertex on the ground and on a sphere, a rod's node on a triangle, a rod's edge across the sheet's
    edge. Solved exactly: ADMM's elastic splitting leaves float32 noise that a light sheet's differences do not
    resolve."""
    if scene != "ground":
        monkeypatch.setattr(solver, "TOLERANCE", 1.0e-7)
    iters = 500 if scene == "ground" else 2000
    SheetContactRollout(rng, scene, mu).check(dict(OPTIONS, iterations=iters, exact=True, self_contact=False))


def test_pinch_adjoint_matches_finite_differences(rng, monkeypatch):
    """Two proxy boxes shorter than a rod edge squeeze a rod (friction ``mu``, random) closed over 8 steps, then lift
    it: each holds it by one edge at its face's end, with a line contact's other end at the face's far end. The rod's
    nodes shifted (mostly along it) at the last step's start slide those points along the edge, which moves with the
    face's end; loss ``w . q`` one step later. Within 1% of ``|dL/dq| |shift|`` over 16 seeds (measured 0.5% at most;
    holding those points, as the adjoint did, up to 17%: 14 of 16 seeds fail)."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-3)
    T, dt, r, half = 16, 2.0e-3, 0.005, np.array([0.01, 0.009, 0.02])
    mu, tilt = rng.uniform(0.3, 1.0), rng.uniform(-0.1, 0.1)
    n = 6
    y = np.linspace(-0.1, 0.1, n)
    shift = np.array([0.3 * rng.normal(), 1.0, 0.3 * rng.normal()])

    def build():
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        bodies = []
        for s in (1.0, -1.0):
            body = b.add_body(xform=wp.transform(wp.vec3(s * (r + half[0] + 0.2 * r), 0.0, r), wp.quat_identity()))
            b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2],
                            cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=mu))
            bodies.append(body)
        SolverDiSMech.add_rod(b, rod=newton.Rod(np.column_stack([tilt * y, y, np.full(n, r)]), radius=r),
                              cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu), stretch_stiffness=1.0e4,
                              bend_stiffness=2.0e-3, bend_damping=1.0e-4)
        model = b.finalize()
        f = model.body_flags.numpy()
        f[bodies] |= int(BodyFlags.PROXY)
        model.body_flags.assign(f)
        return model, bodies

    def run(X, grad=False):
        model, bodies = build()
        sim = SolverDiSMech(model, iterations=40)
        states = [model.state(requires_grad=grad) for _ in range(T + 1)]
        for st in states:
            flatten_state(st)
        tape = wp.Tape() if grad else None
        with tape if grad else nullcontext():
            for t in range(T):
                bq, bqd = states[t].body_q.numpy(), states[t].body_qd.numpy()
                for body, s in zip(bodies, (1.0, -1.0), strict=True):
                    sq = 0.6 * r * min(t / 8, 1.0)
                    bq[body] = [s * (1.2 * r + half[0] - sq), 0.0, r + 0.05 * max(t - 10, 0) * dt, 0.0, 0.0, 0.0, 1.0]
                    bqd[body] = [-s * 0.6 * r / (8 * dt) if t < 8 else 0.0, 0.0, 0.05 if t >= 10 else 0.0, 0, 0, 0]
                states[t].body_q.assign(bq)
                states[t].body_qd.assign(bqd)
                if t == T - 1 and X != 0.0:
                    q = states[t].dismech.q.numpy()
                    q[: 3 * n].reshape(n, 3)[:] += X * shift
                    states[t].dismech.q.assign(q)
                sim.step(states[t], states[t + 1], None, None, dt)
        return states, tape

    states, tape = run(0.0, grad=True)
    w = rng.normal(size=states[T].dismech.q.shape[0])
    tape.backward(grads={states[T].dismech.q: wp.array(w.astype(np.float32), dtype=float)})
    g = states[T - 1].dismech.q.grad.numpy()[: 3 * n].reshape(n, 3).astype(np.float64)
    adjoint = float(g.sum(0) @ shift)
    scale = np.linalg.norm(g.sum(0)) * np.linalg.norm(shift)  # (a random loss's derivative can all but cancel)

    def loss(X):
        return run(X)[0][T].dismech.q.numpy().astype(np.float64) @ w

    h = 1.0e-4
    c = {e: (loss(e) - loss(-e)) / (2 * e) for e in (h, h / 2)}
    fd = (4 * c[h / 2] - c[h]) / 3
    assert_derivative(adjoint, fd, abs(c[h] - c[h / 2]), 1.0e-2, "rod shift", scale=scale)


@wp.kernel(enable_backward=False)
def _weights_kernel(q: wp.array[float], pairs: wp.array[wp.vec4i], w: wp.array[wp.vec4], out: wp.array[wp.vec4],
                    bary: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = _closest_weights(q, pairs[i], w[i])
    x = wp.vec3(q[0], q[1], q[2])
    a = wp.vec3(q[3], q[4], q[5])
    b = wp.vec3(q[6], q[7], q[8])
    c = wp.vec3(q[9], q[10], q[11])
    bary[i] = G.triangle_closest_point_barycentric(a, b, c, x)


def test_closest_weights_match_the_collision():
    """A node over a triangle's interior or one of its edges: the adjoint's weights (a projection onto the plane or
    line of the feature the step found) are the collision's closest point."""
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.05], [0.2, 0.9, -0.05]])
    for x, pair in (([0.4, 0.3, 0.2], [0, 1, 2, 3]), ([0.5, -0.2, 0.1], [0, 1, 2, -1]),
                    ([0.7, 0.8, 0.1], [0, -1, 2, 3])):
        q = wp.array(np.concatenate([x, tri.ravel()]), dtype=float)
        out, bary = wp.zeros(1, dtype=wp.vec4), wp.zeros(1, dtype=wp.vec3)
        guess = wp.vec4(1.0, *(-0.3 if p >= 0 else 0.0 for p in pair[1:]))  # only the signs matter
        wp.launch(_weights_kernel, dim=1, inputs=[q, wp.array([pair], dtype=wp.vec4i), wp.array([guess], dtype=wp.vec4)],
                  outputs=[out, bary])
        expected = np.concatenate([[1.0], -bary.numpy()[0]])
        np.testing.assert_allclose(out.numpy()[0], expected, atol=1.0e-5, err_msg=str(pair))


@wp.kernel
def _tactile_force_loss_kernel(f: wp.array[wp.vec3], out: wp.array[float]):
    out[0] = f[0][0] + 1.5 * f[0][1] - 0.1 * f[0][2]


def test_tactile_multistep_adjoint_matches_finite_differences(monkeypatch):
    """Multi-step ``Tactile.step(state_out, state_in)`` differentiability with respect to initial velocity and
    sensor shape friction across 2 sliding contact steps (CPU + CUDA)."""
    import sys
    from pathlib import Path

    examples_dir = str(Path(__file__).resolve().parents[1] / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from utils.tactile import Tactile  # noqa: PLC0415

    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-7)
    dt, r, n = 2.0e-3, 0.005, 5

    def run(vy, vz, mu, grad=False):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -50.0))
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.005), wp.quat_identity()))
        sensor_shape = b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.005,
                                       cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu))
        pts = np.column_stack([np.linspace(-0.02, 0.02, n), np.zeros(n), np.full(n, r * 1.01)])
        SolverDiSMech.add_rod(b, rod=newton.Rod(pts, radius=r),
                              cfg=newton.ModelBuilder.ShapeConfig(density=20000.0, mu=0.3),
                              stretch_stiffness=1.0e3, bend_stiffness=1.0e-3, bend_damping=1.0e-4)
        model = b.finalize()
        flags = model.body_flags.numpy()
        flags[body] |= int(BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        if grad:
            model.shape_material_mu.requires_grad = True
        sim = SolverDiSMech(model, iterations=200)
        tact = Tactile(model, sim, {"index_dip_link": sensor_shape})
        states = [model.state(requires_grad=grad) for _ in range(3)]
        for st in states:
            flatten_state(st)
        qd = states[0].particle_qd.numpy()
        qd[:, 0] = 2.5
        qd[:, 1] = vy
        qd[:, 2] = vz
        states[0].particle_qd.assign(qd)
        if grad:
            states[0].particle_qd.requires_grad = True
        loss_arr = wp.zeros(1, dtype=float, requires_grad=grad)
        tape = wp.Tape() if grad else None
        with tape if grad else nullcontext():
            for t in range(2):
                sim.step(states[t], states[t + 1], None, None, dt)
                tact.step(states[t + 1], states[t])
            wp.launch(_tactile_force_loss_kernel, dim=1, inputs=[tact.cur_total], outputs=[loss_arr])
        return float(loss_arr.numpy()[0]), tape, states[0].particle_qd, model.shape_material_mu, loss_arr

    vy0, vz0, mu0 = 1.0, -0.3, 0.3
    val, tape, qd0, shape_mu, loss_arr = run(vy0, vz0, mu0, grad=True)
    assert abs(val) > 1.0e-2, f"expected nonzero contact loss, got {val}"
    tape.backward(loss_arr)
    g_qd = qd0.grad.numpy()
    checks = [
        ("tactile dF/dvy", float(g_qd[:, 1].sum()), lambda e: run(vy0 + e, vz0, mu0)[0], 2.0e-2),
        ("tactile dF/dvz", float(g_qd[:, 2].sum()), lambda e: run(vy0, vz0 + e, mu0)[0], 5.0e-3),
        ("tactile dF/dmu", float(shape_mu.grad.numpy()[0]), lambda e: run(vy0, vz0, mu0 + e)[0], 2.0e-3),
    ]
    for label, ad, fn, h in checks:
        c = {e: (fn(e) - fn(-e)) / (2 * e) for e in (h, h / 2)}
        fd = (4 * c[h / 2] - c[h]) / 3
        assert_derivative(ad, fd, abs(c[h] - c[h / 2]), 2.0e-2, label)


@wp.kernel
def _tribology_loss_kernel(f: wp.array[wp.vec3], q: wp.array[wp.vec3], out: wp.array[float]):
    out[0] = f[0][0] + 0.7 * f[0][1] + 5.0 * (q[0][0] + q[2][1])


@pytest.mark.parametrize("alpha0,beta0", [(0.0, 0.0), (0.25, 0.35)])
def test_tribology_adjoint_matches_finite_differences(monkeypatch, alpha0, beta0):
    """Power-law load dependence ``(N/N0)^(-alpha)`` and directional anisotropy ``1 + beta (t . e)^2``: physical
    parallel/perpendicular force ratio and power-law scaling against closed-form theory, plus multi-step ``StepAdjoint``
    + ``Tactile`` gradients wrt ``(mu0, alpha, beta, vy0)`` against Richardson finite differences."""
    import sys
    from pathlib import Path

    examples_dir = str(Path(__file__).resolve().parents[1] / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from utils.tactile import Tactile  # noqa: PLC0415

    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-7)
    dt, r, n, n0 = 2.0e-3, 0.005, 5, 2.0

    def run(alpha, beta, mu, vx, vy, exact=True, grad=False):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -50.0))
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.005), wp.quat_identity()))
        sensor_shape = b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.005,
                                       cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu))
        pts = np.column_stack([np.linspace(-0.02, 0.02, n), np.zeros(n), np.full(n, r * 1.01)])
        SolverDiSMech.add_rod(b, rod=newton.Rod(pts, radius=r),
                              cfg=newton.ModelBuilder.ShapeConfig(density=20000.0, mu=0.35),
                              stretch_stiffness=1.0e3, bend_stiffness=1.0e-3, bend_damping=1.0e-4)
        model = b.finalize()
        flags = model.body_flags.numpy()
        flags[body] |= int(BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        if grad:
            model.shape_material_mu.requires_grad = True
        sim = SolverDiSMech(model, iterations=200, exact=exact, friction_alpha=alpha, friction_beta=beta,
                            friction_n0=n0, friction_dir=(1.0, 0.0, 0.0))
        tact = Tactile(model, sim, {"index_dip_link": sensor_shape})
        states = [model.state(requires_grad=grad) for _ in range(3)]
        for st in states:
            flatten_state(st)
        qd = states[0].particle_qd.numpy()
        qd[:, 0] = vx
        qd[:, 1] = vy
        qd[:, 2] = -0.3
        states[0].particle_qd.assign(qd)
        if grad:
            states[0].particle_qd.requires_grad = True
        loss_arr = wp.zeros(1, dtype=float, requires_grad=grad)
        tape = wp.Tape() if grad else None
        with tape if grad else nullcontext():
            for t in range(2):
                sim.step(states[t], states[t + 1], None, None, dt)
                tact.step(states[t + 1], states[t])
            wp.launch(_tribology_loss_kernel, dim=1, inputs=[tact.cur_total, states[2].particle_q], outputs=[loss_arr])
        return (
            float(loss_arr.numpy()[0]),
            tact.cur_total.numpy()[0].copy(),
            float(sim.contact.s.rho.numpy()[:n].sum()),
            tape,
            sim.contact.trib,
            model.shape_material_mu,
            states[0].particle_qd,
            loss_arr,
        )

    # Theory check on parallel vs perpendicular anisotropy + power-law scaling over baseline (alpha=0, beta=0)
    if alpha0 > 0.0:
        _, f_par, rho_par, *_ = run(alpha0, beta0, 0.4, 4.0, 0.0)
        _, f_perp, rho_perp, *_ = run(alpha0, beta0, 0.4, 0.0, 4.0)
        _, f_iso, rho_iso, *_ = run(0.0, 0.0, 0.4, 4.0, 0.0)
        np.testing.assert_allclose(rho_par, rho_perp, rtol=1.0e-3)
        np.testing.assert_allclose(abs(f_par[0]) / abs(f_perp[1]), 1.0 + beta0, rtol=5.0e-3)
        ratio = (abs(f_par[0]) / max(abs(f_iso[0]), 1.0e-12))
        expected_ratio = ((rho_par / n) / n0) ** (-alpha0) * (1.0 + beta0)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=2.0e-2)

    mu0, vx0, vy0 = 0.35, 2.5, 1.0
    val, _, _, tape, trib, shape_mu, qd0, loss_arr = run(alpha0, beta0, mu0, vx0, vy0, grad=True)
    assert abs(val) > 1.0e-2
    tape.backward(loss_arr)
    g_trib = trib.grad.numpy()
    checks = [
        ("trib dL/dalpha", float(g_trib[0]), lambda e: run(alpha0 + e, beta0, mu0, vx0, vy0)[0], 2.0e-3),
        ("trib dL/dbeta", float(g_trib[1]), lambda e: run(alpha0, beta0 + e, mu0, vx0, vy0)[0], 2.0e-3),
        ("trib dL/dmu", float(shape_mu.grad.numpy()[0]), lambda e: run(alpha0, beta0, mu0 + e, vx0, vy0)[0], 2.0e-3),
        ("trib dL/dvy", float(qd0.grad.numpy()[:, 1].sum()),
         lambda e: run(alpha0, beta0, mu0, vx0, vy0 + e)[0], 5.0e-3),
    ]
    for label, ad, fn, h in checks:
        c = {e: (fn(e) - fn(-e)) / (2 * e) for e in (h, h / 2)}
        fd = (4 * c[h / 2] - c[h]) / 3
        assert_derivative(ad, fd, abs(c[h] - c[h / 2]), 2.0e-2, label)


@wp.kernel
def _coupled_tactile_loss_kernel(tot: wp.array[wp.vec3], force: wp.array[wp.vec3], n_tax: int, out: wp.array[float]):
    s = tot[0][0] + 0.25 * tot[0][2]
    for k in range(n_tax):
        s += 0.2 * (force[k][0] + 0.5 * force[k][1] + 0.3 * force[k][2])
    out[0] = s


def test_coupled_tactile_and_actuator_adjoint(monkeypatch):
    """Multi-step coupled MuJoCo + DiSMech + ``Tactile`` adjoint wrt ``joint_target_q`` at both ``t=0`` and ``t=1``."""
    import sys
    from pathlib import Path

    from newton.solvers import SolverMuJoCo  # noqa: PLC0415
    from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy  # noqa: PLC0415

    examples_dir = str(Path(__file__).resolve().parents[1] / "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from utils import tactile  # noqa: PLC0415

    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-8)
    dt, r_rod, r_cap = 2.0e-3, 0.005, 0.01

    def run(acts, grad=False):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        b.add_ground_plane()
        pts = np.array([[-0.03, 0.0, r_rod], [0.0, 0.0, r_rod], [0.03, 0.0, r_rod]])
        caps, _ = SolverDiSMech.add_rod(b, rod=newton.Rod(pts, radius=r_rod),
                                        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5),
                                        stretch_stiffness=1.0e5, bend_stiffness=1.0e-2)
        SolverDiSMech.fix_segment(b, caps[0])
        car = b.add_link(label="carriage", mass=0.5, inertia=wp.mat33(np.eye(3) * 1.0e-3))
        jz = b.add_joint_prismatic(
            parent=-1, child=car, axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.003, 0.0, 2.0 * r_rod + r_cap - 2.0e-4), wp.quat_identity()),
            target_ke=20000.0, target_kd=200.0, armature=0.01, actuator_mode=newton.JointTargetMode.POSITION,
        )
        box = b.add_link(label="thumb_dip")
        sh = b.add_shape_capsule(
            box, radius=r_cap, half_height=0.015, label="thumb_dip_paxini",
            xform=wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)),
            cfg=newton.ModelBuilder.ShapeConfig(density=5000.0, mu=0.5),
        )
        jry = b.add_joint_revolute(
            parent=car, child=box, axis=wp.vec3(0.0, 1.0, 0.0),
            target_ke=200.0, target_kd=2.0, armature=0.001, actuator_mode=newton.JointTargetMode.POSITION,
        )
        arm_bodies, arm_joints = [car, box], [jz, jry]
        b.add_articulation(arm_joints)
        m = b.finalize()
        soft_bodies = [bi for bi in range(m.body_count) if bi not in arm_bodies]
        soft_joints = [ji for ji in range(m.joint_count) if ji not in arm_joints]
        coupled = SolverCoupledProxy(
            m,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(v, solver="newton", integrator="implicitfast",
                                                  use_mujoco_contacts=False, njmax=64, nconmax=32),
                    bodies=arm_bodies, joints=arm_joints,
                ),
                SolverCoupled.Entry(
                    name="soft",
                    solver=lambda v: SolverDiSMech(v, iterations=500, exact=True),
                    particles=list(range(m.particle_count)), bodies=soft_bodies, joints=soft_joints,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="mjc", destination="soft", bodies=[box], mode="lagged")],
                iterations=1,
            ),
        )
        tact = tactile.Tactile(m, coupled._entries["soft"].solver, {"thumb_dip": sh})
        tact.hook()
        pipeline = newton.CollisionPipeline(m)
        contacts = pipeline.contacts()
        states = [m.state(requires_grad=grad) for _ in range(3)]
        for st in states:
            flatten_state(st)
        newton.eval_fk(m, m.joint_q, m.joint_qd, states[0])
        ctrls = []
        for a in acts:
            c = m.control()
            arr = np.zeros(m.joint_coord_count, dtype=np.float32)
            arr[-2], arr[-1] = a[0], a[1]
            c.joint_target_q = wp.array(arr, dtype=float, device=m.device, requires_grad=grad)
            ctrls.append(c)
        loss_arr = wp.zeros(1, dtype=float, device=m.device, requires_grad=grad)
        tape = wp.Tape() if grad else None
        with tape if grad else nullcontext():
            for t, c in enumerate(ctrls):
                states[t].clear_forces()
                pipeline.collide(states[t], contacts)
                coupled.step(states[t], states[t + 1], c, contacts, dt)
                tact.commit(dt)
            tact.sample()
            wp.launch(_coupled_tactile_loss_kernel, dim=1,
                      inputs=[tact.total, tact.force, tact.count], outputs=[loss_arr], device=m.device)
        return float(loss_arr.numpy()[0]), tape, ctrls, loss_arr

    x0 = np.array([[-0.0005, 0.05], [-0.001, 0.10]], dtype=np.float64)
    val, tape, ctrls, loss_arr = run(x0, grad=True)
    assert abs(val) > 1.0
    tape.backward(loss_arr)
    g0 = ctrls[0].joint_target_q.grad.numpy()[-2:]
    g1 = ctrls[1].joint_target_q.grad.numpy()[-2:]

    def bump(t, k, e):
        x = x0.copy()
        x[t, k] += e
        return run(x)[0]

    checks = [
        ("coupled tactile t=0 jz", float(g0[0]), 0, 0, 2.0e-4),
        ("coupled tactile t=0 jry", float(g0[1]), 0, 1, 5.0e-3),
        ("coupled tactile t=1 jz", float(g1[0]), 1, 0, 2.0e-4),
        ("coupled tactile t=1 jry", float(g1[1]), 1, 1, 5.0e-3),
    ]
    for label, ad, t_idx, k_idx, h in checks:
        c = {e: (bump(t_idx, k_idx, e) - bump(t_idx, k_idx, -e)) / (2 * e) for e in (h, h / 2)}
        fd = (4 * c[h / 2] - c[h]) / 3
        assert_derivative(ad, fd, abs(c[h] - c[h / 2]), 2.0e-2, label)
