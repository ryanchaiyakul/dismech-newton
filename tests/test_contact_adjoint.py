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
from rollouts import CrossingRollout, GroundRollout, SheetContactRollout, SphereRollout

from dismech_newton import SolverDiSMech, flatten_state, solver
from dismech_newton.adjoint import _closest_weights

# Converged: 12 step lengths, as a step from beyond the barrier's onset can need far shorter steps than 4 give.
OPTIONS = {"iterations": 500}
# A two-sided pair's majorized A converges the pair's common motion slowly.
TWO_SIDED = pytest.mark.xfail(strict=True, reason="a two-sided pair's step converges too slowly for the differences")


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
    steps, so no node moves half a radius in one (the pairs found again mid-step), and converged further: the pair's
    common motion converges slowly (:data:`TWO_SIDED`). Tilt and yaw turn the normal; a random position's stretching
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
@pytest.mark.parametrize("scene", ["ground", pytest.param("sphere", marks=TWO_SIDED),
                                   pytest.param("foot", marks=TWO_SIDED), pytest.param("edge", marks=TWO_SIDED)])
def test_sheet_contact_adjoint_matches_finite_differences(rng, scene, mu):
    """A sheet's vertex on the ground and on a sphere, a rod's node on a triangle, a rod's edge across the sheet's
    edge. Solved exactly: ADMM's elastic splitting leaves float32 noise that a light sheet's differences do not
    resolve."""
    SheetContactRollout(rng, scene, mu).check(dict(OPTIONS, exact=True))


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


@wp.kernel
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
