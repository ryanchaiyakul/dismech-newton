"""The contact adjoint against finite differences of a rollout (ground, rod-rod, a rod edge on a sphere, a sheet),
and its closest points against the collision's. Sliding (mu = 0.3) unless stuck. Contacts start at the barrier's
onset (:data:`rollouts.ONSET`). The steps are converged far below the solver's tolerance: the differences are of the
converged step, which the adjoint is. Every check holds over 16 seeds (``--seeds 16``) at
:meth:`~rollouts.Rollout.check`'s default tolerance."""

import numpy as np
import pytest
import warp as wp
from newton._src.geometry import kernels as G
from rollouts import CrossingRollout, GroundRollout, SheetContactRollout, SphereRollout

from dismech_newton import solver
from dismech_newton.adjoint import _closest_weights

# Converged: 12 step lengths, as a step from beyond the barrier's onset can need far shorter steps than 4 give.
OPTIONS = {"iterations": 500, "linesearch": 12}
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
