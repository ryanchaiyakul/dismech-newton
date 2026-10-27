"""The rod element: its Jacobian against central differences of its strains, with the reference frames current and
lagged (a step's start behind its iterate), and its strains at rest; a cantilever against beam theory."""

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_close

from dismech_newton import DiSMechSolver, flatten_state
from dismech_newton.builder import vec5f
from dismech_newton.rod import Rod, mat5x11f, rod_jacobian, rod_strain, vec5i, vec11f


# -O2: nvrtc -O3 reads the struct argument through a local-memory pointer (CUDA 700) for sm_75-sm_90 (Warp 1.17).
@wp.kernel(module="unique", module_options={"optimization_level": 2})
def _evaluate(x: wp.array[vec11f], v: Rod, strain: wp.array[vec5f], jac: wp.array[mat5x11f]):
    i = wp.tid()
    strain[i] = rod_strain(x[i], v, 0)
    eps, J = rod_jacobian(x[i], v, 0)
    jac[i] = J


def _perpendicular(t, rng):
    a = rng.normal(size=3)
    a -= (a @ t) * t
    return a / np.linalg.norm(a)


def _triplet(rng, lag: float):
    """DOFs ``[x0, theta_e, x1, theta_f, x2]`` of a bent, twisted triplet and its view: the start-of-step positions
    ``lag`` away (``0``: current frames), directors perpendicular to the start's tangents."""
    pts = np.array([[0.0, 0.0, 0.0], [0.11, 0.02, -0.01], [0.2, 0.08, 0.03]]) + 0.01 * rng.normal(size=(3, 3))
    old = pts + lag * 0.05 * rng.normal(size=(3, 3))
    x = np.concatenate([pts[0], [0.3], pts[1], [-0.2], pts[2]])
    v = Rod()
    v.q_old = wp.array(old.ravel(), dtype=float)
    d1 = [_perpendicular((old[k + 1] - old[k]) / np.linalg.norm(old[k + 1] - old[k]), rng) for k in range(2)]
    v.edge_d1_old = wp.array(d1, dtype=wp.vec3)
    v.ref_twist_old = wp.array([0.1], dtype=float)
    v.conn = wp.array([[0, 1, 0, 1, 2]], dtype=vec5i)
    v.edge_length = wp.array([0.1, 0.12], dtype=float)
    v.theta_dof_offset = 0  # unused: the element reads its gathered DOFs
    return x, v


def _evaluate_at(xs: np.ndarray, v: Rod):
    n = len(xs)
    strain, jac = wp.zeros(n, dtype=vec5f), wp.zeros(n, dtype=mat5x11f)
    wp.launch(_evaluate, dim=n, inputs=[wp.array(xs, dtype=vec11f), v], outputs=[strain, jac])
    return strain.numpy().astype(np.float64), jac.numpy().astype(np.float64)


@pytest.mark.parametrize("lag", [0.0, 1.0], ids=["current", "lagged"])
def test_jacobian_matches_finite_differences(rng, lag):
    x, v = _triplet(rng, lag)
    h = 1.0e-3
    xs = np.concatenate([x[None], x + h * np.eye(11), x - h * np.eye(11)])
    strain, jac = _evaluate_at(xs, v)
    fd = (strain[1:12] - strain[12:]).T / (2.0 * h)
    assert_close(jac[0], fd, 2.0e-3, "jacobian")


def test_strains_at_rest(rng):
    """A straight, untwisted triplet at its rest lengths has zero strains; stretching an edge by ``s`` gives
    ``s / l0``."""
    _, v = _triplet(rng, 0.0)
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.22, 0.0, 0.0]])
    v.q_old = wp.array(pts.ravel(), dtype=float)
    v.edge_d1_old = wp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3)
    v.ref_twist_old = wp.array([0.0], dtype=float)
    straight = np.concatenate([pts[0], [0.0], pts[1], [0.0], pts[2]])
    stretched = straight.copy()
    stretched[8] += 0.006
    strain, _ = _evaluate_at(np.stack([straight, stretched]), v)
    np.testing.assert_allclose(strain[0], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(strain[1], [0.0, 0.05, 0.0, 0.0, 0.0], atol=1.0e-5)


def test_cantilever_converges_to_beam_theory():
    """Tip deflection of a rod clamped by its first segment under its weight against ``w L^4 / (8 EI)`` (the clamp at
    the second node; ``bend_stiffness = EI / h``), deflection ~1% of the length: within 10%, closer on the finer mesh
    (1.07 and 1.012 on 2026-10-27)."""
    length, r, rho, E = 0.5, 0.005, 1000.0, 1.0e8
    area, EI = np.pi * r**2, E * np.pi * r**4 / 4.0
    ratios = []
    for n in (20, 40):
        h = length / n
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), length, segment_count=n, radius=r)
        bodies = DiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=rho),
                                       stretch_stiffness=E * area, bend_stiffness=EI / h, twist_stiffness=EI / h)
        DiSMechSolver.fix_segment(builder, bodies[0])
        model = builder.finalize()
        solver = DiSMechSolver(model, self_contact=False)
        s0, s1 = model.state(), model.state()
        flatten_state(s0)
        flatten_state(s1)
        for _ in range(200):  # implicit Euler's damping settles it
            solver.step(s0, s1, None, None, 0.02)
            s0, s1 = s1, s0
        tip = 1.0 - s0.particle_q.numpy()[-1, 2]
        ratios.append(tip / (rho * area * 9.81 * (length - h) ** 4 / (8.0 * EI)))
    assert 1.0 < ratios[1] < ratios[0] < 1.1, ratios
