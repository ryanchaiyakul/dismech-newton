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


def _dahl_triplet(eps_max: float, tau: float):
    """One triplet of stiffness ``K = 1`` (edges 0.1 m), its first edge clamped, the tip node nearly massless."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rod = newton.Rod.create_straight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.2, segment_count=2, radius=0.01)
    DiSMechSolver.add_rod(builder, rod, cfg=newton.ModelBuilder.ShapeConfig(density=1.0), bend_stiffness=1.0,
                          collide=False, proxies=False, dahl_eps_max=eps_max, dahl_tau=tau)
    DiSMechSolver.fix_segment(builder, edge=0)
    model = builder.finalize()
    return model, DiSMechSolver(model, self_contact=False)


@wp.kernel
def _tip_force(force: wp.array[float], i: int, f: wp.array[wp.vec3]):
    f[2] = wp.vec3(0.0, 0.0, force[i])


@wp.kernel
def _record_kappa(strain: wp.array[vec5f], i: int, out: wp.array[float]):
    out[i] = -strain[0][2]


def _dahl_cycle(eps_max, tau, force, dt):
    """The curvature ``kappa = -kappa_1`` (the turning angle's ``2 tan(phi / 2)``) under a tip force ``force[i]``
    along +z at the end of step ``i`` (no host sync per step)."""
    model, solver = _dahl_triplet(eps_max, tau)
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    f, out = wp.array(force, dtype=float), wp.zeros(len(force), dtype=float)
    for i in range(len(force)):
        wp.launch(_tip_force, dim=1, inputs=[f, i], outputs=[s0.particle_f])
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        wp.launch(_record_kappa, dim=1, inputs=[s0.dismech.triplet_strain_q, i], outputs=[out])
    return out.numpy().astype(np.float64)


def _dahl(d, sigma, sigma_max, tau):
    s = np.sign(d) if abs(d) > 1.0e-6 else (-1.0 if sigma < 0.0 else 1.0)
    e = np.exp(-s * d / tau)
    return s * sigma_max * (1.0 - e) + sigma * e


def _dahl_quasistatic(force, eps_max, tau, l=0.1):
    """The exact quasi-static loop: per sample, ``F l cos(phi) / (1 + kappa^2 / 4) = kappa + sigma(kappa)`` (the
    moment about the clamp over ``d kappa / d phi``, ``K = 1``), ``sigma`` the Dahl law from the last sample."""
    kappa, sigma, out = 0.0, 0.0, []
    for f in force:
        def residual(x, kappa=kappa, sigma=sigma, f=f):
            return f * l * np.cos(2.0 * np.arctan(0.5 * x)) / (1.0 + 0.25 * x * x) - x - _dahl(x - kappa, sigma,
                                                                                                 eps_max, tau)
        lo, hi = -2.0, 2.0
        for _ in range(60):
            lo, hi = ((lo + hi) / 2, hi) if residual((lo + hi) / 2) > 0.0 else (lo, (lo + hi) / 2)
        sigma, kappa = _dahl(lo - kappa, sigma, eps_max, tau), lo
        out.append(kappa)
    return np.array(out)


@pytest.mark.parametrize("period", [1.0, 2.0])
def test_dahl_loop_matches_closed_form(period):
    """Newton's Dahl hysteresis on one triplet under a tip force ``3 sin(2 pi t / T)`` N (moment ~0.3 K), without
    damping: the curvature against the exact quasi-static loop, whose branches are exponentials toward
    ``+-sigma_max = +-K eps_max`` at rate ``1 / tau``. Rate independent: the same loop at either period ``T``, its
    area (moment over curvature, the last full cycle) and its curvature at zero force (the set) within 3%, the
    curvature within 2% of its peak. Over a step the Dahl stress is linearized with the branches' mean tangent
    ``sigma_max / tau`` (an implicit damping ``dt sigma_max / tau``): at 2000 steps per period, area +2.6% (T 1 s,
    dt 1e-3; frozen sigma breaks down there: 0.003 for 0.057), +1.3% (dt/T 5e-4) and +0.6% (2.5e-4)
    (2026-11-01)."""
    eps_max, tau, dt = 0.1, 0.05, 5.0e-4
    t = dt * np.arange(1, int(round(1.25 * period / dt)) + 1)
    force = 3.0 * np.sin(2.0 * np.pi * t / period)
    kappa, exact = _dahl_cycle(eps_max, tau, force, dt), _dahl_quasistatic(force, eps_max, tau)
    assert_close(kappa, exact, 2.0e-2, "curvature")
    cycle = t > 0.25 * period
    moment = [0.1 * force[cycle] * np.cos(2.0 * np.arctan(0.5 * k[cycle])) / (1.0 + 0.25 * k[cycle] ** 2)
              for k in (kappa, exact)]
    area = [0.5 * abs(k[cycle] @ np.roll(m, -1) - np.roll(k[cycle], -1) @ m) for k, m in zip((kappa, exact), moment)]
    assert abs(area[0] / area[1] - 1.0) < 0.03, area
    zero = np.argmin(np.abs(t - 0.5 * period))  # the force back to zero after the first peak: the set
    assert abs(kappa[zero] / exact[zero] - 1.0) < 0.03, (kappa[zero], exact[zero])
    assert exact[zero] > 0.5 * eps_max  # a set of order eps_max: the loop is open


def test_dahl_off_is_elastic():
    """Dahl is off where ``eps_max`` or ``tau`` is 0: the trajectory bit-identical to a rod without it."""
    force = 3.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, 200))
    runs = [_dahl_cycle(eps_max, tau, force, 1.0e-3) for eps_max, tau in ((0.0, 0.0), (0.0, 0.05), (0.1, 0.0))]
    assert np.abs(runs[0]).max() > 0.1
    for run in runs[1:]:
        assert np.array_equal(run, runs[0])
