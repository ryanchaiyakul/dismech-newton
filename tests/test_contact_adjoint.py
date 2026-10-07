"""The contact adjoint (ADMM, ground and rod-rod) against finite differences of a rollout."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp
from fd import assert_close
from rollouts import CrossingRollout, GroundRollout, contact_options

from dismech_newton import ADMMDiSMechSolver
from dismech_newton.adjoint import StepAdjoint, _ContactSystem

SMOOTHING = pytest.mark.parametrize("smoothing", [0.0, 1.0e-3], ids=["exact", "smoothed"])


@pytest.mark.parametrize(
    "mu, v_mean, checks",
    [
        (0.3, (0.3, 0.1, 0.0), ("mass", "friction", "velocity", "position")),
        (0.0, (0.3, 0.1, 0.0), ("mass", "velocity", "position")),
        # Stuck: a stretching perturbation would break the stick (its linear range is ~1e-6 m).
        (2.0, (0.0, 0.0, 0.0), ("velocity", "translation")),
    ],
    ids=["slide", "frictionless", "stick"],
)
@SMOOTHING
def test_ground_adjoint_matches_finite_differences(request, device, rng, mu, v_mean, checks, smoothing):
    """Perturbations stay clear of the stick/slip and on/off switches. The adjoint is exact to ~1e-5 here; the
    tolerance is the differences' round-off: ~1e-2 for a freely sliding (frictionless) rod, ~2% for the small
    mass derivative (~3% on the CPU, whose float32 rounding moves the differences, not the adjoint)."""
    if device.is_cpu and mu == 0.3 and smoothing == 0.0:
        # Every interior node has two ground contacts (both capsules' end spheres): the same constraint twice
        # leaves the contact adjoint system singular. cuDSS still solves it; SciPy's regularized fallback does not.
        request.applymarker(pytest.mark.xfail(reason="duplicate ground contacts: singular adjoint system on the CPU"))
    GroundRollout(rng, v_mean, checks).check(ADMMDiSMechSolver, contact_options(mu, smoothing), rtol=3.0e-2,
                                             rtols={"mass": 5.0e-2} if device.is_cpu else None,
                                             wrt_friction="friction" in checks)


@pytest.mark.parametrize("mu", [0.0, 0.3], ids=["frictionless", "slide"])
@SMOOTHING
def test_rod_contact_adjoint_matches_finite_differences(rng, mu, smoothing):
    """Rod-rod contact, whose closest points and normal the step takes from ``q_in``: tilting the top rod turns
    the normal (missing that dependence was a 2.5-3% error). The mass and random-position derivatives are small
    against their differences' round-off (~4e-3; they converge toward the adjoint only at larger steps, beyond
    which the contact turns nonlinear)."""
    CrossingRollout(rng).check(ADMMDiSMechSolver, contact_options(mu, smoothing), rtol=2.0e-3,
                               rtols={"mass": 1.0e-2, "position": 1.0e-2}, wrt_friction=mu > 0.0)


def test_contact_adjoint_refinement_agrees(monkeypatch, rng):
    """Refining the contact system against the exact ``J^T`` leaves the solve's gradients (round-off apart)."""
    r = GroundRollout(rng, (0.3, 0.1, 0.0), ("mass", "velocity", "position", "translation"))
    options = contact_options(0.3)
    solved = r.adjoint(ADMMDiSMechSolver, **options)
    monkeypatch.setattr(StepAdjoint, "refine", 2)
    refined = r.adjoint(ADMMDiSMechSolver, **options)
    for name in solved:
        assert_close(refined[name], solved[name], 1.0e-3, name, scale=max(abs(solved[name]), 1.0e-12))


def test_contact_pattern_ignores_slot_order(rng):
    """The narrow phase fills contact slots in a varying order (across worlds): the same contacts in other slots
    must reuse the cached pattern (a new one costs a cuDSS analysis) and give the same adjoint."""
    r = GroundRollout(rng, (0.3, 0.1, 0.0), ("velocity",))
    r.steps = 1
    run = r.setup(ADMMDiSMechSolver, contact_options(0.3), grad=True)
    r.record(run)
    adjoint, snapshot, state_in = run.solver._adjoint, run.solver.contact_snapshot(), run.states[0]
    h = run.solver.theta * r.dt

    def solve(snap):
        system = _ContactSystem.build(adjoint, state_in, h, snap)
        system.solve(state_in, h)
        return system.pattern, adjoint._lam.numpy().copy()

    perm = rng.permutation(snapshot.count)
    permuted = replace(snapshot, **{f: wp.array(getattr(snapshot, f).numpy()[perm], dtype=getattr(snapshot, f).dtype)
                                    for f in ("pairs", "bary", "normal", "anchor", "shift", "thickness",
                                              "force", "rho")})
    assert snapshot.count > 1
    pattern, lam = solve(snapshot)
    pattern_permuted, lam_permuted = solve(permuted)
    assert pattern_permuted is pattern
    np.testing.assert_allclose(lam_permuted, lam, rtol=1.0e-5, atol=1.0e-6 * np.abs(lam).max())
