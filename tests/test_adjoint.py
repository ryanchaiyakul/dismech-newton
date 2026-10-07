"""The step adjoint (no contact) against finite differences of a rollout, and its CUDA graph capture."""

import numpy as np
import pytest
import warp as wp
from fd import assert_close
from rollouts import ClampedRollout, SpinningRollout

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver
from dismech_newton.adjoint import StepAdjoint


@pytest.mark.parametrize(
    "solver_cls, options",
    [
        (DiSMechSolver, {"newton_tol": 1.0e-7}),
        (DiSMechSolver, {"newton_tol": 1.0e-7, "theta": 0.5}),
        (ADMMDiSMechSolver, {"tol": 1.0e-6, "iterations": 5000}),
        (ADMMDiSMechSolver, {"tol": 1.0e-6, "iterations": 5000, "theta": 0.5}),
    ],
    ids=["newton", "newton-midpoint", "admm", "admm-midpoint"],
)
def test_step_adjoint_matches_finite_differences(rng, solver_cls, options):
    """Every differentiable input, the clamp and the external force included. Differences use Newton-Raphson:
    the adjoint is of the exact root."""
    ClampedRollout(rng).check(solver_cls, options, rtol=1.0e-2, fd_solver_cls=DiSMechSolver,
                              fd_options={"newton_tol": 1.0e-7, "theta": options.get("theta", 1.0)})


@pytest.mark.parametrize("refine", [0, 1], ids=["solve", "refined"])
@pytest.mark.parametrize("theta", [1.0, 0.5], ids=["euler", "midpoint"])
def test_step_adjoint_solves_exact_transpose(monkeypatch, rng, theta, refine):
    """``|adj_q - J^T lam| / |adj_q|`` against the residual's tape ``J^T`` with the reference frames far from
    current: the Hessian ``A`` is ``J``, so one solve is exact to round-off."""
    monkeypatch.setattr(StepAdjoint, "refine", refine)
    r = SpinningRollout(rng)
    run = r.setup(DiSMechSolver, dict(newton_tol=1.0e-7, theta=theta), grad=True)
    r.simulate(run)
    before, after = run.states[-2:]
    after.dismech.q.grad.fill_(1.0)
    run.solver.vjp(before, after, r.dt)

    adj = run.solver._adjoint
    free = ~r.fixed
    b = adj._rhs.numpy()[free]
    adj._jt_lam(before, theta * r.dt)  # J^T of the lam the solve left
    assert np.linalg.norm(b - adj.q_theta.grad.numpy()[free]) / np.linalg.norm(b) < 2.0e-3


@pytest.mark.parametrize(
    "solver_cls, options",
    [
        (DiSMechSolver, {"newton_tol": 0.0, "newton_iterations": 5}),  # cuDSS captures without a tolerance
        (ADMMDiSMechSolver, {"tol": 1.0e-5}),
    ],
    ids=["newton", "admm"],
)
def test_gradient_graph_capture(request, rng, solver_cls, options):
    """A whole gradient evaluation (rollout, loss, backward) in one CUDA graph: replays give the eager
    gradients, and do not accumulate."""
    request.getfixturevalue("needs_cudss" if solver_cls is DiSMechSolver else "needs_cuda")
    r = ClampedRollout(rng)
    run = r.setup(solver_cls, options, grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    inputs = [*run.arrays.values(), run.states[0].dismech.q, run.states[0].dismech.qd]

    tape = r.record(run, loss)  # eager: the one-time host work (adjoint, cuDSS analysis, ADMM factorization)
    assert run.solver.graph_capturable
    expected = [a.grad.numpy().copy() for a in inputs]
    with wp.ScopedCapture() as capture:
        tape.zero()
        r.record(run, loss)
    for _ in range(2):
        wp.capture_launch(capture.graph)
    for a, g in zip(inputs, expected, strict=True):
        assert_close(a.grad.numpy(), g, 1.0e-3)


@pytest.mark.usefixtures("needs_cudss")
def test_graph_capturable_reports_cudss_loops(rng):
    """A tolerance iterates in a device-side loop, which cuDSS cannot be captured in."""
    model = ClampedRollout(rng).model(1, DiSMechSolver)
    assert not DiSMechSolver(model, newton_tol=1.0e-6).graph_capturable
    assert DiSMechSolver(model, newton_tol=0.0).graph_capturable
