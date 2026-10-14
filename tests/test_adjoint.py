"""The step adjoint (no contact) against finite differences of a rollout, rods and midedge shells, and its CUDA graph
capture."""

import pytest
import warp as wp
from fd import assert_close
from rollouts import ClampedRollout, ShellRollout

# The differences: Gauss-Newton to convergence. No self-contact: a rollout's copies overlap.
EXACT = {"exact": True, "iterations": 30, "self_contact": False}


@pytest.mark.parametrize(
    "options, drive",
    [(EXACT, 0.0), ({"iterations": 300, "self_contact": False}, 0.0), (EXACT, 2.0e-3),
     ({**EXACT, "theta": 0.5}, 0.0)],
    ids=["exact", "admm", "driven", "midpoint"],
)
def test_step_adjoint_matches_finite_differences(rng, options, drive):
    """Every differentiable input, the clamp and the external force included; the element Hessians are Warp's
    reverse mode through their Jacobians. ``driven``: the clamp moves between steps, written into each step's input,
    which is the last step's output (a step's fixed DOFs are its input's: reading the output's was a 20% error). The
    moved clamp leaves its neighbour's frame off its tangent, where the rod's hand-written Jacobian is not exactly the
    strain's: ~1e-3 per step, up to 4% here. ``midpoint``: the theta method's extrapolation, theta = 1/2."""
    r = ClampedRollout(rng)
    r.drive = drive
    r.check(options, rtol=6.0e-2 if drive else 1.0e-2, fd_options={**EXACT, "theta": options.get("theta", 1.0)})


def test_shell_adjoint_matches_finite_differences(rng):
    """Membrane, bending and damping stiffness, rest strains, mass, velocity, the clamp and a force. A random
    position's derivative cancels ~200x over the DOFs: its float32 round-off is a few percent."""
    ShellRollout(rng).check(EXACT, rtol=1.0e-2, rtols={"position": 6.0e-2})


@pytest.mark.usefixtures("needs_cudss")
def test_gradient_graph_capture(rng):
    """A whole gradient evaluation (rollout, loss, backward) in one CUDA graph: replays give the eager gradients,
    and do not accumulate. No contacts possible, so recording reads no count."""
    r = ClampedRollout(rng)
    run = r.setup({"iterations": 10, "self_contact": False}, grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    inputs = [*run.arrays.values(), run.states[0].dismech.q, run.states[0].dismech.qd]

    tape = r.record(run, loss)  # eager: the one-time host work (adjoint, cuDSS analysis)
    assert run.solver.graph_capturable
    expected = [a.grad.numpy().copy() for a in inputs]
    with wp.ScopedCapture() as capture:
        tape.zero()
        r.record(run, loss)
    for _ in range(2):
        wp.capture_launch(capture.graph)
    for a, g in zip(inputs, expected, strict=True):
        assert_close(a.grad.numpy(), g, 1.0e-3)
