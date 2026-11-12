"""The step adjoint (no contact) against finite differences of a rollout, rods and midedge shells, and its CUDA graph
capture."""

import pytest
import warp as wp
from fd import assert_close
from rollouts import ClampedRollout, JunctionRollout, PinRollout, ShellRollout

from dismech_newton import solver

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
    drive turns the clamp's edges off their frames' tangents: untransported, the hand-written Jacobian assumed the
    frames orthonormal, and the error grew linearly with the drive (up to 14% over 16 seeds). ``midpoint``: the theta
    method's extrapolation, theta = 1/2."""
    r = ClampedRollout(rng)
    r.drive = drive
    r.check(options, fd_options={**EXACT, "theta": options.get("theta", 1.0)})


def test_junction_adjoint_matches_finite_differences(rng):
    """A rod graph: junction triplets, edges stored against them (their signs in the element Jacobian)."""
    JunctionRollout(rng).check(EXACT, fd_options=EXACT)


def test_shell_adjoint_matches_finite_differences(rng):
    """Membrane, bending and damping stiffness, rest strains, mass, velocity, the clamp and a force."""
    ShellRollout(rng).check(EXACT)


@pytest.mark.usefixtures("needs_cudss")
def test_gradient_graph_capture(rng):
    """A whole gradient evaluation (the solver's reset, rollout, loss, backward) in one CUDA graph: replays give the
    eager gradients to round-off, and do not accumulate. Without the reset, each rollout starts from the last one's
    warm start: unconverged (10 iterations), up to 0.6% off the first's (16 seeds). No contacts possible, so recording
    reads no count."""
    r = ClampedRollout(rng)
    run = r.setup({"iterations": 10, "self_contact": False}, grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    inputs = [*run.arrays.values(), run.states[0].dismech.q, run.states[0].dismech.qd]

    run.solver.reset(run.states[0])
    tape = r.record(run, loss)  # eager: the one-time host work (adjoint, cuDSS analysis)
    assert run.solver.graph_capturable
    expected = [a.grad.numpy().copy() for a in inputs]
    with wp.ScopedCapture() as capture:
        tape.zero()
        run.solver.reset(run.states[0])
        r.record(run, loss)
    for _ in range(2):
        wp.capture_launch(capture.graph)
    for a, g in zip(inputs, expected, strict=True):
        assert_close(a.grad.numpy(), g, 1.0e-6)


class DahlRollout(ClampedRollout):
    rod_options = dict(ClampedRollout.rod_options, dahl_eps_max=0.1, dahl_tau=0.05)


@pytest.mark.parametrize("options", [EXACT, {"iterations": 300, "self_contact": False}], ids=["exact", "admm"])
def test_dahl_adjoint_matches_finite_differences(rng, options):
    """With Dahl hysteresis (its tangent ``sigma_max / tau`` twice the bend stiffness): the stress is state, carried
    across steps by the exact law, and its adjoint with it, so gradients reach the stiffness through ``sigma_max``
    too. Over 16 seeds within 1e-3 (2026-11-01); an adjoint that holds the stresses frozen fails."""
    DahlRollout(rng).check(options, fd_options=EXACT)


def test_proxy_pin_adjoint_matches_finite_differences(rng, monkeypatch):
    """A rod pinned by a ball joint (:class:`~dismech_newton.coupling.ProxyPins`) to a free 6-DOF proxy body
    (:class:`~dismech_newton.coupling.ProxyArm`), chained across multiple steps: gradients wrt rod parameters, states,
    external force, and initial proxy body twist ``body_qd``."""
    monkeypatch.setattr(solver, "TOLERANCE", 1.0e-6)
    opts = {**EXACT, "iterations": 200}
    PinRollout(rng).check(opts, fd_options=opts)
