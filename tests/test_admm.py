"""ADMMDiSMechSolver: new masses refactorise in place, so a step captured in a CUDA graph stays valid."""

import newton
import numpy as np
import pytest
import warp as wp

from dismech_newton import ADMMDiSMechSolver, add_rod, fix_segment, flatten_state


def bent_rod():
    """A clamped rod, its tip lifted: released, it moves by its elastic forces over its masses."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.5, segment_count=8, radius=0.01)
    ids = add_rod(builder, rod, stretch_stiffness=1.0e4, bend_stiffness=1.0, twist_stiffness=1.0, proxies=False)
    fix_segment(builder, edge=ids[0])
    model = builder.finalize()
    state = model.state()
    flatten_state(state)
    q = state.dismech.q.numpy()
    q[3 * (model.particle_count - 1) + 2] += 0.02
    state.dismech.q.assign(q)
    return model, state


def step_from(solver, model, state_in):
    out = model.state()
    flatten_state(out)
    solver.reset(state_in)
    solver.step(state_in, out, None, None, 1.0 / 240.0)
    return out.dismech.q.numpy()


@pytest.mark.usefixtures("needs_cuda")
@pytest.mark.parametrize("linear_solver", ["auto", "dense", "cudss"])
def test_refresh_mass_keeps_a_captured_step(linear_solver):
    """A step captured before ``refresh_mass`` = a new solver's step at the new masses."""
    model, start = bent_rod()
    if linear_solver == "cudss":
        pytest.importorskip("nvmath")
    options = dict(linear_solver=linear_solver, tol=0.0 if linear_solver == "cudss" else 1.0e-4)  # cuDSS: no loop
    solver = ADMMDiSMechSolver(model, **options)
    out = model.state()
    flatten_state(out)
    step_from(solver, model, start)  # the solver sets itself up (factorizes) eagerly
    assert solver.graph_capturable
    solver.reset(start)
    with wp.ScopedCapture() as capture:
        solver.step(start, out, None, None, 1.0 / 240.0)
    solver.reset(start)
    wp.capture_launch(capture.graph)
    before = out.dismech.q.numpy()

    model.particle_mass.assign(model.particle_mass.numpy() * 3.0)
    solver.refresh_mass()
    solver.reset(start)
    wp.capture_launch(capture.graph)
    replayed = out.dismech.q.numpy()
    fresh = step_from(ADMMDiSMechSolver(model, **options), model, start)

    moved = np.abs(fresh - before).max()
    assert moved > 1.0e-5, f"the masses barely matter here ({moved:.1e})"
    err = np.abs(replayed - fresh).max()
    assert err < 1.0e-3 * moved, f"replayed step off the new masses' by {err:.1e} (the change: {moved:.1e})"
