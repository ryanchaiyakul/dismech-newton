"""What holds whatever the elements: a free body's momentum and energy, and the same rollout on both devices."""

import newton
import numpy as np
import pytest
import warp as wp
from rollouts import ClampedRollout, cloth, grid

from dismech_newton import DiSMechSolver, flatten_state
from dismech_newton.state import dof_constants


def _free(kind: str):
    """A straight rod or a flat sheet, no gravity, damping or contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if kind == "rod":
        rod = newton.Rod.create_straight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5, segment_count=10, radius=0.01)
        DiSMechSolver.add_rod(builder, rod, stretch_stiffness=1.0e3, bend_stiffness=0.05, twist_stiffness=0.05)
    else:
        v, f = grid(5, 4, 0.3, 0.2)
        cloth(builder, v, f, density=0.5, tri_ke=2.0e3, tri_ka=1.0e3, edge_ke=1.0e-2, damping=0.0)
    return builder.finalize()


@pytest.mark.parametrize(
    "kind, options",
    [("rod", {}), ("rod", {"exact": True, "iterations": 30, "theta": 0.5}), ("sheet", {}),
     ("sheet", {"exact": True, "iterations": 30})],
    ids=["rod-admm", "rod-midpoint", "sheet-admm", "sheet-exact"],
)
def test_free_body_keeps_momentum_and_loses_energy(rng, kind, options):
    """Thrown at random from rest shape (no potential energy) for 100 steps: the internal forces cancel, so the linear
    momentum stays to float32 round-off (5e-5 to 3e-4 of it), and the kinetic energy never exceeds the start's. A
    sheet on midpoint is left out: it gains energy (1.7x here, more when thrown harder), the midpoint rule's
    nonlinear instability on a membrane."""
    model = _free(kind)
    solver = DiSMechSolver(model, self_contact=False, **options)
    mass = dof_constants(model)[0].numpy().astype(np.float64)
    nd = 3 * model.particle_count
    s0, s1 = model.state(), model.state()
    flatten_state(s0)
    flatten_state(s1)
    qd0 = 0.3 * rng.normal(size=mass.size)
    s0.dismech.qd.assign(qd0)
    p0, e0 = (mass[:nd] * qd0[:nd]).reshape(-1, 3).sum(0), 0.5 * mass @ qd0**2
    for _ in range(100):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        qd = s0.dismech.qd.numpy().astype(np.float64)
        p = (mass[:nd] * qd[:nd]).reshape(-1, 3).sum(0)
        assert np.linalg.norm(p - p0) < 1.0e-3 * np.linalg.norm(p0), (p, p0)
        assert 0.5 * mass @ qd**2 <= 1.001 * e0


@pytest.mark.usefixtures("needs_cuda")
def test_devices_agree(rng):
    """A clamped rod's rollout (Gauss-Newton to convergence) on the CPU and on CUDA: the same to float32 round-off
    (1.8e-7 on 2026-10-27), so a device's kernels cannot go wrong alone."""
    out = []
    seed = rng.integers(1 << 31)
    for device in ("cpu", "cuda"):
        with wp.ScopedDevice(device):
            r = ClampedRollout(np.random.default_rng(seed))
            run = r.setup({"exact": True, "iterations": 30, "self_contact": False}, grad=False)
            r.simulate(run)
            out.append(np.concatenate([s.dismech.q.numpy() for s in run.states[1:]]))
    assert np.abs(out[0] - out[1]).max() < 1.0e-5 * np.abs(out[1]).max()
