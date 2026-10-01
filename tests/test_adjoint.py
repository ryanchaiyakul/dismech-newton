"""The step adjoint (:mod:`dismech_newton.adjoint`) against central finite differences of a rollout."""

import newton
import numpy as np
import pytest
import warp as wp

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, flatten_state

STEPS, DT = 5, 1.0e-2


@wp.kernel
def _dot(x: wp.array[float], w: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i] * w[i])


class Rollout:
    """A clamped rod thrown with a random velocity; ``loss = w . (q_T + dt qd_T)`` (``dt``: velocities
amplify the float32 rounding of positions by ``1 / dt``, which would drown the finite differences)."""

    def __init__(self, solver_cls, **options):
        builder = newton.ModelBuilder()
        rod = newton.Rod.create_straight((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.5, segment_count=10, radius=0.01)
        bodies = solver_cls.add_rod(builder, rod, stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0,
                                    bend_damping=0.01)
        solver_cls.fix_segment(builder, bodies[0])
        self.model = builder.finalize()
        self.solver = solver_cls(self.model, **options)
        self.params = self.model.dismech.triplet_params
        self.params.requires_grad = True
        self.k0 = self.params.numpy()
        self.states = [self.model.state(requires_grad=True) for _ in range(STEPS + 1)]
        for state in self.states:
            flatten_state(state)
        rng = np.random.default_rng(0)
        self.fixed = self.solver.fixed.numpy() != 0
        n = len(self.fixed)
        self.v0 = np.where(self.fixed, 0.0, 0.5 * rng.normal(size=n))
        self.q0 = np.concatenate([self.model.particle_q.numpy().ravel(), self.model.dismech.edge_q.numpy()])
        w = rng.normal(size=n)
        self.w = wp.array(w, dtype=float)
        self.w_dt = wp.array(DT * w, dtype=float)
        self.direction = np.where(self.fixed, 0.0, rng.normal(size=n))

    def loss(self, bend=1.0, dq=0.0, dv=0.0, grad=False):
        k = self.k0.copy()
        k[:, 2:4] *= bend
        self.params.assign(k)
        s0 = self.states[0]
        s0.dismech.q.assign(self.q0 + dq)
        s0.dismech.qd.assign(self.v0 + dv)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
            for t in range(STEPS):
                self.solver.step(self.states[t], self.states[t + 1], None, None, DT)
            end = self.states[STEPS].dismech
            wp.launch(_dot, dim=len(self.fixed), inputs=[end.q, self.w], outputs=[loss])
            wp.launch(_dot, dim=len(self.fixed), inputs=[end.qd, self.w_dt], outputs=[loss])
        if not grad:
            return float(loss.numpy()[0])
        tape.backward(loss)
        d = self.direction
        return (float(np.sum(self.params.grad.numpy()[:, 2:4] * k[:, 2:4])), float(s0.dismech.qd.grad.numpy() @ d),
                float(s0.dismech.q.grad.numpy() @ d))


def _derivative(f, h: float) -> float:
    """``f'(0)`` by Richardson-extrapolated central differences (steps ``h`` and ``h / 2``)."""
    def central(e):
        return (f(e) - f(-e)) / (2.0 * e)

    return (4.0 * central(0.5 * h) - central(h)) / 3.0


@pytest.mark.parametrize(
    "solver_cls, options",
    [
        (DiSMechSolver, {"newton_tol": 1.0e-7}),
        (DiSMechSolver, {"newton_tol": 1.0e-7, "theta": 0.5}),
        (ADMMDiSMechSolver, {"tol": 1.0e-6, "iterations": 5000}),
    ],
    ids=["newton", "newton-midpoint", "admm"],
)
def test_step_adjoint_matches_finite_differences(solver_cls, options):
    """The adjoint is the derivative of the step's exact root, so the finite differences are taken
    with Newton-Raphson (ADMM's own carry its truncation error)."""
    ad_bend, ad_qd, ad_q = Rollout(solver_cls, **options).loss(grad=True)
    r = Rollout(DiSMechSolver, newton_tol=1.0e-7, theta=options.get("theta", 1.0))
    d = r.direction
    fd_bend = _derivative(lambda e: r.loss(bend=1.0 + e), 0.2)  # a small effect over five steps: a large step
    fd_qd = _derivative(lambda e: r.loss(dv=e * d), 2.0e-2)
    fd_q = _derivative(lambda e: r.loss(dq=e * d), 6.0e-4)  # stiff stretching: nonlinear in positions
    for name, ad, fd in (("bend stiffness", ad_bend, fd_bend), ("velocity", ad_qd, fd_qd), ("position", ad_q, fd_q)):
        assert abs(ad - fd) <= 1.0e-2 * abs(fd), f"{name}: adjoint {ad:.6e}, finite differences {fd:.6e}"
