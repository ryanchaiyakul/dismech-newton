"""The step adjoint (:mod:`dismech_newton.adjoint`) against central finite differences of a rollout."""

import newton
import numpy as np
import pytest
import warp as wp

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, flatten_state
from dismech_newton.adjoint import suspended_tape

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
        d = self.model.dismech
        self.params = d.triplet_params
        self.rest = self.solver.triplets.rest
        self.length = d.edge_length
        self.particle_mass = self.model.particle_mass
        self.inertia = d.edge_inertia
        for a in (self.params, self.rest, self.length, self.particle_mass, self.inertia):
            a.requires_grad = True
        self.k0, self.rest0, self.length0 = self.params.numpy(), self.rest.numpy(), self.length.numpy()
        self.particle_mass0, self.inertia0 = self.particle_mass.numpy(), self.inertia.numpy()
        self.states = [self.model.state(requires_grad=True) for _ in range(STEPS + 1)]
        for state in self.states:
            flatten_state(state)
        rng = np.random.default_rng(0)
        self.fixed = self.solver.fixed.numpy() != 0
        n = len(self.fixed)
        self.v0 = np.where(self.fixed, 0.0, 0.5 * rng.normal(size=n))
        self.q0 = np.concatenate([self.model.particle_q.numpy().ravel(), self.model.dismech.edge_q.numpy()])
        w = rng.normal(size=n)
        self.w64 = w
        self.w = wp.array(w, dtype=float)
        self.w_dt = wp.array(DT * w, dtype=float)
        self.direction = np.where(self.fixed, 0.0, rng.normal(size=n))
        self.rest_direction = rng.normal(size=self.rest0.shape)
        self.length_direction = rng.normal(size=self.length0.shape)  # relative
        self.particle_mass_direction = rng.normal(size=self.particle_mass0.shape)  # relative
        self.inertia_direction = rng.normal(size=self.inertia0.shape)  # relative

    def loss(self, bend=1.0, rest=0.0, length=0.0, mass=0.0, dq=0.0, dv=0.0, grad=False):
        k = self.k0.copy()
        k[:, 2:4] *= bend
        self.params.assign(k)
        self.rest.assign(self.rest0 + rest * self.rest_direction)
        self.length.assign(self.length0 * (1.0 + length * self.length_direction))
        self.particle_mass.assign(self.particle_mass0 * (1.0 + mass * self.particle_mass_direction))
        self.inertia.assign(self.inertia0 * (1.0 + mass * self.inertia_direction))
        self.solver.refresh_mass()
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
            # In float64: the float32 atomic sum rounds in a run-dependent order, by more than the
            # finite differences resolve.
            q, qd = end.q.numpy().astype(np.float64), end.qd.numpy().astype(np.float64)
            return float(self.w64 @ (q + DT * qd))
        tape.backward(loss)
        d = self.direction
        return {
            "bend stiffness": float(np.sum(self.params.grad.numpy()[:, 2:4] * k[:, 2:4])),
            "rest strain": float(np.sum(self.rest.grad.numpy() * self.rest_direction)),
            "rest length": float(self.length.grad.numpy() @ (self.length0 * self.length_direction)),
            "mass": float(self.particle_mass.grad.numpy() @ (self.particle_mass0 * self.particle_mass_direction)
                          + self.inertia.grad.numpy() @ (self.inertia0 * self.inertia_direction)),
            "velocity": float(s0.dismech.qd.grad.numpy() @ d),
            "position": float(s0.dismech.q.grad.numpy() @ d),
        }


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
    adjoint = Rollout(solver_cls, **options).loss(grad=True)
    r = Rollout(DiSMechSolver, newton_tol=1.0e-7, theta=options.get("theta", 1.0))
    d = r.direction
    fd = {
        "bend stiffness": _derivative(lambda e: r.loss(bend=1.0 + e), 0.2),  # a small effect over five steps
        "rest strain": _derivative(lambda e: r.loss(rest=e), 5.0e-2),  # nonlinear in curvature
        "rest length": _derivative(lambda e: r.loss(length=e), 8.0e-3),  # a small effect: float32 noise below
        "mass": _derivative(lambda e: r.loss(mass=e), 0.1),
        "velocity": _derivative(lambda e: r.loss(dv=e * d), 2.0e-2),
        "position": _derivative(lambda e: r.loss(dq=e * d), 6.0e-4),  # stiff stretching: nonlinear in positions
    }
    for name, f in fd.items():
        ad = adjoint[name]
        assert abs(ad - f) <= 1.0e-2 * abs(f), f"{name}: adjoint {ad:.6e}, finite differences {f:.6e}"


class GroundRollout:
    """A rod lying on the ground (ADMM, contact and friction), its nodes sliding with ``v_mean``
    plus a little noise, never upwards; the loss as in :class:`Rollout`. Every evaluation builds a
    fresh solver, so ADMM's warm starts do not carry over between the finite differences."""

    radius = 0.01

    def __init__(self, mu: float, v_mean, smoothing: float):
        self.mu, self.smoothing = mu, smoothing
        model = self._build()
        nd = 3 * model.particle_count
        n = nd + model.dismech.edge_q.shape[0]
        rng = np.random.default_rng(0)
        v = np.zeros(n)
        v[:nd] = np.tile(v_mean, model.particle_count) + 0.02 * rng.normal(size=nd)
        v[2:nd:3] = 0.0
        v[nd:] = 0.5 * rng.normal(size=n - nd)
        self.v0 = v
        self.q0 = np.concatenate([model.particle_q.numpy().ravel(), model.dismech.edge_q.numpy()])
        self.w = rng.normal(size=n)
        self.direction = rng.normal(size=n)
        self.direction[2:nd:3] = 0.0  # in the ground's plane
        self.rigid = np.zeros(n)  # a horizontal translation, which keeps a stuck rod stuck
        self.rigid[0:nd:3], self.rigid[1:nd:3] = 0.6, -0.8
        self.m0 = model.particle_mass.numpy()
        self.mass_direction = rng.normal(size=self.m0.shape)

    def _build(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        rod = newton.Rod.create_straight((0.0, 0.0, self.radius), (1.0, 0.0, 0.0), 0.5, segment_count=10,
                                         radius=self.radius)
        ADMMDiSMechSolver.add_rod(builder, rod, stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
        return builder.finalize()

    def loss(self, mass=0.0, mu=0.0, dq=0.0, dv=0.0, grad=False):
        model = self._build()
        model.particle_mass.assign(self.m0 * (1.0 + mass * self.mass_direction))
        solver = ADMMDiSMechSolver(model, tol=1.0e-7, iterations=20000, friction=self.mu + mu,
                                   contact_smoothing=self.smoothing)
        solver.refresh_mass()
        pipeline = newton.CollisionPipeline(model, soft_contact_max=0, verify_buffers=False,
                                            speculative_contact_gap_max=2.0 * self.radius, contact_matching="latest")
        contacts = pipeline.contacts()
        model.particle_mass.requires_grad = True
        solver.contact.friction.requires_grad = True
        states = [model.state(requires_grad=True) for _ in range(STEPS + 1)]
        for state in states:
            flatten_state(state)
        s0 = states[0].dismech
        s0.q.assign(self.q0 + dq)
        s0.qd.assign(self.v0 + dv)
        n = len(self.w)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
            for t in range(STEPS):
                with suspended_tape():
                    pipeline.collide(states[t], contacts, dt=2.0 * DT)
                solver.step(states[t], states[t + 1], None, contacts, DT)
            end = states[STEPS].dismech
            wp.launch(_dot, dim=n, inputs=[end.q, wp.array(self.w, dtype=float)], outputs=[loss])
            wp.launch(_dot, dim=n, inputs=[end.qd, wp.array(DT * self.w, dtype=float)], outputs=[loss])
        if not grad:
            q, qd = end.q.numpy().astype(np.float64), end.qd.numpy().astype(np.float64)
            return float(self.w @ (q + DT * qd))
        assert solver.contact.active.numpy().sum() > 0
        tape.backward(loss)
        return {
            "mass": float(model.particle_mass.grad.numpy() @ (self.m0 * self.mass_direction)),
            "friction": float(solver.contact.friction.grad.numpy()[0]),
            "velocity": float(s0.qd.grad.numpy() @ self.direction),
            "position": float(s0.q.grad.numpy() @ self.direction),
            "translation": float(s0.q.grad.numpy() @ self.rigid),
        }


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
@pytest.mark.parametrize("smoothing", [0.0, 1.0e-3], ids=["exact", "smoothed"])
def test_contact_adjoint_matches_finite_differences(mu, v_mean, checks, smoothing):
    """Sliding, frictionless and stuck contact with the ground, against central finite differences
    of ADMM; the perturbations keep every contact on its side of the stick/slip and on/off switches."""
    r = GroundRollout(mu, v_mean, smoothing)
    adjoint = r.loss(grad=True)
    fd = {
        "mass": lambda: _derivative(lambda e: r.loss(mass=e), 0.1),
        "friction": lambda: _derivative(lambda e: r.loss(mu=e), 0.05),
        "velocity": lambda: _derivative(lambda e: r.loss(dv=e * r.direction), 4.0e-2),  # small in the stuck case
        "position": lambda: _derivative(lambda e: r.loss(dq=e * r.direction), 6.0e-4),
        "translation": lambda: _derivative(lambda e: r.loss(dq=e * r.rigid), 2.0e-3),
    }
    for name in checks:
        f, ad = fd[name](), adjoint[name]
        assert abs(ad - f) <= 1.0e-2 * abs(f), f"{name}: adjoint {ad:.6e}, finite differences {f:.6e}"
