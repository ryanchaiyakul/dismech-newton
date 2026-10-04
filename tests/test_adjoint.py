"""The step adjoint against finite differences of a rollout, batched: one model holds a rod per evaluation."""

import copy

import newton
import numpy as np
import pytest
import warp as wp
from fd import assert_close, derivative, jacobian

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, flatten_state
from dismech_newton.dofs import dof_constants
from dismech_newton.solver import suspended_tape

STEPS, DT, SEGMENTS, RADIUS = 3, 1.0e-2, 4, 0.01


@pytest.fixture(autouse=True)
def _on_device(device):
    """Models and solvers on ``--device``."""
    with wp.ScopedDevice(device):
        yield


@wp.kernel
def _dot(x: wp.array[float], w: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i] * w[i])


def _parameters(model, solver) -> dict:
    """The differentiable arrays, rods contiguous in each."""
    d = model.dismech
    return {"params": d.triplet_params, "rest": solver.triplets.rest, "length": d.edge_length,
            "particle_mass": model.particle_mass, "inertia": d.edge_inertia}


def _group(builder, group: int):
    cfg = copy.copy(builder.default_shape_cfg)
    cfg.collision_group = group
    return cfg


def _dofs(model, copies: int) -> np.ndarray:
    """``(copies, n)``: each rod's DOFs, nodes then edges."""
    p, e = model.particle_count // copies, model.dismech.edge_q.shape[0] // copies
    c = np.arange(copies)[:, None]
    return np.hstack([3 * p * c + np.arange(3 * p), 3 * model.particle_count + e * c + np.arange(e)])


class Rollout:
    """Copies of a rod, overlapping but not colliding, each perturbed along its row of ``X``; per copy
    ``loss = w . (q_T + dt qd_T)`` (``dt`` damps velocity round-off).

    ``directions`` maps a name to ``(fd step, {array: perturbation per unit})``, arrays one rod's slice of
    :func:`_parameters`, ``q`` or ``qd``. Every evaluation builds a fresh model, so warm starts do not carry over.
    """

    ground = False
    height = 1.0
    rod_options = dict(stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)

    def __init__(self, rng, solver_cls):
        model = self._model(1, solver_cls)
        self.fixed = dof_constants(model)[1].numpy() != 0
        self.w = rng.normal(size=len(self.fixed))
        self.v0 = self._velocity(rng, model)
        self.directions = self._directions(rng, model)

    def _model(self, copies: int, solver_cls):
        builder = newton.ModelBuilder()
        if self.ground:
            builder.add_ground_plane(cfg=_group(builder, -1))
        for c in range(copies):
            rod = newton.Rod.create_straight((0.0, 0.0, self.height), (1.0, 0.0, 0.0), 0.5, segment_count=SEGMENTS,
                                             radius=RADIUS)
            cfg = _group(builder, c + 1)  # copies touch the ground only
            bodies = solver_cls.add_rod(builder, rod, cfg=cfg, **self.rod_options)
            if not self.ground:
                solver_cls.fix_segment(builder, bodies[0])
        return builder.finalize()

    def _matrix(self, key: str, size: int) -> np.ndarray:
        """``(k, size)``: the directions' perturbations of one rod's ``key``."""
        return np.stack([d[key].ravel() if key in d else np.zeros(size) for _, d in self.directions.values()])

    @property
    def steps(self) -> np.ndarray:
        return np.array([h for h, _ in self.directions.values()])

    def losses(self, X: np.ndarray, solver_cls, **options) -> np.ndarray:
        """``(N,)`` at the rows of ``X`` ``(N, k)``."""
        return self._run(np.atleast_2d(X), solver_cls, options, grad=False)

    def adjoint(self, solver_cls, **options) -> dict:
        """The derivatives along ``directions`` (and in ``friction``, on the ground)."""
        return self._run(np.zeros((1, len(self.directions))), solver_cls, options, grad=True)

    def _run(self, X, solver_cls, options, grad):
        copies = len(X)
        model = self._model(copies, solver_cls)
        solver = solver_cls(model, **options)
        arrays = _parameters(model, solver)
        for key, a in arrays.items():
            base = a.numpy()
            a.assign((base.reshape(copies, -1) + X @ self._matrix(key, base.size // copies)).reshape(base.shape))
            a.requires_grad = grad
        solver.refresh_mass()
        states = [model.state(requires_grad=grad) for _ in range(STEPS + 1)]
        for state in states:
            flatten_state(state)
        s0 = states[0].dismech
        dofs = _dofs(model, copies)
        n = self.fixed.size
        q, qd = s0.q.numpy().astype(np.float64), np.zeros(s0.qd.shape[0])
        q[dofs] += X @ self._matrix("q", n)
        qd[dofs] = self.v0 + X @ self._matrix("qd", n)
        s0.q.assign(q)
        s0.qd.assign(qd)

        pipeline = contacts = None
        if self.ground:
            pipeline = newton.CollisionPipeline(model, soft_contact_max=0, verify_buffers=False,
                                                speculative_contact_gap_max=2.0 * RADIUS, contact_matching="latest")
            contacts = pipeline.contacts()
            solver.contact.friction.requires_grad = grad
        w = np.tile(self.w, copies)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
            for t in range(STEPS):
                if pipeline is not None:
                    with suspended_tape():
                        pipeline.collide(states[t], contacts, dt=2.0 * DT)
                solver.step(states[t], states[t + 1], None, contacts, DT)
            end = states[STEPS].dismech
            if grad:
                wp.launch(_dot, dim=len(w), inputs=[end.q, wp.array(w, dtype=float)], outputs=[loss])
                wp.launch(_dot, dim=len(w), inputs=[end.qd, wp.array(DT * w, dtype=float)], outputs=[loss])
        if not grad:
            # In float64: the float32 atomic sum rounds in a run-dependent order, by more than the
            # finite differences resolve.
            q, qd = end.q.numpy().astype(np.float64), end.qd.numpy().astype(np.float64)
            return (q[dofs] + DT * qd[dofs]) @ self.w
        if self.ground:
            assert solver.contact.active.numpy().sum() > 0
        tape.backward(loss)
        grads = {key: a.grad.numpy().ravel() for key, a in arrays.items()}
        grads["q"], grads["qd"] = s0.q.grad.numpy()[dofs[0]], s0.qd.grad.numpy()[dofs[0]]
        out = {name: sum(float(grads[key] @ delta.ravel()) for key, delta in deltas.items())
               for name, (_, deltas) in self.directions.items()}
        if self.ground:
            out["friction"] = float(solver.contact.friction.grad.numpy()[0])
        return out

    def finite_differences(self, solver_cls, **options) -> dict:
        J = jacobian(lambda X: self.losses(X, solver_cls, **options)[:, None], np.zeros(len(self.steps)), self.steps)
        return dict(zip(self.directions, J[0]))


class ClampedRollout(Rollout):
    """A clamped rod thrown at random."""

    rod_options = dict(Rollout.rod_options, bend_damping=0.01)

    def _velocity(self, rng, model):
        return np.where(self.fixed, 0.0, 0.5 * rng.normal(size=self.fixed.size))

    def _directions(self, rng, model):
        d = model.dismech
        k = d.triplet_params.numpy()
        bend = np.zeros_like(k)
        bend[:, 2:4] = k[:, 2:4]
        length, mass, inertia = d.edge_length.numpy(), model.particle_mass.numpy(), d.edge_inertia.numpy()
        free = np.where(self.fixed, 0.0, rng.normal(size=self.fixed.size))
        return {
            "bend stiffness": (0.2, {"params": bend}),  # relative; a small effect over the steps
            "rest strain": (5.0e-2, {"rest": rng.normal(size=(k.shape[0], 5))}),  # nonlinear in curvature
            "rest length": (8.0e-3, {"length": length * rng.normal(size=length.shape)}),  # float32 noise below
            "mass": (0.1, {"particle_mass": mass * rng.normal(size=mass.shape),
                           "inertia": inertia * rng.normal(size=inertia.shape)}),
            "velocity": (2.0e-2, {"qd": free}),
            "position": (6.0e-4, {"q": free}),  # stiff stretching: nonlinear in positions
        }


class GroundRollout(Rollout):
    """A rod sliding on the ground (ADMM, friction)."""

    ground = True
    height = RADIUS

    def __init__(self, rng, v_mean, checks):
        self.v_mean, self.checks = np.asarray(v_mean, dtype=float), checks
        super().__init__(rng, ADMMDiSMechSolver)

    def _velocity(self, rng, model):
        nd = 3 * model.particle_count
        v = np.zeros(self.fixed.size)
        v[:nd] = np.tile(self.v_mean, model.particle_count) + 0.02 * rng.normal(size=nd)
        v[2:nd:3] = 0.0  # never upwards
        v[nd:] = 0.5 * rng.normal(size=v.size - nd)
        return v

    def _directions(self, rng, model):
        nd = 3 * model.particle_count
        planar = rng.normal(size=self.fixed.size)
        planar[2:nd:3] = 0.0  # in the ground's plane
        rigid = np.zeros(self.fixed.size)  # a horizontal translation, which keeps a stuck rod stuck
        rigid[0:nd:3], rigid[1:nd:3] = 0.6, -0.8
        mass = model.particle_mass.numpy()
        directions = {
            "mass": (0.1, {"particle_mass": mass * rng.normal(size=mass.shape)}),
            "velocity": (4.0e-2, {"qd": planar}),  # small in the stuck case
            "position": (6.0e-4, {"q": planar}),
            "translation": (2.0e-3, {"q": rigid}),
        }
        return {name: directions[name] for name in self.checks if name in directions}


def _assert_matches(adjoint: dict, fd: dict, rtol: float = 1.0e-2, rtols: dict | None = None):
    """``rtols`` overrides ``rtol`` per name."""
    for name, f in fd.items():
        assert_close(adjoint[name], f, (rtols or {}).get(name, rtol), name)


@pytest.mark.parametrize(
    "solver_cls, options",
    [
        (DiSMechSolver, {"newton_tol": 1.0e-7}),
        (DiSMechSolver, {"newton_tol": 1.0e-7, "theta": 0.5}),
        (ADMMDiSMechSolver, {"tol": 1.0e-6, "iterations": 5000}),
    ],
    ids=["newton", "newton-midpoint", "admm"],
)
def test_step_adjoint_matches_finite_differences(rng, solver_cls, options):
    """Differences use Newton-Raphson: the adjoint is of the exact root."""
    r = ClampedRollout(rng, DiSMechSolver)
    adjoint = r.adjoint(solver_cls, **options)
    _assert_matches(adjoint, r.finite_differences(DiSMechSolver, newton_tol=1.0e-7, theta=options.get("theta", 1.0)))


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
def test_contact_adjoint_matches_finite_differences(device, rng, mu, v_mean, checks, smoothing):
    """Perturbations stay clear of the stick/slip and on/off switches.

    A fixed iteration count keeps the batched copies identical to lone rods (a tolerance stops on their joint
    residual); the mass derivative, small against round-off, still differs by up to ~2% (~3% on the CPU,
    whose float32 rounding moves the differences, not the adjoint)."""
    r = GroundRollout(rng, v_mean, checks)
    options = dict(tol=0.0, iterations=500, friction=mu, contact_smoothing=smoothing)
    adjoint = r.adjoint(ADMMDiSMechSolver, **options)
    fd = r.finite_differences(ADMMDiSMechSolver, **options)
    if "friction" in checks:
        zero = np.zeros((1, len(r.directions)))
        fd["friction"] = derivative(
            lambda e: r.losses(zero, ADMMDiSMechSolver, **dict(options, friction=mu + e))[0], 0.05)
    _assert_matches(adjoint, fd, rtol=3.0e-2, rtols={"mass": 5.0e-2} if device.is_cpu else None)
