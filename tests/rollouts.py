"""Batched rollouts for the adjoint tests: one model holds a perturbed copy of the scene per evaluation, so a
row of finite differences is one simulation."""

import copy
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from fd import assert_close, derivative, jacobian

from dismech_newton import ADMMDiSMechSolver, DiSMechSolver, flatten_state, suspended_tape
from dismech_newton.dofs import dof_constants

RADIUS = 0.01


def contact_options(friction: float, smoothing: float = 1.0e-3) -> dict:
    """ADMM with contact: a fixed iteration count keeps batched copies identical to lone rods."""
    return dict(tol=0.0, iterations=500, friction=friction, contact_smoothing=smoothing)


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
    """``(copies, n)``: each copy's DOFs, nodes then edges."""
    p, e = model.particle_count // copies, model.dismech.edge_q.shape[0] // copies
    c = np.arange(copies)[:, None]
    return np.hstack([3 * p * c + np.arange(3 * p), 3 * model.particle_count + e * c + np.arange(e)])


class Rollout:
    """Copies of a scene, overlapping but not colliding, each perturbed along its row of ``X``; per copy
    ``loss = sum_t w . (q_t + dt qd_t)`` over every step's end (``dt`` damps velocity round-off; the intermediate
    states take the loss's adjoint as well as the next step's).

    ``directions`` maps a name to ``(fd step, {array: perturbation per unit})``, arrays one copy's slice of
    :func:`_parameters`, ``q`` or ``qd`` (the first state's) or ``force`` (``particle_f`` on every step). Every
    evaluation builds a fresh model, so warm starts do not carry over.
    """

    ground = False
    colliding = False  # a collision pipeline every step (ADMM)
    height = 1.0
    segments, steps, dt = 4, 3, 1.0e-2
    rod_options = dict(stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)

    def __init__(self, rng, solver_cls=DiSMechSolver):
        model = self.model(1, solver_cls)
        self.fixed = dof_constants(model)[1].numpy() != 0
        self.nodes = model.particle_count
        self.w = rng.normal(size=len(self.fixed))
        self.v0 = self._velocity(rng, model)
        self.directions = self._directions(rng, model)

    def _rods(self) -> list:
        """A copy's rods: ``(start, direction, clamped)``."""
        return [((0.0, 0.0, self.height), (1.0, 0.0, 0.0), not self.ground)]

    def model(self, copies: int, solver_cls):
        builder = newton.ModelBuilder()
        if self.ground:
            builder.add_ground_plane(cfg=_group(builder, -1))
        for c in range(copies):
            cfg = _group(builder, c + 1)  # copies apart; a copy's rods touch each other
            for start, direction, clamped in self._rods():
                rod = newton.Rod.create_straight(start, direction, 0.5, segment_count=self.segments, radius=RADIUS)
                bodies = solver_cls.add_rod(builder, rod, cfg=cfg, **self.rod_options)
                if clamped:
                    solver_cls.fix_segment(builder, bodies[0])
        return builder.finalize()

    def _matrix(self, key: str, size: int) -> np.ndarray:
        """``(k, size)``: the directions' perturbations of one copy's ``key``."""
        return np.array([d[key].ravel() if key in d else np.zeros(size) for _, d in self.directions.values()]
                        ).reshape(len(self.directions), size)

    @property
    def fd_steps(self) -> np.ndarray:
        return np.array([h for h, _ in self.directions.values()])

    def setup(self, solver_cls, options: dict, grad: bool, X: np.ndarray | None = None) -> SimpleNamespace:
        """``len(X)`` perturbed copies (default one, unperturbed), their solver and states (the first set, the
        rest to step into)."""
        if X is None:
            X = np.zeros((1, len(self.directions)))
        copies = len(X)
        model = self.model(copies, solver_cls)
        solver = solver_cls(model, **options)
        arrays = _parameters(model, solver)
        for key, a in arrays.items():
            base = a.numpy()
            a.assign((base.reshape(copies, -1) + X @ self._matrix(key, base.size // copies)).reshape(base.shape))
            a.requires_grad = grad
        solver.refresh_mass()
        states = [model.state(requires_grad=grad) for _ in range(self.steps + 1)]
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
        force = (X @ self._matrix("force", 3 * self.nodes)).reshape(-1, 3)
        for state in states[:-1]:
            state.particle_f.assign(force)
        run = SimpleNamespace(model=model, solver=solver, states=states, arrays=arrays, dofs=dofs,
                              pipeline=None, contacts=None)
        if self.colliding:
            run.pipeline = newton.CollisionPipeline(model, soft_contact_max=0, verify_buffers=False,
                                                    speculative_contact_gap_max=2.0 * RADIUS,
                                                    contact_matching="latest")
            run.contacts = run.pipeline.contacts()
            solver.contact.friction.requires_grad = grad
        return run

    def simulate(self, run: SimpleNamespace) -> None:
        """Every step, after its collision detection (off the tape)."""
        for t in range(self.steps):
            if run.pipeline is not None:
                with suspended_tape():
                    run.pipeline.collide(run.states[t], run.contacts, dt=2.0 * self.dt)
            run.solver.step(run.states[t], run.states[t + 1], None, run.contacts, self.dt)

    def losses(self, X: np.ndarray, solver_cls, **options) -> np.ndarray:
        """``(N,)`` at the rows of ``X`` ``(N, k)``, summed in float64: the float32 atomic sum rounds in a
        run-dependent order, by more than the finite differences resolve."""
        run = self.setup(solver_cls, options, grad=False, X=np.atleast_2d(X))
        self.simulate(run)
        total = 0.0
        for state in run.states[1:]:
            q, qd = state.dismech.q.numpy().astype(np.float64), state.dismech.qd.numpy().astype(np.float64)
            total = total + (q[run.dofs] + self.dt * qd[run.dofs]) @ self.w
        return total

    def record(self, run: SimpleNamespace, loss: wp.array | None = None) -> wp.Tape:
        """Simulate, ``loss``, backward: device work only, so a CUDA graph can hold it once ``run`` has stepped."""
        if not hasattr(run, "weights"):  # copied from the host before any capture
            run.weights = wp.array(self.w, dtype=float), wp.array(self.dt * self.w, dtype=float)
        w, w_dt = run.weights
        if loss is None:
            loss = wp.zeros(1, dtype=float, requires_grad=True)
        loss.zero_()
        tape = wp.Tape()
        with tape:
            self.simulate(run)
            for state in run.states[1:]:
                wp.launch(_dot, dim=len(self.w), inputs=[state.dismech.q, w], outputs=[loss])
                wp.launch(_dot, dim=len(self.w), inputs=[state.dismech.qd, w_dt], outputs=[loss])
        tape.backward(loss)
        return tape

    def adjoint(self, solver_cls, **options) -> dict:
        """The derivatives along ``directions`` (and in ``friction``, with contacts)."""
        run = self.setup(solver_cls, options, grad=True)
        self.record(run)
        if self.colliding:
            assert run.solver.contact.active.numpy().sum() > 0
        s0 = run.states[0].dismech
        grads = {key: a.grad.numpy().ravel() for key, a in run.arrays.items()}
        grads["q"], grads["qd"] = s0.q.grad.numpy()[run.dofs[0]], s0.qd.grad.numpy()[run.dofs[0]]
        grads["force"] = sum(state.particle_f.grad.numpy() for state in run.states[:-1]).ravel()
        out = {name: sum(float(grads[key] @ delta.ravel()) for key, delta in deltas.items())
               for name, (_, deltas) in self.directions.items()}
        if self.colliding:
            out["friction"] = float(run.solver.contact.friction.grad.numpy()[0])
        return out

    def finite_differences(self, solver_cls, wrt_friction: bool = False, **options) -> dict:
        """Along ``directions``; with ``wrt_friction``, also in the friction coefficient."""
        J = jacobian(lambda X: self.losses(X, solver_cls, **options)[:, None], np.zeros(len(self.fd_steps)),
                     self.fd_steps)
        out = dict(zip(self.directions, J[0]))
        if wrt_friction:
            zero, mu = np.zeros((1, len(self.directions))), options["friction"]
            out["friction"] = derivative(lambda e: self.losses(zero, solver_cls, **dict(options, friction=mu + e))[0],
                                         0.05)
        return out

    def check(self, solver_cls, options: dict, rtol: float, rtols: dict | None = None, fd_solver_cls=None,
              fd_options: dict | None = None, wrt_friction: bool = False) -> None:
        """The adjoint of ``solver_cls`` against finite differences (of ``fd_solver_cls``, ``fd_options`` if
        given); ``rtols`` overrides ``rtol`` per name."""
        adjoint = self.adjoint(solver_cls, **options)
        fd = self.finite_differences(fd_solver_cls or solver_cls, wrt_friction=wrt_friction, **(fd_options or options))
        for name, f in fd.items():
            assert_close(adjoint[name], f, (rtols or {}).get(name, rtol), name)


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
        free_nodes = ~self.fixed[: 3 * self.nodes].reshape(-1, 3)
        return {
            "bend stiffness": (0.2, {"params": bend}),  # relative; a small effect over the steps
            "rest strain": (5.0e-2, {"rest": rng.normal(size=(k.shape[0], 5))}),  # nonlinear in curvature
            "rest length": (8.0e-3, {"length": length * rng.normal(size=length.shape)}),  # float32 noise below
            "mass": (0.1, {"particle_mass": mass * rng.normal(size=mass.shape),
                           "inertia": inertia * rng.normal(size=inertia.shape)}),
            "velocity": (2.0e-2, {"qd": free}),
            "position": (6.0e-4, {"q": free}),  # stiff stretching: nonlinear in positions
            # The prescribed DOFs (the clamp), which every step carries over from its input.
            "clamp": (6.0e-4, {"q": np.where(self.fixed, rng.normal(size=self.fixed.size), 0.0)}),
            # [N] on every free node along one direction (a random force per node can all but miss the loss).
            "force": (0.3, {"force": np.where(free_nodes, 0.3 * rng.normal(size=3), 0.0)}),
        }


class SpinningRollout(ClampedRollout):
    """A long clamped rod swinging and spinning at a long step: its reference frames lag far behind."""

    segments, steps, dt = 30, 2, 1.0 / 30.0
    rod_options = dict(Rollout.rod_options, bend_damping=0.05)

    def _velocity(self, rng, model):
        nd, x = 3 * self.nodes, np.linspace(0.0, 1.0, self.nodes)
        v = np.zeros(self.fixed.size)
        v[1:nd:3], v[2:nd:3], v[nd:] = 0.5 * x**2, 0.3 * x, np.linspace(0.0, 5.0, v.size - nd)
        return np.where(self.fixed, 0.0, v)

    def _directions(self, rng, model):
        return {}


class GroundRollout(Rollout):
    """A rod sliding on the ground (ADMM, friction); ``checks`` picks the directions."""

    ground = colliding = True
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


class CrossingRollout(Rollout):
    """Per copy a cantilever along x and a free rod along y lying across it, pressed down and sliding along x: one
    rod-rod contact, mid-edge on both (no cell switch), whose normal turns with the rods (unlike the ground's)."""

    colliding = True
    half = 0.3125  # nodes at +-0.0625 around the crossing at the origin

    def __init__(self, rng):
        super().__init__(rng, ADMMDiSMechSolver)

    def _rods(self) -> list:
        return [((-self.half, 0.0, 0.0), (1.0, 0.0, 0.0), True),
                ((0.0, -self.half, 2.0 * RADIUS), (0.0, 1.0, 0.0), False)]

    def _velocity(self, rng, model):
        nd, half = 3 * model.particle_count, model.particle_count // 2
        v = np.zeros(self.fixed.size)
        top = slice(3 * half, nd)
        v[top][0::3], v[top][2::3] = 0.3, -0.1  # sliding along the bottom rod, pressed onto it
        v[nd:] = np.where(self.fixed[nd:], 0.0, 0.5 * rng.normal(size=v.size - nd))
        return v

    def _directions(self, rng, model):
        nd, half = 3 * model.particle_count, model.particle_count // 2
        x = model.particle_q.numpy()[half:]
        free = np.where(self.fixed, 0.0, rng.normal(size=self.fixed.size))
        tilt = np.zeros(self.fixed.size)  # the top rod turned about x: its tangent, so the normal, turns
        tilt[3 * half + 2 : nd : 3] = x[:, 1]
        yaw = np.zeros(self.fixed.size)  # turned about z: the crossing angle
        yaw[3 * half : nd : 3], yaw[3 * half + 1 : nd : 3] = -x[:, 1], x[:, 0]
        mass = model.particle_mass.numpy()
        return {
            "mass": (0.1, {"particle_mass": mass * rng.normal(size=mass.shape)}),
            "velocity": (4.0e-2, {"qd": free}),
            "position": (1.2e-3, {"q": free}),  # small effects: smaller steps round off (4x larger is nonlinear)
            "tilt": (2.0e-2, {"q": tilt}),
            "yaw": (2.0e-2, {"q": yaw}),
        }
