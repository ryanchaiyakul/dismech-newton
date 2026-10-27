"""Batched rollouts for the adjoint tests: one model holds a perturbed copy of the scene per evaluation, so a
row of finite differences is one simulation."""

import copy
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from fd import assert_derivative, jacobian
from utils.common import grid

from dismech_newton import DiSMechSolver, add_cloth_mesh, flatten_state, suspended_tape
from dismech_newton.state import dof_constants

RADIUS = 0.01


@wp.kernel
def _drive(fixed: wp.array[wp.int32], num_node_dofs: int, dz: float, q: wp.array[float]):
    i = wp.tid()
    if i < num_node_dofs and i % 3 == 2 and fixed[i] != 0:
        q[i] = q[i] + dz


@wp.kernel
def _dot(x: wp.array[float], w: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i] * w[i])


def cloth(builder, v, f, **kwargs):
    return add_cloth_mesh(builder, vertices=v.tolist(), indices=f.ravel().tolist(), **kwargs)


def _cfg(builder, mu: float):
    cfg = copy.copy(builder.default_shape_cfg)
    cfg.mu = mu
    return cfg


def contact_signatures(solver, copies: int, nodes: int) -> list:
    """Per copy (``nodes`` each) the step's force-carrying contacts: their nodes (copy-local), feature (the weights'
    signs) and whether they slide. What the step's solution is smooth in only while it holds."""
    out = [[] for _ in range(copies)]
    if solver.contact.count:
        s = solver.contact.s
        pair, w, n, u, z = s.pair.numpy(), s.w.numpy(), s.normal.numpy(), s.u.numpy(), s.z.numpy()
        for i in np.flatnonzero(s.active.numpy()):
            un = u[i] @ n[i]  # the scaled normal force, negative when pressing
            if un >= 0.0:
                continue
            slip = np.linalg.norm(z[i] - (z[i] @ n[i]) * n[i])  # zero inside the friction cone
            c = pair[i][pair[i] >= 0].min() // nodes
            local = tuple(int(k - c * nodes) if k >= 0 else -1 for k in pair[i])
            out[c].append((local, tuple(int(x) for x in np.sign(w[i])), bool(slip > 1.0e-6 * -un)))
    return [tuple(sorted(o)) for o in out]


def _dofs(model, copies: int) -> np.ndarray:
    """``(copies, n)``: each copy's DOFs, nodes, edges, then hinges."""
    d = model.dismech
    ne, nh = d.edge_length.shape[0], d.hinge_fixed.shape[0]
    p, e, h = model.particle_count // copies, ne // copies, nh // copies
    c, nd = np.arange(copies)[:, None], 3 * model.particle_count
    return np.hstack([3 * p * c + np.arange(3 * p), nd + e * c + np.arange(e), nd + ne + h * c + np.arange(h)])


class Rollout:
    """Copies of a scene, each perturbed along its row of ``X``; per copy
    ``loss = sum_t w . (q_t + dt qd_t)`` over every step's end (``dt`` damps velocity round-off; the intermediate
    states take the loss's adjoint as well as the next step's).

    ``directions`` maps a name to ``(fd step, {array: perturbation per unit})``, arrays one copy's slice of
    :meth:`parameters`, ``q`` or ``qd`` (the first state's) or ``force`` (``particle_f`` on every step). Every
    evaluation builds a fresh model, so warm starts do not carry over. The solver's contacts ignore collision groups:
    copies that may touch need ``spacing`` (else ``self_contact=False``). The copies share the solver's line search,
    so a perturbation must keep every copy tame: one thrown far shifts the others' differences.
    """

    ground = False
    colliding = False  # the scene has contacts (asserted active, and the same across each stencil)
    shaped = frozenset({"translation", "tilt", "yaw", "speed", "shift"})  # directions without an ascent copy
    witnesses = 6  # unperturbed copies per finite-difference batch, whose spread is the loss's noise
    drive = 0.0  # [m] the clamp raised before every step but the first, in place (a fit's drive moves the last output)
    spacing = 0.0  # copies this far apart on a grid
    height = 1.0
    segments, steps, dt = 4, 3, 1.0e-2
    rod_options = dict(stretch_stiffness=1.0e4, bend_stiffness=2.0, twist_stiffness=1.0)
    throw = 0.5  # [m/s] the free DOFs' random velocities

    def __init__(self, rng, mu: float = 0.3):
        self.mu = mu  # the shapes' and rods' friction coefficient
        model = self.model(1)
        self.fixed = dof_constants(model)[1].numpy() != 0
        self.nodes = model.particle_count
        self.w = rng.normal(size=len(self.fixed))
        self.v0 = self._velocity(rng, model)
        self.directions = self._directions(rng, model)

    # -- the directions' and velocities' common parts, each drawing from ``rng`` once

    def _free(self, rng, scale: float = 1.0) -> np.ndarray:
        """Random on the free DOFs."""
        return np.where(self.fixed, 0.0, scale * rng.normal(size=self.fixed.size))

    def _clamp(self, rng) -> np.ndarray:
        """Random on the prescribed DOFs."""
        return np.where(self.fixed, rng.normal(size=self.fixed.size), 0.0)

    def _force(self, rng, scale: float) -> np.ndarray:
        """``scale`` [N] on every free node along one direction (a random force per node can all but miss the
        loss)."""
        return np.where(~self.fixed[: 3 * self.nodes].reshape(-1, 3), scale * rng.normal(size=3), 0.0)

    @staticmethod
    def _mass(rng, model) -> tuple:
        mass = model.particle_mass.numpy()
        return 0.1, {"particle_mass": mass * rng.normal(size=mass.shape)}

    def _velocity(self, rng, model) -> np.ndarray:
        return self._free(rng, self.throw)

    def _slide(self, rng, model, first: int = 0) -> np.ndarray:
        """Nodes from ``first`` on sliding along x, pressed down; the free twists random."""
        nd = 3 * model.particle_count
        v = np.zeros(self.fixed.size)
        v[3 * first : nd : 3], v[3 * first + 2 : nd : 3] = 0.3, -0.1
        v[nd:] = np.where(self.fixed[nd:], 0.0, 0.5 * rng.normal(size=v.size - nd))
        return v

    def offset(self, c: int, copies: int) -> np.ndarray:
        """Copy ``c``'s offset: a square grid about the origin (small coordinates round off less in float32)."""
        side = int(np.ceil(np.sqrt(copies)))
        return self.spacing * np.array([c % side - 0.5 * (side - 1), c // side - 0.5 * (side - 1), 0.0])

    def parameters(self, model, solver) -> dict:
        """The differentiable arrays, copies contiguous in each."""
        d = model.dismech
        return {"params": d.triplet_params, "rest": solver.triplets.rest, "length": d.edge_length,
                "particle_mass": model.particle_mass, "inertia": d.edge_inertia}

    def _rods(self) -> list:
        """A copy's rods: ``(start, direction, clamped)``."""
        return [((0.0, 0.0, self.height), (1.0, 0.0, 0.0), not self.ground)]

    def model(self, copies: int):
        builder = newton.ModelBuilder()
        if self.ground:
            builder.add_ground_plane(cfg=_cfg(builder, self.mu))
        for c in range(copies):
            cfg = _cfg(builder, self.mu)
            for start, direction, clamped in self._rods():
                start = np.add(start, self.offset(c, copies))
                rod = newton.Rod.create_straight(start, direction, 0.5, segment_count=self.segments, radius=RADIUS)
                bodies = DiSMechSolver.add_rod(builder, rod, cfg=cfg, **self.rod_options)
                if clamped:
                    DiSMechSolver.fix_segment(builder, bodies[0])
        return builder.finalize()

    def _matrix(self, key: str, size: int) -> np.ndarray:
        """``(k, size)``: the directions' perturbations of one copy's ``key``."""
        return np.array([d[key].ravel() if key in d else np.zeros(size) for _, d in self.directions.values()]
                        ).reshape(len(self.directions), size)

    def setup(self, options: dict, grad: bool, X: np.ndarray | None = None) -> SimpleNamespace:
        """``len(X)`` perturbed copies (default one, unperturbed), their solver and states (the first set, the
        rest to step into)."""
        if X is None:
            X = np.zeros((1, len(self.directions)))
        copies = len(X)
        model = self.model(copies)
        solver = DiSMechSolver(model, **options)
        arrays = self.parameters(model, solver)
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
        return SimpleNamespace(model=model, solver=solver, states=states, arrays=arrays, dofs=dofs)

    def simulate(self, run: SimpleNamespace, signatures: list | None = None) -> None:
        """Every step, after :attr:`drive` (off the tape); each step's :func:`contact_signatures` into
        ``signatures``."""
        for t in range(self.steps):
            if self.drive and t > 0:
                with suspended_tape():
                    wp.launch(_drive, dim=run.solver.num_dofs, inputs=[run.solver.fixed, run.solver.num_node_dofs,
                              self.drive], outputs=[run.states[t].dismech.q])
            run.solver.step(run.states[t], run.states[t + 1], None, None, self.dt)
            if signatures is not None:
                signatures.append(contact_signatures(run.solver, len(run.dofs), self.nodes))

    def _assert_smooth(self, X: np.ndarray, signatures: list) -> None:
        """Every copy of a direction's stencil (``X``'s rows along it) has the same contacts: else its loss has a
        kink or jump inside the stencil, and the differences say nothing."""
        names = list(self.directions)
        for k in range(X.shape[1]):
            rows = np.flatnonzero(X[:, k] != 0.0)
            for t, sig in enumerate(signatures):
                changed = {i for i in rows if sig[i] != sig[rows[0]]}
                assert not changed, (
                    f"{names[k]}: the contacts change across its finite-difference stencil at step {t}: "
                    f"{sorted(set(sig[rows[0]]) ^ set(sig[min(changed)]))} (step {X[min(changed), k]:+.1e}); "
                    "the loss is not smooth there, so take a smaller step or a firmer scene")

    def losses(self, X: np.ndarray, **options) -> np.ndarray:
        """``(N,)`` at the rows of ``X`` ``(N, k)``, summed in float64: the float32 atomic sum rounds in a
        run-dependent order, by more than the finite differences resolve. A contact scene's stencils must keep their
        contacts (:meth:`_assert_smooth`)."""
        X = np.atleast_2d(X)
        run = self.setup(options, grad=False, X=X)
        signatures = [] if self.colliding else None
        self.simulate(run, signatures)
        if signatures:
            self._assert_smooth(X, signatures)
        copies = len(run.dofs)
        shift = np.zeros(run.dofs.shape)  # each copy's offset, out of its loss
        shift[:, : 3 * self.nodes] = [np.tile(self.offset(c, copies), self.nodes) for c in range(copies)]
        total = 0.0
        for state in run.states[1:]:
            q, qd = state.dismech.q.numpy().astype(np.float64), state.dismech.qd.numpy().astype(np.float64)
            total = total + (q[run.dofs] - shift + self.dt * qd[run.dofs]) @ self.w
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

    def gradients(self, **options) -> dict:
        """The loss's gradient per perturbable array (one copy's)."""
        run = self.setup(options, grad=True)
        self.record(run)
        if self.colliding:
            assert run.solver.contact.s.active.numpy().sum() > 0
        s0 = run.states[0].dismech
        grads = {key: a.grad.numpy().ravel().astype(np.float64) for key, a in run.arrays.items()}
        grads["q"], grads["qd"] = s0.q.grad.numpy()[run.dofs[0]], s0.qd.grad.numpy()[run.dofs[0]]
        grads["force"] = sum(state.particle_f.grad.numpy() for state in run.states[:-1]).ravel()
        return grads

    def along(self, grads: dict) -> dict:
        """The derivatives along ``directions``."""
        return {name: sum(float(grads[key] @ np.ravel(delta)) for key, delta in deltas.items())
                for name, (_, deltas) in self.directions.items()}

    def adjoint(self, **options) -> dict:
        """The derivatives along ``directions``."""
        return self.along(self.gradients(**options))

    def add_ascent(self, grads: dict) -> None:
        """Per random direction (not :attr:`shaped`), ``name ^``: its magnitudes with the gradient's signs, at its
        step. Never larger than the random one anywhere, its derivative ``sum |delta g|`` is never small, so a relative
        tolerance means something (a random direction's can all but cancel)."""
        for name, (h, deltas) in list(self.directions.items()):
            if name not in self.shaped:
                self.directions[name + " ^"] = (h, {key: np.abs(d) * np.sign(grads[key]).reshape(np.shape(d))
                                                    for key, d in deltas.items()})

    def finite_differences(self, **options) -> tuple[dict, dict]:
        """Along ``directions``, and their error estimates: :func:`fd.jacobian`'s, or the loss's own noise if larger
        (the spread ``s`` of :attr:`witnesses` unperturbed copies in each batch, off the solver's float32 floor and the
        copies' rounding, which the two steps share: ``1.9 s / h`` is the extrapolation's standard deviation)."""
        h = np.array([h for h, _ in self.directions.values()])
        spread = []

        def f(X):
            y = self.losses(np.vstack([X, np.zeros((self.witnesses, X.shape[1]))]), **options)
            spread.append(y[len(X):].std(ddof=1))
            return y[: len(X), None]

        J, est = jacobian(f, np.zeros(len(h)), h, error=True)
        est = np.maximum(est[0], 1.9 * max(spread) / h)
        return dict(zip(self.directions, J[0])), dict(zip(self.directions, est))

    def check(self, options: dict, rtol: float = 1.0e-2, rtols: dict | None = None, fd_options: dict | None = None,
              resolved: float = 0.1) -> None:
        """The adjoint against finite differences (with ``fd_options`` if given) along ``directions`` and their
        ascents (:meth:`add_ascent`), within ``rtol`` and the differences' own error (:func:`fd.assert_derivative`);
        ``rtols`` overrides ``rtol`` per name (an ascent's: its direction's). A random direction's error is relative to
        its ascent's derivative, the largest its magnitudes allow (its own can all but cancel); an ascent must be
        ``resolved`` too. Over 16 seeds every test needs at most 3.5e-3 (2026-10-27,
        ``--seeds 16``)."""
        grads = self.gradients(**options)
        self.add_ascent(grads)
        adjoint = self.along(grads)
        fd, est = self.finite_differences(**(fd_options or options))
        for name, f in fd.items():
            base = name.removesuffix(" ^")
            scale = abs(adjoint[name + " ^"]) if name + " ^" in adjoint else None
            assert_derivative(adjoint[name], f, est[name], (rtols or {}).get(base, rtol), name, scale=scale,
                              resolved=resolved if base != name else None)


class ClampedRollout(Rollout):
    """A clamped rod thrown at random."""

    rod_options = dict(Rollout.rod_options, bend_damping=0.01)

    def _directions(self, rng, model):
        d = model.dismech
        k = d.triplet_params.numpy()
        bend = np.zeros_like(k)
        bend[:, 2:4] = k[:, 2:4]
        length, mass, inertia = d.edge_length.numpy(), model.particle_mass.numpy(), d.edge_inertia.numpy()
        free = self._free(rng)
        return {
            "bend stiffness": (0.2, {"params": bend}),  # relative; a small effect over the steps
            "rest strain": (5.0e-2, {"rest": rng.normal(size=(k.shape[0], 5))}),  # nonlinear in curvature
            "rest length": (8.0e-3, {"length": length * rng.normal(size=length.shape)}),  # float32 noise below
            "mass": (0.1, {"particle_mass": mass * rng.normal(size=mass.shape),
                           "inertia": inertia * rng.normal(size=inertia.shape)}),
            "velocity": (2.0e-2, {"qd": free}),
            "position": (6.0e-4, {"q": free}),  # stiff stretching: nonlinear in positions
            # The prescribed DOFs (the clamp), which every step carries over from its input.
            "clamp": (6.0e-4, {"q": self._clamp(rng)}),
            "force": (0.3, {"force": self._force(rng, 0.3)}),
        }


class GroundRollout(Rollout):
    """A rod sliding on the ground (friction ``mu``); ``checks`` picks the directions."""

    ground = colliding = True
    height = RADIUS

    def __init__(self, rng, v_mean, checks, mu: float = 0.3):
        self.v_mean, self.checks = np.asarray(v_mean, dtype=float), checks
        super().__init__(rng, mu)

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
        directions = {
            "mass": self._mass(rng, model),
            "velocity": (4.0e-2, {"qd": planar}),  # small in the stuck case
            "position": (6.0e-4, {"q": planar}),
            "translation": (2.0e-3, {"q": rigid}),
        }
        return {name: directions[name] for name in self.checks if name in directions}


class CrossingRollout(Rollout):
    """Per copy a cantilever along x and a free rod along y lying across it, pressed down and sliding along x: one
    rod-rod contact, mid-edge on both (no cell switch), whose normal turns with the rods (unlike the ground's)."""

    colliding = True
    spacing = 0.8
    half = 0.3125  # nodes at +-0.0625 around the crossing at the origin

    def _rods(self) -> list:
        return [((-self.half, 0.0, 0.0), (1.0, 0.0, 0.0), True),
                ((0.0, -self.half, 2.0 * RADIUS), (0.0, 1.0, 0.0), False)]

    def _velocity(self, rng, model):
        return self._slide(rng, model, model.particle_count // 2)  # the top rod along the bottom one, onto it

    def _directions(self, rng, model):
        nd, half = 3 * model.particle_count, model.particle_count // 2
        x = model.particle_q.numpy()[half:]
        free = self._free(rng)
        tilt = np.zeros(self.fixed.size)  # the top rod turned about x: its tangent, so the normal, turns
        tilt[3 * half + 2 : nd : 3] = x[:, 1]
        yaw = np.zeros(self.fixed.size)  # turned about z: the crossing angle
        yaw[3 * half : nd : 3], yaw[3 * half + 1 : nd : 3] = -x[:, 1], x[:, 0]
        return {
            "mass": self._mass(rng, model),
            "velocity": (4.0e-2, {"qd": free}),
            "position": (3.0e-4, {"q": free}),  # 4x larger lifts the top rod off within the stencil (16 seeds)
            "tilt": (4.0e-2, {"q": tilt}),
            "yaw": (2.0e-2, {"q": yaw}),
        }


class ShellRollout(Rollout):
    """A sheet clamped along one edge (and its neighbouring row), thrown at random (no self-contact: the copies
    overlap)."""

    nx, ny, size = 4, 3, 0.3
    throw = 0.3
    cloth_options = dict(density=0.5, tri_ke=2.0e3, tri_ka=1.0e3, tri_kd=2.0, edge_ke=1.0e-2, edge_kd=1.0e-3)

    def model(self, copies: int):
        builder = newton.ModelBuilder()
        v, f = grid(self.nx, self.ny, self.size, 0.75 * self.size)
        for _ in range(copies):
            ids = cloth(builder, v, f, **self.cloth_options)
            for k in ids[: 2 * self.ny]:  # x = 0 and its neighbours
                builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
        return builder.finalize()

    def parameters(self, model, solver) -> dict:
        return {"params": model.dismech.face_params, "rest": solver.shells.rest, "particle_mass": model.particle_mass}

    def _directions(self, rng, model):
        k = model.dismech.face_params.numpy()
        free = self._free(rng)

        def columns(cols):
            return np.where(np.isin(np.arange(12), cols), k, 0.0)

        return {
            "membrane": (0.1, {"params": columns([0, 1, 2])}),
            "bending": (0.1, {"params": columns([3, 4, 5])}),
            "damping": (0.1, {"params": columns(range(6, 12))}),
            "rest strain": (5.0e-4, {"rest": rng.normal(size=(k.shape[0], 6))}),
            "mass": self._mass(rng, model),
            "velocity": (2.0e-2, {"qd": free}),
            "position": (3.0e-4, {"q": free}),
            "clamp": (7.5e-5, {"q": self._clamp(rng)}),  # stiff: small
            "force": (0.03, {"force": self._force(rng, 0.03)}),
        }


class SheetContactRollout(ShellRollout):
    """A sheet's contacts, one kind per ``scene``, each pressed on and sliding (``free``: the sheet's or rod's free
    DOFs; ``along``: the slide's direction):

    - ``ground``: a free sheet stretching as it slides on the ground, a contact per vertex;
    - ``sphere``: a small free sheet's centre vertex on top of a sphere, whose normal turns as it slides;
    - ``foot``: an upright rod's foot inside a triangle of the clamped sheet (vertex-triangle);
    - ``edge``: a rod tilted down across the clamped sheet's free edge, sliding along it (edge-edge, mid-edge on both).
    """

    colliding = True
    spacing = 0.8
    thickness = 0.004  # the sheet's particle radius
    checks = {"ground": ("mass", "velocity", "position"), "sphere": ("mass", "speed", "shift"),
              "foot": ("mass", "velocity", "position", "shift"), "edge": ("mass", "velocity", "shift")}

    def __init__(self, rng, scene: str, mu: float):
        self.scene = scene
        super().__init__(rng, mu)

    def _rod(self, o: np.ndarray):
        if self.scene == "foot":
            return newton.Rod.create_straight(o + (0.23, 0.09, RADIUS + self.thickness), (0.0, 0.0, 1.0), 0.2,
                                              segment_count=2, radius=RADIUS)
        d = np.array([1.0, 0.0, -0.3]) / np.linalg.norm([1.0, 0.0, -0.3])
        touch = np.array([0.3, 0.05, 0.0]) + (RADIUS + self.thickness) * np.array([-d[2], 0.0, d[0]])
        return newton.Rod.create_straight(o + touch - 0.15 / d[0] * d, d, 0.4, segment_count=4, radius=RADIUS)

    def model(self, copies: int):
        builder = newton.ModelBuilder()
        if self.scene == "ground":
            builder.add_ground_plane(cfg=_cfg(builder, self.mu))
        if self.scene == "sphere":
            v, f = grid(3, 3, 0.1, 0.1)
            v += (-0.05, -0.05, self.thickness)
        else:
            v, f = grid(self.nx, self.ny, self.size, 0.75 * self.size)
            v[:, 2] = self.thickness if self.scene == "ground" else 0.0
        for c in range(copies):
            o = self.offset(c, copies)
            if self.scene == "sphere":
                builder.add_shape_sphere(-1, xform=wp.transform(o + (0.0, 0.0, -0.06), wp.quat_identity()),
                                         radius=0.06, cfg=_cfg(builder, self.mu))
            ids = cloth(builder, v + o, f, particle_radius=self.thickness, **self.cloth_options)
            if self.scene in ("foot", "edge"):
                for k in ids[: 2 * self.ny]:
                    builder.particle_flags[k] &= ~newton.ParticleFlags.ACTIVE
                DiSMechSolver.add_rod(builder, self._rod(o), cfg=_cfg(builder, self.mu), **self.rod_options)
        model = builder.finalize()
        model.particle_mu = self.mu
        return model

    def _moving(self, model) -> tuple[int, int]:
        """The first moving node (the rod's, or the sheet's) and the slide's axis."""
        rod = {"foot": 3, "edge": 5}.get(self.scene, model.particle_count)
        return model.particle_count - rod, int(self.scene == "edge")

    def _velocity(self, rng, model):
        nd, (first, axis) = 3 * model.particle_count, self._moving(model)
        v = self._slide(rng, model, first)
        if axis:
            v[3 * first : nd : 3], v[3 * first + 1 : nd : 3] = 0.0, 0.3
        if self.scene == "ground":  # stretching too: a rigid slide's mass derivative vanishes
            v[:nd] += 0.1 * rng.normal(size=nd) * (np.arange(nd) % 3 != 2)
        return v

    def _directions(self, rng, model):
        nd, (first, axis) = 3 * model.particle_count, self._moving(model)
        free = self._free(rng)
        if self.scene == "ground":
            free[2:nd:3] = 0.0  # in the ground's plane
        along = np.zeros(self.fixed.size)
        along[3 * first + axis : nd : 3] = 1.0
        directions = {
            "mass": self._mass(rng, model),
            # The rod on the sheet's edge, which springs back: 8x larger lifts it off within the stencil (16 seeds).
            "velocity": (5.0e-3 if self.scene == "edge" else 4.0e-2, {"qd": free}),
            # The clamped sheet's membrane: nonlinear at 6e-4; the foot lifts off within a 1.5e-4 stencil (16 seeds).
            "position": (7.5e-5 if self.scene == "foot" else 1.5e-4, {"q": free}),
            "speed": (4.0e-2, {"qd": along}),
            "shift": (2.0e-3, {"q": along}),
        }
        return {name: directions[name] for name in self.checks[self.scene]}


class SphereRollout(Rollout):
    """A rod across the top of a sphere, pressed onto it and sliding over it: its contact holds an
    edge's interior (the top mid-edge: a node over it would hand the contact to an edge under any shift), whose
    closest point and normal turn with the rod."""

    colliding = True
    spacing = 0.8
    radius = 0.3

    def model(self, copies: int):
        builder = newton.ModelBuilder()
        for c in range(copies):
            o = self.offset(c, copies)
            builder.add_shape_sphere(-1, xform=wp.transform(o + (0.0, 0.0, -self.radius), wp.quat_identity()),
                                     radius=self.radius, cfg=_cfg(builder, self.mu))
            rod = newton.Rod.create_straight(o + (-0.3125, 0.0, RADIUS), (1.0, 0.0, 0.0), 0.5,
                                             segment_count=self.segments, radius=RADIUS)
            DiSMechSolver.add_rod(builder, rod, cfg=_cfg(builder, self.mu), **self.rod_options)
        return builder.finalize()

    def _velocity(self, rng, model):
        return self._slide(rng, model)

    def _directions(self, rng, model):
        along = np.zeros(self.fixed.size)  # along the rod: over the sphere, its contact's normal turns
        along[0 : 3 * model.particle_count : 3] = 1.0
        return {
            "mass": self._mass(rng, model),
            "speed": (4.0e-2, {"qd": along}),  # random velocities tumble it from edge to edge
            "shift": (2.0e-3, {"q": along}),
        }
