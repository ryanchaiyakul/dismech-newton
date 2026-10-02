"""Discrete elastic rods with contact and friction, solved with ADMM (``admm.md``).

Per step the solver minimises ``1/2 |q - y|^2_{M alpha} + sum_t W_t(S_t q)`` subject to the contact
cones, with ``y`` the implicit-Euler prediction. Every triplet and every contact has a local
variable ``z = S q``, a scaled dual ``u`` and a prox; one iteration is

    global:   (M alpha + S^T P S) q = M alpha y + f_ext + sum S^T P (z - u)
    local:    z = prox(S q + u)          (every triplet and contact, in parallel)
    duals:    u += S q - z

``H = M alpha + S^T P S`` depends only on masses, ``dt``, the penalties and the topology, so it is
factorised once. Contacts do not enter it (:mod:`~dismech_newton.contact`). Fixed DOFs have identity
rows of ``H``; their coupling moves to the right-hand side, so ``H`` stays constant as they move.
The triplets' local step is :class:`~dismech_newton.triplet.TripletTerm`.
"""


import numpy as np
import scipy.sparse as sp
import warp as wp
from newton import Contacts, Model, State

from .builder import add_colliding_rod
from .contact import ContactSnapshot, ContactTerm
from .frames import external_force
from .linear import BlockInverseSolver, CudssSolver, SymmetricCSR
from .solver import DiSMechSolver
from .triplet import TripletTerm, linear_energy


class ADMMDiSMechSolver(DiSMechSolver):
    """Implicit discrete elastic rods with contact and friction, solved with ADMM.

    Build rods with :meth:`add_rod`. Every other collision shape is an obstacle moving with its
    body (see ``obstacle_motion``). Contacts are the caller's: run a :class:`~newton.CollisionPipeline`
    on ``state_in`` (``soft_contact_max=0``; ``contact_matching="latest"`` warm-starts the forces) and
    pass its :class:`~newton.Contacts` to :meth:`step`. The proxies are posed after every step, so
    the next ``collide`` sees the current rod. With ``theta < 1`` contacts act at ``q_theta``.

    Args:
        model: Model built with :meth:`add_rod`; must be on a CUDA device.
        energy: Triplet energy (see :mod:`dismech_newton.triplet`).
        iterations: Maximum ADMM iterations per step.
        tol: Stop when the local residuals and the change of ``q`` (relative to the mean edge
            length) are below ``tol``; ``0`` always runs ``iterations``. The test runs on the
            device (``wp.capture_while``), so a step can be graph-captured either way.
        check_every: Test convergence every this many iterations (``iterations`` rounds up to a multiple).
        rho_scale, rho_abs_ratio, local_iterations: See :class:`TripletTerm`.
        self_contact: Keep rod-rod contacts (rod-obstacle contacts are always kept).
        friction: Coulomb coefficient ``mu`` (``solver.contact.friction``, a one-element array).
        contact_smoothing: Adjoint only: a velocity [m/s] that, times the step, smooths the contact
            law's switches (:mod:`~dismech_newton.adjoint`); ``0`` differentiates the exact law.
        contact_rho_scale: Contact penalties relative to the largest that provably converges, set per
            contact from its coupling to the others through ``H`` (see :class:`ContactTerm`).
        pose_proxies: See :class:`~dismech_newton.DiSMechSolver`; collision detection needs them.
        linear_solver: ``"dense"`` (one dense inverse per block of ``H``), ``"cudss"``, or
            ``"auto"``: dense while the blocks are small.
        obstacle_motion: ``"velocity"``: bodies move with ``state_in.body_qd``; ``"pose"``: to the
            pose already in ``state_out.body_q`` (a rigid solver stepped first, :mod:`~dismech_newton.coupling`).
        theta: See :class:`~dismech_newton.DiSMechSolver`. With ``theta = 1/2`` nothing damps the
            stiff modes, so the residual ``tol`` leaves must be small: ``1e-5`` holds an undamped
            cantilever, ``1e-4`` lets it blow up within seconds.
    """

    add_rod = staticmethod(add_colliding_rod)

    def __init__(
        self, model: Model, *, energy=linear_energy, iterations: int = 200, tol: float = 1.0e-4, check_every: int = 10,
        rho_scale: float = 0.3, rho_abs_ratio: float = 0.1, local_iterations: int = 2, self_contact: bool = True,
        friction: float = 0.3, contact_rho_scale: float = 1.8, contact_smoothing: float = 1.0e-3,
        pose_proxies: bool = True, linear_solver: str = "auto", theta: float = 1.0, obstacle_motion: str = "velocity",
    ):
        if obstacle_motion not in ("velocity", "pose"):
            raise ValueError(f"obstacle_motion must be 'velocity' or 'pose', got {obstacle_motion!r}")
        self.obstacle_motion = obstacle_motion
        if linear_solver not in ("auto", "dense", "cudss"):
            raise ValueError(f"linear_solver must be 'auto', 'dense' or 'cudss', got {linear_solver!r}")
        # Read by _build_system, which the base constructor calls.
        self.linear_solver = linear_solver
        self.iterations = iterations
        self.tol = tol
        self.check_every = max(1, check_every)
        self._term_options = dict(rho_scale=rho_scale, rho_abs_ratio=rho_abs_ratio, local_iterations=local_iterations)
        self.contact_smoothing = contact_smoothing
        self._contact_options = dict(friction=friction, self_contact=self_contact, rho_scale=contact_rho_scale)
        super().__init__(model, energy=energy, pose_proxies=pose_proxies, theta=theta)

    def _build_system(self) -> None:
        dev = self.device
        self.elastic = TripletTerm(self.triplets, **self._term_options)
        self.contact = ContactTerm(self.model, self.fixed, **self._contact_options)
        self._length_scale = float(np.mean(self.der.edge_length.numpy()))
        self._b = wp.zeros(self.num_dofs, dtype=float, device=dev)
        self._rhs = [wp.zeros(self.num_dofs, dtype=float, device=dev) for _ in range(2)]  # read / being built
        self._q_prev = wp.zeros(self.num_dofs, dtype=float, device=dev)
        self._stats = wp.zeros(2, dtype=float, device=dev)  # [local residuals, change of q]
        self._linear = None
        self._factored_alpha = None

    def refresh_mass(self) -> None:
        super().refresh_mass()
        self._factored_alpha = None  # H holds M alpha

    def contact_snapshot(self) -> ContactSnapshot | None:
        return self.contact.snapshot(self.contact_smoothing)

    def contact_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """``(pairs, force)`` of the active contacts of the last step (see :meth:`ContactTerm.forces`)."""
        return self.contact.forces()

    def add_contact_reactions(self, body_f: wp.array) -> None:
        """Add the last step's contact wrenches on rigid bodies to ``body_f`` (:attr:`newton.State.body_f`)."""
        self.contact.add_reactions(body_f)

    def _factorize(self, alpha: float) -> None:
        """``H = M alpha + S^T P S`` on the free DOFs, identity rows on the fixed ones."""
        n = self.num_dofs
        fixed = self.fixed.numpy() != 0
        mass = self.mass.numpy().astype(np.float64)
        r, c, v = self.elastic.penalty()
        rows, cols = np.concatenate([np.arange(n), r]), np.concatenate([np.arange(n), c])
        vals = np.concatenate([np.where(fixed, 1.0, mass * alpha), v])
        keep = (~fixed[rows] & ~fixed[cols]) | (np.arange(len(rows)) < n)
        H = SymmetricCSR.from_scipy(sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n)), self.device)
        dense = self.linear_solver == "dense" or (self.linear_solver == "auto" and BlockInverseSolver.fits(H))
        self._linear = BlockInverseSolver(H) if dense else CudssSolver(H, refactorize=False)
        self._factored_alpha = alpha

    def _solve(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float) -> None:
        """ADMM iterations on ``state_out.dismech.q``, with a convergence test every ``check_every``."""
        q = state_out.dismech.q
        if self._factored_alpha != self.alpha:
            self._factorize(self.alpha)
        wp.launch(
            _inertia_rhs_kernel,
            dim=self.num_dofs,
            inputs=[self.q_pred, state_in.dismech.q, self.mass, self.alpha, self.fixed, self.model.gravity,
                    self._particle_f(state_in), self.num_node_dofs],
            outputs=[self._b],
            device=self.device,
        )
        self.elastic.begin_step(state_in)
        body_q_end = state_out.body_q if self.obstacle_motion == "pose" else None
        self.contact.begin_step(state_in, contacts, self._linear.solve, dt, body_q_end, self.theta)

        # The first right-hand side comes from the warm-started z and u.
        wp.copy(self._rhs[0], self._b)
        self._local(state_in, q, dt, self._rhs[0], update=0)
        self._iterate(self._admm_block, self.iterations, self.check_every, self.tol, self._stats,
                      state_in=state_in, q=q, dt=dt)
        self.contact.end_step()

    def _admm_block(self, state_in: State, q: wp.array, dt: float) -> None:
        """``check_every`` ADMM iterations; with ``tol``, the residuals of the last one in ``self._stats``."""
        check = self.tol > 0.0
        n = self.check_every
        for i in range(n):
            if check and i == n - 1:
                wp.copy(self._q_prev, q)
                self._stats.zero_()
            rhs, nxt = self._rhs[i % 2], self._rhs[(i + 1) % 2]
            self._linear.solve(rhs, q, reset=(nxt, self._b))  # the next right-hand side starts from b
            self._local(state_in, q, dt, nxt, update=1)
        if n % 2:  # every block starts from self._rhs[0]
            wp.copy(self._rhs[0], self._rhs[1])
        if check:
            wp.launch(_step_change_kernel, dim=self.num_node_dofs, inputs=[q, self._q_prev, self._length_scale,
                      self._stats], device=self.device)

    def _local(self, state_in: State, q: wp.array, dt: float, rhs: wp.array, update: int) -> None:
        self.elastic.local(state_in, q, dt, rhs, self._stats, update)
        self.contact.local(q, rhs, self._stats, update)


# -- kernels ------------------------------------------------------------------------------


@wp.kernel
def _inertia_rhs_kernel(
    q_pred: wp.array[float], q_in: wp.array[float], mass: wp.array[float], alpha: float, dof_fixed: wp.array[wp.int32],
    gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], num_node_dofs: int,
    # outputs
    b: wp.array[float],
):
    """``b = M alpha q_pred + f_ext`` on free DOFs, the prescribed value on fixed ones (identity rows)."""
    i = wp.tid()
    if dof_fixed[i] != 0:
        b[i] = q_in[i]
        return
    v = mass[i] * alpha * q_pred[i]
    if i < num_node_dofs:
        v = v + external_force(i, mass, gravity, particle_f)
    b[i] = v


@wp.kernel
def _step_change_kernel(q: wp.array[float], q_prev: wp.array[float], scale: float, stats: wp.array[float]):
    i = wp.tid()
    wp.atomic_max(stats, 1, wp.abs(q[i] - q_prev[i]) / scale)
