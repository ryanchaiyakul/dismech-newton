"""Discrete elastic rods with contact and friction, solved with ADMM.

Global ``(M alpha + S^T P S) q = M alpha y + f_ext + sum S^T P (z - u)``, local ``z = prox(S q + u)``, dual
``u += S q - z``. ``H`` is constant (factorised once); fixed DOFs get identity rows.
"""

import numpy as np
import scipy.sparse as sp
import warp as wp
from newton import Contacts, Model, State

from .builder import add_colliding_rod
from .contact import ContactSnapshot, ContactTerm
from .dofs import external_force, flatten_state
from .linear import BlockInverseSolver, CudssSolver, ScipySolver, TridiagonalSolver, sparse_solver
from .solver import DiSMechSolver
from .sparse import SymmetricCSR
from .triplet import TripletTerm, linear_energy


class ADMMDiSMechSolver(DiSMechSolver):
    """Implicit discrete elastic rods with contact and friction, solved with ADMM.

    Contacts are the caller's: pass ``Contacts`` from a :class:`~newton.CollisionPipeline` (``soft_contact_max=0``).

    Args:
        iterations: Maximum ADMM iterations per step.
        tol: Residual and ``q``-change tolerance (relative to the mean edge length); ``0`` runs ``iterations``.
        check_every: Iterations between convergence tests.
        rho_scale, rho_abs_ratio, local_iterations: See :class:`TripletTerm`.
        self_contact: Keep rod-rod contacts.
        friction: Coulomb ``mu``.
        contact_smoothing: Adjoint only: a velocity [m/s] smoothing the contact law's switches; ``0`` is exact.
        contact_rho_scale: Contact penalties relative to the largest that provably converges.
        linear_solver: ``"tridiagonal"``, ``"dense"``, ``"cudss"``, ``"scipy"`` or ``"auto"`` (tridiagonal on CUDA
            with nvmath, else dense while the blocks are small, else cuDSS on CUDA, SciPy otherwise). All solve
            the global step in increment form.
        obstacle_motion: ``"velocity"`` (``state_in.body_qd``) or ``"pose"`` (``state_out.body_q``).
        theta: See :class:`DiSMechSolver`; ``1/2`` needs a small ``tol`` (``1e-5``).
    """

    add_rod = staticmethod(add_colliding_rod)

    def __init__(
        self, model: Model, *, energy=linear_energy, iterations: int = 50, tol: float = 1.0e-4, check_every: int = 10,
        rho_scale: float = 0.3, rho_abs_ratio: float = 0.1, local_iterations: int = 2, self_contact: bool = True,
        friction: float = 0.3, contact_rho_scale: float = 1.8, contact_smoothing: float = 1.0e-3,
        pose_proxies: bool = True, linear_solver: str = "auto", theta: float = 1.0, obstacle_motion: str = "velocity",
    ):
        if obstacle_motion not in ("velocity", "pose"):
            raise ValueError(f"obstacle_motion must be 'velocity' or 'pose', got {obstacle_motion!r}")
        self.obstacle_motion = obstacle_motion
        if linear_solver not in ("auto", "tridiagonal", "dense", "cudss", "scipy"):
            raise ValueError("linear_solver must be 'auto', 'tridiagonal', 'dense', 'cudss' or 'scipy', "
                             f"got {linear_solver!r}")
        if linear_solver in ("cudss", "tridiagonal") and not wp.get_device(model.device).is_cuda:
            raise ValueError(f"linear_solver={linear_solver!r} needs a CUDA model")
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

    def reset(self, state: State, world_mask: wp.array | None = None, flags: int | None = None) -> None:
        """Start the next step from ``state`` without the last step's warm start (``z = S q``, ``u = 0``, no
        contact duals): for a step that does not continue the last one, such as a restart from data."""
        if world_mask is not None:
            raise NotImplementedError("ADMMDiSMechSolver.reset resets every world")
        flatten_state(state)
        self.elastic.reset(state)
        self.contact.reset()

    def contact_snapshot(self) -> ContactSnapshot | None:
        return self.contact.snapshot(self.contact_smoothing)

    @property
    def _device_loop(self) -> bool:
        return self.tol > 0.0

    def add_contact_reactions(self, body_f: wp.array) -> None:
        """Add the last step's contact wrenches on rigid bodies to ``body_f``."""
        self.contact.add_reactions(body_f)

    def _factorize(self, alpha: float) -> None:
        """``H = M alpha + S^T P S``, identity rows on fixed DOFs."""
        n = self.num_dofs
        fixed = self.fixed.numpy() != 0
        mass = self.mass.numpy().astype(np.float64)
        r, c, v = self.elastic.penalty()
        rows, cols = np.concatenate([np.arange(n), r]), np.concatenate([np.arange(n), c])
        vals = np.concatenate([np.where(fixed, 1.0, mass * alpha), v])
        keep = (~fixed[rows] & ~fixed[cols]) | (np.arange(len(rows)) < n)
        H = SymmetricCSR.from_scipy(sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n)), self.device)
        kind = self.linear_solver
        host = H.to_scipy() if kind in ("auto", "tridiagonal", "dense") else None  # one copy for fits and setup
        if kind == "auto":
            if TridiagonalSolver.fits(H, host):
                kind = "tridiagonal"
            elif BlockInverseSolver.fits(H, host):
                kind = "dense"
        if kind == "tridiagonal":
            self._linear = TridiagonalSolver(H, increment=True, H=host)
        elif kind == "dense":
            self._linear = BlockInverseSolver(H, increment=True, H=host)
        else:
            sparse = {"cudss": CudssSolver, "scipy": ScipySolver}.get(kind, sparse_solver)
            self._linear = sparse(H, refactorize=False, increment=True)
        self._factored_alpha = alpha

    def _solve(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float) -> None:
        """ADMM iterations on ``state_out.dismech.q``."""
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
        """``check_every`` iterations; with ``tol``, the last one's residuals in ``_stats``."""
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
    """``b = M alpha q_pred + f_ext`` on free DOFs, ``q_in`` on fixed ones."""
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
