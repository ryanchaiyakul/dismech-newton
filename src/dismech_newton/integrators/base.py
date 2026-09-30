"""Interface shared by all time integrators."""

from abc import ABC, abstractmethod
from typing import Any

import warp as wp

from ..system import SymmetricCSR


@wp.kernel
def _assemble_kernel(
    q: wp.array[float],
    q_pred: wp.array[float],
    mass: wp.array[float],
    alpha: float,
    # outputs
    residual: wp.array[float],
    hess_indptr: wp.array[wp.int32],
    hess_vals: wp.array[Any],
):
    i = wp.tid()
    k = mass[i] * alpha
    residual[i] = residual[i] + k * (q[i] - q_pred[i])
    slot = hess_indptr[i]  # the diagonal leads each row of the upper-triangle pattern
    hess_vals[slot] = hess_vals[slot] + type(hess_vals[0])(k)


class IntegratorBase(ABC):
    """Inertial part of the Newton system ``inertia(q) + dE/dq(q) - f_ext = 0``.

    A one-step integrator writes the inertia as ``M * alpha * (q - q_pred)``: ``q_pred`` is where
    the DOFs would land with no elastic force and ``alpha`` is the coefficient relating
    acceleration to position (``1 / dt^2`` for backward Euler, ``1 / (beta dt^2)`` for Newmark).
    A subclass only has to provide ``q_pred``, ``alpha`` and the Newton initial guess in
    :meth:`begin_step` and the velocity update in :meth:`end_step`; the residual and diagonal Hessian are assembled here.

    It works on the flat DOF vectors (``state.dismech.q`` / ``qd``, see
    :mod:`dismech_newton.system`) and knows nothing about Dirichlet conditions or external
    forces; the solver applies those. Per step the solver calls :meth:`begin_step` once,
    :meth:`assemble` every Newton iteration and :meth:`end_step`.

    Integrators that carry extra state (e.g. accelerations) keep it themselves, which assumes
    steps are taken in sequence; call :meth:`reset` after replacing the state.
    """

    def bind(self, mass: wp.array) -> None:
        """Attach the per-DOF mass (which fixes the size and device) and allocate state
        (see :meth:`_allocate`)."""
        self.mass = mass
        self.num_dofs = mass.shape[0]
        self.device = mass.device
        self.q_pred = wp.zeros(self.num_dofs, dtype=float, device=self.device)
        self.alpha = 0.0
        self._allocate()

    def _allocate(self) -> None:
        """Allocate subclass buffers; ``self.num_dofs`` and ``self.device`` are set."""

    def reset(self) -> None:
        """Forget carried state."""

    @abstractmethod
    def begin_step(self, q0: wp.array, v0: wp.array, dt: float, q: wp.array) -> None:
        """Set ``self.q_pred`` and ``self.alpha`` from the start-of-step state, write the Newton
        initial guess into ``q``, and cache what :meth:`end_step` needs."""

    @abstractmethod
    def end_step(self, q: wp.array, v: wp.array) -> None:
        """Write the end-of-step velocities into ``v`` (``q`` is the converged state)."""

    def assemble(self, q: wp.array, residual: wp.array, hessian: SymmetricCSR) -> None:
        """Add the inertia residual and diagonal Hessian at ``q``."""
        wp.launch(
            _assemble_kernel,
            dim=self.num_dofs,
            inputs=[q, self.q_pred, self.mass, self.alpha],
            outputs=[residual, hessian.indptr, hessian.vals],
            device=self.device,
        )
