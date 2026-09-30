"""Implicit (backward) Euler."""

import warp as wp

from .base import IntegratorBase


@wp.kernel
def _predict_kernel(
    q0: wp.array[float], v0: wp.array[float], dt: float, q_pred: wp.array[float], q: wp.array[float]
):
    """The prediction ``q_n + dt v_n`` doubles as the initial guess."""
    i = wp.tid()
    q_pred[i] = q0[i] + dt * v0[i]
    q[i] = q_pred[i]


@wp.kernel
def _velocity_kernel(q: wp.array[float], q0: wp.array[float], dt: float, v: wp.array[float]):
    i = wp.tid()
    v[i] = (q[i] - q0[i]) / dt


class ImplicitEuler(IntegratorBase):
    """``M / dt^2 (q - q_n - dt v_n)``; first order, strongly dissipative."""

    def _allocate(self) -> None:
        self._q0 = wp.zeros(self.num_dofs, dtype=float, device=self.device)
        self._dt = 1.0

    def begin_step(self, q0: wp.array, v0: wp.array, dt: float, q: wp.array) -> None:
        self._dt = dt
        self.alpha = 1.0 / (dt * dt)
        wp.copy(self._q0, q0)
        wp.launch(
            _predict_kernel,
            dim=self.num_dofs,
            inputs=[q0, v0, dt],
            outputs=[self.q_pred, q],
            device=self.device,
        )

    def end_step(self, q: wp.array, v: wp.array) -> None:
        wp.launch(
            _velocity_kernel,
            dim=self.num_dofs,
            inputs=[q, self._q0, self._dt],
            outputs=[v],
            device=self.device,
        )
