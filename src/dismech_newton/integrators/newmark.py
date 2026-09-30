"""Newmark-beta."""

import warp as wp

from .base import IntegratorBase


@wp.kernel
def _predict_kernel(
    q0: wp.array[float],
    v0: wp.array[float],
    accel_prev: wp.array[float],
    beta: float,
    dt: float,
    # outputs
    q_pred: wp.array[float],
    q: wp.array[float],
):
    """``q_pred`` is ``q`` with the new acceleration set to zero, ``q_n + dt v_n + dt^2 (1/2 - beta) a_n``;
    the initial guess ``q`` extrapolates at constant acceleration, ``q_n + dt v_n + dt^2 / 2 a_n``."""
    i = wp.tid()
    q_pred[i] = q0[i] + dt * v0[i] + dt * dt * (0.5 - beta) * accel_prev[i]
    q[i] = q_pred[i] + beta * dt * dt * accel_prev[i]


@wp.kernel
def _end_kernel(
    q: wp.array[float],
    q_pred: wp.array[float],
    v0: wp.array[float],
    alpha: float,
    gamma: float,
    dt: float,
    # in/out
    accel: wp.array[float],
    # outputs
    v: wp.array[float],
):
    i = wp.tid()
    a_prev = accel[i]
    a = alpha * (q[i] - q_pred[i])
    v[i] = v0[i] + dt * ((1.0 - gamma) * a_prev + gamma * a)
    accel[i] = a


class NewmarkBeta(IntegratorBase):
    """Newmark-beta with ``q = q_n + dt v_n + dt^2 [(1/2 - beta) a_n + beta a]`` and
    ``v = v_n + dt [(1 - gamma) a_n + gamma a]``.

    The default ``beta = 1/4, gamma = 1/2`` (average acceleration) is second order and
    unconditionally stable with no numerical damping. ``gamma > 1/2`` adds damping (keep
    ``beta >= (gamma + 1/2)^2 / 4`` for stability). The previous acceleration is carried
    between steps and starts at zero.
    """

    def __init__(self, beta: float = 0.25, gamma: float = 0.5) -> None:
        if beta <= 0.0:
            raise ValueError("beta must be positive (beta = 0 is explicit)")
        self.beta = beta
        self.gamma = gamma

    def _allocate(self) -> None:
        n, dev = self.num_dofs, self.device
        self._accel = wp.zeros(n, dtype=float, device=dev)
        self._v0 = wp.zeros(n, dtype=float, device=dev)
        self._dt = 1.0

    def reset(self) -> None:
        self._accel.zero_()

    def begin_step(self, q0: wp.array, v0: wp.array, dt: float, q: wp.array) -> None:
        self._dt = dt
        self.alpha = 1.0 / (self.beta * dt * dt)
        wp.copy(self._v0, v0)
        wp.launch(
            _predict_kernel,
            dim=self.num_dofs,
            inputs=[q0, v0, self._accel, self.beta, dt],
            outputs=[self.q_pred, q],
            device=self.device,
        )

    def end_step(self, q: wp.array, v: wp.array) -> None:
        wp.launch(
            _end_kernel,
            dim=self.num_dofs,
            inputs=[q, self.q_pred, self._v0, self.alpha, self.gamma, self._dt],
            outputs=[self._accel, v],
            device=self.device,
        )
