"""Single-dataclass constitutive residual corrector for 6D BAC shells."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp
from newton import Model

from ..constants import vec6f, vec12f
from ..shell import Shell
from .base import as_element_param, as_rff_weights, elem_world_indices, model_world_info
from .rff import shell_rff_sum, warp8_bcast


@wp.struct
class ShellResidualData:
    """Device data passed to the 6D shell constitutive residual step function."""

    rff_freq: wp.array[vec12f]
    w_shell: wp.array2d[vec6f]
    d_rest: wp.array[vec6f]
    has_d_rest: int
    scales: vec12f
    clamp_lo: float
    clamp_hi: float


@wp.func
def _shell_bounds(data: ShellResidualData) -> wp.vec2:
    return wp.vec2(data.clamp_lo, data.clamp_hi)


@wp.func
def _shell_con_step(
    t: int,
    lane: int,
    st: int,
    v: Shell,
    eps: vec6f,
    ed: vec6f,
    rb_i: vec6f,
    aux: wp.array[float],
    ew: wp.array[wp.int32],
    data: ShellResidualData,
):
    """Evaluate 6D BAC shell constitutive residual ``(sig_c, dk_c, dc_c)`` and effective rest strain."""
    r0, er, za, zb = rb_i, vec6f(0.0), vec6f(0.0), vec6f(0.0)
    sc = data.scales
    if lane == 0:
        r0 = rb_i + data.d_rest[t] if data.has_d_rest != 0 else rb_i
        er = eps - r0
        za = vec6f(er[0] / sc[0], er[1] / sc[1], er[2] / sc[2], er[3] / sc[3], er[4] / sc[4], er[5] / sc[5])
        zb = vec6f(ed[0] / sc[6], ed[1] / sc[7], ed[2] / sc[8], ed[3] / sc[9], ed[4] / sc[10], ed[5] / sc[11])

    z_a, z_b = za, zb
    if st > 1:
        z_a = vec6f(
            warp8_bcast(za[0]),
            warp8_bcast(za[1]),
            warp8_bcast(za[2]),
            warp8_bcast(za[3]),
            warp8_bcast(za[4]),
            warp8_bcast(za[5]),
        )
        z_b = vec6f(
            warp8_bcast(zb[0]),
            warp8_bcast(zb[1]),
            warp8_bcast(zb[2]),
            warp8_bcast(zb[3]),
            warp8_bcast(zb[4]),
            warp8_bcast(zb[5]),
        )

    rff_dim = data.rff_freq.shape[0]
    sp, dk, dc = vec6f(0.0), vec6f(0.0), vec6f(0.0)
    if rff_dim > 0:
        wi = ew[t] if data.w_shell.shape[0] > 1 else 0
        cn = wp.sqrt(2.0 / float(rff_dim))
        sp, dk, dc = shell_rff_sum(z_a, z_b, data.rff_freq, data.w_shell, wi, rff_dim, lane, st, cn)

    sig_c, dk_c, dc_c = vec6f(0.0), vec6f(0.0), vec6f(0.0)
    if lane == 0 and rff_dim > 0:
        ls = vec6f(aux[t])
        sig_c = wp.cw_mul(ls, sp)
        dk_c = wp.cw_mul(
            ls,
            vec6f(dk[0] / sc[0], dk[1] / sc[1], dk[2] / sc[2], dk[3] / sc[3], dk[4] / sc[4], dk[5] / sc[5]),
        )
        dc_c = wp.cw_mul(
            ls,
            vec6f(dc[0] / sc[6], dc[1] / sc[7], dc[2] / sc[8], dc[3] / sc[9], dc[4] / sc[10], dc[5] / sc[11]),
        )
    return r0, er, sig_c, dk_c, dc_c


@dataclass
class ShellResidual:
    """6D BAC shell constitutive RFF residual and optional per-element rest strain delta."""

    rff_dim: int = 64
    rff_bandwidth: float = 1.5
    clamp_lo: float = -0.35
    clamp_hi: float = 8.0
    scales: tuple[float, ...] = (0.05, 0.05, 0.05, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 20.0, 20.0, 20.0)
    seed: int = 1
    w_shell: np.ndarray | wp.array | None = None
    spatial_rest_delta: np.ndarray | wp.array | None = None

    data: ShellResidualData = field(init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    step_fn: Any = field(default=None, init=False, repr=False)
    bounds_fn: Any = field(default=None, init=False, repr=False)
    rff_freq: wp.array | None = field(default=None, init=False, repr=False)
    d_rest: wp.array | None = field(default=None, init=False, repr=False)
    has_d_rest: bool = field(default=False, init=False, repr=False)
    elem_world: wp.array | None = field(default=None, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.rff_dim < 0
            or self.rff_bandwidth < 0.0
            or len(self.scales) != 12
            or min(self.scales) <= 0.0
            or not (-1.0 < self.clamp_lo <= self.clamp_hi)
        ):
            raise ValueError("Invalid ShellResidual parameters")

    def bind(self, model: Model, elem, device=None) -> "ShellResidual":
        """Allocate/bind shell residual buffers onto ``model`` and ``elem``."""
        self._src = {"w_shell": self.w_shell, "spatial_rest_delta": self.spatial_rest_delta}
        dev = wp.get_device(model.device if device is None else device)
        _, _, nw = model_world_info(model)
        n = int(elem.count)

        rff = np.random.default_rng(self.seed).normal(0.0, self.rff_bandwidth, (self.rff_dim, 12)).astype(np.float32)
        self.rff_freq = wp.array(rff, dtype=vec12f, device=dev)
        has_w = self._src["w_shell"] is not None
        self.w_shell = as_rff_weights(self.w_shell, vec6f, self.rff_dim, 6, nw, dev, "w_shell")
        self.has_d_rest = bool(self.spatial_rest_delta is not None and n > 0)
        dr_val = self.spatial_rest_delta if self.has_d_rest else None
        self.d_rest = as_element_param(dr_val, vec6f, 6, n, dev, "spatial_rest_delta")
        self.elem_world = elem_world_indices(model, elem)

        data = ShellResidualData()
        data.rff_freq = self.rff_freq
        data.w_shell = self.w_shell
        data.d_rest = self.d_rest
        data.has_d_rest = int(self.has_d_rest)
        data.scales = vec12f(*map(float, self.scales))
        data.clamp_lo = float(self.clamp_lo)
        data.clamp_hi = float(self.clamp_hi)
        self.data = data

        params = []
        if self.rff_dim and (n or has_w):
            params.append(self.w_shell)
        if self.has_d_rest:
            params.append(self.d_rest)
        self.params = tuple(params)
        self.step_fn = _shell_con_step
        self.bounds_fn = _shell_bounds
        self._bound = True
        return self

    def stride_info(self, device) -> tuple[int, int]:
        """Return ``(lanes_per_element, block_dim)`` for launching the step kernel."""
        return (1, 1) if device.is_cpu else ((1, 128) if not self.rff_dim else (8, 128))

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays tracked by adjoint recording."""
        return list(self.params)
