"""Single-dataclass constitutive residual corrector for 5D DER rods."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp
from newton import Model

from ..constants import vec5f, vec10f
from ..rod import Rod
from .base import as_element_param, as_rff_weights, elem_world_indices, model_world_info
from .rff import rod_rff_sum, warp8_bcast


@wp.struct
class RodResidualData:
    """Device data passed to the 5D rod constitutive residual step function."""

    rff_freq: wp.array[vec10f]
    w_rod: wp.array2d[vec5f]
    d_rest: wp.array[vec5f]
    has_d_rest: int
    scales: vec10f
    iso_bending: int
    clamp_lo: float
    clamp_hi: float


@wp.func
def _rod_bounds(data: RodResidualData) -> wp.vec2:
    return wp.vec2(data.clamp_lo, data.clamp_hi)


@wp.func
def _rod_con_step(
    t: int,
    lane: int,
    st: int,
    v: Rod,
    eps: vec5f,
    ed: vec5f,
    rb_i: vec5f,
    aux: wp.array[wp.vec2],
    ew: wp.array[wp.int32],
    data: RodResidualData,
):
    """Evaluate 5D DER rod constitutive residual ``(sig_c, dk_c, dc_c)`` and effective rest strain."""
    r0, er, za, zb = rb_i, vec5f(0.0), vec5f(0.0), vec5f(0.0)
    le, lf, dl = 0.0, 0.0, 0.0
    sc = data.scales
    if lane == 0:
        c = v.conn[t]
        r0 = rb_i + data.d_rest[t] if data.has_d_rest != 0 else rb_i
        er = eps - r0
        le, lf = float(v.edge_length[c[0]]), float(v.edge_length[c[1]])
        dl = 0.5 * (le + lf)
        za = vec5f(er[0] / sc[0], er[1] / sc[1], er[2] / (dl * sc[2]), er[3] / (dl * sc[3]), er[4] / (dl * sc[4]))
        zb = vec5f(ed[0] / sc[5], ed[1] / sc[6], ed[2] / (dl * sc[7]), ed[3] / (dl * sc[8]), ed[4] / (dl * sc[9]))

    z_a, z_b = za, zb
    if st > 1:
        z_a = vec5f(warp8_bcast(za[0]), warp8_bcast(za[1]), warp8_bcast(za[2]), warp8_bcast(za[3]), warp8_bcast(za[4]))
        z_b = vec5f(warp8_bcast(zb[0]), warp8_bcast(zb[1]), warp8_bcast(zb[2]), warp8_bcast(zb[3]), warp8_bcast(zb[4]))

    rff_dim = data.rff_freq.shape[0]
    sp, dk, dc = vec5f(0.0), vec5f(0.0), vec5f(0.0)
    if rff_dim > 0:
        wi = ew[t] if data.w_rod.shape[0] > 1 else 0
        cn = wp.sqrt(2.0 / float(rff_dim))
        sp, dk, dc = rod_rff_sum(z_a, z_b, data.rff_freq, data.w_rod, wi, rff_dim, lane, st, cn, data.iso_bending)

    sig_c, dk_c, dc_c = vec5f(0.0), vec5f(0.0), vec5f(0.0)
    if lane == 0 and rff_dim > 0:
        sv = aux[t]
        ls = vec5f(sv[0] * le, sv[1] * lf, 1.0, 1.0, 1.0)
        sa = vec5f(sc[0], sc[1], dl * sc[2], dl * sc[3], dl * sc[4])
        sb = vec5f(sc[5], sc[6], dl * sc[7], dl * sc[8], dl * sc[9])
        sig_c = wp.cw_mul(ls, sp)
        dk_c = vec5f(
            ls[0] * dk[0] / sa[0],
            ls[1] * dk[1] / sa[1],
            ls[2] * dk[2] / sa[2],
            ls[3] * dk[3] / sa[3],
            ls[4] * dk[4] / sa[4],
        )
        dc_c = vec5f(
            ls[0] * dc[0] / sb[0],
            ls[1] * dc[1] / sb[1],
            ls[2] * dc[2] / sb[2],
            ls[3] * dc[3] / sb[3],
            ls[4] * dc[4] / sb[4],
        )
    return r0, er, sig_c, dk_c, dc_c


@dataclass
class RodResidual:
    """5D DER rod constitutive RFF residual and optional per-element rest strain delta."""

    rff_dim: int = 64
    rff_bandwidth: float = 1.5
    iso_bending: bool = True
    clamp_lo: float = -0.35
    clamp_hi: float = 8.0
    scales: tuple[float, ...] = (0.05, 0.05, 5.0, 5.0, 5.0, 0.5, 0.5, 50.0, 50.0, 50.0)
    seed: int = 0
    w_rod: np.ndarray | wp.array | None = None
    spatial_rest_delta: np.ndarray | wp.array | None = None

    data: RodResidualData = field(init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    step_fn: Any = field(default=None, init=False, repr=False)
    bounds_fn: Any = field(default=None, init=False, repr=False)
    rff_freq: wp.array | None = field(default=None, init=False, repr=False)
    d_rest: wp.array | None = field(default=None, init=False, repr=False)
    has_d_rest: bool = field(default=False, init=False, repr=False)
    edge_share: wp.array | None = field(default=None, init=False, repr=False)
    elem_world: wp.array | None = field(default=None, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.rff_dim < 0
            or self.rff_bandwidth < 0.0
            or len(self.scales) != 10
            or min(self.scales) <= 0.0
            or not (-1.0 < self.clamp_lo <= self.clamp_hi)
        ):
            raise ValueError("Invalid RodResidual parameters")

    def bind(self, model: Model, elem, device=None) -> "RodResidual":
        """Allocate/bind rod residual buffers onto ``model`` and ``elem``."""
        self._src = {"w_rod": self.w_rod, "spatial_rest_delta": self.spatial_rest_delta}
        dev = wp.get_device(model.device if device is None else device)
        _, _, nw = model_world_info(model)
        n = int(elem.count)

        if n > 0:
            d = model.dismech
            e, f = d.triplet_edge0.numpy(), d.triplet_edge1.numpy()
            cnt = np.bincount(np.concatenate([e, f]), minlength=d.edge_length.shape[0])
            c = np.divide(1.0, cnt, out=np.zeros_like(cnt, dtype=np.float32), where=cnt > 0)
            self.edge_share = wp.array(np.c_[c[e], c[f]], dtype=wp.vec2, device=dev)
        else:
            self.edge_share = wp.zeros(0, dtype=wp.vec2, device=dev)

        rff = np.random.default_rng(self.seed).normal(0.0, self.rff_bandwidth, (self.rff_dim, 10)).astype(np.float32)
        self.rff_freq = wp.array(rff, dtype=vec10f, device=dev)
        has_w = self._src["w_rod"] is not None
        self.w_rod = as_rff_weights(self.w_rod, vec5f, self.rff_dim, 5, nw, dev, "w_rod")
        self.has_d_rest = bool(self.spatial_rest_delta is not None and n > 0)
        dr_val = self.spatial_rest_delta if self.has_d_rest else None
        self.d_rest = as_element_param(dr_val, vec5f, 5, n, dev, "spatial_rest_delta")
        self.elem_world = elem_world_indices(model, elem)

        data = RodResidualData()
        data.rff_freq = self.rff_freq
        data.w_rod = self.w_rod
        data.d_rest = self.d_rest
        data.has_d_rest = int(self.has_d_rest)
        data.scales = vec10f(*map(float, self.scales))
        data.iso_bending = int(bool(self.iso_bending))
        data.clamp_lo = float(self.clamp_lo)
        data.clamp_hi = float(self.clamp_hi)
        self.data = data

        params = []
        if self.rff_dim and (n or has_w):
            params.append(self.w_rod)
        if self.has_d_rest:
            params.append(self.d_rest)
        self.params = tuple(params)
        self.step_fn = _rod_con_step
        self.bounds_fn = _rod_bounds
        self._bound = True
        return self

    def stride_info(self, device) -> tuple[int, int]:
        """Return ``(lanes_per_element, block_dim)`` for launching the step kernel."""
        return (1, 1) if device.is_cpu else ((1, 128) if not self.rff_dim else (8, 128))

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays tracked by adjoint recording."""
        return list(self.params)
