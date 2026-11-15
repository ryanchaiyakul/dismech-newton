"""External pluggable ``Memory`` laws (``Prony``, ``Elastoplastic``, ``BoucWen``) exercising ``Residual``."""

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import warp as wp
from newton import Model

from dismech_newton.constants import ADJ_MODULE_OPTS, vec5f, vec6f, vec10f, vec12f
from dismech_newton.residuals.base import as_world_param
from dismech_newton.worlds import elem_world_indices, model_world_info


@cache
def _law_ops(dim: int, mode: int):
    """Build dimension- and law-specialized struct, ``step_term``, and carry kernel."""
    vt = vec5f if dim == 5 else vec6f
    pt = vec10f if dim == 5 else vec12f

    @wp.func(name=f"_gv_mem_{dim}_{mode}")
    def _gv(a: wp.array[vt], w: int) -> vt:
        return a[wp.where(a.shape[0] > 1, w, 0)]

    @wp.func(name=f"_yg_mem_{dim}_{mode}")
    def _yg(d: vt, mk: vt, yv: vt, ye: float, yd: float) -> wp.vec3:
        r2 = float(0.0)
        for j in range(dim):
            r2 = r2 + yv[j] * yv[j] * mk[j] * d[j] * d[j]
        if r2 <= ye * ye:
            return wp.vec3(0.0)
        ex = wp.sqrt(r2) - ye
        return wp.vec3(ex + ye, ex * ex / (ex + yd), ex * (ex + 2.0 * yd) / ((ex + yd) * (ex + yd)))

    @wp.struct
    class LawData:
        k_mem: wp.array[vt]
        flow: wp.array[vt]
        tau: wp.array[vt]
        bw_sat: wp.array[vt]
        yield_weight: wp.array[vt]
        mask: vt
        yield_eps: float
        yield_delta: float

    LawData.__name__ = f"MemLawData_{dim}_{mode}"
    LawData.__qualname__ = f"MemLawData_{dim}_{mode}"

    @wp.func(name=f"_mem_step_term_{dim}_{mode}")
    def step_term(
        t: int,
        pb_i: pt,
        r0: vt,
        er: vt,
        mem_i: vt,
        ew: wp.array[wp.int32],
        chi: float,
        dt: float,
        data: LawData,
    ):
        w = ew[t]
        dk_m, dc_m, sig_m = vt(0.0), vt(0.0), vt(0.0)
        if wp.static(mode == 1):
            flv = _gv(data.flow, w)
            yg = _yg(er - mem_i, data.mask, _gv(data.yield_weight, w), data.yield_eps, data.yield_delta)
        if wp.static(mode == 2):
            kmv, tav = _gv(data.k_mem, w), _gv(data.tau, w)
        if wp.static(mode == 3):
            kmv, flv, bsv = _gv(data.k_mem, w), _gv(data.flow, w), _gv(data.bw_sat, w)

        for j in range(dim):
            if data.mask[j] != 0.0:
                k0 = pb_i[j]
                if wp.static(mode == 1):
                    dk_m[j] = -k0 * wp.max(flv[j], 0.0) * yg[2]
                    sig_m[j] = -k0 * mem_i[j]
                if wp.static(mode == 2):
                    dkj = wp.max(kmv[j], 0.0) * k0 * wp.exp(-dt / wp.max(tav[j], 1.0e-6))
                    dk_m[j] = dkj
                    sig_m[j] = dkj * (er[j] - mem_i[j])
                if wp.static(mode == 3):
                    kp = wp.max(kmv[j], 0.0) * k0
                    u = wp.clamp(mem_i[j] / wp.max(bsv[j], 1.0e-6), -1.0, 1.0)
                    dc_m[j] = wp.min(dt * kp * wp.max(flv[j], 0.0) * (1.0 - 0.5 * u * u), chi * k0 * dt)
                    sig_m[j] = kp * mem_i[j]
        return dk_m, dc_m, sig_m

    @wp.kernel(name=f"_mem_carry_{dim}_{mode}", module="unique", module_options=ADJ_MODULE_OPTS)
    def carry_kernel(
        prev: wp.array[vt],
        strain: wp.array[vt],
        rb: wp.array[vt],
        dr: wp.array[vt],
        hdr: int,
        mem: wp.array[vt],
        ew: wp.array[wp.int32],
        data: LawData,
        dt: float,
        mem_out: wp.array[vt],
    ):
        t = wp.tid()
        h, out = mem[t], vt(0.0)
        if dt > 0.0:
            ep, w = strain[t], ew[t]
            r0 = rb[t] + dr[t] if hdr != 0 else rb[t]
            if wp.static(mode == 1):
                flv = _gv(data.flow, w)
                yg = _yg(ep - r0 - h, data.mask, _gv(data.yield_weight, w), data.yield_eps, data.yield_delta)
            if wp.static(mode == 2):
                tav = _gv(data.tau, w)
            if wp.static(mode == 3):
                flv, bsv = _gv(data.flow, w), _gv(data.bw_sat, w)

            for j in range(dim):
                v = h[j]
                if data.mask[j] != 0.0:
                    if wp.static(mode == 1):
                        if yg[0] > 0.0:
                            v = h[j] + wp.max(flv[j], 0.0) * yg[1] * (ep[j] - r0[j] - h[j]) / yg[0]
                    if wp.static(mode == 2):
                        v = h[j] + (1.0 - wp.exp(-dt / wp.max(tav[j], 1.0e-6))) * (ep[j] - r0[j] - h[j])
                    if wp.static(mode == 3):
                        d, b_val, ad = ep[j] - prev[t][j], wp.max(bsv[j], 1.0e-6), wp.abs(ep[j] - prev[t][j])
                        if ad > data.yield_eps:
                            sm = ad if data.yield_eps <= 0.0 else (ad - data.yield_eps) ** 2.0 / (ad - data.yield_eps + data.yield_delta)
                            u = wp.clamp(h[j] / b_val, -1.0, 1.0)
                            sn = wp.where(d >= 0.0, 1.0, -1.0)
                            f = 1.0 - 0.5 * u * u * (1.0 + sn * wp.where(h[j] >= 0.0, 1.0, -1.0))
                            v = wp.clamp(h[j] + sn * sm * wp.max(flv[j], 0.0) * f, -b_val, b_val)
                out[j] = v
        mem_out[t] = out if dt > 0.0 else h

    return LawData, step_term, carry_kernel


class _MemoryLawMixin:
    _mode: int
    _param_names: tuple[str, ...]

    def _bind_law(self, model: Model, elem, device=None, *, yield_eps: float = 0.0, yield_delta: float = 1.0e-3):
        if yield_eps < 0.0 or yield_delta <= 0.0:
            raise ValueError(f"Invalid yield parameters: yield_eps={yield_eps}, yield_delta={yield_delta}")
        dev = wp.get_device(model.device if device is None else device)
        dim = 5 if elem.stored == "triplet_strain_q" else 6
        vt = vec5f if dim == 5 else vec6f
        default_dims = (2, 3, 4) if dim == 5 else (3, 4, 5)
        dims = default_dims if self.enabled_dims is None else self.enabled_dims
        if any(i < 0 or i >= dim for i in dims):
            raise ValueError(f"Invalid enabled_dims={dims} for {dim}D elements")

        _, _, nw = model_world_info(model)
        data_cls, step_fn, carry_kernel = _law_ops(dim, self._mode)
        d = data_cls()
        self._src = {k: getattr(self, k) for k in ("k_mem", "flow", "tau", "bw_sat", "yield_weight") if hasattr(self, k)}
        for k, default in (("k_mem", 0.25), ("flow", 0.5), ("tau", 0.1), ("bw_sat", 0.1), ("yield_weight", 1.0)):
            val = getattr(self, k, default)
            arr = as_world_param(val, vt, dim, nw, dev, k, default=default)
            setattr(d, k, arr)
            if hasattr(self, k):
                setattr(self, k, arr)
        d.mask = vt(*(float(i in dims) for i in range(dim)))
        d.yield_eps = float(yield_eps)
        d.yield_delta = float(yield_delta)

        self.data = d
        self.params = tuple(getattr(d, k) for k in self._param_names) if elem.count else ()
        self._dim = dim
        self._device = dev
        self._elem_world = elem_world_indices(model, elem)
        self._step_fn = step_fn
        self._carry_kernel = carry_kernel
        self._prev_buf = wp.empty(int(elem.count) if self._mode == 3 else 0, dtype=vt, device=dev, requires_grad=True)
        self._bound = True
        return self

    def step_term(self, dim: int):
        if dim != self._dim:
            raise ValueError(f"{type(self).__name__} bound for dim={self._dim}, got dim={dim}")
        return self._step_fn

    def carry(self, elem, prev_state, next_state, dt: float, d_rest: wp.array, has_d_rest: int) -> None:
        if not elem.count:
            return
        prev = getattr(prev_state, elem.stored)
        out = getattr(next_state, elem.memory)
        if self._mode == 3 and dt > 0.0:
            if out.requires_grad and self._prev_buf.grad is not None:
                self._prev_buf.grad.zero_()
            wp.launch(
                elem.kernels.previous,
                elem.count,
                inputs=[prev, prev_state.q, elem.bind(prev_state)],
                outputs=[self._prev_buf],
                device=self._device,
            )
            prev = self._prev_buf
        wp.launch(
            self._carry_kernel,
            elem.count,
            block_dim=1 if self._device.is_cpu else 128,
            device=self._device,
            outputs=[out],
            inputs=[
                prev,
                getattr(next_state, elem.stored),
                elem.rest,
                d_rest,
                has_d_rest,
                getattr(prev_state, elem.memory),
                self._elem_world,
                self.data,
                dt,
            ],
        )


@dataclass
class Elastoplastic(_MemoryLawMixin):
    """Smooth Perzyna-type elastoplastic yield internal memory law."""

    _mode = 1
    _param_names = ("flow", "yield_weight")

    flow: float | np.ndarray | wp.array = 0.5
    yield_eps: float = 0.0
    yield_delta: float = 1.0e-3
    yield_weight: float | np.ndarray | wp.array = 1.0
    enabled_dims: tuple[int, ...] | None = None

    data: Any = field(default=None, init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    _elem_world: wp.array | None = field(default=None, init=False, repr=False)
    _step_fn: Any = field(default=None, init=False, repr=False)
    _carry_kernel: Any = field(default=None, init=False, repr=False)
    _prev_buf: wp.array | None = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _dim: int = field(default=5, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.yield_eps < 0.0 or self.yield_delta <= 0.0:
            raise ValueError("Invalid Elastoplastic parameters")

    def bind(self, model: Model, elem, device=None) -> "Elastoplastic":
        return self._bind_law(model, elem, device=device, yield_eps=self.yield_eps, yield_delta=self.yield_delta)


@dataclass
class Prony(_MemoryLawMixin):
    """Linear viscoelastic Prony internal variable memory law."""

    _mode = 2
    _param_names = ("k_mem", "tau")

    k_mem: float | np.ndarray | wp.array = 0.25
    tau: float | np.ndarray | wp.array = 0.1
    enabled_dims: tuple[int, ...] | None = None

    data: Any = field(default=None, init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    _elem_world: wp.array | None = field(default=None, init=False, repr=False)
    _step_fn: Any = field(default=None, init=False, repr=False)
    _carry_kernel: Any = field(default=None, init=False, repr=False)
    _prev_buf: wp.array | None = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _dim: int = field(default=5, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def bind(self, model: Model, elem, device=None) -> "Prony":
        return self._bind_law(model, elem, device=device)


@dataclass
class BoucWen(_MemoryLawMixin):
    """Smooth hysteretic Bouc-Wen internal memory law."""

    _mode = 3
    _param_names = ("k_mem", "flow", "bw_sat")

    k_mem: float | np.ndarray | wp.array = 0.25
    flow: float | np.ndarray | wp.array = 0.5
    bw_sat: float | np.ndarray | wp.array = 0.1
    yield_eps: float = 0.0
    yield_delta: float = 1.0e-3
    enabled_dims: tuple[int, ...] | None = None

    data: Any = field(default=None, init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    _elem_world: wp.array | None = field(default=None, init=False, repr=False)
    _step_fn: Any = field(default=None, init=False, repr=False)
    _carry_kernel: Any = field(default=None, init=False, repr=False)
    _prev_buf: wp.array | None = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _dim: int = field(default=5, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.yield_eps < 0.0 or self.yield_delta <= 0.0:
            raise ValueError("Invalid BoucWen parameters")

    def bind(self, model: Model, elem, device=None) -> "BoucWen":
        return self._bind_law(model, elem, device=device, yield_eps=self.yield_eps, yield_delta=self.yield_delta)
