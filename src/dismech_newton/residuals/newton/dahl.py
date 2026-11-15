"""Newton's Dahl rate-independent hysteresis residual (mirrors ``vbd:dahl_*`` / ``dismech:triplet_dahl``)."""

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import warp as wp
from newton import Model

from ...constants import ADJ_MODULE_OPTS, DAHL_DEADBAND, vec5f, vec6f, vec10f, vec12f
from ...worlds import elem_world_indices, model_world_info


@wp.func
def dahl_advance(d: float, sigma: float, sigma_max: float, tau: float) -> float:
    """``sigma`` after a strain change ``d``: the exact flow along the branch of ``d``'s sign (in the deadband,
    ``sigma``'s, where alternating noise returns ``sigma`` exactly)."""
    s = wp.where(wp.abs(d) > DAHL_DEADBAND, wp.sign(d), wp.where(sigma < 0.0, -1.0, 1.0))
    e = wp.exp(-s * d / tau)
    return s * sigma_max * (1.0 - e) + sigma * e


@cache
def _dahl_ops(dim: int):
    """Build dimension-specialized Dahl struct, step term, and carry kernel (``dim`` in ``{5, 6}``)."""
    vt = vec5f if dim == 5 else vec6f
    pt = vec10f if dim == 5 else vec12f

    @wp.struct
    class DahlData:
        eps_max: wp.array[float]
        tau: wp.array[float]
        eps_per_elem: int
        tau_per_elem: int
        mask: vt

    DahlData.__name__ = f"DahlData{dim}"
    DahlData.__qualname__ = f"DahlData{dim}"

    @wp.func(name=f"_dahl_step_{dim}")
    def step_term(
        t: int,
        pb_i: pt,
        r0: vt,
        er: vt,
        mem_i: vt,
        ew: wp.array[wp.int32],
        chi: float,
        dt: float,
        data: DahlData,
    ):
        ie = t if data.eps_per_elem != 0 else (ew[t] if data.eps_max.shape[0] > 1 else 0)
        it = t if data.tau_per_elem != 0 else (ew[t] if data.tau.shape[0] > 1 else 0)
        eps_max, tau = data.eps_max[ie], data.tau[it]
        on = wp.where(eps_max > 0.0 and tau > 0.0, 1.0, 0.0)
        rate = on * dt * eps_max / wp.max(tau, 1.0e-30)
        dk_m, dc_m, sig_m = vt(0.0), vt(0.0), vt(0.0)
        for j in range(dim):
            mk = data.mask[j]
            dc_m[j] = mk * rate * pb_i[j]
            sig_m[j] = mk * on * mem_i[j]
        return dk_m, dc_m, sig_m

    @wp.kernel(name=f"_dahl_carry_{dim}", module="unique", module_options=ADJ_MODULE_OPTS)
    def carry_kernel(
        prev: wp.array[vt],
        strain: wp.array[vt],
        params: wp.array[pt],
        rest: wp.array[vt],
        dr: wp.array[vt],
        hdr: int,
        mem: wp.array[vt],
        ew: wp.array[wp.int32],
        data: DahlData,
        dt: float,
        mem_out: wp.array[vt],
    ):
        t = wp.tid()
        ie = t if data.eps_per_elem != 0 else (ew[t] if data.eps_max.shape[0] > 1 else 0)
        it = t if data.tau_per_elem != 0 else (ew[t] if data.tau.shape[0] > 1 else 0)
        eps_max, tau = data.eps_max[ie], data.tau[it]
        a, b, p, m = prev[t], strain[t], params[t], mem[t]
        out = vt(0.0)
        if eps_max > 0.0 and tau > 0.0 and not wp.isnan(a[0]):
            for j in range(dim):
                if data.mask[j] != 0.0:
                    out[j] = dahl_advance(b[j] - a[j], m[j], eps_max * p[j], tau)
                else:
                    out[j] = m[j]
        elif eps_max > 0.0 and tau > 0.0:
            out = m
        mem_out[t] = out

    return DahlData, step_term, carry_kernel


def _as_dahl_param(val, n_worlds: int, n_elems: int, device, name: str, *, grad: bool) -> tuple[wp.array, int]:
    """Convert scalar, per-world, or per-element Dahl parameter into a 1D float32 ``wp.array``."""
    if isinstance(val, wp.array):
        arr_np = val.numpy()
        if np.any(arr_np < 0.0):
            raise ValueError(f"Dahl.{name} must be non-negative")
        is_same = val.device == device and val.dtype == wp.float32
        if val.shape == (n_elems,) and n_elems > 0:
            out = val if is_same else wp.array(arr_np.astype(np.float32), dtype=float, device=device)
            out.requires_grad = grad
            return out, 1
        if val.shape in ((1,), (n_worlds,)):
            out = val if is_same else wp.array(arr_np.astype(np.float32), dtype=float, device=device)
            out.requires_grad = grad
            return out, 0
        raise ValueError(f"Dahl.{name} shape mismatch: {val.shape}")

    arr = np.asarray(val, dtype=np.float32)
    if np.any(arr < 0.0):
        raise ValueError(f"Dahl.{name} must be non-negative")
    if arr.ndim == 0:
        out = wp.array([float(arr.item())], dtype=float, device=device, requires_grad=grad)
        return out, 0
    if arr.ndim == 1 and arr.shape[0] == n_elems and n_elems > 0:
        return wp.array(arr, dtype=float, device=device, requires_grad=grad), 1
    if arr.ndim == 1 and arr.shape[0] in (1, n_worlds):
        return wp.array(arr, dtype=float, device=device, requires_grad=grad), 0
    raise ValueError(f"Dahl.{name} shape mismatch: {arr.shape}")


@dataclass
class Dahl:
    """Rate-independent Dahl hysteresis on element bend/twist (or selected) strain channels."""

    eps_max: float | np.ndarray | wp.array = 0.0
    tau: float | np.ndarray | wp.array = 0.0
    enabled_dims: tuple[int, ...] | None = None
    _from_model: bool = field(default=False, repr=False)

    data: Any = field(default=None, init=False, repr=False)
    params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    eps_max_arr: wp.array | None = field(default=None, init=False, repr=False)
    tau_arr: wp.array | None = field(default=None, init=False, repr=False)
    _elem_world: wp.array | None = field(default=None, init=False, repr=False)
    _carry_kernel: Any = field(default=None, init=False, repr=False)
    _step_fn: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _dim: int = field(default=5, init=False, repr=False)
    _src: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.eps_max, (int, float)) and self.eps_max < 0.0:
            raise ValueError("Dahl.eps_max must be non-negative")
        if isinstance(self.tau, (int, float)) and self.tau < 0.0:
            raise ValueError("Dahl.tau must be non-negative")

    @classmethod
    def from_model(cls, model: Model) -> "Dahl | None":
        """Create a :class:`Dahl` law from model attributes (``dismech:triplet_dahl`` or ``vbd:dahl_*``), if active."""
        der = getattr(model, "dismech", None)
        td_wp = getattr(der, "triplet_dahl", None) if der is not None else None
        if td_wp is not None and td_wp.shape[0]:
            td = td_wp.numpy()
            if (td > 0.0).all(1).any():
                return cls(eps_max=td[:, 0], tau=td[:, 1], _from_model=True)
        vbd = getattr(model, "vbd", None)
        if vbd is not None:
            em = getattr(vbd, "dahl_eps_max", None)
            ta = getattr(vbd, "dahl_tau", None)
            if em is not None and ta is not None and em.shape[0] and ((em.numpy() > 0.0) & (ta.numpy() > 0.0)).any():
                return cls(eps_max=em, tau=ta, _from_model=True)
        return None

    def bind(self, model: Model, elem, device=None) -> "Dahl":
        """Allocate/bind Dahl buffers for ``model`` and ``elem`` on ``device``."""
        dev = wp.get_device(model.device if device is None else device)
        dim = 5 if elem.stored == "triplet_strain_q" else 6
        default_dims = (2, 3, 4) if dim == 5 else (3, 4, 5)
        dims = default_dims if self.enabled_dims is None else self.enabled_dims
        if any(i < 0 or i >= dim for i in dims):
            raise ValueError(f"Invalid Dahl enabled_dims={dims} for {dim}D elements")

        _, _, nw = model_world_info(model)
        grad = not self._from_model
        eps_arr, eps_mode = _as_dahl_param(self.eps_max, nw, int(elem.count), dev, "eps_max", grad=grad)
        tau_arr, tau_mode = _as_dahl_param(self.tau, nw, int(elem.count), dev, "tau", grad=grad)

        data_cls, step_fn, carry_kernel = _dahl_ops(dim)
        vt = vec5f if dim == 5 else vec6f
        d = data_cls()
        d.eps_max = eps_arr
        d.tau = tau_arr
        d.eps_per_elem = eps_mode
        d.tau_per_elem = tau_mode
        d.mask = vt(*(float(i in dims) for i in range(dim)))

        self._dim = dim
        self._device = dev
        self._elem_world = elem_world_indices(model, elem)
        self.eps_max_arr = eps_arr
        self.tau_arr = tau_arr
        self.data = d
        self._step_fn = step_fn
        self._carry_kernel = carry_kernel
        self.params = (eps_arr, tau_arr) if grad else ()
        self._bound = True
        return self

    def step_term(self, dim: int):
        """Return the ``@wp.func`` computing ``(dk_mem, dc_mem, sig_mem)`` for strain dimension ``dim``."""
        if dim != self._dim:
            raise ValueError(f"Dahl bound for dim={self._dim}, got dim={dim}")
        return self._step_fn

    def carry(self, elem, prev_state, next_state, dt: float, d_rest: wp.array, has_d_rest: int) -> None:
        """Advance Dahl stresses from ``prev_state`` into ``next_state`` (independent of ``dt``)."""
        if not elem.count:
            return
        bd = 1 if self._device.is_cpu else 128
        wp.launch(
            self._carry_kernel,
            elem.count,
            block_dim=bd,
            device=self._device,
            inputs=[
                getattr(prev_state, elem.stored),
                getattr(next_state, elem.stored),
                elem.params,
                elem.rest,
                d_rest,
                has_d_rest,
                getattr(prev_state, elem.memory),
                self._elem_world,
                self.data,
                dt,
            ],
            outputs=[getattr(next_state, elem.memory)],
        )
