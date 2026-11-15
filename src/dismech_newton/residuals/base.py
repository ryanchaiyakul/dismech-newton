"""Shared residual orchestrator, Memory/Corrector protocols, parameter builders, and composed element kernels."""

import copy
from dataclasses import dataclass, field, is_dataclass
from functools import cache
from typing import Any, Protocol, runtime_checkable

import numpy as np
import warp as wp
from newton import Model

from ..constants import ADJ_MODULE_OPTS, vec5f, vec6f, vec10f, vec11f, vec11i, vec12f, vec21f, vec21i
from ..elements.rod import Rod, rod_dofs, rod_strain_rate
from ..elements.shell import Shell, shell_dofs, shell_strain_rate
from ..worlds import elem_world_indices


@wp.struct
class EmptyData:
    """Placeholder struct when no constitutive or memory residual is active for a channel."""

    dummy: int


@runtime_checkable
class ElementCorrector(Protocol):
    """Protocol for constitutive residual correctors evaluated in the implicit element step."""

    params: tuple[wp.array, ...]
    data: Any
    step_fn: Any
    bounds_fn: Any

    def bind(self, model: Model, elem, device=None) -> "ElementCorrector":
        """Allocate/bind GPU buffers for ``model`` and ``elem`` on ``device`` and return the bound corrector."""
        ...

    def stride_info(self, device) -> tuple[int, int]:
        """Return ``(lanes_per_element, block_dim)`` for launching the step kernel."""
        ...


@runtime_checkable
class Memory(Protocol):
    """Protocol for carried internal memory laws pluggable into :class:`Residual`."""

    params: tuple[wp.array, ...]
    data: Any

    def bind(self, model: Model, elem, device=None) -> "Memory":
        """Allocate/bind GPU buffers for ``model`` and ``elem`` on ``device`` and return the bound memory law."""
        ...

    def step_term(self, dim: int):
        """Return the ``@wp.func`` computing ``(dk_mem, dc_mem, sig_mem)`` for strain dimension ``dim``."""
        ...

    def carry(self, elem, prev_state, next_state, dt: float, d_rest: wp.array, has_d_rest: int) -> None:
        """Advance carried memory from ``prev_state`` into ``next_state`` over step ``dt``."""
        ...


def clone_if_bound(obj):
    """Return a fresh shallow copy of a residual dataclass if it was already bound to another solver."""
    if not getattr(obj, "_bound", False):
        return obj
    cloned = copy.copy(obj)
    for key, val in getattr(obj, "_src", {}).items():
        setattr(cloned, key, val)
    cloned._src = {}
    cloned._bound = False
    return cloned


def _to_grad_array(arr, dtype, expected_shapes: tuple[tuple[int, ...], ...], device, name: str) -> wp.array:
    """Validate shape/dtype, move to ``device`` if needed, and enable gradients."""
    is_wp = isinstance(arr, wp.array)
    if is_wp and arr.device == device and arr.dtype == dtype:
        out = arr
    else:
        np_val = arr.numpy() if is_wp else arr
        out = wp.array(np_val, dtype=dtype, device=device)
    if out.shape not in expected_shapes:
        raise ValueError(f"{name} shape/dtype mismatch: shape={out.shape}, expected one of {expected_shapes}")
    out.requires_grad = True
    return out


def _as_vec_array(val, vec_dtype, nv: int, n_worlds: int, device, name: str, default: float = 0.0) -> wp.array:
    """Convert scalar, ``(nv,)``, or ``(1|W, nv)`` array to differentiable ``vec_dtype`` array on ``device``."""
    if val is None:
        arr = np.full((1, nv), float(default), dtype=np.float32)
        return _to_grad_array(arr, vec_dtype, ((1,),), device, name) if vec_dtype is not float else _to_grad_array(
            arr, float, ((1, nv),), device, name
        )
    if isinstance(val, wp.array):
        if vec_dtype is float:
            if val.dtype != wp.float32:
                raise ValueError(f"{name} expected float32 wp.array, got {val.dtype}")
            arr = val.reshape((1, nv)) if val.shape == (nv,) else val
            return _to_grad_array(arr, float, ((1, nv), (n_worlds, nv)), device, name)
        if val.dtype == vec_dtype:
            return _to_grad_array(val, vec_dtype, ((1,), (n_worlds,)), device, name)
        if val.dtype == wp.float32 and val.shape[-1] == nv and val.ndim in (1, 2):
            viewed = (val.reshape(-1) if val.ndim == 1 else val).view(vec_dtype)
            return _to_grad_array(viewed, vec_dtype, ((1,), (n_worlds,)), device, name)
        raise ValueError(f"{name} shape/dtype mismatch: shape={val.shape}, dtype={val.dtype}")

    arr = np.asarray(val, dtype=np.float32)
    if arr.size == 1 and arr.ndim == 0:
        arr = np.full((1, nv), float(arr.item()), dtype=np.float32)
    elif arr.shape == (nv,):
        arr = arr[None]
    elif arr.shape not in ((1, nv), (n_worlds, nv)):
        raise ValueError(f"{name} shape mismatch: shape={arr.shape}")
    if vec_dtype is float:
        return _to_grad_array(arr, float, ((1, nv), (n_worlds, nv)), device, name)
    return _to_grad_array(arr, vec_dtype, ((1,), (n_worlds,)), device, name)


def as_world_param(val, vec_dtype, nv: int, n_worlds: int, device, name: str, default: float = 0.0) -> wp.array:
    """Convert scalar or per-world parameter ``val`` into a differentiable ``wp.array``."""
    return _as_vec_array(val, vec_dtype, nv, n_worlds, device, name, default=default)


def as_rff_weights(val, vec_dtype, rows: int, nv: int, n_worlds: int, device, name: str) -> wp.array:
    """Convert ``(rows, nv)`` or ``(1|W, rows, nv)`` weight table into a differentiable ``wp.array``."""
    r0 = max(int(rows), 0)
    if val is None:
        if vec_dtype is float:
            return wp.zeros((1, r0, nv), dtype=float, device=device, requires_grad=True)
        return wp.zeros((1, r0), dtype=vec_dtype, device=device, requires_grad=True)

    if isinstance(val, wp.array):
        if vec_dtype is float:
            if val.dtype != wp.float32:
                raise ValueError(f"{name} expected float32 wp.array, got {val.dtype}")
            arr = val.reshape((1, r0, nv)) if val.shape == (r0, nv) else val
            return _to_grad_array(arr, float, ((1, r0, nv), (n_worlds, r0, nv)), device, name)
        if val.dtype == vec_dtype:
            arr = val.reshape((1, r0)) if val.shape == (r0,) else val
            return _to_grad_array(arr, vec_dtype, ((1, r0), (n_worlds, r0)), device, name)
        if val.dtype == wp.float32 and val.shape in ((r0, nv), (1, r0, nv), (n_worlds, r0, nv)):
            viewed = val.reshape((1, r0, nv)).view(vec_dtype) if val.ndim == 2 else val.view(vec_dtype)
            return _to_grad_array(viewed, vec_dtype, ((1, r0), (n_worlds, r0)), device, name)
        raise ValueError(f"{name} shape/dtype mismatch: shape={val.shape}, dtype={val.dtype}")

    arr = np.asarray(val, dtype=np.float32)
    if arr.shape == (r0, nv):
        arr = arr[None]
    elif arr.shape not in ((1, r0, nv), (n_worlds, r0, nv)):
        raise ValueError(f"{name} shape mismatch: shape={arr.shape}")
    if vec_dtype is float:
        return _to_grad_array(arr, float, ((1, r0, nv), (n_worlds, r0, nv)), device, name)
    return _to_grad_array(arr, vec_dtype, ((1, r0), (n_worlds, r0)), device, name)


def as_element_param(val, vec_dtype, nv: int, n_elems: int, device, name: str) -> wp.array:
    """Convert ``(n_elems, nv)`` per-element shift into a differentiable ``wp.array`` of ``vec_dtype``."""
    if val is None:
        return wp.zeros(n_elems, dtype=vec_dtype, device=device, requires_grad=True)
    if isinstance(val, wp.array):
        if val.dtype == vec_dtype:
            return _to_grad_array(val, vec_dtype, ((n_elems,),), device, name)
        if val.dtype == wp.float32 and val.shape == (n_elems, nv):
            return _to_grad_array(val.view(vec_dtype), vec_dtype, ((n_elems,),), device, name)
        raise ValueError(f"{name} shape/dtype mismatch: shape={val.shape}, dtype={val.dtype}")
    arr = np.asarray(val, dtype=np.float32)
    if arr.shape != (n_elems, nv):
        raise ValueError(f"{name} shape mismatch: shape={arr.shape}, expected ({n_elems}, {nv})")
    return _to_grad_array(arr, vec_dtype, ((n_elems,),), device, name)


@wp.func
def clamp_channel(
    k0: float,
    c0: float,
    sig: float,
    dkr: float,
    dcr: float,
    r0: float,
    er: float,
    ed: float,
    sig_mem: float,
    dc_mem: float,
    clo: float,
    chi: float,
    dt: float,
) -> wp.vec3:
    """SPD tangent clamping and semi-implicit rest shift for one strain channel."""
    dk = wp.where(k0 > 0.0, wp.clamp(dkr, clo * k0, chi * k0), 0.0)
    ks, den = k0 + dk, wp.max(k0 + dk, 1.0e-30)
    dc = wp.where(
        c0 > 0.0,
        wp.clamp(dcr, clo * c0, chi * c0),
        wp.where(k0 > 0.0, wp.clamp(dcr, 0.0, chi * k0 * dt), 0.0),
    )
    return wp.vec3(ks, c0 + dc + dc_mem, wp.where(ks > 0.0, r0 - sig_mem / den - (sig - dk * er - dc * ed) / den, r0))


@wp.func
def _gather11(a: wp.array[float], d: vec11i) -> vec11f:
    return vec11f(
        a[d[0]], a[d[1]], a[d[2]], a[d[3]], a[d[4]], a[d[5]], a[d[6]], a[d[7]], a[d[8]], a[d[9]], a[d[10]]
    )


@wp.func
def _gather21(a: wp.array[float], d: vec21i) -> vec21f:
    return vec21f(
        a[d[0]], a[d[1]], a[d[2]], a[d[3]], a[d[4]], a[d[5]], a[d[6]], a[d[7]], a[d[8]], a[d[9]],
        a[d[10]], a[d[11]], a[d[12]], a[d[13]], a[d[14]], a[d[15]], a[d[16]], a[d[17]], a[d[18]], a[d[19]], a[d[20]],
    )


@cache
def _empty_fns(dim: int):
    """Typed no-op constitutive and memory terms for element dimension ``dim``."""
    vt = vec5f if dim == 5 else vec6f
    pt = vec10f if dim == 5 else vec12f
    v_type = Rod if dim == 5 else Shell
    aux_type = wp.array[wp.vec2] if dim == 5 else wp.array[float]

    @wp.func(name=f"_no_con_{dim}")
    def no_con(
        t: int,
        lane: int,
        st: int,
        v: v_type,
        eps: vt,
        ed: vt,
        rb_i: vt,
        aux: aux_type,
        ew: wp.array[wp.int32],
        data: EmptyData,
    ):
        z = vt(0.0)
        return rb_i, eps - rb_i, z, z, z

    @wp.func(name=f"_no_bounds_{dim}")
    def no_bounds(data: EmptyData) -> wp.vec2:
        return wp.vec2(-0.35, 8.0)

    @wp.func(name=f"_no_mem_{dim}")
    def no_mem(
        t: int, pb_i: pt, r0: vt, er: vt, mem_i: vt, ew: wp.array[wp.int32], chi: float, dt: float, data: EmptyData
    ):
        z = vt(0.0)
        return z, z, z

    return no_con, no_bounds, no_mem


@cache
def make_element_step_kernel(dim: int, con_fn, bounds_fn, con_type, mem_fn, mem_type):
    """Build and cache a composed ``module='unique'`` element step kernel for ``(con_fn, mem_fn)``."""
    if dim == 5:
        vt, pt, v_type = vec5f, vec10f, Rod
        dofs_fn, rate_fn, gather_fn = rod_dofs, rod_strain_rate, _gather11
        aux_type = wp.array[wp.vec2]
    else:
        vt, pt, v_type = vec6f, vec12f, Shell
        dofs_fn, rate_fn, gather_fn = shell_dofs, shell_strain_rate, _gather21
        aux_type = wp.array[float]

    tag = f"{dim}_{con_fn.func.__name__}_{mem_fn.func.__name__}"

    @wp.kernel(name=f"_elem_step_{tag}", module="unique", module_options=ADJ_MODULE_OPTS)
    def step(
        q: wp.array[float],
        qd: wp.array[float],
        v: v_type,
        pb: wp.array[pt],
        rb: wp.array[vt],
        sq: wp.array[vt],
        mem: wp.array[vt],
        aux: aux_type,
        ew: wp.array[wp.int32],
        con_data: con_type,
        mem_data: mem_type,
        st: int,
        dt: float,
        pout: wp.array[pt],
        rout: wp.array[vt],
    ):
        t, lane = wp.tid() // st, wp.tid() % st
        eps, ed, rb_i = vt(0.0), vt(0.0), vt(0.0)
        if lane == 0:
            d = dofs_fn(v, t)
            eps, ed = rate_fn(gather_fn(q, d), gather_fn(qd, d), v, t)
            rb_i = rb[t]
        r0, er, sig_c, dk_c, dc_c = con_fn(t, lane, st, v, eps, ed, rb_i, aux, ew, con_data)
        if lane == 0:
            pb_i = pb[t]
            b = bounds_fn(con_data)
            clo, chi = b[0], b[1]
            dk_m, dc_m, sig_m = mem_fn(t, pb_i, r0, er, mem[t], ew, chi, dt, mem_data)
            ef = (eps - sq[t]) / dt if not wp.isnan(sq[t][0]) else vt(0.0)
            po, ro = pt(), vt()
            for j in range(dim):
                r = clamp_channel(
                    pb_i[j],
                    pb_i[dim + j],
                    sig_c[j],
                    dk_c[j] + dk_m[j],
                    dc_c[j],
                    r0[j],
                    er[j],
                    ef[j],
                    sig_m[j],
                    dc_m[j],
                    clo,
                    chi,
                    dt,
                )
                po[j] = r[0]
                po[dim + j] = r[1]
                ro[j] = r[2]
            pout[t] = po
            rout[t] = ro

    return step


@dataclass
class Residual:
    """Top-level modular residual configuration attaching any combination of domain correctors to ``SolverDiSMech``."""

    rod: ElementCorrector | None = None
    shell: ElementCorrector | None = None
    rod_memory: Memory | None = None
    shell_memory: Memory | None = None
    joint: Any | None = None
    contact: Any | None = None

    _device: Any = field(default=None, init=False, repr=False)
    _elem_world: dict[int, wp.array] = field(default_factory=dict, init=False, repr=False)
    _zero_aux: dict[int, wp.array] = field(default_factory=dict, init=False, repr=False)
    _empty: EmptyData = field(default_factory=EmptyData, init=False, repr=False)
    _params: tuple[wp.array, ...] = field(default=(), init=False, repr=False)
    _bound: bool = field(default=False, init=False, repr=False)
    _rods: Any = field(default=None, init=False, repr=False)
    _shells: Any = field(default=None, init=False, repr=False)

    @classmethod
    def coerce(cls, spec, *, rod_count: int = 1, shell_count: int = 0) -> "Residual | None":
        """Normalize a single corrector or memory law directly passed to ``SolverDiSMech(residual=...)``."""
        if spec is None or isinstance(spec, cls):
            return spec
        from .contact import ContactResidual  # noqa: PLC0415
        from .joint import JointResidual  # noqa: PLC0415
        from .rod import RodResidual  # noqa: PLC0415
        from .shell import ShellResidual  # noqa: PLC0415

        if isinstance(spec, RodResidual):
            return cls(rod=spec)
        if isinstance(spec, ShellResidual):
            return cls(shell=spec)
        if isinstance(spec, JointResidual):
            return cls(joint=spec)
        if isinstance(spec, ContactResidual):
            return cls(contact=spec)
        if is_dataclass(spec) and isinstance(spec, Memory):
            return cls(rod_memory=spec) if (rod_count > 0 or shell_count == 0) else cls(shell_memory=spec)
        raise TypeError(f"Unsupported residual specification: {type(spec).__name__}")

    @property
    def has_rod(self) -> bool:
        """Whether rod constitutive or memory residuals are active."""
        return self.rod is not None or self.rod_memory is not None

    @property
    def has_shell(self) -> bool:
        """Whether shell constitutive or memory residuals are active."""
        return self.shell is not None or self.shell_memory is not None

    @property
    def has_rod_memory(self) -> bool:
        """Whether a rod internal memory law is active."""
        return self.rod_memory is not None

    @property
    def has_shell_memory(self) -> bool:
        """Whether a shell internal memory law is active."""
        return self.shell_memory is not None

    def bind(self, model: Model, triplets, shells, device=None) -> "Residual":
        """Bind all active domain correctors and memory laws onto ``model`` and element blocks."""
        if self._bound:
            return Residual(
                rod=self.rod,
                shell=self.shell,
                rod_memory=self.rod_memory,
                shell_memory=self.shell_memory,
                joint=self.joint,
                contact=self.contact,
            ).bind(model, triplets, shells, device=device)

        dev = wp.get_device(model.device if device is None else device)
        self._device = dev
        self._empty = EmptyData()
        params: list[wp.array] = []

        if self.rod is not None:
            self.rod = clone_if_bound(self.rod).bind(model, triplets, device=dev)
            params.extend(self.rod.params)
        if self.rod_memory is not None:
            self.rod_memory = clone_if_bound(self.rod_memory).bind(model, triplets, device=dev)
            params.extend(self.rod_memory.params)
        if self.shell is not None:
            self.shell = clone_if_bound(self.shell).bind(model, shells, device=dev)
            params.extend(self.shell.params)
        if self.shell_memory is not None:
            self.shell_memory = clone_if_bound(self.shell_memory).bind(model, shells, device=dev)
            params.extend(self.shell_memory.params)
        if self.joint is not None:
            self.joint = clone_if_bound(self.joint).bind(model, device=dev)
            params.extend(self.joint.params)
        if self.contact is not None:
            self.contact = clone_if_bound(self.contact).bind(model, device=dev)
            params.extend(self.contact.params)

        self._elem_world[5] = elem_world_indices(model, triplets)
        self._elem_world[6] = elem_world_indices(model, shells)
        self._zero_aux[5] = wp.zeros(0, dtype=wp.vec2, device=dev) if not triplets.count else (
            self.rod.edge_share if self.rod is not None else wp.zeros(triplets.count, dtype=wp.vec2, device=dev)
        )
        self._zero_aux[6] = shells.area
        self._params = tuple(params)
        self._bound = True
        return self

    def grad_inputs(self) -> list[wp.array]:
        """Differentiable parameter arrays tracked by ``SolverDiSMech._record``."""
        return list(self._params)

    def _eval(self, dim: int, el, s, dt: float, pb, rb, po, ro) -> None:
        if not el.count:
            return
        con = self.rod if dim == 5 else self.shell
        mem = self.rod_memory if dim == 5 else self.shell_memory
        no_con, no_bounds, no_mem = _empty_fns(dim)
        if con is not None:
            con_fn, bounds_fn, con_data = con.step_fn, con.bounds_fn, con.data
            st, bd = con.stride_info(self._device)
        else:
            con_fn, bounds_fn, con_data = no_con, no_bounds, self._empty
            st, bd = (1, 1) if self._device.is_cpu else (1, 128)
        if mem is not None:
            mem_fn, mem_data = mem.step_term(dim), mem.data
        else:
            mem_fn, mem_data = no_mem, self._empty

        con_type = getattr(con_data, "_cls", type(con_data))
        mem_type = getattr(mem_data, "_cls", type(mem_data))
        kern = make_element_step_kernel(dim, con_fn, bounds_fn, con_type, mem_fn, mem_type)
        wp.launch(
            kern,
            el.count * st,
            block_dim=bd,
            device=self._device,
            inputs=[
                s.q,
                s.qd,
                el.bind(s),
                pb,
                rb,
                getattr(s, el.stored),
                getattr(s, el.memory),
                self._zero_aux[dim],
                self._elem_world[dim],
                con_data,
                mem_data,
                st,
                dt,
            ],
            outputs=[el.params_step if po is None else po, el.rest_step if ro is None else ro],
        )

    def _carry(self, dim: int, el, p, s, dt: float) -> None:
        if not el.count:
            return
        con = self.rod if dim == 5 else self.shell
        mem = self.rod_memory if dim == 5 else self.shell_memory
        if mem is None:
            return
        has_dr = 1 if (con is not None and con.has_d_rest) else 0
        dr = con.d_rest if has_dr else el.rest
        mem.carry(el, p, s, dt, dr, has_dr)

    def eval_rod(self, rods, s, dt: float, pb, rb, po=None, ro=None) -> None:
        """Evaluate effective ``(params, rest)`` for DER rods."""
        self._eval(5, rods, s, dt, pb, rb, po, ro)

    def eval_shell(self, shells, s, dt: float, pb, rb, po=None, ro=None) -> None:
        """Evaluate effective ``(params, rest)`` for BAC shells."""
        self._eval(6, shells, s, dt, pb, rb, po, ro)

    def carry_rod(self, rods, p, s, dt: float) -> None:
        """Advance rod internal memory across ``(p -> s)``."""
        self._carry(5, rods, p, s, dt)

    def carry_shell(self, shells, p, s, dt: float) -> None:
        """Advance shell internal memory across ``(p -> s)``."""
        self._carry(6, shells, p, s, dt)
