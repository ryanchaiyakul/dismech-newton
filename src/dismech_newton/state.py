"""The flat DOF vector and a step's per-DOF terms.

DOF ``3 * node + k`` is ``particle_q[node][k]``; DOF ``3 * N + edge`` is ``edge_q[edge]``; DOF ``3 * N + E + hinge``
is ``hinge_q[hinge]`` (a shell's midedge angle).
"""

from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp
from newton import Model, ParticleFlags, State

from .builder import ATTRIBUTES, STATE_ATTRIBUTES, complete


def dof_constants(model: Model) -> tuple[wp.array, wp.array]:
    """Per-DOF ``(mass, fixed)`` (:func:`dof_mass`)."""
    der = model.dismech
    complete(der, ATTRIBUTES, model.device)
    active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
    fixed = np.concatenate([np.repeat(~active, 3), der.edge_fixed.numpy() != 0, der.hinge_fixed.numpy() != 0])
    mass = wp.zeros(len(fixed), dtype=float, device=model.device)
    dof_mass(model, mass)
    return mass, wp.array(fixed, dtype=wp.int32, device=model.device)


def dof_mass(model: Model, mass: wp.array) -> None:
    """The per-DOF masses into ``mass`` (differentiable): the particles', the twist inertias, massless midedge
    angles."""
    wp.launch(_dof_mass_kernel, dim=mass.shape[0], inputs=[model.particle_mass, model.dismech.edge_inertia],
              outputs=[mass], device=model.device)


def flatten_state(state: State) -> None:
    """Make ``state`` own flat ``dismech.q`` / ``qd``, node, edge and hinge arrays viewing them (idempotent)."""
    ns = state.dismech
    if getattr(ns, "q", None) is not None:
        return
    complete(ns, STATE_ATTRIBUTES, state.particle_q.device)
    n = state.particle_q.shape[0]
    nd = 3 * n
    ne = ns.edge_q.shape[0]
    grad = state.particle_q.requires_grad
    for flat_name, node_name, edge_name, hinge_name in (("q", "particle_q", "edge_q", "hinge_q"),
                                                         ("qd", "particle_qd", "edge_qd", "hinge_qd")):
        flat = wp.zeros(nd + ne + ns.hinge_q.shape[0], dtype=float, device=state.particle_q.device,
                        requires_grad=grad)
        nodes = flat[:nd].reshape((n, 3)).view(wp.vec3)
        nodes.assign(getattr(state, node_name))
        setattr(state, node_name, nodes)
        for name, start, stop in ((edge_name, nd, nd + ne), (hinge_name, nd + ne, flat.shape[0])):
            if stop > start:  # Warp has no empty slice; an empty array stays
                view = flat[start:stop]
                view.assign(getattr(ns, name))
                setattr(ns, name, view)
        setattr(ns, flat_name, flat)


BLOCK = 256  # threads per block of a kernel that sums over its threads (tile reductions, wp.launch's default)


def padded(n: int) -> int:
    """``n`` rounded up to whole blocks: a launch ``(k, padded(n))`` keeps each block within one ``k``."""
    return -(-n // BLOCK) * BLOCK


@contextmanager
def suspended_tape():
    """Pause the active ``wp.Tape`` (if any) inside the block; yields it."""
    runtime = wp._src.context.runtime
    tape, runtime.tape = runtime.tape, None
    try:
        yield tape
    finally:
        runtime.tape = tape


@wp.func
def node(q: wp.array[float], n: int) -> wp.vec3:
    return wp.vec3(q[3 * n], q[3 * n + 1], q[3 * n + 2])


@wp.func
def scatter_node(b: wp.array(dtype=Any), fixed: wp.array[wp.int32], n: int, v: wp.vec3):
    """``b[node n] += v`` on its free components."""
    for k in range(3):
        if fixed[3 * n + k] == 0:
            wp.atomic_add(b, 3 * n + k, b.dtype(v[k]))


@wp.func
def block_sum(a: wp.array[wp.float64], k: int, v: wp.float64):
    """``a[k] += v`` summed over the block first (one atomic per block, not per thread); every thread of the block
    calls it with the same ``k``."""
    wp.tile_atomic_add(a, wp.tile_sum(wp.tile(v)), offset=k)


@wp.struct
class Dofs:
    """Per DOF its mass and whether it is fixed; the external force's inputs (gravity, ``particle_f`` on the first
    ``num_node_dofs``); ``1/dt^2``."""

    mass: wp.array[float]
    fixed: wp.array[wp.int32]
    gravity: wp.array[wp.vec3]
    particle_f: wp.array[wp.vec3]
    num_node_dofs: int
    inv_dt2: float


@wp.func
def _external(f: Dofs, i: int) -> float:
    """``m g + particle_f`` on node DOF ``i``."""
    if i < f.num_node_dofs:
        n = i // 3
        k = i - 3 * n
        return f.mass[i] * f.gravity[0][k] + f.particle_f[n][k]
    return 0.0


@wp.func
def inertia_grad(f: Dofs, i: int, r: float) -> float:
    """DOF ``i``'s part of the step's gradient at ``r = q - q_pred``: ``M r / dt^2 - f_ext``."""
    return f.mass[i] * f.inv_dt2 * r - _external(f, i)


@wp.func
def inertia_energy(f: Dofs, i: int, r: float) -> float:
    return 0.5 * f.mass[i] * f.inv_dt2 * r * r - _external(f, i) * r


@wp.kernel
def _dof_mass_kernel(particle_mass: wp.array[float], edge_inertia: wp.array[float], mass: wp.array[float]):
    i = wp.tid()
    nd = 3 * particle_mass.shape[0]
    if i < nd:
        mass[i] = particle_mass[i // 3]
    elif i < nd + edge_inertia.shape[0]:
        mass[i] = edge_inertia[i - nd]
    else:
        mass[i] = 0.0


def bind_dofs(f: Dofs, model: Model, mass: wp.array, fixed: wp.array, no_force: wp.array, state_in: State,
              dt: float) -> Dofs:
    """``f`` set for a step of ``dt`` from ``state_in`` (``no_force``: zeros for a state without ``particle_f``)."""
    f.mass, f.fixed, f.gravity, f.num_node_dofs = mass, fixed, model.gravity, 3 * model.particle_count
    f.inv_dt2 = 1.0 / (dt * dt)
    f.particle_f = state_in.particle_f if state_in.particle_f is not None else no_force
    return f


@wp.kernel
def predict_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], dt: float,
                   q_pred: wp.array[float], q: wp.array[float]):
    """``q_pred = q0 + dt v0``; given ``q``, also the initial guess (fixed DOFs keep ``q0``)."""
    i = wp.tid()
    q_pred[i] = q0[i] + dt * v0[i]
    if q.shape[0] > 0:
        q[i] = wp.where(fixed[i] != 0, q0[i], q_pred[i])


@wp.kernel
def diagonal_kernel(f: Dofs, indptr: wp.array[wp.int32], vals: wp.array[wp.float64]):
    """``A``'s diagonal: ``+= M / dt^2``, ``1`` on fixed DOFs."""
    i = wp.tid()
    slot = indptr[i]
    if f.fixed[i] != 0:
        vals[slot] = wp.float64(1.0)
    else:
        vals[slot] = vals[slot] + wp.float64(f.mass[i] * f.inv_dt2)


@wp.kernel
def end_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], v0: wp.array[float], q_star: wp.array[float],
               theta: float, dt: float, q: wp.array[float], qd: wp.array[float]):
    """The step's end from its solution, the implicit Euler step ``theta dt`` (``q`` may be ``q_star``): the theta
    method's extrapolation to ``dt`` (``theta = 1``: backward Euler), ``0`` velocity on fixed DOFs."""
    i = wp.tid()
    d = q_star[i] - q0[i]
    free = fixed[i] == 0
    q[i] = wp.where(free, q0[i] + d / theta, q_star[i])
    qd[i] = wp.where(free, v0[i] + (d - theta * dt * v0[i]) / (theta * theta * dt), 0.0)
