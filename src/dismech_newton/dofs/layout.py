"""DOF layout constants, state flattening, per-node helpers, and tape suspension.

DOF ``3 * node + k`` is ``particle_q[node][k]``; DOF ``3 * N + edge`` is ``edge_q[edge]``; DOF ``3 * N + E + hinge``
is ``hinge_q[hinge]`` (a shell's midedge angle).
"""

from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp
from newton import Model, ParticleFlags, State

from ..builder import ATTRIBUTES, STATE_ATTRIBUTES, complete
from ..constants import BLOCK

DOFS_MODULE = wp.get_module("dismech_newton.dofs")


def dof_constants(model: Model) -> tuple[wp.array, wp.array]:
    """Per-DOF ``(mass, fixed)`` (:func:`dof_mass`)."""
    from .inertia import dof_mass  # noqa: PLC0415

    der = model.dismech
    complete(der, ATTRIBUTES, model.device)
    active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
    fixed = np.concatenate([np.repeat(~active, 3), der.edge_fixed.numpy() != 0, der.hinge_fixed.numpy() != 0])
    mass = wp.zeros(len(fixed), dtype=float, device=model.device)
    dof_mass(model, mass)
    return mass, wp.array(fixed, dtype=wp.int32, device=model.device)


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
