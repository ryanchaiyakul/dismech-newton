"""The flat DOF vector.

DOF ``3 * node + k`` is ``particle_q[node][k]``; DOF ``3 * N + edge`` is ``edge_q[edge]``; DOF ``3 * N + E + hinge``
is ``hinge_q[hinge]`` (a shell's midedge angle).
"""

import numpy as np
import warp as wp
from newton import Model, ParticleFlags, State

from .builder import ATTRIBUTES, STATE_ATTRIBUTES, complete


def dof_constants(model: Model) -> tuple[wp.array, wp.array]:
    """Per-DOF ``(mass, fixed)``; the midedge angles are massless."""
    der = model.dismech
    complete(der, ATTRIBUTES, model.device)
    active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
    hinge_fixed = der.hinge_fixed.numpy()
    fixed = np.concatenate([np.repeat(~active, 3), der.edge_fixed.numpy() != 0, hinge_fixed != 0])
    mass = np.concatenate([np.repeat(model.particle_mass.numpy(), 3), der.edge_inertia.numpy(),
                           np.zeros(len(hinge_fixed))])
    return (
        wp.array(mass.astype(np.float32), dtype=float, device=model.device),
        wp.array(fixed, dtype=wp.int32, device=model.device),
    )


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


@wp.func
def node(q: wp.array[float], n: int) -> wp.vec3:
    return wp.vec3(q[3 * n], q[3 * n + 1], q[3 * n + 2])


@wp.func
def fixed_node(q: wp.array[float], fixed: wp.array[wp.int32], n: int) -> wp.vec3:
    """Node ``n`` with its free components zeroed."""
    out = wp.vec3()
    for k in range(3):
        if fixed[3 * n + k] != 0:
            out[k] = q[3 * n + k]
    return out


@wp.func
def scatter_node(rhs: wp.array[float], fixed: wp.array[wp.int32], n: int, v: wp.vec3):
    """``rhs[node n] += v`` on its free components."""
    for k in range(3):
        if fixed[3 * n + k] == 0:
            wp.atomic_add(rhs, 3 * n + k, v[k])


@wp.func
def scatter_dof(rhs: wp.array[float], fixed: wp.array[wp.int32], i: int, v: float):
    """``rhs[i] += v`` unless DOF ``i`` is fixed."""
    if fixed[i] == 0:
        wp.atomic_add(rhs, i, v)


@wp.func
def external_force(i: int, mass: wp.array[float], gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3]) -> float:
    """``m g + particle_f`` on node DOF ``i``."""
    n = i // 3
    k = i - 3 * n
    return mass[i] * gravity[0][k] + particle_f[n][k]
