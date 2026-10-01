"""The flat DOF vector and rod frame geometry.

DOF ``3 * node + k`` is component ``k`` of ``particle_q[node]`` and DOF ``3 * N + edge`` is the
twist angle ``edge_q[edge]``. A flattened state owns that vector as ``state.dismech.q`` (velocities
``qd``), and ``particle_q``, ``edge_q`` and their velocities are views into it.

Each edge has a unit tangent ``t`` (from its nodes) and a reference director ``d1`` that is
parallel-transported in time; the material directors ``(m1, m2)`` are that frame rotated by the
twist angle.
"""

import numpy as np
import warp as wp
from newton import Model, ParticleFlags, State

# -- flat DOF vector ----------------------------------------------------------------------


def dof_constants(model: Model) -> tuple[wp.array, wp.array]:
    """Per-DOF ``(mass, fixed)``: node mass / edge twist inertia, and the Dirichlet flag.

    A node DOF is fixed when its particle is not ``ACTIVE``, a twist DOF when ``edge_fixed`` is set.
    """
    der = model.dismech
    active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
    fixed = np.concatenate([np.repeat(~active, 3), der.edge_fixed.numpy() != 0])
    mass = np.concatenate([np.repeat(model.particle_mass.numpy(), 3), der.edge_inertia.numpy()])
    return (
        wp.array(mass.astype(np.float32), dtype=float, device=model.device),
        wp.array(fixed, dtype=wp.int32, device=model.device),
    )


def flatten_state(state: State) -> None:
    """Make ``state`` own the flat vectors ``dismech.q`` / ``dismech.qd`` (idempotent).

    ``particle_q``, ``edge_q``, ``particle_qd`` and ``edge_qd`` become views into them. Call it
    before any CUDA graph capture; the solver does so on the first step.
    """
    ns = state.dismech
    if getattr(ns, "q", None) is not None:
        return
    n = state.particle_q.shape[0]
    nd = 3 * n
    for flat_name, node_name, edge_name in (("q", "particle_q", "edge_q"), ("qd", "particle_qd", "edge_qd")):
        flat = wp.zeros(nd + ns.edge_q.shape[0], dtype=float, device=state.particle_q.device)
        nodes = flat[:nd].reshape((n, 3)).view(wp.vec3)
        edges = flat[nd:]
        nodes.assign(getattr(state, node_name))
        edges.assign(getattr(ns, edge_name))
        setattr(state, node_name, nodes)
        setattr(ns, edge_name, edges)
        setattr(ns, flat_name, flat)


@wp.func
def node(q: wp.array[float], n: int) -> wp.vec3:
    """Position of node ``n`` in the flat DOF vector."""
    return wp.vec3(q[3 * n], q[3 * n + 1], q[3 * n + 2])


@wp.func
def fixed_node(q: wp.array[float], fixed: wp.array[wp.int32], n: int) -> wp.vec3:
    """The fixed components of node ``n`` (free ones zeroed)."""
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


# -- frame math ---------------------------------------------------------------------------


@wp.func
def parallel_transport(m: wp.vec3, t0: wp.vec3, t1: wp.vec3) -> wp.vec3:
    """Transport ``m`` (orthogonal to ``t0``) to ``t1`` along the shortest arc."""
    out = m - wp.dot(m, t1) / (1.0 + wp.dot(t0, t1)) * (t0 + t1)
    return wp.normalize(out)


@wp.func
def signed_angle(a: wp.vec3, b: wp.vec3, axis: wp.vec3) -> float:
    """Angle from ``a`` to ``b`` about ``axis``, in ``(-pi, pi]``."""
    return wp.atan2(wp.dot(axis, wp.cross(a, b)), wp.dot(a, b))


@wp.func
def wrap_angle(a: float) -> float:
    """Wrap ``a`` into ``[-pi, pi)``."""
    two_pi = 2.0 * wp.PI
    return a - two_pi * wp.floor((a + wp.PI) / two_pi)


@wp.func
def skew(a: wp.vec3) -> wp.mat33:
    """``skew(a) @ b == cross(a, b)``."""
    # fmt: off
    return wp.mat33(
          0.0, -a[2],  a[1],
         a[2],   0.0, -a[0],
        -a[1],  a[0],   0.0,
    )
    # fmt: on


@wp.func
def material_frame(d1: wp.vec3, t: wp.vec3, theta: float):
    """Material directors ``(m1, m2)``: reference frame ``(d1, t x d1)`` rotated by ``theta``."""
    d2 = wp.cross(t, d1)
    c = wp.cos(theta)
    s = wp.sin(theta)
    return c * d1 + s * d2, -s * d1 + c * d2


@wp.func
def reference_twist(d1e: wp.vec3, te: wp.vec3, d1f: wp.vec3, tf: wp.vec3, ref_twist_old: float) -> float:
    """Reference twist between two edges, unwrapped to be continuous in time."""
    angle = signed_angle(parallel_transport(d1e, te, tf), d1f, tf)
    return ref_twist_old + wrap_angle(angle - ref_twist_old)


# -- end-of-step kernels (one thread per edge) --------------------------------------------


@wp.kernel
def advance_frames_kernel(
    node_q_prev: wp.array[wp.vec3],
    node_q: wp.array[wp.vec3],
    edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32],
    edge_d1_prev: wp.array[wp.vec3],
    # outputs
    edge_d1: wp.array[wp.vec3],
):
    """Parallel-transport every reference director from the previous tangent to the current one."""
    e = wp.tid()
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    t_prev = wp.normalize(node_q_prev[n1] - node_q_prev[n0])
    edge_d1[e] = parallel_transport(edge_d1_prev[e], t_prev, wp.normalize(node_q[n1] - node_q[n0]))


@wp.kernel
def pose_proxies_kernel(
    node_q: wp.array[wp.vec3],
    node_qd: wp.array[wp.vec3],
    edge_q: wp.array[float],
    edge_d1: wp.array[wp.vec3],
    edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32],
    edge_body: wp.array[wp.int32],
    set_velocity: int,
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Pose every proxy (origin at the edge midpoint, +Z along the tangent, +X along ``m1``) and,
    with ``set_velocity``, set its rigid velocity (midpoint velocity, edge rotation rate)."""
    e = wp.tid()
    body = edge_body[e]
    if body < 0:
        return
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    x0 = node_q[n0]
    x1 = node_q[n1]
    t = wp.normalize(x1 - x0)
    m1, m2 = material_frame(edge_d1[e], t, edge_q[e])
    R = wp.mat33(
        m1[0], m2[0], t[0],
        m1[1], m2[1], t[1],
        m1[2], m2[2], t[2],
    )
    body_q[body] = wp.transform(0.5 * (x0 + x1), wp.quat_from_matrix(R))
    if set_velocity != 0:
        v0 = node_qd[n0]
        v1 = node_qd[n1]
        d = x1 - x0
        body_qd[body] = wp.spatial_vector(0.5 * (v0 + v1), wp.cross(d, v1 - v0) / wp.dot(d, d))
