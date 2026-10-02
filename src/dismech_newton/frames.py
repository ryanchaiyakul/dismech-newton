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
from newton import JointType, Model, ParticleFlags, State

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


def proxy_joints(model: Model) -> np.ndarray:
    """Per edge, the ``(joint_q, joint_qd)`` starts of its proxy's free root joint (``-1`` without one)."""
    out = np.full((model.dismech.edge_body.shape[0], 2), -1, dtype=np.int32)
    if not model.joint_count:
        return out
    free = (model.joint_type.numpy() == int(JointType.FREE)) & (model.joint_parent.numpy() < 0)
    joint_of = np.full(max(model.body_count, 1), -1)
    joint_of[model.joint_child.numpy()[free]] = np.nonzero(free)[0]
    edge_body = model.dismech.edge_body.numpy()
    j = np.where(edge_body >= 0, joint_of[np.maximum(edge_body, 0)], -1)
    has = j >= 0
    out[has, 0] = model.joint_q_start.numpy()[j[has]]
    out[has, 1] = model.joint_qd_start.numpy()[j[has]]
    return out


def flatten_state(state: State) -> None:
    """Make ``state`` own the flat vectors ``dismech.q`` / ``dismech.qd`` (idempotent).

    ``particle_q``, ``edge_q``, ``particle_qd`` and ``edge_qd`` become views into them. Call it
    before any CUDA graph capture; the solver does so on the first step. A state made with
    ``requires_grad`` keeps it: the views share the flat vectors' gradients.
    """
    ns = state.dismech
    if getattr(ns, "q", None) is not None:
        return
    n = state.particle_q.shape[0]
    nd = 3 * n
    grad = state.particle_q.requires_grad
    for flat_name, node_name, edge_name in (("q", "particle_q", "edge_q"), ("qd", "particle_qd", "edge_qd")):
        flat = wp.zeros(nd + ns.edge_q.shape[0], dtype=float, device=state.particle_q.device, requires_grad=grad)
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


@wp.func
def external_force(i: int, mass: wp.array[float], gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3]) -> float:
    """``m g + particle_f`` on node DOF ``i = 3 * node + k``."""
    n = i // 3
    k = i - 3 * n
    return mass[i] * gravity[0][k] + particle_f[n][k]


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
    node_q_prev: wp.array[wp.vec3], node_q: wp.array[wp.vec3], edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32], edge_d1_prev: wp.array[wp.vec3],
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
    node_q: wp.array[wp.vec3], node_qd: wp.array[wp.vec3], edge_q: wp.array[float], edge_d1: wp.array[wp.vec3],
    edge_node0: wp.array[wp.int32], edge_node1: wp.array[wp.int32], edge_body: wp.array[wp.int32],
    edge_joint: wp.array[wp.vec2i], set_velocity: int,
    # outputs
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Pose every proxy (origin at the edge midpoint, +Z along the tangent, +X along ``m1``) and,
    with ``set_velocity``, set its rigid velocity (midpoint velocity, edge rotation rate). Given
    ``joint_q``, a proxy's free root joint (``edge_joint`` starts, ``-1`` without) gets the same."""
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
    X = wp.transform(0.5 * (x0 + x1), wp.quat_from_matrix(R))
    body_q[body] = X
    v0 = node_qd[n0]
    v1 = node_qd[n1]
    d = x1 - x0
    twist = wp.spatial_vector(0.5 * (v0 + v1), wp.cross(d, v1 - v0) / wp.dot(d, d))
    if set_velocity != 0:
        body_qd[body] = twist
    j = edge_joint[e]
    if joint_q.shape[0] > 0 and j[0] >= 0:  # a free root joint: joint_q = body_q, joint_qd = body_qd
        for k in range(7):
            joint_q[j[0] + k] = X[k]
        for k in range(6):
            joint_qd[j[1] + k] = twist[k]
