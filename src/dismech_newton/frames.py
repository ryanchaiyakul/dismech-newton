"""Rod frame geometry: the math primitives and the end-of-step frame update.

Each edge has a unit tangent ``t`` (from its node positions) and a stored reference director
``d1`` that is parallel-transported in time; the material directors ``(m1, m2)`` are that
frame rotated by the edge twist angle. Strain kernels build on the primitives here;
:func:`advance_edge_frames` brings the stored reference directors up to date and
:func:`pose_edge_proxies` places the render-only capsule proxies.
"""

import warp as wp
from newton import State

# -- math primitives ----------------------------------------------------------------------


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


# -- frames -------------------------------------------------------------------------------


@wp.func
def material_frame(d1: wp.vec3, t: wp.vec3, theta: float):
    """Material directors ``(m1, m2)``: reference frame ``(d1, t x d1)`` rotated by ``theta``."""
    d2 = wp.cross(t, d1)
    c = wp.cos(theta)
    s = wp.sin(theta)
    return c * d1 + s * d2, -s * d1 + c * d2


@wp.func
def edge_reference_frame(
    x0: wp.vec3, x1: wp.vec3, d1_old: wp.vec3, t_old: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """Current unit tangent and time-parallel-transported reference director."""
    t = wp.normalize(x1 - x0)
    return t, parallel_transport(d1_old, t_old, t)


@wp.func
def reference_twist(
    d1e: wp.vec3, te: wp.vec3, d1f: wp.vec3, tf: wp.vec3, ref_twist_old: float
) -> float:
    """Reference twist between two edges, unwrapped to be continuous in time."""
    angle = signed_angle(parallel_transport(d1e, te, tf), d1f, tf)
    return ref_twist_old + wrap_angle(angle - ref_twist_old)


# -- end-of-step update -------------------------------------------------------------------


@wp.kernel
def _advance_edge_frames_kernel(
    node_q_prev: wp.array[wp.vec3],
    node_q: wp.array[wp.vec3],
    edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32],
    edge_d1_prev: wp.array[wp.vec3],
    # outputs
    edge_d1: wp.array[wp.vec3],
):
    e = wp.tid()
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    t_prev = wp.normalize(node_q_prev[n1] - node_q_prev[n0])
    _t, d1 = edge_reference_frame(node_q[n0], node_q[n1], edge_d1_prev[e], t_prev)
    edge_d1[e] = d1


@wp.kernel
def _pose_proxies_kernel(
    node_q: wp.array[wp.vec3],
    edge_q: wp.array[float],
    edge_d1: wp.array[wp.vec3],
    edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32],
    edge_body: wp.array[wp.int32],
    # outputs
    body_q: wp.array[wp.transform],
):
    e = wp.tid()
    body = edge_body[e]
    if body < 0:
        return
    x0 = node_q[edge_node0[e]]
    x1 = node_q[edge_node1[e]]
    t = wp.normalize(x1 - x0)
    m1, m2 = material_frame(edge_d1[e], t, edge_q[e])
    R = wp.mat33(
        m1[0], m2[0], t[0],
        m1[1], m2[1], t[1],
        m1[2], m2[2], t[2],
    )
    body_q[body] = wp.transform(0.5 * (x0 + x1), wp.quat_from_matrix(R))


def advance_edge_frames(der, state_in: State, state_out: State, device) -> None:
    """Parallel-transport every edge's reference director from ``state_in`` to ``state_out``.

    ``der`` is the model's ``dismech`` namespace. Tangents are not stored; they are the
    normalized node differences of each state's ``particle_q``.
    """
    wp.launch(
        _advance_edge_frames_kernel,
        dim=der.edge_length.shape[0],
        inputs=[state_in.particle_q, state_out.particle_q, der.edge_node0, der.edge_node1, state_in.dismech.edge_d1_q],
        outputs=[state_out.dismech.edge_d1_q],
        device=device,
    )


def pose_edge_proxies(der, state: State, device) -> None:
    """Pose the kinematic capsule proxies (``state.body_q``) from ``state``'s rod configuration.

    Proxy frame: origin at the segment midpoint, local +Z along the tangent, +X along material
    director m1. Edges without a proxy (``edge_body < 0``) are skipped. The proxies are
    kinematic, so ``body_qd`` is never written.
    """
    wp.launch(
        _pose_proxies_kernel,
        dim=der.edge_length.shape[0],
        inputs=[state.particle_q, state.dismech.edge_q, state.dismech.edge_d1_q, der.edge_node0, der.edge_node1, der.edge_body],
        outputs=[state.body_q],
        device=device,
    )
