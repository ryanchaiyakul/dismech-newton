"""Surface anisotropy frame pushforward and slot surface-pair resolution (zero-kernel @wp.func module)."""

import warp as wp

from ..dofs import node
from .slots import Bodies, Mesh, shape_of, shape_pose


@wp.func
def _unit3(v: wp.vec3) -> wp.vec3:
    return v / wp.sqrt(wp.max(wp.length_sq(v), 1.0e-24))


@wp.func
def _face_u(q: wp.array[float], m: Mesh, f: int, u: wp.vec3) -> wp.vec3:
    """Push rest anisotropy vector ``u`` forward from rest triangle ``f`` to deformed frame at ``q``."""
    if wp.length_sq(u) == 0.0:
        return wp.vec3(0.0)
    tri = m.faces[f]
    r0, e1r, e2r = node(m.rest, tri[0]), node(m.rest, tri[1]), node(m.rest, tri[2])
    re1, re2 = e1r - r0, e2r - r0
    rn = _unit3(wp.cross(re1, re2))
    rt1 = _unit3(re1)
    rt2 = wp.cross(rn, rt1)
    x0, e1q, e2q = node(q, tri[0]), node(q, tri[1]), node(q, tri[2])
    qe1, qe2 = e1q - x0, e2q - x0
    qn = _unit3(wp.cross(qe1, qe2))
    qt1 = _unit3(qe1)
    qt2 = wp.cross(qn, qt1)
    return wp.dot(u, rt1) * qt1 + wp.dot(u, rt2) * qt2 + wp.dot(u, rn) * qn


@wp.func
def _prim_surf(q: wp.array[float], m: Mesh, idx: int, st: wp.vec3, num_nodes: int):
    """Resolve material ID and world surface axis for deformable node, edge, or triangle ``idx``."""
    ne = m.edges.shape[0]
    if idx < num_nodes:
        p = idx
        if m.rod[p] >= 0:
            if wp.length_sq(st) > 0.0:
                return m.node_surf[p], st
            if m.adj_ptr[p + 1] > m.adj_ptr[p]:
                return m.node_surf[p], node(q, m.adj[m.adj_ptr[p]]) - node(q, p)
            return m.node_surf[p], m.node_u[p]
        f = m.node_face[p]
        if f >= 0:
            return m.node_surf[p], _face_u(q, m, f, m.node_u[p])
        return m.node_surf[p], m.node_u[p]
    if idx < num_nodes + ne:
        ev = m.edges[idx - num_nodes]
        p = ev[0]
        if m.rod[p] >= 0:
            return m.node_surf[p], wp.where(wp.length_sq(st) > 0.0, st, node(q, ev[1]) - node(q, p))
        f = wp.where(m.edge_face[idx - num_nodes] >= 0, m.edge_face[idx - num_nodes], m.node_face[p])
        if f >= 0:
            return m.node_surf[p], _face_u(q, m, f, m.node_u[p])
        return m.node_surf[p], m.node_u[p]
    f = idx - num_nodes - ne
    p = m.faces[f][0]
    return m.node_surf[p], _face_u(q, m, f, m.node_u[p])


@wp.func
def _slot_edge_t(
    q: wp.array[float],
    pair: wp.vec4i,
    w: wp.vec4,
    n: wp.vec3,
    side: int,
    fallback: wp.vec3,
) -> wp.vec3:
    """Recompute projected edge tangent on ``q`` for two-node rod slots during adjoint replay."""
    out = fallback
    if wp.length_sq(fallback) > 0.0:
        k0, k1, count = int(-1), int(-1), int(0)
        for k in range(4):
            if pair[k] >= 0 and (w[k] > 0.0) == (side == 0):
                count += 1
                if k0 < 0:
                    k0 = k
                else:
                    k1 = k
        if count == 2:
            e = node(q, pair[k1]) - node(q, pair[k0])
            t = e - wp.dot(e, n) * n
            if wp.length_sq(t) > 1.0e-16:
                out = wp.normalize(t)
    return out


@wp.func
def slot_surface_pair(
    q: wp.array[float],
    poses: wp.array[wp.transform],
    m: Mesh,
    b: Bodies,
    num_nodes: int,
    shape_key0: int,
    own: int,
    key: int,
    ta: wp.vec3,
    tb: wp.vec3,
):
    """Resolve ``(m_A, m_B, u_A, u_B)`` in O(1) registers with SE(3) covariance and swap symmetry."""
    ma, ua = _prim_surf(q, m, own, ta, num_nodes)
    if key >= shape_key0:
        sk = shape_of(key, shape_key0)
        mb = m.shape_surf[sk]
        ub = wp.transform_vector(shape_pose(b, sk, poses), m.shape_u[sk])
        return ma, mb, ua, ub
    mb, ub = _prim_surf(q, m, key, tb, num_nodes)
    return ma, mb, ua, ub
