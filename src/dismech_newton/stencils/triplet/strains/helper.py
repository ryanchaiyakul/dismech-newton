"""Shared types and DOF-scatter helpers for the strain kernels."""

from functools import cache

import warp as wp


@cache
def param_vec(n: int):
    """The Warp ``n``-vector type of per-stencil parameters (one class per length, so types compare equal)."""
    return wp.types.vector(n, float)


vec5f = param_vec(5)
mat55f = wp.types.matrix((5, 5), float)
vec5i = wp.types.vector(5, wp.int32)
vec11f = wp.types.vector(11, float)
mat11f = wp.types.matrix((11, 11), float)
mat5_11f = wp.types.matrix((5, 11), float)


@wp.func
def add_block(H: mat11f, r: int, c: int, B: wp.mat33f, scale: float) -> mat11f:
    for i in range(3):
        for j in range(3):
            H[r + i, c + j] = H[r + i, c + j] + scale * B[i, j]
    return H


@wp.func
def add_edge_hessian(H: mat11f, Dee: wp.mat33f, Def: wp.mat33f, Dff: wp.mat33f) -> mat11f:
    """Scatter the edge-space Hessian of a scalar into node DOFs 0:3, 4:7, 8:11.

    With ``e = x1 - x0`` and ``f = x2 - x1``: ``dx0 = -de``, ``dx1 = de - df``,
    ``dx2 = df``. ``Dfe = Def^T``.
    """
    Dfe = wp.transpose(Def)
    one = 1.0
    H = add_block(H, 0, 0, Dee, one)
    H = add_block(H, 0, 4, -Dee + Def, one)
    H = add_block(H, 0, 8, -Def, one)
    H = add_block(H, 4, 0, -Dee + Dfe, one)
    H = add_block(H, 4, 4, Dee - Def - Dfe + Dff, one)
    H = add_block(H, 4, 8, Def - Dff, one)
    H = add_block(H, 8, 0, -Dfe, one)
    H = add_block(H, 8, 4, Dfe - Dff, one)
    H = add_block(H, 8, 8, Dff, one)
    return H


@wp.func
def add_edge_theta_hessian(H: mat11f, col: int, De: wp.vec3, Df: wp.vec3) -> mat11f:
    """Scatter mixed (edge vectors, theta) second derivatives, symmetric."""
    for k in range(3):
        v0 = -De[k]
        v1 = De[k] - Df[k]
        v2 = Df[k]
        H[k, col] = H[k, col] + v0
        H[col, k] = H[col, k] + v0
        H[4 + k, col] = H[4 + k, col] + v1
        H[col, 4 + k] = H[col, 4 + k] + v1
        H[8 + k, col] = H[8 + k, col] + v2
        H[col, 8 + k] = H[col, 8 + k] + v2
    return H


@wp.func
def edge_gradient(De: wp.vec3, Df: wp.vec3, dtheta_e: float, dtheta_f: float) -> vec11f:
    g = vec11f()
    for k in range(3):
        g[k] = -De[k]
        g[4 + k] = De[k] - Df[k]
        g[8 + k] = Df[k]
    g[3] = dtheta_e
    g[7] = dtheta_f
    return g


@wp.func
def unpack_conn(conn: vec5i):
    """``(e, f, n0, n1, n2)``: the two edges of a triplet and its three nodes."""
    return conn[0], conn[1], conn[2], conn[3], conn[4]


@wp.func
def edge_direction(q: wp.array[wp.vec3], a: int, b: int) -> wp.vec3:
    """Unit tangent from node ``a`` to node ``b`` (the state stores no tangents; they follow from positions)."""
    return wp.normalize(q[b] - q[a])
