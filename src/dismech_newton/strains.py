"""Triplet strains ``[eps_e, eps_f, kappa1, kappa2, tau]`` and their derivatives.

A triplet is two consecutive edges ``e = x1 - x0``, ``f = x2 - x1``; its 11 DOFs are
``[x0, theta_e, x1, theta_f, x2]``.
"""

import warp as wp

from .frames import material_frame, parallel_transport, reference_twist, skew

vec5f = wp.types.vector(5, float)
vec10f = wp.types.vector(10, float)
mat55f = wp.types.matrix((5, 5), float)
vec5i = wp.types.vector(5, wp.int32)
vec11f = wp.types.vector(11, float)
mat11f = wp.types.matrix((11, 11), float)
mat5_11f = wp.types.matrix((5, 11), float)


@wp.func
def unpack_conn(conn: vec5i):
    """``(e, f, n0, n1, n2)``: the two edges of a triplet and its three nodes."""
    return conn[0], conn[1], conn[2], conn[3], conn[4]


@wp.func
def edge_direction(q: wp.array[wp.vec3], a: int, b: int) -> wp.vec3:
    """Unit tangent from node ``a`` to node ``b``."""
    return wp.normalize(q[b] - q[a])


@wp.func
def rest_strain(r: wp.vec3) -> vec5f:
    """Rest strains from rest ``[kappa1, kappa2, tau]``; stretch is measured against the rest length."""
    return vec5f(0.0, 0.0, r[0], r[1], r[2])


# -- scatter from edge space into the 11 DOFs ---------------------------------------------


@wp.func
def add_block(H: mat11f, r: int, c: int, B: wp.mat33f) -> mat11f:
    for i in range(3):
        for j in range(3):
            H[r + i, c + j] = H[r + i, c + j] + B[i, j]
    return H


@wp.func
def add_edge_hessian(H: mat11f, Dee: wp.mat33f, Def: wp.mat33f, Dff: wp.mat33f) -> mat11f:
    """Scatter an edge-space Hessian into node DOFs 0:3, 4:7, 8:11 (``dx0 = -de``, ``dx1 = de - df``, ``dx2 = df``)."""
    Dfe = wp.transpose(Def)
    H = add_block(H, 0, 0, Dee)
    H = add_block(H, 0, 4, -Dee + Def)
    H = add_block(H, 0, 8, -Def)
    H = add_block(H, 4, 0, -Dee + Dfe)
    H = add_block(H, 4, 4, Dee - Def - Dfe + Dff)
    H = add_block(H, 4, 8, Def - Dff)
    H = add_block(H, 8, 0, -Dfe)
    H = add_block(H, 8, 4, Dfe - Dff)
    H = add_block(H, 8, 8, Dff)
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
    d = De - Df
    return vec11f(-De[0], -De[1], -De[2], dtheta_e, d[0], d[1], d[2], dtheta_f, Df[0], Df[1], Df[2])


# -- geometry -----------------------------------------------------------------------------


@wp.struct
class TripletGeometry:
    te: wp.vec3
    tf: wp.vec3
    ne: float
    nf: float
    chi: float
    tt: wp.vec3  # (te + tf) / chi
    td1: wp.vec3  # (m1e + m1f) / chi
    td2: wp.vec3  # (m2e + m2f) / chi
    kb: wp.vec3  # curvature binormal
    m1e: wp.vec3
    m2e: wp.vec3
    m1f: wp.vec3
    m2f: wp.vec3
    strain: vec5f  # [eps_e, eps_f, kappa1, kappa2, tau]


@wp.func
def triplet_geometry(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    theta_e: float,
    theta_f: float,
    d1e_old: wp.vec3,
    te_old: wp.vec3,
    d1f_old: wp.vec3,
    tf_old: wp.vec3,
    ref_twist_old: float,
    l0e: float,
    l0f: float,
):
    """Frames and strains of a triplet; the reference frames are transported from the start of the step."""
    g = TripletGeometry()
    ee = x1 - x0
    ef = x2 - x1
    g.ne = wp.length(ee)
    g.nf = wp.length(ef)
    g.te = ee / g.ne
    g.tf = ef / g.nf
    d1e = parallel_transport(d1e_old, te_old, g.te)
    d1f = parallel_transport(d1f_old, tf_old, g.tf)
    m1e, m2e = material_frame(d1e, g.te, theta_e)
    m1f, m2f = material_frame(d1f, g.tf, theta_f)
    g.m1e = m1e
    g.m2e = m2e
    g.m1f = m1f
    g.m2f = m2f

    g.chi = 1.0 + wp.dot(g.te, g.tf)
    g.tt = (g.te + g.tf) / g.chi
    g.td1 = (m1e + m1f) / g.chi
    g.td2 = (m2e + m2f) / g.chi
    g.kb = 2.0 * wp.cross(g.te, g.tf) / g.chi

    eps_e = g.ne / l0e - 1.0
    eps_f = g.nf / l0f - 1.0
    kappa1 = 0.5 * wp.dot(g.kb, m2e + m2f)
    kappa2 = -0.5 * wp.dot(g.kb, m1e + m1f)
    tau = theta_f - theta_e + reference_twist(d1e, g.te, d1f, g.tf, ref_twist_old)
    g.strain = vec5f(eps_e, eps_f, kappa1, kappa2, tau)
    return g


# -- derivatives --------------------------------------------------------------------------


@wp.func
def deps(te: wp.vec3, tf: wp.vec3, ne: float, nf: float, l0e: float, l0f: float):
    """Gradients and Hessians of the two stretch strains."""
    Je = vec11f()
    Jf = vec11f()
    He = mat11f()
    Hf = mat11f()
    de = te / l0e
    df = tf / l0f
    Pe = (wp.identity(3, dtype=float) - wp.outer(te, te)) / (l0e * ne)
    Pf = (wp.identity(3, dtype=float) - wp.outer(tf, tf)) / (l0f * nf)
    for k in range(3):
        Je[k] = -de[k]
        Je[4 + k] = de[k]
        Jf[4 + k] = -df[k]
        Jf[8 + k] = df[k]
        for l in range(3):
            He[k, l] = Pe[k, l]
            He[k, 4 + l] = -Pe[k, l]
            He[4 + k, l] = -Pe[k, l]
            He[4 + k, 4 + l] = Pe[k, l]
            Hf[4 + k, 4 + l] = Pf[k, l]
            Hf[4 + k, 8 + l] = -Pf[k, l]
            Hf[8 + k, 4 + l] = -Pf[k, l]
            Hf[8 + k, 8 + l] = Pf[k, l]
    return Je, Jf, He, Hf


@wp.func
def dkappa(g: TripletGeometry):
    """Gradients and Hessians of the two bend curvatures."""
    te = g.te
    tf = g.tf
    ne = g.ne
    nf = g.nf
    chi = g.chi
    tt = g.tt
    kb = g.kb
    k1 = g.strain[2]
    k2 = g.strain[3]

    # Edge-space gradients.
    De1 = (-k1 * tt + wp.cross(tf, g.td2)) / ne
    Df1 = (-k1 * tt - wp.cross(te, g.td2)) / nf
    De2 = (-k2 * tt - wp.cross(tf, g.td1)) / ne
    Df2 = (-k2 * tt + wp.cross(te, g.td1)) / nf
    g1 = edge_gradient(De1, Df1, -0.5 * wp.dot(kb, g.m1e), -0.5 * wp.dot(kb, g.m1f))
    g2 = edge_gradient(De2, Df2, -0.5 * wp.dot(kb, g.m2e), -0.5 * wp.dot(kb, g.m2f))

    # Terms shared by both Hessians.
    I3 = wp.identity(3, dtype=float)
    ne2 = ne * ne
    nf2 = nf * nf
    tt_tt = wp.outer(tt, tt)
    Pe = I3 - wp.outer(te, te)
    Pf = I3 - wp.outer(tf, tf)
    Iet = I3 + wp.outer(te, tf)
    tf_c_td2_tt = wp.outer(wp.cross(tf, g.td2), tt)
    te_c_td2_tt = wp.outer(wp.cross(te, g.td2), tt)
    tf_c_td1_tt = wp.outer(wp.cross(tf, g.td1), tt)
    te_c_td1_tt = wp.outer(wp.cross(te, g.td1), tt)
    kb_m2e = wp.outer(kb, g.m2e)
    kb_m2f = wp.outer(kb, g.m2f)
    kb_m1e = wp.outer(kb, g.m1e)
    kb_m1f = wp.outer(kb, g.m1f)

    # kappa1
    Dee = (
        (2.0 * k1 * tt_tt - tf_c_td2_tt - wp.transpose(tf_c_td2_tt)) / ne2
        - k1 / (chi * ne2) * Pe
        + (kb_m2e + wp.transpose(kb_m2e)) / (4.0 * ne2)
    )
    Dff = (
        (2.0 * k1 * tt_tt + te_c_td2_tt + wp.transpose(te_c_td2_tt)) / nf2
        - k1 / (chi * nf2) * Pf
        + (kb_m2f + wp.transpose(kb_m2f)) / (4.0 * nf2)
    )
    Def = -k1 / (chi * ne * nf) * Iet + (
        2.0 * k1 * tt_tt - tf_c_td2_tt + wp.transpose(te_c_td2_tt) - skew(g.td2)
    ) / (ne * nf)
    H1 = add_edge_hessian(mat11f(), Dee, Def, Dff)
    H1[3, 3] = -0.5 * wp.dot(kb, g.m2e)
    H1[7, 7] = -0.5 * wp.dot(kb, g.m2f)
    H1 = add_edge_theta_hessian(
        H1,
        3,
        (0.5 * wp.dot(kb, g.m1e) * tt - wp.cross(tf, g.m1e) / chi) / ne,
        (0.5 * wp.dot(kb, g.m1e) * tt + wp.cross(te, g.m1e) / chi) / nf,
    )
    H1 = add_edge_theta_hessian(
        H1,
        7,
        (0.5 * wp.dot(kb, g.m1f) * tt - wp.cross(tf, g.m1f) / chi) / ne,
        (0.5 * wp.dot(kb, g.m1f) * tt + wp.cross(te, g.m1f) / chi) / nf,
    )

    # kappa2
    Dee = (
        (2.0 * k2 * tt_tt + tf_c_td1_tt + wp.transpose(tf_c_td1_tt)) / ne2
        - k2 / (chi * ne2) * Pe
        - (kb_m1e + wp.transpose(kb_m1e)) / (4.0 * ne2)
    )
    Dff = (
        (2.0 * k2 * tt_tt - te_c_td1_tt - wp.transpose(te_c_td1_tt)) / nf2
        - k2 / (chi * nf2) * Pf
        - (kb_m1f + wp.transpose(kb_m1f)) / (4.0 * nf2)
    )
    Def = -k2 / (chi * ne * nf) * Iet + (
        2.0 * k2 * tt_tt + tf_c_td1_tt - wp.transpose(te_c_td1_tt) + skew(g.td1)
    ) / (ne * nf)
    H2 = add_edge_hessian(mat11f(), Dee, Def, Dff)
    H2[3, 3] = 0.5 * wp.dot(kb, g.m1e)
    H2[7, 7] = 0.5 * wp.dot(kb, g.m1f)
    H2 = add_edge_theta_hessian(
        H2,
        3,
        (0.5 * wp.dot(kb, g.m2e) * tt - wp.cross(tf, g.m2e) / chi) / ne,
        (0.5 * wp.dot(kb, g.m2e) * tt + wp.cross(te, g.m2e) / chi) / nf,
    )
    H2 = add_edge_theta_hessian(
        H2,
        7,
        (0.5 * wp.dot(kb, g.m2f) * tt - wp.cross(tf, g.m2f) / chi) / ne,
        (0.5 * wp.dot(kb, g.m2f) * tt + wp.cross(te, g.m2f) / chi) / nf,
    )
    return g1, g2, H1, H2


@wp.func
def dtau(g: TripletGeometry):
    """Gradient and Hessian of the twist; theta enters linearly, the rest is the reference twist."""
    te = g.te
    tf = g.tf
    ne = g.ne
    nf = g.nf
    kb = g.kb

    J = edge_gradient(0.5 * kb / ne, 0.5 * kb / nf, -1.0, 1.0)

    te_tt = te + g.tt
    tf_tt = tf + g.tt
    Dee = -(wp.outer(kb, te_tt) + wp.outer(te_tt, kb)) / (4.0 * ne * ne)
    Dff = -(wp.outer(kb, tf_tt) + wp.outer(tf_tt, kb)) / (4.0 * nf * nf)
    Def = (2.0 / g.chi * skew(te) - wp.outer(kb, g.tt)) / (2.0 * ne * nf)
    H = add_edge_hessian(mat11f(), Dee, Def, Dff)
    return J, H


@wp.func
def strain_gradient(g: TripletGeometry, sigma: vec5f, l0e: float, l0f: float) -> vec11f:
    """``J^T sigma`` alone (no Hessians), written without component writes so Warp can differentiate it."""
    k1 = g.strain[2]
    k2 = g.strain[3]
    kb = g.kb
    tt = g.tt
    De = (
        sigma[0] / l0e * g.te
        + sigma[2] * (-k1 * tt + wp.cross(g.tf, g.td2)) / g.ne
        + sigma[3] * (-k2 * tt - wp.cross(g.tf, g.td1)) / g.ne
        + sigma[4] * 0.5 * kb / g.ne
    )
    Df = (
        sigma[1] / l0f * g.tf
        + sigma[2] * (-k1 * tt - wp.cross(g.te, g.td2)) / g.nf
        + sigma[3] * (-k2 * tt + wp.cross(g.te, g.td1)) / g.nf
        + sigma[4] * 0.5 * kb / g.nf
    )
    dth_e = -0.5 * (sigma[2] * wp.dot(kb, g.m1e) + sigma[3] * wp.dot(kb, g.m2e)) - sigma[4]
    dth_f = -0.5 * (sigma[2] * wp.dot(kb, g.m1f) + sigma[3] * wp.dot(kb, g.m2f)) + sigma[4]
    return edge_gradient(De, Df, dth_e, dth_f)


@wp.func
def strain_derivatives(g: TripletGeometry, sigma: vec5f, l0e: float, l0f: float):
    """The strain Jacobian ``J`` (5 x 11) and ``sum_i sigma_i H_i``, the stress-weighted strain Hessians."""
    Jse, Jsf, Hse, Hsf = deps(g.te, g.tf, g.ne, g.nf, l0e, l0f)
    Jb1, Jb2, Hb1, Hb2 = dkappa(g)
    Ja, Ha = dtau(g)
    J = mat5_11f()
    for i in range(11):
        J[0, i] = Jse[i]
        J[1, i] = Jsf[i]
        J[2, i] = Jb1[i]
        J[3, i] = Jb2[i]
        J[4, i] = Ja[i]
    return J, sigma[0] * Hse + sigma[1] * Hsf + sigma[2] * Hb1 + sigma[3] * Hb2 + sigma[4] * Ha
