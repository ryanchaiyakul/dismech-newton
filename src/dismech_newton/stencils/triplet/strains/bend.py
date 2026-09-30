import warp as wp

from ....frames import skew
from .geometry import TripletGeometry
from .helper import (
    add_edge_hessian,
    add_edge_theta_hessian,
    edge_gradient,
    mat11f,
    vec11f,
)


@wp.func
def dkappa(g: TripletGeometry) -> tuple[vec11f, vec11f, mat11f, mat11f]:
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
