import warp as wp

from ....frames import material_frame, parallel_transport, reference_twist
from .helper import vec5f


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
