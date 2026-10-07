"""Triplet strains ``[eps_e, eps_f, kappa1, kappa2, tau]`` and their derivatives, taken in edge space
(``e = x1 - x0``, ``f = x2 - x1``): the gradient in the DOFs ``[x0, theta_e, x1, theta_f, x2]``, the Jacobian and
Hessian in ADMM's local variable ``z = [e, theta_e, f, theta_f - theta_e]``.
"""

import warp as wp

vec5f = wp.types.vector(5, float)
vec10f = wp.types.vector(10, float)
mat55f = wp.types.matrix((5, 5), float)
vec11f = wp.types.vector(11, float)
vec8f = wp.types.vector(8, float)
mat58f = wp.types.matrix((5, 8), float)
vec36f = wp.types.vector(36, float)  # packed upper triangle of a symmetric 8 x 8, row-major (see packed_index)

# -- frames -------------------------------------------------------------------------------


@wp.func
def parallel_transport(m: wp.vec3, t0: wp.vec3, t1: wp.vec3) -> wp.vec3:
    """Transport ``m`` from ``t0`` to ``t1`` along the shortest arc."""
    out = m - wp.dot(m, t1) / (1.0 + wp.dot(t0, t1)) * (t0 + t1)
    return wp.normalize(out)


@wp.func
def signed_angle(a: wp.vec3, b: wp.vec3, axis: wp.vec3) -> float:
    """Angle from ``a`` to ``b`` about ``axis``, in ``(-pi, pi]``."""
    return wp.atan2(wp.dot(axis, wp.cross(a, b)), wp.dot(a, b))


@wp.func
def wrap_angle(a: float) -> float:
    """Into ``[-pi, pi)``."""
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
    """``(m1, m2)``: ``(d1, t x d1)`` rotated by ``theta``."""
    d2 = wp.cross(t, d1)
    c = wp.cos(theta)
    s = wp.sin(theta)
    return c * d1 + s * d2, -s * d1 + c * d2


@wp.func
def reference_twist(d1e: wp.vec3, te: wp.vec3, d1f: wp.vec3, tf: wp.vec3, ref_twist_old: float) -> float:
    """Unwrapped against ``ref_twist_old`` to be continuous in time."""
    angle = signed_angle(parallel_transport(d1e, te, tf), d1f, tf)
    return ref_twist_old + wrap_angle(angle - ref_twist_old)


# -- geometry -----------------------------------------------------------------------------


@wp.struct
class TripletGeometry:
    te: wp.vec3
    tf: wp.vec3
    ne: float
    nf: float
    l0e: float  # rest lengths
    l0f: float
    chi: float
    tt: wp.vec3  # (te + tf) / chi
    kb: wp.vec3  # curvature binormal
    te_old: wp.vec3  # the tangents the reference directors are transported from
    tf_old: wp.vec3
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
    """Frames and strains; reference frames transported from the start of the step."""
    g = TripletGeometry()
    ee = x1 - x0
    ef = x2 - x1
    g.ne = wp.length(ee)
    g.nf = wp.length(ef)
    g.l0e = l0e
    g.l0f = l0f
    g.te = ee / g.ne
    g.tf = ef / g.nf
    g.te_old = te_old
    g.tf_old = tf_old
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
    g.kb = 2.0 * wp.cross(g.te, g.tf) / g.chi

    eps_e = g.ne / l0e - 1.0
    eps_f = g.nf / l0f - 1.0
    kappa1 = 0.5 * wp.dot(g.kb, m2e + m2f)
    kappa2 = -0.5 * wp.dot(g.kb, m1e + m1f)
    tau = theta_f - theta_e + reference_twist(d1e, g.te, d1f, g.tf, ref_twist_old)
    g.strain = vec5f(eps_e, eps_f, kappa1, kappa2, tau)
    return g


# -- strain derivatives -------------------------------------------------------------------
#
# Both curvatures are ``kappa = kb . (Me + Mf) / 2`` with directors that turn as ``dM/dtheta = N``
# (so ``d^2M/dtheta^2 = -M``): ``kappa1`` has ``M = m2, N = -m1``, ``kappa2`` has ``M = -m1, N = -m2``.
#
# Those formulas are for current frames (directors transported from the current tangents). Transported from
# ``t_old`` instead, a director differs by a turn ``phi`` about ``t`` (the holonomy of the transport), which acts as
# ``theta + phi``: the strains are ``F(x, theta + phi(x))`` with ``F`` the current-frame strains, and every
# derivative chains through ``phi``. The gradient is taken in the DOFs (:func:`strain_gradient`, differentiable);
# the Jacobian and Hessian in ADMM's ``z`` (:func:`local_strain_derivatives`), which the assembly maps to the DOFs.


@wp.func
def transport_turn(t_old: wp.vec3, t: wp.vec3):
    """``(w, c)`` of the director's turn ``phi`` about ``t`` between transport from ``t_old`` and from ``t``:
    ``d phi/de = w / (c |e|)``, ``w = t x t_old``, ``c = 1 + t . t_old``."""
    return wp.cross(t, t_old), 1.0 + wp.dot(t_old, t)


@wp.func
def strain_gradient(g: TripletGeometry, sigma: vec5f) -> vec11f:
    """``J^T sigma`` without component writes, so Warp can differentiate it."""
    De1, Df1, a1, b1 = kappa_gradient(g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = kappa_gradient(g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    De = sigma[0] / g.l0e * g.te + sigma[2] * De1 + sigma[3] * De2 + sigma[4] * 0.5 * g.kb / g.ne
    Df = sigma[1] / g.l0f * g.tf + sigma[2] * Df1 + sigma[3] * Df2 + sigma[4] * 0.5 * g.kb / g.nf
    torque_e = sigma[2] * a1 + sigma[3] * a2 - sigma[4]
    torque_f = sigma[2] * b1 + sigma[3] * b2 + sigma[4]
    # T^T: each edge's turn moves its torque onto the nodes (d phi/de, see transport_turn).
    we, ce = transport_turn(g.te_old, g.te)
    wf, cf = transport_turn(g.tf_old, g.tf)
    w_e = we / (ce * g.ne)
    w_f = wf / (cf * g.nf)
    return edge_gradient(De + torque_e * w_e, Df + torque_f * w_f, torque_e, torque_f)


@wp.func
def kappa_gradient(g: TripletGeometry, k: float, Me: wp.vec3, Mf: wp.vec3, Ne: wp.vec3, Nf: wp.vec3):
    """``(d/de, d/df, d/dtheta_e, d/dtheta_f)`` of the curvature ``k``."""
    td = (Me + Mf) / g.chi
    De = (-k * g.tt + wp.cross(g.tf, td)) / g.ne
    Df = (-k * g.tt - wp.cross(g.te, td)) / g.nf
    return De, Df, 0.5 * wp.dot(g.kb, Ne), 0.5 * wp.dot(g.kb, Nf)


@wp.func
def edge_gradient(De: wp.vec3, Df: wp.vec3, dtheta_e: float, dtheta_f: float) -> vec11f:
    d = De - Df
    return vec11f(-De[0], -De[1], -De[2], dtheta_e, d[0], d[1], d[2], dtheta_f, Df[0], Df[1], Df[2])


# -- derivatives in the local variable z --------------------------------------------------
#
# ADMM's local variable is ``z = [e, theta_e, f, theta_f - theta_e]`` (8). In z the edges are coordinates, so the
# derivatives are the edge-space ones with ``theta_e = z3``, ``theta_f = z3 + z7`` and no node scatter. With
# ``A = theta_e + phi_e(e)``, ``B = theta_f + phi_f(f)`` (the transport turns) and ``W = sum_i sigma_i F_i`` in the
# current-frame variables ``(e, f, A, B)`` (``W_AB = 0``):
#
#   H_ee = W_ee + ge W_eA^T + W_eA ge^T + W_AA ge ge^T + W_A d^2phi_e     H_e3 = W_eA + W_eB + W_AA ge   H_e7 = W_eB
#   H_ff = W_ff + gf W_fB^T + W_fB gf^T + W_BB gf gf^T + W_B d^2phi_f     H_f3 = W_fA + H_f7             H_f7 = W_fB + W_BB gf
#   H_ef = W_ef + W_eB gf^T + ge W_fA^T                                   H_33 = W_AA + W_BB             H_37 = H_77 = W_BB
#
# ``ge = d phi_e/de = w_e / (c_e n_e)`` and ``d^2phi_e = -(w_e a_e^T + a_e w_e^T) / n_e^2`` (``transport_turn``,
# with ``w = t x t_old``, ``c = 1 + t . t_old``, ``a = (I - t t^T) t_old / (2 c^2) + t / c``), so every turn term of
# ``H_ee`` is ``w_e X_e^T + X_e w_e^T``. The curvature Hessians are linear in ``(k, M, N)``, so both are taken at
# once from the sigma-weighted ``kw, Mw, Nw``; twist and stretch fold into the same terms.


@wp.func
def packed_index(i: int, j: int) -> int:
    """Slot of ``(i, j)``, ``i <= j``, in a ``vec36f``: ``i * 8 - i * (i - 1) / 2 + (j - i)``."""
    return i * (15 - i) / 2 + j


@wp.func
def _sym_upper(alpha: float, t: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, u: wp.vec3, v: wp.vec3):
    """Upper ``(00, 01, 02, 11, 12, 22)`` of ``alpha (I - t t^T) + a b^T + b a^T + c d^T + d c^T + u v^T + v u^T``."""
    s00 = alpha * (1.0 - t[0] * t[0]) + 2.0 * (a[0] * b[0] + c[0] * d[0] + u[0] * v[0])
    s11 = alpha * (1.0 - t[1] * t[1]) + 2.0 * (a[1] * b[1] + c[1] * d[1] + u[1] * v[1])
    s22 = alpha * (1.0 - t[2] * t[2]) + 2.0 * (a[2] * b[2] + c[2] * d[2] + u[2] * v[2])
    s01 = -alpha * t[0] * t[1] + a[0] * b[1] + b[0] * a[1] + c[0] * d[1] + d[0] * c[1] + u[0] * v[1] + v[0] * u[1]
    s02 = -alpha * t[0] * t[2] + a[0] * b[2] + b[0] * a[2] + c[0] * d[2] + d[0] * c[2] + u[0] * v[2] + v[0] * u[2]
    s12 = -alpha * t[1] * t[2] + a[1] * b[2] + b[1] * a[2] + c[1] * d[2] + d[1] * c[2] + u[1] * v[2] + v[1] * u[2]
    return s00, s01, s02, s11, s12, s22


@wp.func
def _local_jacobian_row(De: wp.vec3, Df: wp.vec3, a: float, b: float, ge: wp.vec3, gf: wp.vec3) -> vec8f:
    """A strain's z-gradient from its current-frame ``(d/de, d/df, d/dA, d/dB)``."""
    u = De + a * ge
    v = Df + b * gf
    return vec8f(u[0], u[1], u[2], a + b, v[0], v[1], v[2], b)


@wp.func
def local_strain_derivatives(g: TripletGeometry, sigma: vec5f):
    """``J = d eps/dz`` (5 x 8) and the packed upper triangle (``vec36f``, see :func:`packed_index`) of
    ``sum_i sigma_i d^2 eps_i/dz^2`` in ``z = [e, theta_e, f, theta_f - theta_e]``, frames transported from
    ``g.te_old``, ``g.tf_old``."""
    te = g.te
    tf = g.tf
    ne = g.ne
    nf = g.nf
    chi = g.chi
    tt = g.tt
    kb = g.kb

    # Transport turns: d phi/de = w / (c n).
    we, ce = transport_turn(g.te_old, te)
    wf, cf = transport_turn(g.tf_old, tf)
    ge = we / (ce * ne)
    gf = wf / (cf * nf)

    # J: stretch, the curvatures (M, N) = (m2, -m1) and (-m1, -m2), twist.
    k1 = g.strain[2]
    k2 = g.strain[3]
    De1, Df1, a1, b1 = kappa_gradient(g, k1, g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = kappa_gradient(g, k2, -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    ue = te / g.l0e
    uf = tf / g.l0f
    J = wp.matrix_from_rows(
        vec8f(ue[0], ue[1], ue[2], 0.0, 0.0, 0.0, 0.0, 0.0),
        vec8f(0.0, 0.0, 0.0, 0.0, uf[0], uf[1], uf[2], 0.0),
        _local_jacobian_row(De1, Df1, a1, b1, ge, gf),
        _local_jacobian_row(De2, Df2, a2, b2, ge, gf),
        _local_jacobian_row(0.5 * kb / ne, 0.5 * kb / nf, -1.0, 1.0, ge, gf),
    )

    # Sigma-weighted curvature data (one Hessian for both curvatures).
    s2 = sigma[2]
    s3 = sigma[3]
    s4 = sigma[4]
    kw = s2 * k1 + s3 * k2
    Me = s2 * g.m2e - s3 * g.m1e
    Mf = s2 * g.m2f - s3 * g.m1f
    Ne = -(s2 * g.m1e + s3 * g.m2e)
    Nf = -(s2 * g.m1f + s3 * g.m2f)
    td = (Me + Mf) / chi
    ne2 = ne * ne
    nf2 = nf * nf
    nef = ne * nf

    # theta terms
    W_AA = -0.5 * wp.dot(kb, Me)
    W_BB = -0.5 * wp.dot(kb, Mf)
    sA = -0.5 * wp.dot(kb, Ne)
    sB = -0.5 * wp.dot(kb, Nf)
    W_eA = (sA * tt + wp.cross(tf, Ne) / chi) / ne
    W_fA = (sA * tt - wp.cross(te, Ne) / chi) / nf
    W_eB = (sB * tt + wp.cross(tf, Nf) / chi) / ne
    W_fB = (sB * tt - wp.cross(te, Nf) / chi) / nf
    torque_A = -sA - s4
    torque_B = -sB + s4

    # ee, ff: stretch + curvature projections, the tt and kb symmetric products, the turn terms.
    pe = (kw * tt - wp.cross(tf, td)) / ne2
    pf = (kw * tt + wp.cross(te, td)) / nf2
    ae = (g.te_old - wp.dot(te, g.te_old) * te) / (2.0 * ce * ce) + te / ce
    af = (g.tf_old - wp.dot(tf, g.tf_old) * tf) / (2.0 * cf * cf) + tf / cf
    Xe = (W_eA + 0.5 * W_AA * ge) / (ce * ne) - torque_A / ne2 * ae
    Xf = (W_fB + 0.5 * W_BB * gf) / (cf * nf) - torque_B / nf2 * af
    ee00, ee01, ee02, ee11, ee12, ee22 = _sym_upper(
        sigma[0] / (g.l0e * ne) - kw / (chi * ne2), te, tt, pe, kb, (Me - s4 * (te + tt)) / (4.0 * ne2), we, Xe
    )
    ff00, ff01, ff02, ff11, ff12, ff22 = _sym_upper(
        sigma[1] / (g.l0f * nf) - kw / (chi * nf2), tf, tt, pf, kb, (Mf - s4 * (tf + tt)) / (4.0 * nf2), wf, Xf
    )

    # ef (full 3 x 3): c (I + te tf^T) + qe tt^T + tt qf^T + skew(r) + W_eB gf^T + ge W_fA^T.
    c = -kw / (chi * nef)
    qe = pe * (ne / nf) - (0.5 * s4 / nef) * kb
    qf = pf * (nf / ne)
    r = (s4 / chi * te - td) / nef
    ef = (
        c * (wp.identity(3, dtype=float) + wp.outer(te, tf))
        + wp.outer(qe, tt)
        + wp.outer(tt, qf)
        + wp.outer(W_eB, gf)
        + wp.outer(ge, W_fA)
    )

    e3 = W_eA + W_eB + W_AA * ge
    f7 = W_fB + W_BB * gf
    f3 = W_fA + f7
    # fmt: off
    H = vec36f(
        ee00, ee01, ee02, e3[0], ef[0, 0], ef[0, 1] - r[2], ef[0, 2] + r[1], W_eB[0],
        ee11, ee12, e3[1], ef[1, 0] + r[2], ef[1, 1], ef[1, 2] - r[0], W_eB[1],
        ee22, e3[2], ef[2, 0] - r[1], ef[2, 1] + r[0], ef[2, 2], W_eB[2],
        W_AA + W_BB, f3[0], f3[1], f3[2], W_BB,
        ff00, ff01, ff02, f7[0],
        ff11, ff12, f7[1],
        ff22, f7[2],
        W_BB,
    )
    # fmt: on
    return J, H
