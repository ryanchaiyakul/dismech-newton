"""Triplet strains ``[eps_e, eps_f, kappa1, kappa2, tau]`` and their derivatives in the DOFs
``[x0, theta_e, x1, theta_f, x2]``, taken in edge space (``e = x1 - x0``, ``f = x2 - x1``).
"""

import warp as wp

vec5f = wp.types.vector(5, float)
vec10f = wp.types.vector(10, float)
mat55f = wp.types.matrix((5, 5), float)
vec11f = wp.types.vector(11, float)
mat11f = wp.types.matrix((11, 11), float)
mat5_11f = wp.types.matrix((5, 11), float)
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
# ``theta + phi``: the strains are ``F(x, theta + phi(x))`` with ``F`` the current-frame strains. So with
# ``T = d(x, theta + phi)/dq``, ``J = J_F T`` and ``sum sigma H = T^T (sum sigma H_F) T + sum_edges (sigma . dF/dtheta)
# d^2 phi`` -- the exact (symmetric) Hessian of the step's energy.


@wp.func
def transport_turn(t_old: wp.vec3, t: wp.vec3, n: float):
    """``(d phi/de, d^2 phi/de^2)`` at ``e = n t``, ``phi`` the director's turn about ``t`` between transport from
    ``t_old`` and from ``t``: the signed area of the spherical triangle ``(t_old, t, t')``, ``t' = e' / |e'|``, from
    ``tan(phi / 2) = (t x t_old) . t' / (1 + t_old . t + t . t' + t' . t_old)``."""
    w = wp.cross(t, t_old)
    c = 1.0 + wp.dot(t_old, t)
    v = w / c  # d phi / dt' (perpendicular to t)
    s = t_old + t
    I3 = wp.identity(3, dtype=float)
    P = I3 - wp.outer(t, t)
    H_t = -(wp.outer(w, s) + wp.outer(s, w)) / (2.0 * c * c)  # d^2 phi / dt'^2
    H = (P * H_t * P - wp.outer(v, t) - wp.outer(t, v)) / (n * n)  # through t' = e' / |e'|
    return v / n, H


@wp.func
def _transport_map(ge: wp.vec3, gf: wp.vec3) -> mat11f:
    """``T = d(x, theta + phi)/dq``: the identity, and ``d phi_e/dq``, ``d phi_f/dq`` in the twist rows."""
    T = wp.identity(11, dtype=float)
    Ge = edge_gradient(ge, wp.vec3(), 0.0, 0.0)
    Gf = edge_gradient(wp.vec3(), gf, 0.0, 0.0)
    for i in range(11):
        T[3, i] = T[3, i] + Ge[i]
        T[7, i] = T[7, i] + Gf[i]
    return T


@wp.func
def strain_derivatives(g: TripletGeometry, sigma: vec5f):
    """``J`` (5 x 11) and ``sum_i sigma_i H_i``."""
    JF, HF = _current_frame_derivatives(g, sigma)
    ge, He = transport_turn(g.te_old, g.te, g.ne)
    gf, Hf = transport_turn(g.tf_old, g.tf, g.nf)
    T = _transport_map(ge, gf)
    H = wp.transpose(T) * HF * T
    torque_e = float(0.0)
    torque_f = float(0.0)
    for i in range(5):
        torque_e = torque_e + sigma[i] * JF[i, 3]
        torque_f = torque_f + sigma[i] * JF[i, 7]
    H = add_edge_hessian(H, torque_e * He, wp.mat33(), torque_f * Hf)
    return JF * T, H


@wp.func
def _current_frame_derivatives(g: TripletGeometry, sigma: vec5f):
    """``J_F`` and ``sum_i sigma_i H_F,i``: the derivatives for current frames."""
    zero = wp.vec3()
    De1, Df1, a1, b1 = kappa_gradient(g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = kappa_gradient(g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    Jse = edge_gradient(g.te / g.l0e, zero, 0.0, 0.0)
    Jsf = edge_gradient(zero, g.tf / g.l0f, 0.0, 0.0)
    Jb1 = edge_gradient(De1, Df1, a1, b1)
    Jb2 = edge_gradient(De2, Df2, a2, b2)
    Ja = edge_gradient(0.5 * g.kb / g.ne, 0.5 * g.kb / g.nf, -1.0, 1.0)
    J = mat5_11f()
    for i in range(11):
        J[0, i] = Jse[i]
        J[1, i] = Jsf[i]
        J[2, i] = Jb1[i]
        J[3, i] = Jb2[i]
        J[4, i] = Ja[i]

    # Stretch: the Hessian of |e| / l0 is the projection (I - t t^T) / (l0 |e|).
    I3 = wp.identity(3, dtype=float)
    H = add_edge_hessian(
        mat11f(),
        sigma[0] / (g.l0e * g.ne) * (I3 - wp.outer(g.te, g.te)),
        wp.mat33(),
        sigma[1] / (g.l0f * g.nf) * (I3 - wp.outer(g.tf, g.tf)),
    )
    H = add_kappa_hessian(H, sigma[2], g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    H = add_kappa_hessian(H, sigma[3], g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    H = add_tau_hessian(H, sigma[4], g)
    return J, H


@wp.func
def strain_gradient(g: TripletGeometry, sigma: vec5f) -> vec11f:
    """``J^T sigma`` without component writes, so Warp can differentiate it."""
    De1, Df1, a1, b1 = kappa_gradient(g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = kappa_gradient(g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    De = sigma[0] / g.l0e * g.te + sigma[2] * De1 + sigma[3] * De2 + sigma[4] * 0.5 * g.kb / g.ne
    Df = sigma[1] / g.l0f * g.tf + sigma[2] * Df1 + sigma[3] * Df2 + sigma[4] * 0.5 * g.kb / g.nf
    torque_e = sigma[2] * a1 + sigma[3] * a2 - sigma[4]
    torque_f = sigma[2] * b1 + sigma[3] * b2 + sigma[4]
    # T^T: each edge's turn moves its torque onto the nodes (d phi/de of transport_turn, without its Hessian).
    w_e = wp.cross(g.te, g.te_old) / ((1.0 + wp.dot(g.te_old, g.te)) * g.ne)
    w_f = wp.cross(g.tf, g.tf_old) / ((1.0 + wp.dot(g.tf_old, g.tf)) * g.nf)
    return edge_gradient(De + torque_e * w_e, Df + torque_f * w_f, torque_e, torque_f)


@wp.func
def kappa_gradient(g: TripletGeometry, k: float, Me: wp.vec3, Mf: wp.vec3, Ne: wp.vec3, Nf: wp.vec3):
    """``(d/de, d/df, d/dtheta_e, d/dtheta_f)`` of the curvature ``k``."""
    td = (Me + Mf) / g.chi
    De = (-k * g.tt + wp.cross(g.tf, td)) / g.ne
    Df = (-k * g.tt - wp.cross(g.te, td)) / g.nf
    return De, Df, 0.5 * wp.dot(g.kb, Ne), 0.5 * wp.dot(g.kb, Nf)


@wp.func
def add_kappa_hessian(
    H: mat11f, w: float, g: TripletGeometry, k: float, Me: wp.vec3, Mf: wp.vec3, Ne: wp.vec3, Nf: wp.vec3
) -> mat11f:
    """``H + w d^2k/dq^2``."""
    te = g.te
    tf = g.tf
    ne = g.ne
    nf = g.nf
    chi = g.chi
    tt = g.tt
    kb = g.kb
    td = (Me + Mf) / chi

    I3 = wp.identity(3, dtype=float)
    ne2 = ne * ne
    nf2 = nf * nf
    tt_tt = wp.outer(tt, tt)
    tf_c_td_tt = wp.outer(wp.cross(tf, td), tt)
    te_c_td_tt = wp.outer(wp.cross(te, td), tt)
    kb_Me = wp.outer(kb, Me)
    kb_Mf = wp.outer(kb, Mf)
    Dee = (
        (2.0 * k * tt_tt - tf_c_td_tt - wp.transpose(tf_c_td_tt)) / ne2
        - k / (chi * ne2) * (I3 - wp.outer(te, te))
        + (kb_Me + wp.transpose(kb_Me)) / (4.0 * ne2)
    )
    Dff = (
        (2.0 * k * tt_tt + te_c_td_tt + wp.transpose(te_c_td_tt)) / nf2
        - k / (chi * nf2) * (I3 - wp.outer(tf, tf))
        + (kb_Mf + wp.transpose(kb_Mf)) / (4.0 * nf2)
    )
    Def = -k / (chi * ne * nf) * (I3 + wp.outer(te, tf)) + (
        2.0 * k * tt_tt - tf_c_td_tt + wp.transpose(te_c_td_tt) - skew(td)
    ) / (ne * nf)
    H = add_edge_hessian(H, w * Dee, w * Def, w * Dff)
    H[3, 3] = H[3, 3] - 0.5 * w * wp.dot(kb, Me)
    H[7, 7] = H[7, 7] - 0.5 * w * wp.dot(kb, Mf)
    se = -0.5 * w * wp.dot(kb, Ne)
    sf = -0.5 * w * wp.dot(kb, Nf)
    H = add_edge_theta_hessian(
        H, 3, (se * tt + w * wp.cross(tf, Ne) / chi) / ne, (se * tt - w * wp.cross(te, Ne) / chi) / nf
    )
    H = add_edge_theta_hessian(
        H, 7, (sf * tt + w * wp.cross(tf, Nf) / chi) / ne, (sf * tt - w * wp.cross(te, Nf) / chi) / nf
    )
    return H


@wp.func
def add_tau_hessian(H: mat11f, w: float, g: TripletGeometry) -> mat11f:
    """``H + w d^2tau/dq^2``."""
    kb = g.kb
    te_tt = g.te + g.tt
    tf_tt = g.tf + g.tt
    Dee = -(wp.outer(kb, te_tt) + wp.outer(te_tt, kb)) / (4.0 * g.ne * g.ne)
    Dff = -(wp.outer(kb, tf_tt) + wp.outer(tf_tt, kb)) / (4.0 * g.nf * g.nf)
    Def = (2.0 / g.chi * skew(g.te) - wp.outer(kb, g.tt)) / (2.0 * g.ne * g.nf)
    return add_edge_hessian(H, w * Dee, w * Def, w * Dff)


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
# ``ge = d phi_e/de = w_e / (c_e n_e)`` and ``d^2phi_e = -(w_e a_e^T + a_e w_e^T) / n_e^2`` (from transport_turn,
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
    ``g.te_old``, ``g.tf_old``. Equals ``strain_derivatives`` reduced by ``dq/dz`` (``J T``, ``T^T H T``)."""
    te = g.te
    tf = g.tf
    ne = g.ne
    nf = g.nf
    chi = g.chi
    tt = g.tt
    kb = g.kb

    # Transport turns: d phi/de = w / (c n).
    we = wp.cross(te, g.te_old)
    wf = wp.cross(tf, g.tf_old)
    ce = 1.0 + wp.dot(g.te_old, te)
    cf = 1.0 + wp.dot(g.tf_old, tf)
    ge = we / (ce * ne)
    gf = wf / (cf * nf)

    # J: stretch, the curvatures (M, N) = (m2, -m1) and (-m1, -m2), twist.
    k1 = g.strain[2]
    k2 = g.strain[3]
    td1 = (g.m2e + g.m2f) / chi
    td2 = -(g.m1e + g.m1f) / chi
    ue = te / g.l0e
    uf = tf / g.l0f
    J = wp.matrix_from_rows(
        vec8f(ue[0], ue[1], ue[2], 0.0, 0.0, 0.0, 0.0, 0.0),
        vec8f(0.0, 0.0, 0.0, 0.0, uf[0], uf[1], uf[2], 0.0),
        _local_jacobian_row(
            (-k1 * tt + wp.cross(tf, td1)) / ne, (-k1 * tt - wp.cross(te, td1)) / nf,
            -0.5 * wp.dot(kb, g.m1e), -0.5 * wp.dot(kb, g.m1f), ge, gf,
        ),
        _local_jacobian_row(
            (-k2 * tt + wp.cross(tf, td2)) / ne, (-k2 * tt - wp.cross(te, td2)) / nf,
            -0.5 * wp.dot(kb, g.m2e), -0.5 * wp.dot(kb, g.m2f), ge, gf,
        ),
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


# -- scatter from edge space into the 11 DOFs ---------------------------------------------


@wp.func
def edge_gradient(De: wp.vec3, Df: wp.vec3, dtheta_e: float, dtheta_f: float) -> vec11f:
    d = De - Df
    return vec11f(-De[0], -De[1], -De[2], dtheta_e, d[0], d[1], d[2], dtheta_f, Df[0], Df[1], Df[2])


@wp.func
def add_edge_hessian(H: mat11f, Dee: wp.mat33f, Def: wp.mat33f, Dff: wp.mat33f) -> mat11f:
    """Scatter into node DOFs 0:3, 4:7, 8:11 (``dx0 = -de``, ``dx1 = de - df``, ``dx2 = df``)."""
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
    """Scatter mixed (edge, theta) second derivatives, symmetric."""
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
def add_block(H: mat11f, r: int, c: int, B: wp.mat33f) -> mat11f:
    for i in range(3):
        for j in range(3):
            H[r + i, c + j] = H[r + i, c + j] + B[i, j]
    return H
