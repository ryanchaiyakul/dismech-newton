"""Midedge shells (Chen et al. 2018) on Newton's cloth mesh, as an element of the generic block (:mod:`.element`).

A face's DOFs are its 3 vertices, the 3 flap vertices across its edges (a boundary edge's flap is the vertex
itself) and the 3 midedge angles of its hinges (massless, ``hinge_q``): 21 DOFs. Its strains are the first and
second fundamental forms in the rest frame (``L``, ``L a L^T = I``), whitened by the material: the energy
``A/4 (lam + 2 mu) |R (L a L^T - I)|^2 / 2 + A D |R (L b L^T - b0)|^2 / 2``, with ``R^T R = [[1, nu, 0], [nu, 1, 0],
[0, 0, 2 (1 - nu)]]`` (plane-stress StVK in ``(S11, S22, S12)``), ``lam``, ``mu`` the 2D Lame parameters.
"""

import numpy as np
import warp as wp
from newton import Model

from .builder import _P, vec6f
from .element import element_kernels, rest_strains

vec9i = wp.types.vector(9, wp.int32)
vec21i = wp.types.vector(21, wp.int32)
vec21f = wp.types.vector(21, float)
mat63f = wp.types.matrix((6, 3), float)
mat6x21f = wp.types.matrix((6, 21), float)
mat23f = wp.types.matrix((2, 3), float)


@wp.struct
class Shell:
    """The faces as the element block sees them: 6 nodes and 3 hinges, hinge signs and flaps, the strain map ``W``
    (a form's entries to its whitened strains); ``hinge_offset``: the first midedge angle's DOF."""

    conn: wp.array[vec9i]
    sign: wp.array[wp.vec3]
    flap: wp.array[wp.vec3]
    strain_map: wp.array[wp.mat33]
    hinge_offset: int


class Shells:
    """The model's faces as an element type of :mod:`.element`: the view (the same for every state) and the rest
    strains (the model's initial shape, angles 0)."""

    stored = "face_strain_q"

    def __init__(self, model: Model, hinge_offset: int):
        d, dev = model.dismech, wp.get_device(model.device)
        self.count = model.custom_frequency_counts.get(_P + "face", 0)
        self.params = d.face_params
        conn = np.column_stack(
            [getattr(d, f"face_node{k}").numpy() for k in range(6)]
            + [getattr(d, f"face_hinge{k}").numpy() for k in range(3)]
        ).astype(np.int32)
        v = self.view = Shell()
        v.conn = wp.array(conn.reshape(-1, 9), dtype=vec9i, device=dev)
        v.sign, v.flap, v.strain_map = d.face_sign, d.face_flap, d.face_strain_map
        v.hinge_offset = hinge_offset
        self.kernels = element_kernels(Shell, shell_strain, shell_jacobian, shell_dofs, 6, 21, shell_geometric)
        self.rest = rest_strains(self, model, vec6f)

    def bind(self, s) -> Shell:
        return self.view

    def dofs(self) -> np.ndarray:
        """``(F, 21)``: ``[p0, p1, p2, f0, f1, f2]`` (3 each), then the 3 midedge angles."""
        c = self.view.conn.numpy().astype(np.int64).reshape(-1, 9)
        nodes = (3 * c[:, :6, None] + np.arange(3)).reshape(-1, 18)
        return np.concatenate([nodes, self.view.hinge_offset + c[:, 6:]], axis=1)


# -- the element ---------------------------------------------------------------------------------------


@wp.func
def shell_dofs(v: Shell, i: int) -> vec21i:
    c = v.conn[i]
    d = vec21i()
    for r in range(6):
        for k in range(3):
            d[3 * r + k] = 3 * c[r] + k
    for h in range(3):
        d[18 + h] = v.hinge_offset + c[6 + h]
    return d


@wp.func
def shell_strain(x: vec21f, v: Shell, i: int) -> vec6f:
    m, g = _forms(_points(x), _angles(x), v.sign[i], v.flap[i])
    return _whiten(v, i, m, g)


@wp.func
def shell_jacobian(x: vec21f, v: Shell, i: int):
    Wm = v.strain_map[i]
    m, g, J = _derivatives(_points(x), _angles(x), v.sign[i], v.flap[i], Wm, Wm * _b_of_g())
    return _whiten(v, i, m, g), J


@wp.func
def shell_geometric(v: Shell, i: int, sigma: vec6f) -> wp.mat33:
    """The membrane's geometric stiffness: the first form is quadratic in the vertices, so ``sum sigma d^2 eps =
    (D^T S D) (x) I3``, ``S`` the stress in the edge basis (its tensile part), ``D = [[-1, 1, 0], [-1, 0, 1]]``."""
    w = wp.transpose(v.strain_map[i]) * wp.vec3(float(sigma[0]), float(sigma[1]), float(sigma[2]))
    S = wp.mat22(2.0 * float(w[0]), float(w[1]), float(w[1]), 2.0 * float(w[2]))
    # S+ = (S + |S|) / 2, |S| = (S^2 + |det S| I) / (|l1| + |l2|), (|l1| + |l2|)^2 = tr(S^2) + 2 |det S|
    a = wp.abs(wp.determinant(S))
    r = wp.sqrt(wp.trace(S * S) + 2.0 * a)
    T = wp.mat22()
    if r > 0.0:
        T = 0.5 * (S + (S * S + a * wp.identity(2, float)) / r)
    Dm = mat23f(-1.0, 1.0, 0.0, -1.0, 0.0, 1.0)
    return wp.transpose(Dm) * T * Dm


# -- geometry ---------------------------------------------------------------------------------------------


@wp.func
def _points(x: vec21f) -> mat63f:
    P = mat63f()
    for r in range(6):
        for k in range(3):
            P[r, k] = float(x[3 * r + k])
    return P


@wp.func
def _angles(x: vec21f) -> wp.vec3:
    return wp.vec3(float(x[18]), float(x[19]), float(x[20]))


@wp.func
def _b_of_g() -> wp.mat33:
    """``(m00, m01, m11)`` of the second form from the ``g_i``: ``-2 (g0 + g1), -2 g0, -2 (g0 + g2)``."""
    m2 = float(-2.0)
    z = float(0.0)
    return wp.mat33(m2, m2, z, m2, z, z, m2, z, m2)


@wp.func
def _whiten(v: Shell, i: int, m: wp.vec3, g: wp.vec3) -> vec6f:
    Wm = v.strain_map[i]
    em = Wm * m
    eb = Wm * (_b_of_g() * g)
    return vec6f(float(em[0]), float(em[1]), float(em[2]), float(eb[0]), float(eb[1]), float(eb[2]))


@wp.func
def _row(P: mat63f, r: int) -> wp.vec3:
    return wp.vec3(P[r, 0], P[r, 1], P[r, 2])


@wp.func
def _half_dihedral(P: mat63f, n: wp.vec3, i: int) -> float:
    j = (i + 1) % 3
    k = (i + 2) % 3
    pj = _row(P, j)
    pk = _row(P, k)
    t = pk - pj
    ng = wp.cross(pj - pk, _row(P, 3 + i) - pk)
    s = wp.dot(wp.cross(n, ng), t) / wp.length(t)
    return float(0.5) * wp.atan2(s, wp.dot(n, ng))


@wp.func
def _forms(P: mat63f, th: wp.vec3, sign: wp.vec3, flap: wp.vec3):
    """``(m, g)``: the first form's entries ``(eu.eu, eu.ev, ev.ev)`` and ``g_i = tan(psi_i + s_i theta_i) |n| / l_i``
    (``psi_i`` half the dihedral angle at hinge ``i``)."""
    p0 = _row(P, 0)
    eu = _row(P, 1) - p0
    ev = _row(P, 2) - p0
    n = wp.cross(eu, ev)
    nn = wp.length(n)
    g = wp.vec3()
    for i in range(3):
        t = _row(P, (i + 2) % 3) - _row(P, (i + 1) % 3)
        psi = float(0.0)
        if flap[i] != 0.0:
            psi = _half_dihedral(P, n, i)
        g[i] = wp.tan(psi + float(sign[i]) * th[i]) * nn / wp.length(t)
    return wp.vec3(wp.dot(eu, eu), wp.dot(eu, ev), wp.dot(ev, ev)), g


@wp.func
def _put(J: mat6x21f, row: int, w: wp.vec3, node: int, v: wp.vec3) -> mat6x21f:
    """``J[row:row+3, node] += w v^T``: a form entry's gradient ``v`` at ``node``, through its column ``w`` of ``W``
    (``+=``: Warp's adjoint of ``J[i, j] = J[i, j] + v`` is wrong)."""
    for r in range(3):
        for k in range(3):
            J[row + r, 3 * node + k] += float(w[r] * v[k])
    return J


@wp.func
def _derivatives(P: mat63f, th: wp.vec3, sign: wp.vec3, flap: wp.vec3, Wm: wp.mat33, Wb: wp.mat33):
    """``(m, g, J)``: the forms and ``J = [Wm dm/dx; Wb dg/dx]`` in the 21 local DOFs."""
    p0 = _row(P, 0)
    eu = _row(P, 1) - p0
    ev = _row(P, 2) - p0
    m = wp.vec3(wp.dot(eu, eu), wp.dot(eu, ev), wp.dot(ev, ev))
    J = mat6x21f()
    two = float(2.0)
    w0 = wp.vec3(Wm[0, 0], Wm[1, 0], Wm[2, 0])
    w1 = wp.vec3(Wm[0, 1], Wm[1, 1], Wm[2, 1])
    w2 = wp.vec3(Wm[0, 2], Wm[1, 2], Wm[2, 2])
    J = _put(J, 0, w0, 0, -two * eu)
    J = _put(J, 0, w0, 1, two * eu)
    J = _put(J, 0, w1, 0, -(eu + ev))
    J = _put(J, 0, w1, 1, ev)
    J = _put(J, 0, w1, 2, eu)
    J = _put(J, 0, w2, 0, -two * ev)
    J = _put(J, 0, w2, 2, two * ev)

    n = wp.cross(eu, ev)
    nn = wp.length(n)
    nh = n / nn
    g = wp.vec3()
    one = float(1.0)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        w = wp.vec3(Wb[0, i], Wb[1, i], Wb[2, i])
        pj = _row(P, j)
        pk = _row(P, k)
        t = pk - pj
        length = wp.length(t)
        tn = t / length
        h = nn / length
        psi = float(0.0)
        if flap[i] != 0.0:
            psi = _half_dihedral(P, n, i)
        s = float(sign[i])
        c = wp.tan(psi + s * th[i])
        g[i] = c * h
        sec2 = one + c * c
        # c dh: dh = d|n| / l - |n| dl / l^2, d|n|/dp_a = nh x t_a (t_a the edge opposite a)
        for a in range(3):
            J = _put(J, 3, w, a, (c / length) * wp.cross(nh, _row(P, (a + 2) % 3) - _row(P, (a + 1) % 3)))
        J = _put(J, 3, w, k, (-c * nn / (length * length)) * tn)
        J = _put(J, 3, w, j, (c * nn / (length * length)) * tn)
        # h dc = h sec^2 (dpsi + s dtheta); dpsi = -1/2 (u1 p_i + u2 f + u3 p_j + u4 p_k) (Bridson et al. 2003)
        for r in range(3):
            J[3 + r, 18 + i] = float(w[r] * h * sec2 * s)
        if flap[i] != 0.0:
            f = _row(P, 3 + i)
            ng = wp.cross(pj - pk, f - pk)
            u1 = n / wp.dot(n, n)
            u2 = ng / wp.dot(ng, ng)
            pi = _row(P, i)
            sc = float(-0.5) * h * sec2
            J = _put(J, 3, w, i, sc * length * u1)
            J = _put(J, 3, w, 3 + i, sc * length * u2)
            J = _put(J, 3, w, j, sc * (wp.dot(pi - pk, tn) * u1 + wp.dot(f - pk, tn) * u2))
            J = _put(J, 3, w, k, sc * (-wp.dot(pi - pj, tn) * u1 - wp.dot(f - pj, tn) * u2))
    return m, g, J
