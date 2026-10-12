"""Midedge shells (Chen et al. 2018) on Newton's cloth mesh, as an element of the generic block (:mod:`.element`).

A face's DOFs are its 3 vertices, the 3 flap vertices across its edges (a boundary edge's flap is the vertex
itself) and the 3 midedge angles of its hinges (massless, ``hinge_q``): 21 DOFs. Its strains are the first and
second fundamental forms in the rest frame (``L``, ``L a L^T = I``), whitened by the material: the energy
``A/4 (lam + 2 mu) |R (L a L^T - I)|^2 / 2 + A D |R (L b L^T - b0)|^2 / 2``, with ``R^T R = [[1, nu, 0], [nu, 1, 0],
[0, 0, 2 (1 - nu)]]`` (plane-stress StVK in ``(S11, S22, S12)``), ``lam``, ``mu`` the 2D Lame parameters. The
geometry is float64 inside.
"""

import numpy as np
import warp as wp
from newton import Model, ModelBuilder, State

from ..builder import _P, register_custom_attributes
from ..dofs import flatten_state
from ..strains import vec6f, vec12f
from .element import element_kernels

vec9i = wp.types.vector(9, wp.int32)
vec21i = wp.types.vector(21, wp.int32)
vec3d = wp.vec3d
mat33d = wp.mat33d
mat63d = wp.types.matrix((6, 3), wp.float64)
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
    """The model's faces, bound for a solver: the view, the rest strains (the model's initial shape, angles 0), the
    previous step's strains (the damping's reference)."""

    def __init__(self, model: Model, hinge_offset: int):
        d, dev = model.dismech, wp.get_device(model.device)
        self.count = model.custom_frequency_counts.get(_P + "face", 0)
        self.params = d.face_params
        conn = np.column_stack([getattr(d, f"face_node{k}").numpy() for k in range(6)]
                               + [getattr(d, f"face_hinge{k}").numpy() for k in range(3)]).astype(np.int32)
        v = self.view = Shell()
        v.conn = wp.array(conn.reshape(-1, 9), dtype=vec9i, device=dev)
        v.sign, v.flap, v.strain_map = d.face_sign, d.face_flap, d.face_strain_map
        v.hinge_offset = hinge_offset
        self.kernels = element_kernels(Shell, shell_strain, shell_jacobian, shell_dofs, 6, 21, shell_geometric)
        self.rest, self.strain_prev = (wp.zeros(self.count, dtype=vec6f, device=dev) for _ in range(2))
        if self.count:
            rest = model.state()
            flatten_state(rest)
            wp.launch(self.kernels.measure, dim=self.count, inputs=[rest.dismech.q, v], outputs=[self.rest],
                      device=dev)

    def dofs(self) -> np.ndarray:
        """``(F, 21)``: ``[p0, p1, p2, f0, f1, f2]`` (3 each), then the 3 midedge angles."""
        c = self.view.conn.numpy().astype(np.int64).reshape(-1, 9)
        nodes = (3 * c[:, :6, None] + np.arange(3)).reshape(-1, 18)
        return np.concatenate([nodes, self.view.hinge_offset + c[:, 6:]], axis=1)

    def begin_step(self, state_in: State) -> None:
        if self.count:
            s = state_in.dismech
            wp.launch(self.kernels.previous, dim=self.count, inputs=[s.face_strain_q, s.q, self.view],
                      outputs=[self.strain_prev], device=self.rest.device)

    def end_step(self, state_out: State) -> None:
        if self.count:
            s = state_out.dismech
            wp.launch(self.kernels.measure, dim=self.count, inputs=[s.q, self.view], outputs=[s.face_strain_q],
                      device=self.rest.device)


# -- building on Newton's cloth mesh -------------------------------------------------------------------


def add_cloth_mesh(builder: ModelBuilder, *, vertices, indices, pos=(0.0, 0.0, 0.0), rot=None, scale: float = 1.0,
                   vel=(0.0, 0.0, 0.0), bend_stiffness: float | None = None, damping: float | None = None,
                   **kwargs) -> np.ndarray:
    """:meth:`newton.ModelBuilder.add_cloth_mesh` as a midedge shell; returns its vertices' particle indices.

    Newton supplies the particles, their lumped masses (``density`` per area), ``particle_radius`` (the contact
    thickness), the triangles and the bending edges; a hinge is a bending edge. The material comes from Newton's
    parameters as its VBD solver reads them:

    - membrane: ``tri_ke``, ``tri_ka``, the 2D Lame ``mu``, ``lam`` (exact at small strain);
    - bending: the plate's ``D = edge_ke A_e / L_e`` per hinge (``A_e`` the area of its two faces), averaged over a
      face's hinges: Newton's dihedral spring is anisotropic, so this is within about 10% along a regular grid's
      axes and 30% across its diagonals. ``bend_stiffness`` sets ``D`` [N m] instead;
    - damping as a time: ``4 tri_kd / (lam + 2 mu)`` (membrane), ``edge_kd / edge_ke`` (bending), or ``damping`` [s]
      for both.

    The rest shape is the initial one. Fix vertices through their particle flags; a clamped edge needs its
    neighbouring row too (the midedge angles stay free).
    """
    register_custom_attributes(builder)
    v0, t0, e0 = builder.particle_count, len(builder.tri_indices), len(builder.edge_indices)
    builder.add_cloth_mesh(pos=pos, rot=wp.quat_identity() if rot is None else rot, scale=scale, vel=vel,
                           vertices=vertices, indices=indices, **kwargs)
    tris = np.asarray(builder.tri_indices[t0:], dtype=np.int64).reshape(-1, 3)
    edges = np.asarray(builder.edge_indices[e0:], dtype=np.int64).reshape(-1, 4)  # (o0, o1, v0, v1)
    ke, ka, kd = np.asarray(builder.tri_materials[t0:], dtype=np.float64).reshape(-1, 5)[:, :3].T
    edge_ke, edge_kd = np.asarray(builder.edge_bending_properties[e0:], dtype=np.float64).reshape(-1, 2).T
    x = np.asarray(builder.particle_q, dtype=np.float32).astype(np.float64)  # as the model stores them

    # -- topology: hinge i of a face is its edge opposite node i, the flap the other face's opposite node -------
    hinge_of = {(min(a, b), max(a, b)): h for h, (_, _, a, b) in enumerate(edges)}
    face_nodes = np.concatenate([tris, tris], axis=1)
    face_hinges, sign, flap = np.zeros((len(tris), 3), np.int64), np.zeros((len(tris), 3)), np.zeros((len(tris), 3))
    for f, tri in enumerate(tris):
        for i in range(3):
            a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
            h = hinge_of[(min(a, b), max(a, b))]
            o0, o1, h0, _ = edges[h]
            face_hinges[f, i] = h
            sign[f, i] = 1.0 if h0 == a else -1.0
            other = o1 if o0 == tri[i] else o0
            if other >= 0:
                face_nodes[f, 3 + i], flap[f, i] = other, 1.0
    if not np.allclose(np.bincount(face_hinges.ravel(), (sign * flap).ravel(), minlength=len(edges)), 0.0):
        raise ValueError("add_cloth_mesh: inconsistently oriented faces")

    # -- rest frame and material --------------------------------------------------------------------------------
    p = x[tris]
    eu, ev = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    metric = np.stack([np.stack([(eu * eu).sum(1), (eu * ev).sum(1)], -1),
                       np.stack([(eu * ev).sum(1), (ev * ev).sum(1)], -1)], -2)
    area = 0.5 * np.sqrt(np.linalg.det(metric))
    L = np.linalg.inv(np.linalg.cholesky(metric))  # L a L^T = I
    lam_2mu = ka + 2.0 * ke  # 2D Lame: lam = tri_ka, mu = tri_ke
    if bend_stiffness is None:
        hinge_area = np.bincount(face_hinges.ravel(), np.repeat(area, 3), minlength=len(edges))
        hinge_len = np.linalg.norm(x[edges[:, 3]] - x[edges[:, 2]], axis=1)
        bend_stiffness = (edge_ke * hinge_area / hinge_len)[face_hinges].mean(1)
    gamma = (4.0 * kd / lam_2mu, (edge_kd / np.where(edge_ke > 0, edge_ke, np.inf))[face_hinges].mean(1))
    K = np.repeat(np.column_stack([0.25 * area * lam_2mu, area * bend_stiffness]), 3, axis=1)
    C = K * (damping if damping is not None else np.repeat(np.column_stack(gamma), 3, axis=1))
    params, nu = np.concatenate([K, C], axis=1), ka / lam_2mu
    # W = R T: T takes (m00, m01, m11) to (S11, S22, S12) of S = L M L^T, R whitens (module docstring)
    l00, l01, l10, l11 = L[:, 0, 0], L[:, 0, 1], L[:, 1, 0], L[:, 1, 1]
    T = np.stack([np.stack([l00 * l00, 2 * l00 * l01, l01 * l01], -1), np.stack([l10 * l10, 2 * l10 * l11, l11 * l11], -1),
                  np.stack([l00 * l10, l00 * l11 + l01 * l10, l01 * l11], -1)], -2)
    R = np.zeros((len(tris), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 1], R[:, 2, 2] = 1.0, nu, np.sqrt(1.0 - nu**2), np.sqrt(2.0 * (1.0 - nu))
    W = R @ T

    for h, (_, _, h0, h1) in enumerate(edges):
        builder.add_custom_values(**{_P + "hinge_node0": int(h0), _P + "hinge_node1": int(h1), _P + "hinge_fixed": 0,
                                     _P + "hinge_q": 0.0, _P + "hinge_qd": 0.0})
    hinge0 = len(builder.custom_attributes[_P + "hinge_fixed"].values) - len(edges)
    for f in range(len(tris)):
        builder.add_custom_values(
            **{_P + f"face_node{k}": int(face_nodes[f, k]) for k in range(6)},
            **{_P + f"face_hinge{k}": hinge0 + int(face_hinges[f, k]) for k in range(3)},
            **{_P + "face_sign": wp.vec3(*sign[f]), _P + "face_flap": wp.vec3(*flap[f]),
               _P + "face_strain_map": wp.mat33(*W[f].ravel().tolist()),
               _P + "face_params": vec12f(*params[f].tolist())})
    return np.arange(v0, builder.particle_count)


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
def shell_strain(q: wp.array[float], v: Shell, i: int) -> vec6f:
    c = v.conn[i]
    m, g = _forms(_points(q, c), _angles(q, v, c), v.sign[i], v.flap[i])
    return _whiten(v, i, m, g)


@wp.func
def shell_jacobian(q: wp.array[float], v: Shell, i: int):
    c = v.conn[i]
    Wm = _membrane_map(v, i)
    m, g, J = _derivatives(_points(q, c), _angles(q, v, c), v.sign[i], v.flap[i], Wm, Wm * _b_of_g())
    return _whiten(v, i, m, g), J


@wp.func
def shell_geometric(v: Shell, i: int, sigma: vec6f) -> wp.mat33:
    """The membrane's geometric stiffness: the first form is quadratic in the vertices, so ``sum sigma d^2 eps =
    (D^T S D) (x) I3``, ``S`` the stress in the edge basis (its tensile part), ``D = [[-1, 1, 0], [-1, 0, 1]]``."""
    w = wp.transpose(_membrane_map(v, i)) * vec3d(wp.float64(sigma[0]), wp.float64(sigma[1]), wp.float64(sigma[2]))
    S = wp.mat22(2.0 * float(w[0]), float(w[1]), float(w[1]), 2.0 * float(w[2]))
    # S+ = (S + |S|) / 2, |S| = (S^2 + |det S| I) / (|l1| + |l2|), (|l1| + |l2|)^2 = tr(S^2) + 2 |det S|
    a = wp.abs(wp.determinant(S))
    r = wp.sqrt(wp.trace(S * S) + 2.0 * a)
    T = wp.mat22()
    if r > 0.0:
        T = 0.5 * (S + (S * S + a * wp.identity(2, float)) / r)
    Dm = mat23f(-1.0, 1.0, 0.0, -1.0, 0.0, 1.0)
    return wp.transpose(Dm) * T * Dm


# -- geometry (float64) --------------------------------------------------------------------------------


@wp.func
def _points(q: wp.array[float], c: vec9i) -> mat63d:
    P = mat63d()
    for r in range(6):
        for k in range(3):
            P[r, k] = wp.float64(q[3 * c[r] + k])
    return P


@wp.func
def _angles(q: wp.array[float], v: Shell, c: vec9i) -> vec3d:
    o = v.hinge_offset
    return vec3d(wp.float64(q[o + c[6]]), wp.float64(q[o + c[7]]), wp.float64(q[o + c[8]]))


@wp.func
def _membrane_map(v: Shell, i: int) -> mat33d:
    W = v.strain_map[i]
    return mat33d(wp.float64(W[0, 0]), wp.float64(W[0, 1]), wp.float64(W[0, 2]), wp.float64(W[1, 0]),
                  wp.float64(W[1, 1]), wp.float64(W[1, 2]), wp.float64(W[2, 0]), wp.float64(W[2, 1]),
                  wp.float64(W[2, 2]))


@wp.func
def _b_of_g() -> mat33d:
    """``(m00, m01, m11)`` of the second form from the ``g_i``: ``-2 (g0 + g1), -2 g0, -2 (g0 + g2)``."""
    m2 = wp.float64(-2.0)
    z = wp.float64(0.0)
    return mat33d(m2, m2, z, m2, z, z, m2, z, m2)


@wp.func
def _whiten(v: Shell, i: int, m: vec3d, g: vec3d) -> vec6f:
    Wm = _membrane_map(v, i)
    em = Wm * m
    eb = Wm * (_b_of_g() * g)
    return vec6f(float(em[0]), float(em[1]), float(em[2]), float(eb[0]), float(eb[1]), float(eb[2]))


@wp.func
def _row(P: mat63d, r: int) -> vec3d:
    return vec3d(P[r, 0], P[r, 1], P[r, 2])


@wp.func
def _half_dihedral(P: mat63d, n: vec3d, i: int) -> wp.float64:
    j = (i + 1) % 3
    k = (i + 2) % 3
    pj = _row(P, j)
    pk = _row(P, k)
    t = pk - pj
    ng = wp.cross(pj - pk, _row(P, 3 + i) - pk)
    s = wp.dot(wp.cross(n, ng), t) / wp.length(t)
    return wp.float64(0.5) * wp.atan2(s, wp.dot(n, ng))


@wp.func
def _forms(P: mat63d, th: vec3d, sign: wp.vec3, flap: wp.vec3):
    """``(m, g)``: the first form's entries ``(eu.eu, eu.ev, ev.ev)`` and ``g_i = tan(psi_i + s_i theta_i) |n| / l_i``
    (``psi_i`` half the dihedral angle at hinge ``i``)."""
    p0 = _row(P, 0)
    eu = _row(P, 1) - p0
    ev = _row(P, 2) - p0
    n = wp.cross(eu, ev)
    nn = wp.length(n)
    g = vec3d()
    for i in range(3):
        t = _row(P, (i + 2) % 3) - _row(P, (i + 1) % 3)
        psi = wp.float64(0.0)
        if flap[i] != 0.0:
            psi = _half_dihedral(P, n, i)
        g[i] = wp.tan(psi + wp.float64(sign[i]) * th[i]) * nn / wp.length(t)
    return vec3d(wp.dot(eu, eu), wp.dot(eu, ev), wp.dot(ev, ev)), g


@wp.func
def _put(J: mat6x21f, row: int, w: vec3d, node: int, v: vec3d) -> mat6x21f:
    """``J[row:row+3, node] += w v^T``: a form entry's gradient ``v`` at ``node``, through its column ``w`` of ``W``."""
    for r in range(3):
        for k in range(3):
            J[row + r, 3 * node + k] = J[row + r, 3 * node + k] + float(w[r] * v[k])
    return J


@wp.func
def _derivatives(P: mat63d, th: vec3d, sign: wp.vec3, flap: wp.vec3, Wm: mat33d, Wb: mat33d):
    """``(m, g, J)``: the forms and ``J = [Wm dm/dx; Wb dg/dx]`` in the 21 local DOFs, accumulated straight into
    float32 (two float64 3 x 21 gradients make the kernel too large for nvrtc -O3)."""
    p0 = _row(P, 0)
    eu = _row(P, 1) - p0
    ev = _row(P, 2) - p0
    m = vec3d(wp.dot(eu, eu), wp.dot(eu, ev), wp.dot(ev, ev))
    J = mat6x21f()
    two = wp.float64(2.0)
    w0 = vec3d(Wm[0, 0], Wm[1, 0], Wm[2, 0])
    w1 = vec3d(Wm[0, 1], Wm[1, 1], Wm[2, 1])
    w2 = vec3d(Wm[0, 2], Wm[1, 2], Wm[2, 2])
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
    g = vec3d()
    one = wp.float64(1.0)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        w = vec3d(Wb[0, i], Wb[1, i], Wb[2, i])
        pj = _row(P, j)
        pk = _row(P, k)
        t = pk - pj
        length = wp.length(t)
        tn = t / length
        h = nn / length
        psi = wp.float64(0.0)
        if flap[i] != 0.0:
            psi = _half_dihedral(P, n, i)
        s = wp.float64(sign[i])
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
            sc = wp.float64(-0.5) * h * sec2
            J = _put(J, 3, w, i, sc * length * u1)
            J = _put(J, 3, w, 3 + i, sc * length * u2)
            J = _put(J, 3, w, j, sc * (wp.dot(pi - pk, tn) * u1 + wp.dot(f - pk, tn) * u2))
            J = _put(J, 3, w, k, sc * (-wp.dot(pi - pj, tn) * u1 - wp.dot(f - pj, tn) * u2))
    return m, g, J
