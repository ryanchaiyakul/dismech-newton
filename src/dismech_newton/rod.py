"""Discrete elastic rods as an element type of :mod:`.element`: a triplet (two consecutive edges) and its strains
``[eps_e, eps_f, kappa1, kappa2, tau]``, with the reference frames transported in time; the frame and proxy kernels
the step ends with."""

import numpy as np
import warp as wp
from newton import JointType, Model

from .builder import _P, vec5f
from .element import element_kernels, rest_strains
from .state import node

vec5i = wp.types.vector(5, wp.int32)
vec8f = wp.types.vector(8, float)
vec11i = wp.types.vector(11, wp.int32)
vec11f = wp.types.vector(11, float)
mat5x11f = wp.types.matrix((5, 11), float)


@wp.struct
class Rod:
    """The rods as the element block sees them: connectivity ``(e, f, n0, n1, n2)`` per triplet, rest lengths and the
    step's start (positions, reference directors, reference twist)."""

    q_old: wp.array[float]
    edge_d1_old: wp.array[wp.vec3]
    ref_twist_old: wp.array[float]
    conn: wp.array[vec5i]
    edge_length: wp.array[float]
    theta_dof_offset: int


class Rods:
    """The model's triplets as an element type; ``rest`` starts at the initial strains (stretch 0)."""

    stored = "triplet_strain_q"

    def __init__(self, model: Model):
        d, dev = model.dismech, wp.get_device(model.device)
        self.count = model.custom_frequency_counts.get(_P + "triplet", 0)
        self.params, self.edge_length, self.num_node_dofs = d.triplet_params, d.edge_length, 3 * model.particle_count
        e, f = d.triplet_edge0.numpy(), d.triplet_edge1.numpy()
        node0, node1 = d.edge_node0.numpy(), d.edge_node1.numpy()
        conn = np.column_stack((e, f, node0[e], node1[e], node1[f])).astype(np.int32)
        self.conn = wp.array(conn.reshape(-1, 5), dtype=vec5i, device=dev)
        self.kernels = element_kernels(Rod, rod_strain, rod_jacobian, rod_dofs, 5, 11)
        self.rest = rest_strains(self, model, vec5f)
        if self.count:
            r = self.rest.numpy()
            r[:, :2] = 0.0
            self.rest.assign(r)

    def bind(self, s) -> Rod:
        """The view with frames transported from ``s`` (a state's ``dismech``: ``q``, ``edge_d1_q``,
        ``triplet_ref_twist_q``)."""
        v = Rod()
        v.q_old, v.edge_d1_old, v.ref_twist_old = s.q, s.edge_d1_q, s.triplet_ref_twist_q
        v.conn, v.edge_length, v.theta_dof_offset = self.conn, self.edge_length, self.num_node_dofs
        return v

    def dofs(self) -> np.ndarray:
        """``(T, 11)`` DOFs ``[x0, theta_e, x1, theta_f, x2]``."""
        e, f, n0, n1, n2 = self.conn.numpy().astype(np.int64).reshape(-1, 5).T
        th = self.num_node_dofs
        return np.column_stack([3 * n0, 3 * n0 + 1, 3 * n0 + 2, th + e, 3 * n1, 3 * n1 + 1, 3 * n1 + 2, th + f,
                                3 * n2, 3 * n2 + 1, 3 * n2 + 2])


@wp.func
def rod_dofs(v: Rod, t: int) -> vec11i:
    c = v.conn[t]
    d = vec11i()
    for k in range(3):
        d[k] = 3 * c[2] + k
        d[4 + k] = 3 * c[3] + k
        d[8 + k] = 3 * c[4] + k
    d[3] = v.theta_dof_offset + c[0]
    d[7] = v.theta_dof_offset + c[1]
    return d


@wp.func
def rod_strain(x: vec11f, v: Rod, t: int) -> vec5f:
    return _geometry(x, v, t).strain


@wp.func
def rod_jacobian(x: vec11f, v: Rod, t: int):
    g = _geometry(x, v, t)
    ge = _dphi(g.te_old, g.te, g.ne)
    gf = _dphi(g.tf_old, g.tf, g.nf)
    ue = g.te / g.l0e
    uf = g.tf / g.l0f
    De1, Df1, a1, b1 = _dkappa(g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = _dkappa(g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    Jz = wp.matrix_from_rows(
        vec8f(ue[0], ue[1], ue[2], 0.0, 0.0, 0.0, 0.0, 0.0),
        vec8f(0.0, 0.0, 0.0, 0.0, uf[0], uf[1], uf[2], 0.0),
        _row(De1, Df1, a1, b1, ge, gf),
        _row(De2, Df2, a2, b2, ge, gf),
        _row(0.5 * g.kb / g.ne, 0.5 * g.kb / g.nf, -1.0, 1.0, ge, gf),  # the twist
    )
    J = mat5x11f()
    for i in range(11):
        i0, s0, i1, s1 = _z_of_dof(i)
        for r in range(5):
            J[r, i] = s0 * Jz[r, i0] + s1 * Jz[r, i1]
    return g.strain, J


# -- frames and geometry ----------------------------------------------------------------------


@wp.func
def parallel_transport(m: wp.vec3, t0: wp.vec3, t1: wp.vec3) -> wp.vec3:
    """Transport ``m`` from ``t0`` to ``t1`` along the shortest arc, ``m`` first projected off ``t0``: a prescribed
    DOF moved between steps turns its edges off their stored frames, where the hand-written Jacobian is wrong."""
    m = m - wp.dot(m, t0) * t0
    out = m - wp.dot(m, t1) / (1.0 + wp.dot(t0, t1)) * (t0 + t1)
    return wp.normalize(out)


@wp.func
def material_frame(d1: wp.vec3, t: wp.vec3, theta: float):
    """``(m1, m2)``: ``(d1, t x d1)`` rotated by ``theta``."""
    d2 = wp.cross(t, d1)
    c = wp.cos(theta)
    s = wp.sin(theta)
    return c * d1 + s * d2, -s * d1 + c * d2


@wp.func
def reference_twist(d1e: wp.vec3, te: wp.vec3, d1f: wp.vec3, tf: wp.vec3, ref_twist_old: float) -> float:
    """The angle from ``d1e`` (transported to ``tf``) to ``d1f`` about ``tf``, unwrapped against ``ref_twist_old`` to
    be continuous in time."""
    a = parallel_transport(d1e, te, tf)
    angle = wp.atan2(wp.dot(tf, wp.cross(a, d1f)), wp.dot(a, d1f)) - ref_twist_old
    return ref_twist_old + angle - 2.0 * wp.PI * wp.floor((angle + wp.PI) / (2.0 * wp.PI))


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
    strain: vec5f


@wp.func
def _geometry(x: vec11f, v: Rod, t: int) -> TripletGeometry:
    """Frames and strains from the triplet's DOFs ``x`` (:func:`rod_dofs`' order), reference frames transported from
    the step's start."""
    c = v.conn[t]
    e, f = c[0], c[1]
    g = TripletGeometry()
    ee = wp.vec3(x[4], x[5], x[6]) - wp.vec3(x[0], x[1], x[2])
    ef = wp.vec3(x[8], x[9], x[10]) - wp.vec3(x[4], x[5], x[6])
    g.ne = wp.length(ee)
    g.nf = wp.length(ef)
    g.l0e = v.edge_length[e]
    g.l0f = v.edge_length[f]
    g.te = ee / g.ne
    g.tf = ef / g.nf
    g.te_old = wp.normalize(node(v.q_old, c[3]) - node(v.q_old, c[2]))
    g.tf_old = wp.normalize(node(v.q_old, c[4]) - node(v.q_old, c[3]))
    d1e = parallel_transport(v.edge_d1_old[e], g.te_old, g.te)
    d1f = parallel_transport(v.edge_d1_old[f], g.tf_old, g.tf)
    m1e, m2e = material_frame(d1e, g.te, x[3])
    m1f, m2f = material_frame(d1f, g.tf, x[7])
    g.m1e = m1e
    g.m2e = m2e
    g.m1f = m1f
    g.m2f = m2f
    g.chi = 1.0 + wp.dot(g.te, g.tf)
    g.tt = (g.te + g.tf) / g.chi
    g.kb = 2.0 * wp.cross(g.te, g.tf) / g.chi
    g.strain = vec5f(g.ne / g.l0e - 1.0, g.nf / g.l0f - 1.0, 0.5 * wp.dot(g.kb, m2e + m2f),
                     -0.5 * wp.dot(g.kb, m1e + m1f),
                     x[7] - x[3] + reference_twist(d1e, g.te, d1f, g.tf, v.ref_twist_old[t]))
    return g


# -- strain derivatives -------------------------------------------------------------------


@wp.func
def _dphi(t_old: wp.vec3, t: wp.vec3, n: float) -> wp.vec3:
    return wp.cross(t, t_old) / ((1.0 + wp.dot(t_old, t)) * n)


@wp.func
def _dkappa(g: TripletGeometry, k: float, Me: wp.vec3, Mf: wp.vec3, Ne: wp.vec3, Nf: wp.vec3):
    td = (Me + Mf) / g.chi
    De = (-k * g.tt + wp.cross(g.tf, td)) / g.ne
    Df = (-k * g.tt - wp.cross(g.te, td)) / g.nf
    return De, Df, 0.5 * wp.dot(g.kb, Ne), 0.5 * wp.dot(g.kb, Nf)


@wp.func
def _row(De: wp.vec3, Df: wp.vec3, a: float, b: float, ge: wp.vec3, gf: wp.vec3) -> vec8f:
    u = De + a * ge
    w = Df + b * gf
    return vec8f(u[0], u[1], u[2], a + b, w[0], w[1], w[2], b)


@wp.func
def _z_of_dof(i: int):
    """``dz`` per unit of DOF ``i`` in ``z = [e, theta_e, f, theta_f - theta_e]``: ``s0 e_a0 + s1 e_a1`` (constants
    for a constant ``i``)."""
    if i < 3:
        return i, -1.0, i, 0.0
    if i == 3:
        return 3, 1.0, 7, -1.0
    if i < 7:
        return i - 4, 1.0, i, -1.0
    if i == 7:
        return 7, 1.0, 7, 0.0
    return i - 4, 1.0, i - 4, 0.0


# -- end-of-step kernels (one thread per edge or triplet) ---------------------------------


@wp.kernel
def advance_frames_kernel(node_q_prev: wp.array[wp.vec3], node_q: wp.array[wp.vec3], edge_node0: wp.array[wp.int32],
                          edge_node1: wp.array[wp.int32], edge_d1_prev: wp.array[wp.vec3],
                          edge_d1: wp.array[wp.vec3]):
    """Parallel-transport every ``d1`` to the current tangent."""
    e = wp.tid()
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    t_prev = wp.normalize(node_q_prev[n1] - node_q_prev[n0])
    edge_d1[e] = parallel_transport(edge_d1_prev[e], t_prev, wp.normalize(node_q[n1] - node_q[n0]))


@wp.kernel
def advance_ref_twist_kernel(q: wp.array[float], edge_d1: wp.array[wp.vec3], conn: wp.array[vec5i],
                             ref_twist_prev: wp.array[float], ref_twist: wp.array[float]):
    t = wp.tid()
    c = conn[t]
    te = wp.normalize(node(q, c[3]) - node(q, c[2]))
    tf = wp.normalize(node(q, c[4]) - node(q, c[3]))
    ref_twist[t] = reference_twist(edge_d1[c[0]], te, edge_d1[c[1]], tf, ref_twist_prev[t])


def proxy_joints(model: Model) -> np.ndarray:
    """Per edge, the ``(joint_q, joint_qd)`` starts of its proxy's free root joint, ``-1`` without."""
    out = np.full((model.dismech.edge_body.shape[0], 2), -1, dtype=np.int32)
    if not model.joint_count:
        return out
    free = (model.joint_type.numpy() == int(JointType.FREE)) & (model.joint_parent.numpy() < 0)
    joint_of = np.full(max(model.body_count, 1), -1)
    joint_of[model.joint_child.numpy()[free]] = np.nonzero(free)[0]
    edge_body = model.dismech.edge_body.numpy()
    j = np.where(edge_body >= 0, joint_of[np.maximum(edge_body, 0)], -1)
    has = j >= 0
    out[has, 0] = model.joint_q_start.numpy()[j[has]]
    out[has, 1] = model.joint_qd_start.numpy()[j[has]]
    return out


@wp.kernel
def pose_proxies_kernel(node_q: wp.array[wp.vec3], node_qd: wp.array[wp.vec3], edge_q: wp.array[float],
                        edge_d1: wp.array[wp.vec3], edge_node0: wp.array[wp.int32], edge_node1: wp.array[wp.int32],
                        edge_body: wp.array[wp.int32], edge_joint: wp.array[wp.vec2i], set_velocity: int,
                        body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], joint_q: wp.array[float],
                        joint_qd: wp.array[float]):
    """Pose each proxy (midpoint, +Z tangent, +X ``m1``) and optionally its velocity and free joint."""
    e = wp.tid()
    body = edge_body[e]
    if body < 0:
        return
    x0 = node_q[edge_node0[e]]
    x1 = node_q[edge_node1[e]]
    t = wp.normalize(x1 - x0)
    m1, m2 = material_frame(edge_d1[e], t, edge_q[e])
    X = wp.transform(0.5 * (x0 + x1), wp.quat_from_matrix(wp.matrix_from_cols(m1, m2, t)))
    body_q[body] = X
    v0 = node_qd[edge_node0[e]]
    v1 = node_qd[edge_node1[e]]
    d = x1 - x0
    twist = wp.spatial_vector(0.5 * (v0 + v1), wp.cross(d, v1 - v0) / wp.dot(d, d))
    if set_velocity != 0:
        body_qd[body] = twist
    j = edge_joint[e]
    if joint_q.shape[0] > 0 and j[0] >= 0:  # a free root joint: joint_q = body_q, joint_qd = body_qd
        for k in range(7):
            joint_q[j[0] + k] = X[k]
        for k in range(6):
            joint_qd[j[1] + k] = twist[k]
