import warp as wp
from newton import State

from ..dofs import node
from ..strains import TripletGeometry, triplet_geometry, vec5f, vec8f
from ..triplet import Triplets, vec5i

vec11i = wp.types.vector(11, wp.int32)
mat5x11f = wp.types.matrix((5, 11), float)


@wp.struct
class Rod:
    """The rods as the element block sees them: connectivity, rest lengths and the step's start (positions,
    reference directors, reference twist). :func:`_geometry` evaluates one triplet's frames and strains from it."""

    q_old: wp.array[float]
    edge_d1_old: wp.array[wp.vec3]
    ref_twist_old: wp.array[float]
    conn: wp.array[vec5i]
    edge_length: wp.array[float]
    theta_dof_offset: int


def bind_rod(triplets: Triplets, state_in: State) -> Rod:
    s = state_in.dismech
    v = Rod()
    v.q_old, v.edge_d1_old, v.ref_twist_old = s.q, s.edge_d1_q, s.triplet_ref_twist_q
    v.conn, v.edge_length, v.theta_dof_offset = triplets.conn, triplets.der.edge_length, triplets.num_node_dofs
    return v


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
def rod_strain(q: wp.array[float], v: Rod, t: int) -> vec5f:
    return _geometry(q, v, t).strain


@wp.func
def rod_jacobian(q: wp.array[float], v: Rod, t: int):
    g = _geometry(q, v, t)
    ge = dphi(g.te_old, g.te, g.ne)
    gf = dphi(g.tf_old, g.tf, g.nf)
    ue = deps(g.te, g.l0e)
    uf = deps(g.tf, g.l0f)
    De1, Df1, a1, b1 = dkappa(g, g.strain[2], g.m2e, g.m2f, -g.m1e, -g.m1f)
    De2, Df2, a2, b2 = dkappa(g, g.strain[3], -g.m1e, -g.m1f, -g.m2e, -g.m2f)
    Det, Dft, at, bt = dtau(g)
    Jz = wp.matrix_from_rows(
        vec8f(ue[0], ue[1], ue[2], 0.0, 0.0, 0.0, 0.0, 0.0),
        vec8f(0.0, 0.0, 0.0, 0.0, uf[0], uf[1], uf[2], 0.0),
        _row(De1, Df1, a1, b1, ge, gf),
        _row(De2, Df2, a2, b2, ge, gf),
        _row(Det, Dft, at, bt, ge, gf),
    )
    J = mat5x11f()
    for i in range(11):
        i0, s0, i1, s1 = _z_of_dof(i)
        for r in range(5):
            J[r, i] = s0 * Jz[r, i0] + s1 * Jz[r, i1]
    return g.strain, J


# -- strain derivatives -------------------------------------------------------------------


@wp.func
def dphi(t_old: wp.vec3, t: wp.vec3, n: float) -> wp.vec3:
    return wp.cross(t, t_old) / ((1.0 + wp.dot(t_old, t)) * n)


@wp.func
def deps(t: wp.vec3, l0: float) -> wp.vec3:
    return t / l0


@wp.func
def dkappa(g: TripletGeometry, k: float, Me: wp.vec3, Mf: wp.vec3, Ne: wp.vec3, Nf: wp.vec3):
    td = (Me + Mf) / g.chi
    De = (-k * g.tt + wp.cross(g.tf, td)) / g.ne
    Df = (-k * g.tt - wp.cross(g.te, td)) / g.nf
    return De, Df, 0.5 * wp.dot(g.kb, Ne), 0.5 * wp.dot(g.kb, Nf)


@wp.func
def dtau(g: TripletGeometry):
    return 0.5 * g.kb / g.ne, 0.5 * g.kb / g.nf, -1.0, 1.0


# TODO: move TripletGeometry to rod.py

@wp.func
def _geometry(q: wp.array[float], v: Rod, t: int) -> TripletGeometry:
    c = v.conn[t]
    e, f, n0, n1, n2 = c[0], c[1], c[2], c[3], c[4]
    return triplet_geometry(
        node(q, n0),
        node(q, n1),
        node(q, n2),
        q[v.theta_dof_offset + e],
        q[v.theta_dof_offset + f],
        v.edge_d1_old[e],
        wp.normalize(node(v.q_old, n1) - node(v.q_old, n0)),
        v.edge_d1_old[f],
        wp.normalize(node(v.q_old, n2) - node(v.q_old, n1)),
        v.ref_twist_old[t],
        v.edge_length[e],
        v.edge_length[f],
    )


@wp.func
def _row(De: wp.vec3, Df: wp.vec3, a: float, b: float, ge: wp.vec3, gf: wp.vec3) -> vec8f:
    u = De + a * ge
    w = Df + b * gf
    return vec8f(u[0], u[1], u[2], a + b, w[0], w[1], w[2], b)


@wp.func
def _z_of_dof(i: int):
    """``dz`` per unit of DOF ``i``: ``s0 e_a0 + s1 e_a1`` (constants for a constant ``i``)."""
    if i < 3:
        return i, -1.0, i, 0.0
    if i == 3:
        return 3, 1.0, 7, -1.0
    if i < 7:
        return i - 4, 1.0, i, -1.0
    if i == 7:
        return 7, 1.0, 7, 0.0
    return i - 4, 1.0, i - 4, 0.0
