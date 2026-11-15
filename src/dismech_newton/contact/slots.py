"""Contact primitive slots, deformable mesh metadata, rigid body metadata, and yielding-body arm helpers."""

import numpy as np
import scipy.sparse as sp
import warp as wp
from newton import GeoType, Model, ShapeFlags
from newton._src.geometry.sdf_texture import TextureSDFData, create_empty_texture_sdf_data
from scipy.sparse.csgraph import maximum_bipartite_matching

from ..builder import _P
from ..constants import SIDES
from ..dofs import node
from ..solver.linear import row_slot
from ..worlds import world_shift


@wp.struct
class Slots:
    """Per contact: ``pair`` with weights ``w``, the normal from ``b`` to ``a``, the anchor, the start ``c0`` of
    ``C q`` (minus the anchor's motion), the thickness, friction coefficient, lagged normal force ``rho``,
    stiffness and multiplier ``lam``; ``own``, ``key``: its feature and the other one, which the list is sorted and
    the multiplier carried by; ``world``: its world."""

    active: wp.array[wp.int32]
    own: wp.array[wp.int32]
    world: wp.array[wp.int32]
    pair: wp.array[wp.vec4i]
    w: wp.array[wp.vec4]
    normal: wp.array[wp.vec3]
    anchor: wp.array[wp.vec3]
    c0: wp.array[wp.vec3]
    thickness: wp.array[float]
    mu: wp.array[float]
    rho: wp.array[float]
    stiffness: wp.array[float]
    lam: wp.array[float]
    key: wp.array[wp.int32]
    roll: wp.array[wp.vec4]
    ta: wp.array[wp.vec3]
    tb: wp.array[wp.vec3]
    pad: wp.array[wp.vec4]  # a pad's point: (c, p, h, d), its energy c (th + h - d)^p, Hunt-Crossley's d [s/m]
    damp: wp.array[float]  # and its damping d f0 / dt (f0 its pressure's force at the step's start)
    rewound: wp.array[wp.int32]  # sliding pair tried to rewind in this step: use isotropic majorizer
    aniso: wp.array[wp.vec4]  # (e1.x, e1.y, e1.z, gamma), D_t = e^(g/2) e1 e1^T + e^(-g/2) e2 e2^T
    base: wp.array[wp.vec2]  # baseline (mu0, pad_c0) saved at emit (for Phase 4C idempotency)


@wp.struct
class Arm:
    """Bodies that yield to the contacts within the step (:class:`~.coupling.ProxyArm`): ``k`` DOFs ``z`` per world,
    from DOF ``start``, their moves over the step (their anchors are at its start). Body ``b``'s 6 DOFs start at
    ``base[b]`` (-1: it does not yield), its centre of mass's move ``dx`` and a small rotation ``dth``, moving its
    point ``x`` by ``dx + dth x (x - com[b])``; their inertia is ``(z - pred)^T M[w] (z - pred) / 2 h^2``, ``pred``
    their own step's moves, ``M`` block-diagonal by body (its 6 x 6; the kernels read only those blocks). A shape
    pair's body is ``shape_body[shape_of(key, key0)]``."""

    k: int
    start: int
    key0: int
    inv_h2: float
    shape_body: wp.array[wp.int32]
    base: wp.array[wp.int32]
    com: wp.array[wp.vec3]
    M: wp.array3d[float]
    Minv: wp.array3d[float]
    pred: wp.array[float]


def make_arm(model: Model, key0: int, start: int, k: int) -> Arm:
    if k % 6:
        raise ValueError(f"an arm's DOFs are whole bodies, 6 each (M block-diagonal by body), got {k}")
    a, dev, nb = Arm(), model.device, max(model.body_count, 1)
    a.k, a.start, a.key0, a.inv_h2 = k, start, key0, 0.0
    a.shape_body = model.shape_body if model.shape_count else wp.zeros(1, dtype=wp.int32, device=dev)
    a.base = wp.full(nb, -1, dtype=wp.int32, device=dev)
    a.com = wp.zeros(nb, dtype=wp.vec3, device=dev)
    a.M = wp.zeros((max(model.world_count, 1), max(k, 1), max(k, 1)), dtype=float, device=dev)
    a.Minv = wp.zeros_like(a.M)
    a.pred = wp.zeros(max(model.world_count, 1) * max(k, 1), dtype=float, device=dev)
    return a


def _slots(n: int, device) -> Slots:
    s = Slots()
    for name, var in Slots.vars.items():
        setattr(s, name, wp.zeros(n, dtype=var.type.dtype, device=device))
    s.key.fill_(-1)
    return s


# -- pairs: shared with the sources ----------------------------------------------------------------


@wp.func
def contact_point(q: wp.array[float], pair: wp.vec4i, w: wp.vec4, anchor: wp.vec3) -> wp.vec3:
    """``C q = sum_k w_k x_k - anchor``."""
    c = -anchor
    for k in range(4):
        if pair[k] >= 0:
            c += w[k] * node(q, pair[k])
    return c


@wp.func
def roll_point(
    q: wp.array[float], pair: wp.vec4i, w: wp.vec4, roll: wp.vec4, n: wp.vec3, ta: wp.vec3, tb: wp.vec3
) -> wp.vec3:
    """The rods' surface points' motion beyond their centrelines': a side on a rod edge ``x0 -> x1`` (tangent
    ``t``, radius ``r``) turns by ``t0 x t``, moving its contact point ``-r n`` by ``r t (n . (x1 - x0)) / L`` to
    first order (``roll = (-r/L, r/L)`` on the edge's nodes; ``t`` its side's: ``ta`` where ``w > 0``). Tangential
    (``t`` normal to ``n``): it moves only friction's slip, so a cable on a turning pulley rolls with it, its
    centreline at ``w (R + r)``."""
    c = wp.vec3(0.0)
    for k in range(4):
        if roll[k] != 0.0:
            c += roll[k] * wp.dot(n, node(q, pair[k])) * wp.where(w[k] > 0.0, ta, tb)
    return c


@wp.func
def node_map(w: wp.vec4, roll: wp.vec4, n: wp.vec3, ta: wp.vec3, tb: wp.vec3, k: int) -> wp.mat33:
    """``d (C q + roll) / d x_k``: ``w_k I + roll_k t n^T``."""
    return w[k] * wp.identity(3, float) + roll[k] * wp.outer(wp.where(w[k] > 0.0, ta, tb), n)


@wp.func
def slot_map(s: Slots, i: int, k: int) -> wp.mat33:
    return node_map(s.w[i], s.roll[i], s.normal[i], s.ta[i], s.tb[i], k)


@wp.func
def slot_roll(q: wp.array[float], s: Slots, i: int) -> wp.vec3:
    return roll_point(q, s.pair[i], s.w[i], s.roll[i], s.normal[i], s.ta[i], s.tb[i])


@wp.func
def shape_of(key: int, key0: int) -> int:
    """A shape pair's shape from its key ``key0 + SIDES shape + side`` (side 1: a line contact's other end)."""
    return (key - key0) // SIDES


@wp.func
def arm_body(arm: Arm, s: Slots, i: int) -> int:
    """Pair ``i``'s yielding body, or -1."""
    if arm.k == 0 or s.key[i] < arm.key0:
        return -1
    b = arm.shape_body[shape_of(s.key[i], arm.key0)]
    if b < 0 or arm.base[b] < 0:
        return -1
    return b


@wp.func
def arm_jacobian(m: int, r: wp.vec3) -> wp.vec3:
    """The move of a body's point ``com + r`` per unit of its DOF ``m``: ``dx`` (``m < 3``), then ``dth``."""
    e = wp.vec3(0.0)
    e[m % 3] = 1.0
    return wp.where(m < 3, e, wp.cross(e, r))


@wp.func
def arm_move(q: wp.array[float], arm: Arm, b: int, r: wp.vec3) -> wp.vec3:
    """Body ``b``'s point ``com + r`` moved by its DOFs in ``q``."""
    i = arm.base[b]
    return wp.vec3(q[i], q[i + 1], q[i + 2]) + wp.cross(wp.vec3(q[i + 3], q[i + 4], q[i + 5]), r)


@wp.func
def arm_scatter(b: wp.array[wp.float64], arm: Arm, body: int, r: wp.vec3, g: wp.vec3):
    """``b -= P^T g``: a force ``g`` on the body's point ``com + r`` (``P`` its :func:`arm_move`), into its DOFs."""
    i = arm.base[body]
    t = wp.cross(r, g)
    for c in range(3):
        wp.atomic_add(b, i + c, wp.float64(-g[c]))
        wp.atomic_add(b, i + 3 + c, wp.float64(-t[c]))


@wp.func
def arm_assemble(
    vals: wp.array[wp.float64],
    indptr: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    fixed: wp.array[wp.int32],
    arm: Arm,
    body: int,
    r: wp.vec3,
    H: wp.mat33,
    n: int,
    Jn: wp.mat33,
):
    """Into the upper CSR (all in the pattern: a node and its world's arm), the Hessian blocks of an energy with
    Hessian ``H`` in a gap ``Jn x_n - P z`` (``P`` the body's :func:`arm_move` at ``r``): node ``n``'s ``-Jn^T H P``,
    then, with ``n < 0``, the body's ``P^T H P``."""
    for m in range(6):
        HP = H * arm_jacobian(m, r)
        zm = arm.base[body] + m
        if n >= 0:
            v = wp.transpose(Jn) * HP
            for c in range(3):
                if fixed[3 * n + c] == 0:
                    wp.atomic_add(vals, row_slot(indptr, indices, 3 * n + c, zm), wp.float64(-v[c]))
        else:
            for m2 in range(m, 6):
                wp.atomic_add(
                    vals,
                    row_slot(indptr, indices, zm, arm.base[body] + m2),
                    wp.float64(wp.dot(arm_jacobian(m2, r), HP)),
                )


@wp.func
def body_flexibility(arm: Arm, Minv: wp.array3d[float], b: int, n: wp.vec3, r: wp.vec3) -> float:
    """``n P M^-1 P^T n``: body ``b``'s flexibility along ``n`` at its point ``com + r`` (``Minv``: the arm's)."""
    w = (arm.base[b] - arm.start) // arm.k
    o = arm.base[b] - arm.start - w * arm.k  # the body's DOFs in its world's
    flex = float(0.0)
    for m in range(6):
        pm = wp.dot(n, arm_jacobian(m, r))
        for m2 in range(6):
            flex += pm * Minv[w, o + m, o + m2] * wp.dot(n, arm_jacobian(m2, r))
    return flex


@wp.func
def arm_flexibility(arm: Arm, s: Slots, i: int) -> float:
    """The pair's yielding body's part of its flexibility along its normal (:func:`body_flexibility`)."""
    b = arm_body(arm, s, i)
    if b < 0:
        return 0.0
    return body_flexibility(arm, arm.Minv, b, s.normal[i], s.anchor[i] - arm.com[b])


@wp.func
def pair_point(q: wp.array[float], s: Slots, i: int, arm: Arm, anchor: wp.vec3) -> wp.vec3:
    """:func:`contact_point` with the anchor moved by its body's DOFs in ``q`` if it yields, plus the rods' surface
    turn (:func:`roll_point`)."""
    c = contact_point(q, s.pair[i], s.w[i], anchor) + slot_roll(q, s, i)
    b = arm_body(arm, s, i)
    if b >= 0:
        c -= arm_move(q, arm, b, s.anchor[i] - arm.com[b])
    return c


@wp.func
def pair_fixed(fixed: wp.array[wp.int32], pair: wp.vec4i) -> bool:
    """Nothing in ``pair`` can move."""
    for k in range(4):
        n = pair[k]
        if n >= 0 and fixed[3 * n] * fixed[3 * n + 1] * fixed[3 * n + 2] == 0:
            return False
    return True


@wp.func
def flexibility(pair: wp.vec4i, w: wp.vec4, mass: wp.array[float], fixed: wp.array[wp.int32]) -> float:
    """``C M^-1 C^T`` (an x-y-z mean over the pair's free DOFs)."""
    flex = float(0.0)
    for j in range(4):
        p = pair[j]
        if p >= 0:
            for a in range(3):
                if fixed[3 * p + a] == 0:
                    flex += w[j] * w[j] / (3.0 * mass[3 * p + a])
    return flex


@wp.func
def emit(
    s: Slots,
    i: int,
    q: wp.array[float],
    pair: wp.vec4i,
    w: wp.vec4,
    n: wp.vec3,
    xb: wp.vec3,
    db: wp.vec3,
    th: float,
    mu: float,
    radius: wp.vec4,
    pad: wp.vec4,
):
    """Activate slot ``i``; ``xb``: the anchor at the end of the step, ``db``: its displacement over it; ``radius``:
    per node its rod's (0 off rods): a side on two rod nodes (an edge) rolls (:func:`roll_point`); ``pad``: a pad's
    node (:attr:`Slots.pad`), ``th`` to the pad's surface (its core's ``h`` less)."""
    s.active[i] = 1
    s.pair[i] = pair
    s.w[i] = w
    s.normal[i] = n
    s.anchor[i] = xb
    roll = wp.vec4(0.0)
    s.ta[i] = wp.vec3(0.0)
    s.tb[i] = wp.vec3(0.0)
    for side in range(2):
        k0 = int(-1)
        k1 = int(-1)
        count = int(0)
        for k in range(4):
            if pair[k] >= 0 and (w[k] > 0.0) == (side == 0):
                count += 1
                if k0 < 0:
                    k0 = k
                else:
                    k1 = k
        if count == 2:
            if radius[k0] > 0.0 and radius[k1] > 0.0:
                e = node(q, pair[k1]) - node(q, pair[k0])
                t = e - wp.dot(e, n) * n
                if wp.length(t) > 0.0:
                    rl = 0.5 * (radius[k0] + radius[k1]) / wp.length(e)
                    roll[k0] = -rl
                    roll[k1] = rl
                    if side == 0:
                        s.ta[i] = wp.normalize(t)
                    else:
                        s.tb[i] = wp.normalize(t)
    s.roll[i] = roll
    s.c0[i] = contact_point(q, pair, w, xb) + db + slot_roll(q, s, i)
    s.thickness[i] = th - pad[2]
    s.pad[i] = pad
    s.mu[i] = mu
    s.rewound[i] = 0
    s.aniso[i] = wp.vec4(0.0)
    s.base[i] = wp.vec2(mu, pad[0])


@wp.func
def clear(s: Slots, i: int):
    s.active[i] = 0
    s.rho[i] = 0.0
    s.key[i] = -1
    s.rewound[i] = 0
    s.aniso[i] = wp.vec4(0.0)
    s.base[i] = wp.vec2(0.0)


@wp.struct
class Mesh:
    """What collides: per edge (the rods', then the shells') its nodes, whether it collides and (a shell's) its faces'
    opposite nodes (-1) and the triangle whose shape pairs its slots hold (-1; each triangle one edge's); per
    triangle its nodes and edges (edge ``k`` opposite node ``k``); per node its neighbours
    along colliding edges (CSR), radius, ``mu``, rod (-1 off rods), arc length and rest position (flat, the model's
    initial state). A rod here is a branch: a chain between nodes of other than two rod edges (ends, junctions), or a
    loop; per rod edge its branch, its first node's arc and the arc's direction along it; per branch its end nodes
    (equal: a loop) and length."""

    edges: wp.array[wp.vec2i]
    edge_on: wp.array[wp.int32]
    edge_opp: wp.array[wp.vec2i]
    edge_face: wp.array[wp.int32]
    faces: wp.array[wp.vec3i]
    face_edge: wp.array[wp.vec3i]
    adj_ptr: wp.array[wp.int32]
    adj: wp.array[wp.int32]
    radius: wp.array[float]
    mu: wp.array[float]
    rod: wp.array[wp.int32]
    arc: wp.array[float]
    rest: wp.array[float]
    rod_len: wp.array[float]
    rod_end: wp.array[wp.vec2i]
    edge_rod: wp.array[wp.int32]
    edge_arc: wp.array[float]
    edge_dir: wp.array[float]
    world: wp.array[wp.int32]
    shift: wp.array[wp.vec3]  # per node, its world's place in the trees (each world apart: no other world's pairs)
    node_surf: wp.array[wp.int32]
    node_u: wp.array[wp.vec3]
    node_face: wp.array[wp.int32]
    shape_surf: wp.array[wp.int32]
    shape_u: wp.array[wp.vec3]


def _mu_shape(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Per node the shape whose ``shape_material_mu`` is its friction: its rod capsule's (a rod's ``cfg.mu``), or -1:
    ``particle_mu`` (a shell vertex); and per rod edge whether it collides (its capsule has a colliding shape)."""
    der, n = model.dismech, model.particle_count
    n0, n1, edge_body = (getattr(der, k).numpy() for k in ("edge_node0", "edge_node1", "edge_body"))
    shape_body = model.shape_body.numpy()
    own = np.flatnonzero((shape_body >= 0) & ((model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0))
    body_shape = np.full(max(model.body_count, 1), -1)
    body_shape[shape_body[own]] = own
    on = (edge_body >= 0) & (body_shape[np.maximum(edge_body, 0)] >= 0)
    out = np.full(n, -1)
    out[n0[on]] = out[n1[on]] = body_shape[edge_body[on]]
    return out, on


def _mesh(model: Model, device) -> Mesh:
    der = model.dismech
    n0, n1, length, edge_body = (
        getattr(der, k).numpy() for k in ("edge_node0", "edge_node1", "edge_length", "edge_body")
    )
    n = model.particle_count
    mu_shape, on = _mu_shape(model)
    mu = np.where(mu_shape >= 0, model.shape_material_mu.numpy()[np.maximum(mu_shape, 0)] if model.shape_count else 0.0,
                  float(model.particle_mu))
    rod, arc, rod_len, rod_end = np.full(n, -1, dtype=np.int32), np.zeros(n), [], []
    ne = len(n0)
    edge_rod, edge_arc, edge_dir = np.full(ne, -1), np.zeros(ne), np.ones(ne)
    deg = np.bincount(np.concatenate([n0, n1]), minlength=n)
    inc = [[] for _ in range(n)]
    for e, (a, b) in enumerate(zip(n0.tolist(), n1.tolist())):
        inc[a].append(e)
        inc[b].append(e)
    for start in [v for v in range(n) if deg[v] not in (0, 2)] + [v for v in range(n) if deg[v] == 2]:  # then loops
        for e in inc[start]:
            v, a, r = start, 0.0, len(rod_len)
            if edge_rod[e] >= 0:
                continue
            if rod[v] < 0:
                rod[v] = r
            while edge_rod[e] < 0:
                forward = n0[e] == v
                edge_rod[e], edge_dir[e], edge_arc[e] = r, 1.0 if forward else -1.0, a if forward else a + length[e]
                a, v = a + length[e], n1[e] if forward else n0[e]
                if rod[v] < 0:
                    rod[v], arc[v] = r, a
                if deg[v] != 2:
                    break
                e = inc[v][0] if inc[v][1] == e else inc[v][1]
            rod_len.append(a)
            rod_end.append((start, v))
    # The shells' edges: edge k of a triangle is opposite its node k; an edge's faces' opposite nodes.
    nf = model.custom_frequency_counts.get(_P + "face", 0)
    tris = np.stack([getattr(der, f"face_node{k}").numpy() for k in range(3)], 1) if nf else np.zeros((0, 3), int)
    ends = np.stack([tris[:, [1, 2, 0]], tris[:, [2, 0, 1]]], 2).reshape(-1, 2)
    uniq, first, inv, count = np.unique(
        np.sort(ends, 1), axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    inv = inv.ravel()
    last = len(inv) - 1 - np.unique(np.sort(ends, 1)[::-1], axis=0, return_index=True)[1]
    opp = np.stack([tris.ravel()[first], np.where(count > 1, tris.ravel()[last], -1)], 1)
    edges = np.concatenate([np.stack([n0, n1], 1), uniq]).astype(np.int32)
    edge_face = np.full(len(edges), -1)  # a matching: each triangle into one of its edges (one exists: 3 edges a
    if nf:  # triangle, at most 2 triangles an edge)
        inc = sp.csr_matrix((np.ones(3 * nf), (np.repeat(np.arange(nf), 3), inv)), shape=(nf, len(uniq)))
        match = maximum_bipartite_matching(inc, perm_type="column")
        edge_face[len(n0) + match[match >= 0]] = np.flatnonzero(match >= 0)
    edge_on = np.concatenate([on, np.ones(len(uniq), dtype=bool)])
    a, b = edges[edge_on, 0], edges[edge_on, 1]
    src, dst = np.concatenate([a, b]), np.concatenate([b, a])
    m = Mesh()
    for name, value, dtype in (
        ("edges", edges, wp.vec2i),
        ("edge_on", edge_on.astype(np.int32), wp.int32),
        ("edge_opp", np.concatenate([np.full((len(n0), 2), -1), opp]), wp.vec2i),
        ("edge_face", edge_face, wp.int32),
        ("faces", tris, wp.vec3i),
        ("face_edge", len(n0) + inv.reshape(-1, 3), wp.vec3i),
        ("adj_ptr", np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))]), wp.int32),
        ("adj", dst[np.argsort(src, kind="stable")], wp.int32),
        ("radius", model.particle_radius.numpy() if n else np.zeros(1), float),
        ("mu", mu, float),
        ("rod", rod, wp.int32),
        ("arc", arc, float),
        ("rest", model.particle_q.numpy().ravel() if n else np.zeros(3), float),
        ("rod_len", rod_len + [0.0], float),
        ("rod_end", rod_end + [(0, 0)], wp.vec2i),
        ("edge_rod", np.concatenate([edge_rod, np.full(len(uniq), -1)]), wp.int32),
        ("edge_arc", np.concatenate([edge_arc, np.zeros(len(uniq))]), float),
        ("edge_dir", np.concatenate([edge_dir, np.ones(len(uniq))]), float),
        ("world", model.particle_world.numpy() if n else np.zeros(1), wp.int32),
    ):
        value = np.asarray(value)
        setattr(m, name, wp.array(value if dtype is float else value.astype(np.int32), dtype=dtype, device=device))
    m.shift = wp.array(world_shift(model), dtype=wp.vec3, device=device)
    ns_attr = getattr(der, "surface_id", None) if der is not None else None
    nu_attr = getattr(der, "surface_u", None) if der is not None else None
    ss_attr = getattr(der, "shape_surface_id", None) if der is not None else None
    su_attr = getattr(der, "shape_surface_u", None) if der is not None else None
    node_surf = ns_attr.numpy().astype(np.int32) if ns_attr is not None and n else np.zeros(max(n, 1), np.int32)
    node_u = nu_attr.numpy().astype(np.float32) if nu_attr is not None and n else np.zeros((max(n, 1), 3), np.float32)
    sc = model.shape_count
    shape_surf = ss_attr.numpy().astype(np.int32) if ss_attr is not None and sc else np.zeros(max(sc, 1), np.int32)
    shape_u = (su_attr.numpy().astype(np.float32) if su_attr is not None and sc
               else np.zeros((max(sc, 1), 3), np.float32))
    if n and sc:
        e0_np, e1_np, eb_np = n0, n1, edge_body
        mask_eb = eb_np >= 0
        if mask_eb.any():
            sb_np = model.shape_body.numpy()
            for e_idx in np.flatnonzero(mask_eb):
                sids = np.flatnonzero(sb_np == eb_np[e_idx])
                if sids.size:
                    sid0 = int(sids[0])
                    if ns_attr is not None:
                        node_surf[e0_np[e_idx]] = node_surf[e1_np[e_idx]] = int(shape_surf[sid0])
                    if nu_attr is not None:
                        node_u[e0_np[e_idx]] = node_u[e1_np[e_idx]] = shape_u[sid0]
    node_face = np.full(max(n, 1), -1, dtype=np.int32)
    if nf:
        for fi in range(nf):
            for k_f in range(3):
                vi = int(tris[fi, k_f])
                if node_face[vi] < 0:
                    node_face[vi] = fi
    m.node_surf = wp.array(node_surf, dtype=wp.int32, device=device)
    m.node_u = wp.array(node_u, dtype=wp.vec3, device=device)
    m.node_face = wp.array(node_face, dtype=wp.int32, device=device)
    m.shape_surf = wp.array(shape_surf, dtype=wp.int32, device=device)
    m.shape_u = wp.array(shape_u, dtype=wp.vec3, device=device)
    return m


@wp.struct
class Bodies:
    """Newton's shapes as nodes and rod edges meet them: per shape its geometry (type, scale, mesh, mesh properties),
    its frame in its body (-1: the world's), margin, ``mu`` and bounding radius; the shapes that take part (colliding
    with particles, not a rod's capsule); the bodies' poses, velocities and centres of mass at the step's start; the
    step ``h``; ``sweep``: per body, how much further a yielding one's pairs reach: its move since the step's start (an
    iterate's detection sees it at its start pose) plus the move cap it may still take before the next detection."""

    kind: wp.array[wp.int32]
    scale: wp.array[wp.vec3]
    source: wp.array[wp.uint64]
    props: wp.array[wp.int32]
    sdf: wp.array[wp.int32]  # a mesh's in ``sdfs`` (Newton's texture SDF, Mesh.build_sdf), -1: its BVH
    sdfs: wp.array[TextureSDFData]
    xform: wp.array[wp.transform]
    body: wp.array[wp.int32]
    margin: wp.array[float]
    mu: wp.array[float]
    kh: wp.array[float]  # a hydroelastic pad's (Newton's is_hydroelastic) pressure per depth [N/m^3], 0 off pads
    hc: wp.array[float]  # and its Hunt-Crossley dissipation [s/m]
    radius: wp.array[float]
    shapes: wp.array[wp.int32]
    world_ptr: wp.array[wp.int32]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_com: wp.array[wp.vec3]
    sweep: wp.array[float]
    h: float


def _bodies(model: Model, device) -> Bodies:
    b = Bodies()
    body, kind = model.shape_body.numpy(), model.shape_type.numpy()
    on = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
    edge_body = model.dismech.edge_body.numpy()
    on &= ~np.isin(body, edge_body[edge_body >= 0])  # not a rod's capsule
    on &= kind != int(GeoType.HFIELD)  # not supported
    props = getattr(model, "_shape_mesh_properties", None)
    b.kind, b.scale, b.source = model.shape_type, model.shape_scale, model.shape_source_ptr
    b.xform, b.body = model.shape_transform, model.shape_body
    b.margin, b.mu = model.shape_margin, model.shape_material_mu
    # hydroelastic pads: Newton's flag and kh; Hunt-Crossley's d (Newton has none: ``model.shape_pad_dissipation``)
    pad = (model.shape_flags.numpy() & int(ShapeFlags.HYDROELASTIC)) != 0
    kh = model.shape_material_kh.numpy() if pad.any() else np.zeros(len(kind))
    hc = getattr(model, "shape_pad_dissipation", None)
    hc = np.zeros(len(kind)) if hc is None else np.asarray(hc, dtype=float)
    b.kh = wp.array(np.where(pad, kh, 0.0), dtype=float, device=device)
    b.hc = wp.array(np.where(pad, hc, 0.0), dtype=float, device=device)
    # the cull sphere about the shape's origin; Newton's mesh radius is about its AABB's centre: a mesh's from its
    # vertices (unculled without them)
    radius = model.shape_collision_radius.numpy().copy()
    sources, scale = getattr(model, "shape_source", None) or [None] * len(kind), model.shape_scale.numpy()
    for i in np.flatnonzero(on & ((kind == int(GeoType.MESH)) | (kind == int(GeoType.CONVEX_MESH)))):
        v = getattr(sources[i], "vertices", None)
        radius[i] = np.linalg.norm(np.asarray(v) * scale[i], axis=1).max() if v is not None and len(v) else 1.0e30
    b.radius = wp.array(radius, dtype=float, device=device)
    b.props = props if props is not None else wp.zeros(len(kind), dtype=wp.int32, device=device)
    index, tables = getattr(model, "_shape_sdf_index", None), getattr(model, "_texture_sdf_data", None)
    index = index.numpy() if index is not None and tables is not None else np.full(len(kind), -1)
    b.sdf = wp.array(np.where(kind == int(GeoType.MESH), index, -1), dtype=wp.int32, device=device)
    b.sdfs = tables if tables is not None else wp.array([create_empty_texture_sdf_data()], dtype=TextureSDFData,
                                                        device=device)
    world = model.shape_world.numpy()
    ids = np.flatnonzero(on)
    ids = ids[np.argsort(world[ids], kind="stable")]  # the global shapes (-1), then world by world
    b.shapes = wp.array(ids, dtype=wp.int32, device=device)
    W = max(model.world_count, 1)
    b.world_ptr = wp.array(np.concatenate([[0], np.cumsum(np.bincount(world[ids] + 1, minlength=W + 1))]),
                           dtype=wp.int32, device=device)
    b.body_q = model.body_q if model.body_count else wp.zeros(1, dtype=wp.transform, device=device)
    b.body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=device)
    b.body_com = model.body_com if model.body_count else wp.zeros(1, dtype=wp.vec3, device=device)
    b.sweep = wp.zeros(max(model.body_count, 1), dtype=float, device=device)
    b.h = 0.0
    return b


@wp.func
def shape_pose(b: Bodies, k: int, poses: wp.array[wp.transform]) -> wp.transform:
    """Shape ``k``'s frame in the world, its body at ``poses`` (``b.body_q``: the step's start)."""
    if b.body[k] < 0:
        return b.xform[k]
    return wp.transform_multiply(poses[b.body[k]], b.xform[k])


_shape_pose = shape_pose
