"""Where contacts come from: :class:`Offset` fills the :class:`~.contact.Slots` each step.

Rods and shells against each other and against Newton's (non-rod) shapes, by Offset Geometric Contact (Chen et al.
2025). A shell is a triangle mesh (feature blocks: Eqs. 8-10), a rod an edge-only manifold (Eqs. 14-15): a node's ball
cut by the planes perpendicular to its edges (closed); an edge's slab (open), on a shell also cut by the planes through
it perpendicular to its faces; a triangle's prism. A contact is a feature pair whose blocks hold each other's closest
points, so each is found once, with no tolerances: a node against a triangle's closest feature (vertex-facet), a node
against a rod's (node-node kept by the lower node), an edge against an edge (interiors, kept by the higher edge). A
node and an edge decide their shared boundary with the same :func:`_ahead`. A body's surface is offset outward only
(Sec. 3.9): a node meets it at its closest point, the gap signed (negative inside), so one depth holds inside and out;
a rod's node only if its edges lead away from the body, else the edge meets it at its interior's closest point.
"""

import numpy as np
import warp as wp
from newton import GeoType, Model, ShapeFlags, State
from newton._src.geometry import kernels as G  # its shape SDFs; triangle_...: exact zeros off the closest feature

from .builder import _P
from .contact import Slots, clear, contact_point, emit, pair_fixed
from .state import node

SLOTS = 8  # contacts per feature (node or edge) on rods
SHEET_SLOTS = 64  # with sheets: an edge on a sheet crosses several of its edges, and compressed layers stack
REACH = 2.0  # pairs within this many thicknesses: the thickness, plus two move caps (a radius each)
EXCLUSION = 1.5  # rod-rod pairs closer than this many thicknesses along one rod are not contacts
OPEN = 1.0e-4  # an edge-edge closest point this close to an end (fraction of the edge) is the node's: blocks are open
PARALLEL = 1.0e-3  # edges with sin^2 of their angle below this: no edge-edge pair (their nodes' pairs hold it, IPC)
BISECT = 20  # bisections for a rod edge's closest point to a shape


@wp.struct
class Mesh:
    """What collides: per edge (the rods', then the shells') its nodes, whether it collides and (a shell's) its faces'
    opposite nodes (-1); per triangle its nodes and edges (edge ``k`` opposite node ``k``); per node its neighbours
    along colliding edges (CSR), radius, ``mu``, rod (-1 off rods), arc length and rest position (flat, the model's
    initial state); per rod its length if closed, else 0."""

    edges: wp.array[wp.vec2i]
    edge_on: wp.array[wp.int32]
    edge_opp: wp.array[wp.vec2i]
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


def _mesh(model: Model, device) -> Mesh:
    der = model.dismech
    n0, n1, length, edge_body = (
        getattr(der, k).numpy() for k in ("edge_node0", "edge_node1", "edge_length", "edge_body")
    )
    nb, n = max(model.body_count, 1), model.particle_count
    shape_body = model.shape_body.numpy()
    own = (shape_body >= 0) & ((model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0)
    body_mu, body_on = np.zeros(nb), np.zeros(nb, dtype=bool)
    body_mu[shape_body[own]] = model.shape_material_mu.numpy()[own]
    body_on[shape_body[own]] = True
    on = (edge_body >= 0) & body_on[edge_body]
    mu = np.full(n, float(model.particle_mu))
    mu[n0[on]] = mu[n1[on]] = body_mu[edge_body[on]]
    rod, arc, rod_len = np.full(n, -1, dtype=np.int32), np.zeros(n), []
    nxt = dict(zip(n0.tolist(), zip(n1.tolist(), length.tolist())))
    for start in sorted(set(nxt) - set(n1.tolist())) + sorted(nxt):  # open rods' first nodes, then loops
        s, a = start, 0.0
        while rod[s] < 0:
            rod[s], arc[s] = len(rod_len), a
            if s not in nxt:
                break
            s, step = nxt[s]
            a += step
        if rod[start] == len(rod_len):
            rod_len.append(a if s == start else 0.0)
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
    edge_on = np.concatenate([on, np.ones(len(uniq), dtype=bool)])
    a, b = edges[edge_on, 0], edges[edge_on, 1]
    src, dst = np.concatenate([a, b]), np.concatenate([b, a])
    m = Mesh()
    for name, value, dtype in (
        ("edges", edges, wp.vec2i),
        ("edge_on", edge_on.astype(np.int32), wp.int32),
        ("edge_opp", np.concatenate([np.full((len(n0), 2), -1), opp]), wp.vec2i),
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
    ):
        value = np.asarray(value)
        setattr(m, name, wp.array(value if dtype is float else value.astype(np.int32), dtype=dtype, device=device))
    return m


class Offset:
    """Rods and shells against each other and against shapes: ``slots`` (:data:`SLOTS`) per node and per edge, keyed by
    the other feature (node ``w``, edge ``num_nodes + f``, triangle ``num_nodes + num_edges + t``, or shape
    ``num_nodes + num_edges + num_triangles + k``). Two BVHs, of the edges and of the triangles, hold their boxes at a
    detection, grown by a contact's reach, and are refit each detection (graph-capturable); shapes are culled by their
    bounding spheres. Without ``self_contact``, no rod-rod or shell-shell pairs."""

    def __init__(self, model: Model, fixed: wp.array, *, self_contact: bool):
        dev = wp.get_device(model.device)
        self.mesh = mesh = _mesh(model, dev)
        self.fixed, self.self_contact = fixed, int(self_contact)
        self.num_nodes, self.num_edges, self.num_faces = model.particle_count, mesh.edges.shape[0], mesh.faces.shape[0]
        self.shape_key0 = self.num_nodes + self.num_edges + self.num_faces  # shape k's contact key: shape_key0 + k
        self.num_rod_edges = model.dismech.edge_node0.shape[0]
        self.pad = 2.0 * float(mesh.radius.numpy().max()) * REACH
        e, x = mesh.edges.numpy(), model.particle_q.numpy()
        elems = np.concatenate([e[:, [0, 1, 1]], mesh.faces.numpy()])  # the edges, then the triangles
        self._elems = wp.array(elems, dtype=wp.vec3i, device=dev)
        self._lo, self._hi = (
            wp.array(f(x[elems], axis=1) + sign * self.pad, dtype=wp.vec3, device=dev)
            for f, sign in ((np.min, -1.0), (np.max, 1.0))
        )
        E = self.num_edges
        self._bvh = wp.Bvh(self._lo[:E], self._hi[:E])
        self._face_bvh = wp.Bvh(self._lo[E:], self._hi[E:]) if self.num_faces else None
        self.bodies = _bodies(model, dev)
        sc = self.scene = Scene()  # the step's positions and slots set in fill
        sc.m, sc.b, sc.fixed, sc.self_contact = mesh, self.bodies, fixed, self.self_contact
        sc.num_nodes, sc.slots = self.num_nodes, SHEET_SLOTS if self.num_faces else SLOTS
        sc.overflow = self.overflow = wp.zeros(1, dtype=wp.int32, device=dev)  # contacts beyond a feature's slots
        # With a move cap (the barrier's): each node's, lowered so a pair dropped by a full feature cannot close
        sc.limit = self.limit = wp.zeros(self.num_nodes, dtype=float, device=dev)
        self.cap = 0.0  # set by the solver

    def slots(self) -> int:
        """None where no pair can touch (recording a step for its adjoint then reads no count)."""
        on = self.mesh.edge_on.numpy()
        rods, shells = bool(on[: self.num_rod_edges].any()), self.num_faces > 0
        if (rods or shells) and (self.self_contact or self.bodies.shapes.shape[0] or (rods and shells)):
            return (self.num_nodes + self.num_edges) * self.scene.slots
        return 0

    def fill(self, state_in: State, dt: float, q: wp.array, s: Slots):
        """The contacts near ``q`` (the step's start or an iterate), shapes moving over the step ``dt``."""
        sc, b, dev = self.scene, self.bodies, self.fixed.device
        sc.q, sc.s = q, s
        wp.launch(_bounds_kernel, dim=self.num_edges + self.num_faces,
                  inputs=[q, self._elems, self.pad, self._lo, self._hi], device=dev)
        self._bvh.refit()
        faces = wp.uint64(0)
        if self._face_bvh is not None:
            self._face_bvh.refit()
            faces = self._face_bvh.id
        b.body_q, b.h = state_in.body_q if state_in.body_q is not None else b.body_q, dt
        b.body_qd = state_in.body_qd if state_in.body_qd is not None else b.body_qd
        sc.b = b
        self.limit.fill_(self.cap)
        wp.launch(_offset_kernel, dim=self.num_nodes + self.num_edges,
                  inputs=[self._bvh.id, faces, self._lo, self._hi, sc], device=dev)


@wp.struct
class Bodies:
    """Newton's shapes as nodes and rod edges meet them: per shape its geometry (type, scale, mesh, mesh properties),
    its frame in its body (-1: the world's), margin, ``mu`` and bounding radius; the shapes that take part (colliding
    with particles, not a rod's capsule); the bodies' poses, velocities and centres of mass at the step's start; the
    step ``h``."""

    kind: wp.array[wp.int32]
    scale: wp.array[wp.vec3]
    source: wp.array[wp.uint64]
    props: wp.array[wp.int32]
    xform: wp.array[wp.transform]
    body: wp.array[wp.int32]
    margin: wp.array[float]
    mu: wp.array[float]
    radius: wp.array[float]
    shapes: wp.array[wp.int32]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_com: wp.array[wp.vec3]
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
    b.margin, b.mu, b.radius = model.shape_margin, model.shape_material_mu, model.shape_collision_radius
    b.props = props if props is not None else wp.zeros(len(kind), dtype=wp.int32, device=device)
    b.shapes = wp.array(np.flatnonzero(on), dtype=wp.int32, device=device)
    b.body_q = model.body_q if model.body_count else wp.zeros(1, dtype=wp.transform, device=device)
    b.body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=device)
    b.body_com = model.body_com if model.body_count else wp.zeros(1, dtype=wp.vec3, device=device)
    b.h = 0.0
    return b


@wp.func
def _shape_distance(kind: int, scale: wp.vec3, mesh: wp.uint64, props: int, x: wp.vec3, reach: float):
    """The signed distance from a shape's surface to its local point ``x`` (negative inside) and the outward
    normal there; ``1e6`` with no surface within ``reach`` (meshes)."""
    d = float(1.0e6)
    n = wp.vec3(0.0, 0.0, 1.0)
    if kind == GeoType.SPHERE:
        d = G.sdf_sphere(x, scale[0])
        n = G.sdf_sphere_grad(x, scale[0])
    elif kind == GeoType.BOX:
        d = G.sdf_box(x, scale[0], scale[1], scale[2])
        n = G.sdf_box_grad(x, scale[0], scale[1], scale[2])
    elif kind == GeoType.CAPSULE:
        d = G.sdf_capsule(x, scale[0], scale[1], 2)
        n = G.sdf_capsule_grad(x, scale[0], scale[1], 2)
    elif kind == GeoType.CYLINDER:
        d = G.sdf_cylinder(x, scale[0], scale[1], 2, -1.0, scale[2])
        n = G.sdf_cylinder_grad(x, scale[0], scale[1], 2, -1.0, scale[2])
    elif kind == GeoType.CONE:
        d = G.sdf_cone(x, scale[0], scale[1], 2)
        n = G.sdf_cone_grad(x, scale[0], scale[1], 2)
    elif kind == GeoType.ELLIPSOID:
        d = G.sdf_ellipsoid(x, scale)
        n = G.sdf_ellipsoid_grad(x, scale)
    elif kind == GeoType.PLANE:
        d = G.sdf_plane(x, scale[0] * 0.5, scale[1] * 0.5)
    elif kind == GeoType.MESH or kind == GeoType.CONVEX_MESH:
        s = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
        r = G.mesh_query_point_sign(mesh, wp.cw_div(x, scale), reach / s, G.resolve_mesh_sign_method(props))
        if r.result:
            v = x - wp.cw_mul(wp.mesh_eval_position(mesh, r.face, r.u, r.v), scale)
            d = wp.length(v) * r.sign
            n = wp.normalize(v) * r.sign
    return d, n


@wp.func
def _shape_pose(b: Bodies, k: int) -> wp.transform:
    """Shape ``k``'s frame in the world at the step's start."""
    if b.body[k] < 0:
        return b.xform[k]
    return wp.transform_multiply(b.body_q[b.body[k]], b.xform[k])


@wp.func
def _shape_gap(b: Bodies, k: int, X: wp.transform, x: wp.vec3, reach: float):
    """:func:`_shape_distance` at the world point ``x``, the normal in the world."""
    xl = wp.transform_point(wp.transform_inverse(X), x)
    d, n = _shape_distance(b.kind[k], b.scale[k], b.source[k], b.props[k], xl, reach)
    return d, wp.transform_vector(X, n)


@wp.func
def _anchor(b: Bodies, k: int, x: wp.vec3):
    """Shape ``k``'s world point ``x`` (at the step's start) at the end of the step, and its displacement over it."""
    j = b.body[k]
    if j < 0:
        return x, wp.vec3(0.0)
    dx = wp.vec3(0.0)
    if b.body_qd.shape[0] > 0:
        v = b.body_qd[j]
        dx = b.h * (
            wp.spatial_top(v) + wp.cross(wp.spatial_bottom(v), x - wp.transform_point(b.body_q[j], b.body_com[j]))
        )
    return x + dx, dx


@wp.func
def _ahead(x: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    """``(x - a) . (b - a)``: positive iff ``x`` lies past ``a`` toward ``b``."""
    return wp.dot(x - a, b - a)


@wp.func
def _in_node_region(q: wp.array[float], m: Mesh, v: int, x: wp.vec3, tol: float = 0.0) -> bool:
    """``x`` in node ``v``'s block: not past ``v`` toward any neighbour (closed; Eqs. 8, 15), or not by more than
    ``tol`` of the edge to it."""
    for k in range(m.adj_ptr[v], m.adj_ptr[v + 1]):
        nb = node(q, m.adj[k])
        if _ahead(x, node(q, v), nb) > tol * wp.length_sq(nb - node(q, v)):
            return False
    return True


@wp.func
def _edge_holds(q: wp.array[float], m: Mesh, b: Bodies, k: int, X: wp.transform, i: int, n: wp.vec3, far: float) -> bool:
    """Rod node ``i``'s pair with shape ``k`` (normal ``n`` at ``i``) is a rod edge's: the shape's point lies past ``i``
    toward a neighbour, and that edge's closest point to the shape is inside it (:func:`_edge_contacts`'s test), so
    the edge holds the pair. Else the node does, a local minimum along its rod or not."""
    xv = node(q, i)
    for a in range(m.adj_ptr[i], m.adj_ptr[i + 1]):
        xn = node(q, m.adj[a])
        if _ahead(xv - n, xv, xn) > 0.0:
            d1, n1 = _shape_gap(b, k, X, xn, far + wp.length(xn - xv))
            if _ahead(xn - n1, xn, xv) > 0.0:
                return True
    return False


@wp.func
def _in_edge_region(q: wp.array[float], m: Mesh, e: int, x: wp.vec3) -> bool:
    """``x`` in shell edge ``e``'s block: inside its slab (open) and, for each face, not on the face's side of the
    plane through ``e`` perpendicular to it (closed; Eq. 9)."""
    a = node(q, m.edges[e][0])
    b = node(q, m.edges[e][1])
    if _ahead(x, a, b) <= 0.0 or _ahead(x, b, a) <= 0.0:
        return False
    for k in range(2):
        o = m.edge_opp[e][k]
        if o >= 0:
            p = a + _ahead(node(q, o), a, b) / wp.length_sq(b - a) * (b - a)
            if wp.dot(x - p, p - node(q, o)) < 0.0:
                return False
    return True


@wp.func
def _excluded(
    q: wp.array[float], m: Mesh, a0: int, a1: int, sa: float, b0: int, b1: int, sb: float, cutoff: float
) -> bool:
    """Both points on one rod, closer than ``cutoff`` along it (around a loop either way)."""
    r = m.rod[a0]
    if r < 0 or m.rod[b0] != r:
        return False
    d = wp.abs(
        m.arc[a0] + sa * wp.length(node(q, a1) - node(q, a0)) - m.arc[b0] - sb * wp.length(node(q, b1) - node(q, b0))
    )
    if m.rod_len[r] > 0.0:
        d = wp.min(d, m.rod_len[r] - d)
    return d < cutoff


@wp.func
def _closest_st(q: wp.array[float], a0: int, a1: int, b0: int, b1: int) -> wp.vec2:
    """The closest points of the lines through edges ``a`` and ``b``; ``(-1, -1)`` if near parallel (a node's
    pair then, as is any point outside the edges)."""
    d1 = node(q, a1) - node(q, a0)
    d2 = node(q, b1) - node(q, b0)
    r = node(q, a0) - node(q, b0)
    a = wp.dot(d1, d1)
    b = wp.dot(d1, d2)
    e = wp.dot(d2, d2)
    den = a * e - b * b
    if den <= PARALLEL * a * e:
        return wp.vec2(-1.0)
    return wp.vec2(b * wp.dot(d2, r) - e * wp.dot(d1, r), a * wp.dot(d2, r) - b * wp.dot(d1, r)) / den


@wp.struct
class Scene:
    """A detection: the positions ``q``, what collides, the shapes, the fixed DOFs, the slots ``s`` filled, the
    overflow count and the nodes' move limits."""

    q: wp.array[float]
    m: Mesh
    b: Bodies
    fixed: wp.array[wp.int32]
    s: Slots
    overflow: wp.array[wp.int32]
    limit: wp.array[float]
    self_contact: int
    num_nodes: int
    slots: int


@wp.func
def _drop(sc: Scene, pair: wp.vec4i, gap: float):
    """A pair no slot holds, ``gap`` apart along its normal: its nodes move at most 0.45 of it (each side's weights
    sum to 1, so it closes by at most 0.9 of it) until the next detection."""
    for k in range(4):
        if pair[k] >= 0:
            wp.atomic_min(sc.limit, pair[k], wp.max(0.45 * gap, 0.0))


@wp.func
def _keep(sc: Scene, base: int, key: int, pair: wp.vec4i, w: wp.vec4, n: wp.vec3, xb: wp.vec3, db: wp.vec3, th: float,
          mu: float):
    """Emit the contact into a free slot of the feature's from ``base``, unless it is there already (found through
    another triangle or edge) or farther than :data:`REACH` thicknesses; a full feature keeps its closest contacts.
    ``n``: the normal, or zero for ``C q``'s direction; ``xb``, ``db``: a body's point at the end of the step and its
    motion over it. A shell-shell pair's thickness is at most half its rest distance, so none touches at rest (the
    rods' rule is :func:`_excluded`)."""
    s, m = sc.s, sc.m
    d = contact_point(sc.q, pair, w, xb - db)
    if wp.length(n) == 0.0:
        if wp.length(d) < 1.0e-9:
            return
        n = wp.normalize(d)
    if pair_fixed(sc.fixed, pair):
        return
    shells = True  # all shell nodes
    for k in range(4):
        if pair[k] >= 0:
            shells = shells and m.rod[pair[k]] < 0
    if wp.dot(n, contact_point(sc.q, pair, w, xb)) > REACH * th:
        return
    if shells and (pair[1] >= 0 or pair[2] >= 0):  # no pair touches at rest
        th = wp.min(th, 0.5 * wp.length(contact_point(m.rest, pair, w, wp.vec3(0.0))))
    slot = int(-1)
    far = int(-1)  # with every slot taken, the farthest contact (by gap at the start), if farther than this one
    gap = wp.dot(n, d) - th
    for k in range(sc.slots):  # filled from base on (an evicted slot is refilled at once): stop at the first free
        j = base + k
        if s.key[j] == key:
            return
        if s.key[j] < 0:
            slot = j
            break
        if wp.dot(s.normal[j], s.c0[j]) - s.thickness[j] > gap:
            if far < 0 or wp.dot(s.normal[j], s.c0[j]) - s.thickness[j] > wp.dot(s.normal[far], s.c0[far]) - s.thickness[far]:
                far = j
    if slot < 0:
        wp.atomic_add(sc.overflow, 0, 1)
        if far < 0:
            _drop(sc, pair, wp.dot(n, d))
            return
        slot = far
        _drop(sc, s.pair[far], wp.dot(s.normal[far], s.c0[far]))
        clear(s, slot)
    s.key[slot] = key
    emit(s, slot, sc.q, pair, w, n, xb, db, th, mu)


@wp.func
def _keep_pair(sc: Scene, base: int, key: int, pair: wp.vec4i, w: wp.vec4, th: float, mu: float):
    """:func:`_keep` for two features of rods and shells: the normal along ``C q``, no anchor."""
    _keep(sc, base, key, pair, w, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), th, mu)


@wp.func
def _keep_shape(sc: Scene, base: int, k: int, pair: wp.vec4i, w: wp.vec4, x: wp.vec3, n: wp.vec3, th: float):
    """:func:`_keep` against shape ``k``, at its world point ``x`` (the step's start) with the normal ``n``."""
    xb, db = _anchor(sc.b, k, x)
    key = sc.num_nodes + sc.m.edges.shape[0] + sc.m.faces.shape[0] + k
    _keep(sc, base, key, pair, w, n, xb, db, th, 0.5 * (sc.m.mu[pair[0]] + sc.b.mu[k]))


@wp.kernel
def _bounds_kernel(
    q: wp.array[float],
    elems: wp.array[wp.vec3i],
    pad: float,
    lo: wp.array[wp.vec3],
    hi: wp.array[wp.vec3],
):
    """An edge's or triangle's box at ``q``, grown by ``pad``."""
    i = wp.tid()
    a = wp.vec3(1.0e30)
    b = wp.vec3(-1.0e30)
    for j in range(3):
        v = elems[i][j]
        a = wp.min(a, node(q, v))
        b = wp.max(b, node(q, v))
    lo[i] = a - wp.vec3(pad)
    hi[i] = b + wp.vec3(pad)


@wp.func
def _edge_contacts(sc: Scene, bvh: wp.uint64, lo: wp.vec3, hi: wp.vec3, e: int, base: int):
    """Edge ``e``'s contacts: a rod's interior against shapes, and against higher edges' interiors."""
    m, b, q = sc.m, sc.b, sc.q
    a0 = m.edges[e][0]
    a1 = m.edges[e][1]
    xa = node(q, a0)
    xb = node(q, a1)
    for j in range(b.shapes.shape[0] if m.rod[a0] >= 0 else 0):
        k = b.shapes[j]
        X = _shape_pose(b, k)
        th = 0.5 * (m.radius[a0] + m.radius[a1]) + b.margin[k]
        reach = REACH * th
        ya, da = _anchor(b, k, xa)
        yb, db = _anchor(b, k, xb)
        reach += wp.max(wp.length(da), wp.length(db))  # the body's motion over the step
        c = wp.transform_get_translation(X)
        t = wp.clamp(_ahead(c, xa, xb) / wp.length_sq(xb - xa), 0.0, 1.0)
        if wp.length(c - wp.lerp(xa, xb, t)) > b.radius[k] + reach:
            continue
        far = reach + wp.length(xb - xa)  # a mesh's query radius
        d, n = _shape_gap(b, k, X, xa, far)
        d1, n1 = _shape_gap(b, k, X, xb, far)
        if _ahead(xa - n, xa, xb) <= 0.0 or _ahead(xb - n1, xb, xa) <= 0.0:
            continue  # an end is the closest: its node's pair
        t0 = float(0.0)
        t1 = float(1.0)
        for it in range(BISECT):  # the gap's slope along the edge changes sign once on a convex shape
            d, n = _shape_gap(b, k, X, wp.lerp(xa, xb, 0.5 * (t0 + t1)), far)
            if wp.dot(n, xb - xa) < 0.0:
                t0 = 0.5 * (t0 + t1)
            else:
                t1 = 0.5 * (t0 + t1)
        t = 0.5 * (t0 + t1)
        d, n = _shape_gap(b, k, X, wp.lerp(xa, xb, t), far)
        if d <= reach:
            _keep_shape(
                sc, base, k, wp.vec4i(a0, a1, -1, -1), wp.vec4(1.0 - t, t, 0.0, 0.0), wp.lerp(xa, xb, t) - d * n, n, th
            )
    query = wp.bvh_query_aabb(bvh, lo, hi)
    f = int(0)
    while wp.bvh_query_next(query, f):
        b0 = m.edges[f][0]
        b1 = m.edges[f][1]
        if f > e or m.edge_on[f] == 0 or b0 == a0 or b0 == a1 or b1 == a0 or b1 == a1:
            continue  # the higher edge's pair (a shell's, not a long rod edge's); or neighbours
        if (m.rod[a0] >= 0) == (m.rod[b0] >= 0) and sc.self_contact == 0:
            continue
        st = _closest_st(q, a0, a1, b0, b1)
        if wp.min(st) <= OPEN or wp.max(st) >= 1.0 - OPEN:
            continue  # an end: a node's pair (found exactly on a node, rounding would duplicate it)
        th = 0.5 * (m.radius[a0] + m.radius[a1] + m.radius[b0] + m.radius[b1])
        if not _excluded(q, m, a0, a1, st[0], b0, b1, st[1], EXCLUSION * th):
            _keep_pair(sc, base, sc.num_nodes + f, wp.vec4i(a0, a1, b0, b1),
                       wp.vec4(1.0 - st[0], st[0], st[1] - 1.0, -st[1]), th, 0.5 * (m.mu[a0] + m.mu[b0]))


@wp.func
def _node_contacts(sc: Scene, bvh: wp.uint64, face_bvh: wp.uint64, i: int, base: int):
    """Node ``i``'s contacts: against rods (node-node, node-edge), shapes and triangles."""
    m, b, q = sc.m, sc.b, sc.q
    xv = node(q, i)
    on_rod = m.rod[i] >= 0
    query = wp.bvh_query_aabb(bvh, xv, xv)
    f = int(0)
    while wp.bvh_query_next(query, f):  # against rods
        b0 = m.edges[f][0]
        b1 = m.edges[f][1]
        if m.edge_on[f] == 0 or m.rod[b0] < 0 or b0 == i or b1 == i or (on_rod and sc.self_contact == 0):
            continue
        # The other feature: node b0 or b1 (closed blocks) or f's interior (open).
        other = sc.num_nodes + f
        t = float(0.0)
        if _ahead(xv, node(q, b0), node(q, b1)) <= 0.0:
            other = b0
        elif _ahead(xv, node(q, b1), node(q, b0)) <= 0.0:
            other = b1
            t = 1.0
        else:
            t = _ahead(xv, node(q, b0), node(q, b1)) / wp.length_sq(node(q, b1) - node(q, b0))
        pair = wp.vec4i(i, b0, b1, -1)
        w = wp.vec4(1.0, t - 1.0, -t, 0.0)
        th = m.radius[i] + 0.5 * (m.radius[b0] + m.radius[b1])
        c0 = b0
        c1 = b1
        if other < sc.num_nodes:
            if (on_rod and other < i) or not _in_node_region(q, m, other, xv):
                continue  # the lower node's pair; or xv lies past it into its other edge
            pair = wp.vec4i(i, other, -1, -1)
            w = wp.vec4(1.0, -1.0, 0.0, 0.0)
            th = m.radius[i] + m.radius[other]
            c0 = other
            c1 = other
        if on_rod:
            # Not a local minimum along this rod; within OPEN of it, the edge-edge pair is left to it (else a node
            # just over a still rod, rounded past itself, is neither's).
            if not _in_node_region(q, m, i, wp.lerp(node(q, b0), node(q, b1), t), OPEN):
                continue
            if _excluded(q, m, i, i, 0.0, c0, c1, t, EXCLUSION * th):
                continue
        _keep_pair(sc, base, other, pair, w, th, 0.5 * (m.mu[i] + m.mu[c0]))
    for j in range(b.shapes.shape[0]):  # against shapes: their surfaces offset outward (Sec. 3.9)
        k = b.shapes[j]
        X = _shape_pose(b, k)
        th = m.radius[i] + b.margin[k]
        y, dy = _anchor(b, k, xv)
        reach = REACH * th + wp.length(dy)  # and the body's motion
        if wp.length(xv - wp.transform_get_translation(X)) > b.radius[k] + reach:
            continue
        d, n = _shape_gap(b, k, X, xv, reach)
        if d <= reach and not (on_rod and _edge_holds(q, m, b, k, X, i, n, reach)):
            _keep_shape(sc, base, k, wp.vec4i(i, -1, -1, -1), wp.vec4(1.0, 0.0, 0.0, 0.0), xv - d * n, n, th)
    if m.faces.shape[0] == 0:
        return
    query = wp.bvh_query_aabb(face_bvh, xv, xv)
    while wp.bvh_query_next(query, f):  # against triangles: their closest feature (Algorithm 1)
        tri = m.faces[f]
        if tri[0] == i or tri[1] == i or tri[2] == i or (not on_rod and sc.self_contact == 0):
            continue
        bary = G.triangle_closest_point_barycentric(node(q, tri[0]), node(q, tri[1]), node(q, tri[2]), xv)
        pair = wp.vec4i(i, -1, -1, -1)
        w = wp.vec4(1.0, 0.0, 0.0, 0.0)
        th = m.radius[i]
        count = int(0)
        corner = int(0)
        side = int(0)
        for k in range(3):
            if bary[k] > 0.0:
                count += 1
                corner = k
                pair[k + 1] = tri[k]
                w[k + 1] = -bary[k]
                th += bary[k] * m.radius[tri[k]]
            else:
                side = k
        key = sc.num_nodes + m.edges.shape[0] + f
        if count == 1:
            key = tri[corner]
            if not _in_node_region(q, m, key, xv):
                continue
        elif count == 2:
            key = sc.num_nodes + m.face_edge[f][side]
            e = key - sc.num_nodes
            own = m.edge_opp[e][0] == i or m.edge_opp[e][1] == i
            if not _in_edge_region(q, m, e, xv) and not own:
                continue  # (and its own triangle's opposite edge: that triangle's height)
        _keep_pair(sc, base, key, pair, w, th, 0.5 * (m.mu[i] + m.mu[tri[0]]))


@wp.kernel
def _offset_kernel(bvh: wp.uint64, face_bvh: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], sc: Scene):
    """Thread ``i < num_nodes``: node ``i``'s contacts, else edge ``i - num_nodes``'s, into its ``slots`` from
    ``i * slots``. Query: the node's point, or the edge's box (the trees' boxes carry the pad). A rod node's pair
    with another rod must lie in its own node's block; with a shape, unless a rod edge holds it (:func:`_edge_holds`);
    a shell vertex's need not (vertex-facet is one-sided)."""
    i = wp.tid()
    base = i * sc.slots
    for k in range(sc.slots):
        clear(sc.s, base + k)
    if i >= sc.num_nodes:
        e = i - sc.num_nodes
        if sc.m.edge_on[e] != 0:
            _edge_contacts(sc, bvh, lo[e], hi[e], e, base)
    elif sc.m.adj_ptr[i + 1] > sc.m.adj_ptr[i]:  # on a colliding rod or a shell
        _node_contacts(sc, bvh, face_bvh, i, base)
