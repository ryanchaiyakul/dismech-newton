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
from .contact import Slots, clear, contact_point, emit, engaged, pair_fixed
from .state import node

SLOTS = 8  # contacts per feature (node or edge): an edge on a sheet crosses several of its edges
EXCLUSION = 1.5  # rod-rod pairs closer than this many thicknesses along one rod are not contacts
OPEN = 1.0e-4  # an edge-edge closest point this close to an end (fraction of the edge) is the node's: blocks are open
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
    """Rods and shells against each other and against shapes: :data:`SLOTS` per node and per edge, keyed by
    the other feature (node ``w``, edge ``num_nodes + f``, triangle ``num_nodes + num_edges + t``, or shape
    ``num_nodes + num_edges + num_triangles + k``). Two BVHs, of the edges and of the triangles, hold their boxes over
    the step, grown by a contact's reach, and are refit each step (graph-capturable); shapes are culled by their
    bounding spheres. Without ``self_contact``, no rod-rod or shell-shell pairs."""

    def __init__(self, model: Model, fixed: wp.array, *, engage_gap: float, self_contact: bool,
                 friction: float | None = None):
        dev = wp.get_device(model.device)
        self.mesh = mesh = _mesh(model, dev)
        self.fixed, self.self_contact = fixed, int(self_contact)
        self.num_nodes, self.num_edges, self.num_faces = model.particle_count, mesh.edges.shape[0], mesh.faces.shape[0]
        self.shape_key0 = self.num_nodes + self.num_edges + self.num_faces  # shape k's contact key: shape_key0 + k
        self.num_rod_edges = model.dismech.edge_node0.shape[0]
        self.pad = 2.0 * float(mesh.radius.numpy().max()) * (1.0 + engage_gap)
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
        if friction is not None:  # every contact's mu: each side's
            mesh.mu.fill_(friction)
            self.bodies.mu = wp.full(self.bodies.mu.shape[0], friction, dtype=float, device=dev)
        sc = self.scene = Scene()  # the step's positions and slots set in fill
        sc.m, sc.b, sc.fixed, sc.engage, sc.self_contact = mesh, self.bodies, fixed, engage_gap, self.self_contact
        sc.num_nodes = self.num_nodes
        sc.overflow = self.overflow = wp.zeros(1, dtype=wp.int32, device=dev)  # contacts beyond a feature's slots

    def slots(self) -> int:
        """None where no pair can touch (recording a step for its adjoint then reads no count)."""
        on = self.mesh.edge_on.numpy()
        rods, shells = bool(on[: self.num_rod_edges].any()), self.num_faces > 0
        if (rods or shells) and (self.self_contact or self.bodies.shapes.shape[0] or (rods and shells)):
            return (self.num_nodes + self.num_edges) * SLOTS
        return 0

    def fill(self, state_in: State, dt: float, q: wp.array, s: Slots, prev: Slots):
        """The contacts for the step to the prediction ``q``, warm-started from ``prev``."""
        sc, b, dev = self.scene, self.bodies, self.fixed.device
        sc.q, sc.q1, sc.s, sc.prev = state_in.dismech.q, q, s, prev
        wp.launch(_bounds_kernel, dim=self.num_edges + self.num_faces,
                  inputs=[sc.q, q, self._elems, self.pad, self._lo, self._hi], device=dev)
        self._bvh.refit()
        faces = wp.uint64(0)
        if self._face_bvh is not None:
            self._face_bvh.refit()
            faces = self._face_bvh.id
        b.body_q, b.h = state_in.body_q if state_in.body_q is not None else b.body_q, dt
        b.body_qd = state_in.body_qd if state_in.body_qd is not None else b.body_qd
        sc.b = b
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
def _in_node_region(q: wp.array[float], m: Mesh, v: int, x: wp.vec3) -> bool:
    """``x`` in node ``v``'s block: not past ``v`` toward any neighbour (closed; Eqs. 8, 15)."""
    for k in range(m.adj_ptr[v], m.adj_ptr[v + 1]):
        if _ahead(x, node(q, v), node(q, m.adj[k])) > 0.0:
            return False
    return True


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
    if den <= 1.0e-12 * a * e:
        return wp.vec2(-1.0)
    return wp.vec2(b * wp.dot(d2, r) - e * wp.dot(d1, r), a * wp.dot(d2, r) - b * wp.dot(d1, r)) / den


@wp.struct
class Scene:
    """A step's collision: positions at its start ``q`` and its prediction ``q1``, what collides, the shapes, the fixed
    DOFs, the slots ``s`` filled (warm-started from ``prev``), the overflow count; ``engage``: the engage gap."""

    q: wp.array[float]
    q1: wp.array[float]
    m: Mesh
    b: Bodies
    fixed: wp.array[wp.int32]
    s: Slots
    prev: Slots
    overflow: wp.array[wp.int32]
    engage: float
    self_contact: int
    num_nodes: int


@wp.func
def _keep(sc: Scene, base: int, key: int, pair: wp.vec4i, w: wp.vec4, n: wp.vec3, xb: wp.vec3, db: wp.vec3, th: float,
          mu: float):
    """Emit the contact into a free slot of the feature's from ``base``, unless it is there already (found through
    another triangle or edge) or :func:`~.contact.engaged` drops it (a shell's with its motion), as
    :class:`~.contact.ContactBlock` would. ``n``: the normal, or zero for ``C q``'s direction; ``xb``, ``db``: a
    body's point at the end of the step and its motion over it. Rod-rod pairs are warm-started by ``key`` (a light
    shell vertex's warm-started impulse lingers and pumps energy); shell-shell points closer than :data:`EXCLUSION`
    thicknesses at rest are not contacts (the rods' rule is :func:`_excluded`)."""
    s, prev, m = sc.s, sc.prev, sc.m
    d = contact_point(sc.q, pair, w, xb - db)
    if wp.length(n) == 0.0:
        if wp.length(d) < 1.0e-9:
            return
        n = wp.normalize(d)
    if pair_fixed(sc.fixed, pair):
        return
    shells = True  # all shell nodes
    rods = True  # all rod nodes
    for k in range(4):
        if pair[k] >= 0:
            shells = shells and m.rod[pair[k]] < 0
            rods = rods and m.rod[pair[k]] >= 0
    if shells and pair[1] >= 0 and wp.length(contact_point(m.rest, pair, w, wp.vec3(0.0))) < EXCLUSION * th:
        return
    warm = int(-1)
    for k in range(SLOTS):
        if rods and prev.key[base + k] == key:
            warm = base + k
    if (warm < 0 or prev.rho[warm] * wp.length(prev.u[warm]) == 0.0) and not engaged(
        sc.q, sc.q1, pair, w, n, xb, th, sc.engage, not rods
    ):
        return
    slot = int(-1)
    for k in range(SLOTS):
        if s.key[base + k] == key:
            return
        if slot < 0 and s.key[base + k] < 0:
            slot = base + k
    if slot < 0:
        wp.atomic_add(sc.overflow, 0, 1)
        return
    s.key[slot] = key
    emit(s, slot, sc.q, pair, w, n, xb, db, th, mu)
    if warm >= 0:
        s.u[slot] = prev.u[warm]
        s.rho[slot] = prev.rho[warm]


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
    q0: wp.array[float],
    q1: wp.array[float],
    elems: wp.array[wp.vec3i],
    pad: float,
    lo: wp.array[wp.vec3],
    hi: wp.array[wp.vec3],
):
    """An edge's or triangle's box over the step (start ``q0``, prediction ``q1``), grown by ``pad``."""
    i = wp.tid()
    a = wp.vec3(1.0e30)
    b = wp.vec3(-1.0e30)
    for j in range(3):
        v = elems[i][j]
        a = wp.min(a, wp.min(node(q0, v), node(q1, v)))
        b = wp.max(b, wp.max(node(q0, v), node(q1, v)))
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
        reach = (1.0 + sc.engage) * th + wp.max(wp.length(node(sc.q1, a0) - xa), wp.length(node(sc.q1, a1) - xb))
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
    qlo = wp.min(xv, node(sc.q1, i))
    qhi = wp.max(xv, node(sc.q1, i))
    query = wp.bvh_query_aabb(bvh, qlo, qhi)
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
            if not _in_node_region(q, m, i, wp.lerp(node(q, b0), node(q, b1), t)):
                continue  # not a local minimum along this rod
            if _excluded(q, m, i, i, 0.0, c0, c1, t, EXCLUSION * th):
                continue
        _keep_pair(sc, base, other, pair, w, th, 0.5 * (m.mu[i] + m.mu[c0]))
    for j in range(b.shapes.shape[0]):  # against shapes: their surfaces offset outward (Sec. 3.9)
        k = b.shapes[j]
        X = _shape_pose(b, k)
        th = m.radius[i] + b.margin[k]
        reach = (1.0 + sc.engage) * th + wp.length(node(sc.q1, i) - xv)
        if wp.length(xv - wp.transform_get_translation(X)) > b.radius[k] + reach:
            continue
        d, n = _shape_gap(b, k, X, xv, reach)
        if d <= reach and not (on_rod and not _in_node_region(q, m, i, xv - n)):  # else a rod edge holds it
            _keep_shape(sc, base, k, wp.vec4i(i, -1, -1, -1), wp.vec4(1.0, 0.0, 0.0, 0.0), xv - d * n, n, th)
    if m.faces.shape[0] == 0:
        return
    query = wp.bvh_query_aabb(face_bvh, qlo, qhi)
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
            if not _in_edge_region(q, m, key - sc.num_nodes, xv):
                continue
        _keep_pair(sc, base, key, pair, w, th, 0.5 * (m.mu[i] + m.mu[tri[0]]))


@wp.kernel
def _offset_kernel(bvh: wp.uint64, face_bvh: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], sc: Scene):
    """Thread ``i < num_nodes``: node ``i``'s contacts, else edge ``i - num_nodes``'s, into its :data:`SLOTS` from
    ``i * SLOTS``. Query: the node's box over the step, or the edge's own (the trees' boxes carry the pad). A rod's
    point must also be a local minimum along its rod (in its own node's block); a shell vertex's need not
    (vertex-facet is one-sided)."""
    i = wp.tid()
    base = i * SLOTS
    for k in range(SLOTS):
        clear(sc.s, base + k)
    if i >= sc.num_nodes:
        e = i - sc.num_nodes
        if sc.m.edge_on[e] != 0:
            _edge_contacts(sc, bvh, lo[e], hi[e], e, base)
    elif sc.m.adj_ptr[i + 1] > sc.m.adj_ptr[i]:  # on a colliding rod or a shell
        _node_contacts(sc, bvh, face_bvh, i, base)
