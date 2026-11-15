"""Where contacts come from: :class:`Offset` appends them to the :class:`~.contact.Slots` list at each detection.

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
from newton import GeoType, Model, State
from newton._src.geometry import kernels as G  # its shape SDFs; triangle_...: exact zeros off the closest feature
from newton._src.geometry.sdf_texture import texture_sample_sdf_grad

from ..constants import (
    BISECT,
    EXCLUSION,
    FACE_PASSES,
    FLAT,
    FWD_MODULE_OPTS,
    INSIDE,
    NEARER,
    OPEN,
    PAD_DEPTH,
    PAIRS,
    PARALLEL,
    REACH,
    SIDES,
)
from ..dofs import node
from .barrier import _barrier_force
from .slots import (
    Arm,
    Bodies,
    Mesh,
    Slots,
    _bodies,
    _mesh,
    _mu_shape,
    clear,
    contact_point,
    emit,
    pair_fixed,
    pair_point,
    shape_of,
    shape_pose,
)

wp.set_module_options(FWD_MODULE_OPTS)


class Offset:
    """Rods and shells against each other and against shapes, into one list of :data:`PAIRS` places per node and edge,
    each pair keyed by its feature (node ``v`` or edge ``num_nodes + e``, ``own``) and the other (node ``w``, edge
    ``num_nodes + f``, triangle ``num_nodes + num_edges + t``, or shape ``num_nodes + num_edges + num_triangles + 2 k +
    side``). Two BVHs, of the edges and of the triangles, hold their boxes at a detection, grown by a contact's reach,
    and are refit each detection (graph-capturable); shapes are culled by their bounding spheres. Without
    ``self_contact``, no rod-rod or shell-shell pairs. ``world``: per DOF its world (a pair's: its feature's)."""

    def __init__(self, model: Model, fixed: wp.array, *, self_contact: bool, world: np.ndarray):
        dev = wp.get_device(model.device)
        self.mesh = mesh = _mesh(model, dev)
        self.model = model
        self._mu_shape = wp.array(_mu_shape(model)[0], dtype=wp.int32, device=dev)
        self.fixed, self.self_contact = fixed, int(self_contact)
        self.num_nodes, self.num_edges, self.num_faces = model.particle_count, mesh.edges.shape[0], mesh.faces.shape[0]
        self.shape_key0 = self.num_nodes + self.num_edges + self.num_faces  # shape k's: shape_key0 + SIDES k + side
        self.num_rod_edges = model.dismech.edge_node0.shape[0]
        self.pad = 2.0 * float(mesh.radius.numpy().max()) * REACH
        e, x = mesh.edges.numpy(), model.particle_q.numpy()
        elems = np.concatenate([e[:, [0, 1, 1]], mesh.faces.numpy()])  # the edges, then the triangles
        self._elems = wp.array(elems, dtype=wp.vec3i, device=dev)
        shift = mesh.shift.numpy()[elems[:, 0]] if len(elems) else 0.0
        self._lo, self._hi = (
            wp.array(f(x[elems], axis=1) + sign * self.pad + shift, dtype=wp.vec3, device=dev)
            for f, sign in ((np.min, -1.0), (np.max, 1.0))
        )
        E = self.num_edges
        self._bvh = wp.Bvh(self._lo[:E], self._hi[:E])
        self._face_bvh = wp.Bvh(self._lo[E:], self._hi[E:]) if self.num_faces else None
        self.bodies = _bodies(model, dev)
        self.num_prims = self.shape_key0 + SIDES * max(model.shape_count, 1)  # a pair key's range
        sc = self.scene = Scene()  # the step's positions and list set in fill
        sc.m, sc.b, sc.fixed, sc.self_contact = mesh, self.bodies, fixed, self.self_contact
        sc.num_nodes = self.num_nodes
        sc.world = wp.array(world[3 * np.arange(self.num_nodes)] if self.num_nodes else np.zeros(1), dtype=wp.int32,
                            device=dev)
        sc.count = self.count = wp.zeros(1, dtype=wp.int32, device=dev)  # pairs appended (past the list: dropped)
        sc.overflow = self.overflow = wp.zeros(1, dtype=wp.int32, device=dev)  # pairs dropped, the list full
        # With a move cap (the barrier's): each node's, lowered so a pair dropped by a full feature cannot close
        sc.limit = self.limit = wp.zeros(self.num_nodes, dtype=float, device=dev)
        ptr = self.bodies.world_ptr.numpy()  # the most shapes a feature meets: the global ones and a world's
        f = Found()  # (a struct field's read is a copy: filled, then set)
        f.width = int(ptr[1] + np.diff(ptr[1:]).max(initial=0))
        size = max(2 * f.width * (self.num_nodes + self.num_edges), 1)
        for name, var in Found.vars.items():
            if name != "width":
                setattr(f, name, wp.zeros(size, dtype=var.type.dtype, device=dev))
        f.shape.fill_(-1)
        sc.found = f
        self.cap = 0.0  # set by the solver

    def refresh_mu(self) -> None:
        """Re-read the nodes' friction coefficients (their capsules' ``shape_material_mu``, ``particle_mu``)."""
        m = self.model
        if self.num_nodes:
            shape_mu = m.shape_material_mu if m.shape_count else wp.zeros(1, dtype=float, device=self.fixed.device)
            wp.launch(_mu_kernel, dim=self.num_nodes, inputs=[self._mu_shape, shape_mu, float(m.particle_mu)],
                      outputs=[self.mesh.mu], device=self.fixed.device)

    def capacity(self) -> int:
        """The list's size, 0 where no pair can touch (recording a step for its adjoint then reads no count)."""
        on = self.mesh.edge_on.numpy()
        rods, shells = bool(on[: self.num_rod_edges].any()), self.num_faces > 0
        if (rods or shells) and (self.self_contact or self.bodies.shapes.shape[0] or (rods and shells)):
            return (self.num_nodes + self.num_edges) * PAIRS
        return 0

    def fill(self, state_in: State, dt: float, q: wp.array, s: Slots):
        """The contacts near ``q`` (the step's start or an iterate), shapes moving over the step ``dt``."""
        sc, b, dev = self.scene, self.bodies, self.fixed.device
        sc.q, sc.s = q, s
        wp.launch(_clear_kernel, dim=s.active.shape[0], inputs=[s], device=dev)
        self.count.zero_()
        wp.launch(_bounds_kernel, dim=self.num_edges + self.num_faces,
                  inputs=[q, self._elems, self.mesh.shift, self.pad, self._lo, self._hi], device=dev)
        self._bvh.refit()
        faces = wp.uint64(0)
        if self._face_bvh is not None:
            self._face_bvh.refit()
            faces = self._face_bvh.id
        b.body_q, b.h = state_in.body_q if state_in.body_q is not None else b.body_q, dt
        b.body_qd = state_in.body_qd if state_in.body_qd is not None else b.body_qd
        sc.b = b
        self.limit.fill_(self.cap)
        if sc.found.width:
            wp.launch(_shapes_kernel, dim=(self.num_nodes + self.num_edges, sc.found.width), inputs=[sc], device=dev)
        wp.launch(_offset_kernel, dim=self.num_nodes + self.num_edges,
                  inputs=[self._bvh.id, faces, self._lo, self._hi, sc], device=dev)

    def reactions(self, q: wp.array, s: Slots, arm: Arm, eps: float, body_f: wp.array) -> None:
        """Add to ``body_f`` each shape pair's force at ``q`` on its body (equal and opposite to the nodes'), with its
        torque about the body's centre of mass at the step's end: world frame, Newton's ``body_f`` layout."""
        wp.launch(_reaction_kernel, dim=s.active.shape[0],
                  inputs=[q, s, arm, eps, self.shape_key0, self.bodies, body_f], device=self.fixed.device)


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
def _shape_count(b: Bodies, world: int) -> int:
    """How many shapes a feature of ``world`` meets: the global ones and its world's (a global feature: all)."""
    if world < 0:
        return b.shapes.shape[0]
    return b.world_ptr[1] + b.world_ptr[world + 2] - b.world_ptr[world + 1]


@wp.func
def _shape_at(b: Bodies, world: int, j: int) -> int:
    """The ``j``-th of them (:func:`_shape_count`)."""
    if world < 0 or j < b.world_ptr[1]:
        return b.shapes[j]
    return b.shapes[b.world_ptr[world + 1] + j - b.world_ptr[1]]


@wp.func
def _other_world(m: Mesh, a: int, b: int) -> bool:
    wa = m.world[a]
    wb = m.world[b]
    return wa >= 0 and wb >= 0 and wa != wb


@wp.func
def _shape_gap(b: Bodies, k: int, X: wp.transform, x: wp.vec3, reach: float):
    """:func:`_shape_distance` at the world point ``x``, the normal in the world."""
    xl = wp.transform_point(wp.transform_inverse(X), x)
    d, n = _shape_distance(b.kind[k], b.scale[k], b.source[k], b.props[k], xl, reach)
    return d, wp.transform_vector(X, n)


@wp.func
def _gap(b: Bodies, k: int, X: wp.transform, x: wp.vec3, reach: float):
    """:func:`_shape_gap`, a mesh with an SDF (Newton's texture SDF, ``Mesh.build_sdf``) by its trilinear value and
    gradient: detection's (the adjoint keeps the mesh's exact surface: its backward of a texture lookup compiles for
    hours)."""
    j = b.sdf[k]
    if j >= 0 and b.sdfs[j].coarse_texture.width > 0:
        sdf = b.sdfs[j]
        xl = wp.transform_point(wp.transform_inverse(X), x)
        s = wp.where(sdf.scale_baked, 1.0, wp.min(wp.min(wp.abs(b.scale[k][0]), wp.abs(b.scale[k][1])),
                                                  wp.abs(b.scale[k][2])))
        d, g = texture_sample_sdf_grad(sdf, wp.where(sdf.scale_baked, xl, wp.cw_div(xl, b.scale[k])))
        return d * s, wp.transform_vector(X, wp.normalize(g))
    return _shape_gap(b, k, X, x, reach)


@wp.func
def _sweep(b: Bodies, k: int) -> float:
    """How far shape ``k``'s body has moved in the step so far (a yielding body at an iterate), else 0."""
    j = b.body[k]
    return wp.where(j >= 0, b.sweep[j], 0.0)


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


@wp.kernel
def _reaction_kernel(q: wp.array[float], s: Slots, arm: Arm, eps: float, key0: int, b: Bodies,
                     body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    if s.active[i] == 0 or s.key[i] < key0:
        return
    j = b.body[shape_of(s.key[i], key0)]
    if j < 0:
        return
    g, h, fr = _barrier_force(pair_point(q, s, i, arm, s.anchor[i]), s, i, eps)
    com = wp.transform_point(b.body_q[j], b.body_com[j])
    if b.body_qd.shape[0] > 0:
        com += b.h * wp.spatial_top(b.body_qd[j])
    wp.atomic_add(body_f, j, wp.spatial_vector(g, wp.cross(s.anchor[i] - com, g)))


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
def _edge_holds(q: wp.array[float], m: Mesh, b: Bodies, k: int, X: wp.transform, i: int, n: wp.vec3,
                far: float) -> bool:
    """Rod node ``i``'s pair with shape ``k`` (normal ``n`` at ``i``) is a rod edge's: the shape's point lies past ``i``
    toward a neighbour, and that edge's contact (:func:`_edge_scan`, as :func:`_edge_contacts` finds it) is inside it
    and does not reach ``i``. Else the node does, a local minimum along its rod or not."""
    xv = node(q, i)
    for a in range(m.adj_ptr[i], m.adj_ptr[i + 1]):
        xn = node(q, m.adj[a])
        if _ahead(xv - n, xv, xn) > 0.0:
            r = _edge_scan(b, k, X, xv, xn, far + wp.length(xn - xv), 2 * BISECT + 1)  # (its far end unused)
            if r[1] > OPEN and r[0] < 1.0 - OPEN:
                return True
    return False


@wp.func
def _edge_scan(b: Bodies, k: int, X: wp.transform, xa: wp.vec3, xb: wp.vec3, far: float,
               passes: int = 3 * BISECT + 1) -> wp.vec3:
    """The edge ``xa xb``'s closest point ``t`` to shape ``k`` (its gap's slope changes sign once on a convex shape; an
    end's within ``2^-BISECT``), and how far either way from it the shape's normal stays ``t``'s: lying along a face,
    the edge touches it on that segment (on a curved surface, ``t`` alone). One gap per pass, so one inlined copy;
    ``2 BISECT + 1`` passes leave the far end at ``t``."""
    t0 = float(0.0)
    t1 = float(1.0)
    t = float(0.0)
    lo = float(0.0)
    nt = wp.vec3(0.0)
    for it in range(passes):
        u = wp.where(it == BISECT, t, 0.5 * (t0 + t1))
        d, n = _gap(b, k, X, wp.lerp(xa, xb, u), far)
        if it < BISECT:
            if wp.dot(n, xb - xa) < 0.0:
                t0 = u
            else:
                t1 = u
            t = 0.5 * (t0 + t1)
        elif it == BISECT:  # then from t toward 0, toward 1: t0 on the face, t1 off it
            nt = n
            t0 = t
            t1 = 0.0
        else:
            if wp.dot(n, nt) >= 1.0 - FLAT:
                t0 = u
            else:
                t1 = u
            if it == 2 * BISECT:
                lo = t0
                t0 = t
                t1 = 1.0
    return wp.vec3(t, lo, t0)


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
def _along(q: wp.array[float], m: Mesh, e: int, i: int, s: float):
    """Rod edge ``e``'s point ``s`` from its first node (``e < 0``: node ``i``): its branch and arc."""
    if e < 0:
        return m.rod[i], m.arc[i]
    x = node(q, m.edges[e][1]) - node(q, m.edges[e][0])
    return m.edge_rod[e], m.edge_arc[e] + m.edge_dir[e] * s * wp.length(x)


@wp.func
def _excluded(
    q: wp.array[float], m: Mesh, ea: int, ia: int, sa: float, eb: int, ib: int, sb: float, cutoff: float
) -> bool:
    """Two rod points (:func:`_along`'s) closer than ``cutoff`` along their rods: on one branch (around a loop either
    way) or through a node two branches end at."""
    ra, a = _along(q, m, ea, ia, sa)
    rb, b = _along(q, m, eb, ib, sb)
    if ra < 0 or rb < 0:
        return False
    d = float(1.0e30)
    if ra == rb:
        d = wp.abs(a - b)
        if m.rod_end[ra][0] == m.rod_end[ra][1]:
            d = wp.min(d, m.rod_len[ra] - d)
    else:
        for i in range(2):
            for j in range(2):
                if m.rod_end[ra][i] == m.rod_end[rb][j]:
                    d = wp.min(d, wp.where(i == 0, a, m.rod_len[ra] - a) + wp.where(j == 0, b, m.rod_len[rb] - b))
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
class Found:
    """A feature's contacts with shapes, found in parallel per (feature, shape) and kept in order after
    (:func:`_shapes_kernel`): per feature ``width`` shape places of 2 sides; ``shape`` -1 where none."""

    shape: wp.array[wp.int32]
    pair: wp.array[wp.vec4i]
    w: wp.array[wp.vec4]
    x: wp.array[wp.vec3]
    n: wp.array[wp.vec3]
    th: wp.array[float]
    width: int


@wp.struct
class Scene:
    """A detection: the positions ``q``, what collides, the shapes, the fixed DOFs, the list ``s`` appended to (its
    ``count``), the overflow count and the nodes' move limits; per node its world; ``found``: the features' shape
    contacts."""

    q: wp.array[float]
    m: Mesh
    b: Bodies
    fixed: wp.array[wp.int32]
    s: Slots
    count: wp.array[wp.int32]
    overflow: wp.array[wp.int32]
    limit: wp.array[float]
    world: wp.array[wp.int32]
    self_contact: int
    num_nodes: int
    found: Found


@wp.func
def _found(sc: Scene, r: int, k: int, pair: wp.vec4i, w: wp.vec4, x: wp.vec3, n: wp.vec3, th: float):
    f = sc.found
    f.shape[r] = k
    f.pair[r] = pair
    f.w[r] = w
    f.x[r] = x
    f.n[r] = n
    f.th[r] = th


@wp.func
def _keep_found(sc: Scene, i: int):
    """Feature ``i``'s shape contacts (:class:`Found`), kept in shape order, side 0 then 1."""
    f = sc.found
    for r in range(2 * f.width * i, 2 * f.width * (i + 1)):
        if f.shape[r] >= 0:
            _keep_shape(sc, i, f.shape[r], r % 2, f.pair[r], f.w[r], f.x[r], f.n[r], f.th[r])


@wp.func
def _drop(sc: Scene, pair: wp.vec4i, gap: float):
    """A pair the full list drops, ``gap`` apart along its normal: its nodes move at most 0.45 of it (each side's
    weights sum to 1, so it closes by at most 0.9 of it) until the next detection."""
    for k in range(4):
        if pair[k] >= 0:
            wp.atomic_min(sc.limit, pair[k], wp.max(0.45 * gap, 0.0))


@wp.kernel
def _clear_kernel(s: Slots):
    clear(s, wp.tid())


@wp.func
def _keep(sc: Scene, own: int, key: int, pair: wp.vec4i, w: wp.vec4, n: wp.vec3, xb: wp.vec3, db: wp.vec3, th: float,
          mu: float, sweep: float, pad: wp.vec4):
    """Append the contact of feature ``own`` with ``key`` to the list, unless farther than :data:`REACH` thicknesses
    (found twice, through two triangles or edges: the sort drops one); past the list's end, dropped (:func:`_drop`).
    ``n``: the normal, or zero for ``C q``'s direction; ``xb``, ``db``: a body's point at the end of the step and its
    motion over it; ``sweep``: how much further a yielding body's pair reaches (:attr:`Bodies.sweep`). A shell-shell
    pair's thickness is at most half its rest distance, so none touches at rest (the rods' rule is
    :func:`_excluded`)."""
    s, m = sc.s, sc.m
    d = contact_point(sc.q, pair, w, xb - db)
    if wp.length(n) == 0.0:
        if wp.length(d) < 1.0e-9:
            return
        n = wp.normalize(d)
    if pair_fixed(sc.fixed, pair) and sweep == 0.0:  # nothing moves (a yielding body's sweep is at least its cap)
        return
    shells = True  # all shell nodes
    for k in range(4):
        if pair[k] >= 0:
            shells = shells and m.rod[pair[k]] < 0
    if wp.dot(n, contact_point(sc.q, pair, w, xb)) > REACH * th + sweep:
        return
    if shells and (pair[1] >= 0 or pair[2] >= 0):  # no pair touches at rest
        th = wp.min(th, 0.5 * wp.length(contact_point(m.rest, pair, w, wp.vec3(0.0))))
    slot = wp.atomic_add(sc.count, 0, 1)
    if slot >= s.active.shape[0]:
        wp.atomic_add(sc.overflow, 0, 1)
        _drop(sc, pair, wp.dot(n, d))
        return
    s.own[slot] = own
    s.key[slot] = key
    s.world[slot] = sc.world[wp.where(own < sc.num_nodes, own, m.edges[wp.max(own - sc.num_nodes, 0)][0])]
    radius = wp.vec4(0.0)
    for k in range(4):
        if pair[k] >= 0 and m.rod[pair[k]] >= 0:
            radius[k] = m.radius[pair[k]]
    emit(s, slot, sc.q, pair, w, n, xb, db, th, mu, radius, pad)


@wp.func
def _keep_pair(sc: Scene, own: int, key: int, pair: wp.vec4i, w: wp.vec4, th: float, mu: float):
    """:func:`_keep` for two features of rods and shells: the normal along ``C q``, no anchor."""
    _keep(sc, own, key, pair, w, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), th, mu, 0.0, wp.vec4(0.0))


@wp.func
def _keep_shape(sc: Scene, own: int, k: int, side: int, pair: wp.vec4i, w: wp.vec4, x: wp.vec3, n: wp.vec3,
                th: float, c: float = 0.0, power: float = 0.0):
    """:func:`_keep` against shape ``k``, at its world point ``x`` (the step's start) with the normal ``n``; ``side``:
    which end of a rod edge's line contact (0 for a single point), or a pad's point; against a pad, its pressure's
    ``c delta^power`` (:attr:`.slots.Slots.pad`; 0: a node's, its core's barrier only)."""
    xb, db = _anchor(sc.b, k, x)
    key = sc.num_nodes + sc.m.edges.shape[0] + sc.m.faces.shape[0] + SIDES * k + side
    pad = wp.vec4(0.0)
    if sc.b.kh[k] > 0.0:
        pad = wp.vec4(c, power, PAD_DEPTH * th, sc.b.hc[k])
    _keep(sc, own, key, pair, w, n, xb, db, th, 0.5 * (sc.m.mu[pair[0]] + sc.b.mu[k]), _sweep(sc.b, k), pad)


@wp.func
def _pad_points(sc: Scene, own: int, k: int, X: wp.transform, tri: wp.vec3i, th: float, reach: float, far: float):
    """A rod edge's (``tri[2] < 0``) or a triangle's points against pad ``k``: its pressure field (Elandt et al. 2019,
    linear: ``kh`` per depth) over the partner, at fixed points (so the energy is continuous: a point enters at zero
    pressure) about a radius apart: a rod edge's ``q`` points along it, each its length's share of Winkler's cylinder
    across the section (``kh l (8/15) sqrt(2R) delta^(5/2)``, Johnson 1985); a triangle's ``n^2`` sub-triangles'
    centroids, each its area's share (``kh A delta^2 / 2``)."""
    m, b, q = sc.m, sc.b, sc.q
    kh = b.kh[k]
    r = m.radius[tri[0]]
    if tri[2] < 0:
        le = wp.length(node(m.rest, tri[1]) - node(m.rest, tri[0]))
        nq = wp.clamp(int(wp.ceil(le / r)), 1, SIDES)
        c = kh * le / float(nq) * (8.0 / 15.0) * wp.sqrt(2.0 * r)
        for j in range(nq):
            u = (float(j) + 0.5) / float(nq)
            x = wp.lerp(node(q, tri[0]), node(q, tri[1]), u)
            d, n = _gap(b, k, X, x, far)
            if d <= reach:
                _keep_shape(sc, own, k, j, wp.vec4i(tri[0], tri[1], -1, -1), wp.vec4(1.0 - u, u, 0.0, 0.0), x - d * n,
                            n, th, c, 2.5)
        return
    r0 = node(m.rest, tri[0])
    e1 = node(m.rest, tri[1]) - r0
    e2 = node(m.rest, tri[2]) - r0
    size = wp.max(wp.length(e1), wp.max(wp.length(e2), wp.length(e2 - e1)))
    nq = wp.clamp(int(wp.ceil(size / r)), 1, 4)  # n^2 <= SIDES
    c = 0.5 * kh * 0.5 * wp.length(wp.cross(e1, e2)) / float(nq * nq)
    for j in range(nq * nq):  # the sub-triangles: cell (i, l) under the diagonal its up one, (i + 1/3, l + 1/3) / n;
        i = j // nq  # past it the down one of its mirror (n - 1 - i, n - 1 - l), (i + 2/3, l + 2/3) / n
        l = j % nq
        up = i + l < nq
        o = wp.where(up, 1.0 / 3.0, 2.0 / 3.0)
        u = (float(wp.where(up, i, nq - 1 - i)) + o) / float(nq)
        v = (float(wp.where(up, l, nq - 1 - l)) + o) / float(nq)
        wv = wp.vec4(1.0 - u - v, u, v, 0.0)
        x = wv[0] * node(q, tri[0]) + u * node(q, tri[1]) + v * node(q, tri[2])
        d, n = _gap(b, k, X, x, far)
        if d <= reach:
            _keep_shape(sc, own, k, j, wp.vec4i(tri[0], tri[1], tri[2], -1), wv, x - d * n, n, th, c, 2.0)


@wp.kernel
def _mu_kernel(mu_shape: wp.array[wp.int32], shape_mu: wp.array[float], particle_mu: float, mu: wp.array[float]):
    i = wp.tid()
    mu[i] = wp.where(mu_shape[i] >= 0, shape_mu[wp.max(mu_shape[i], 0)], particle_mu)


@wp.kernel
def _bounds_kernel(
    q: wp.array[float],
    elems: wp.array[wp.vec3i],
    shift: wp.array[wp.vec3],
    pad: float,
    lo: wp.array[wp.vec3],
    hi: wp.array[wp.vec3],
):
    """An edge's or triangle's box at ``q``, grown by ``pad``, in its world's place (:attr:`Mesh.shift`)."""
    i = wp.tid()
    a = wp.vec3(1.0e30)
    b = wp.vec3(-1.0e30)
    for j in range(3):
        v = elems[i][j]
        a = wp.min(a, node(q, v))
        b = wp.max(b, node(q, v))
    s = shift[elems[i][0]]
    lo[i] = a - wp.vec3(pad) + s
    hi[i] = b + wp.vec3(pad) + s


@wp.func
def _shell_shape(sc: Scene, e: int, jj: int, at: int):
    """Shell edge ``e``'s interior, and its triangle's (:attr:`Mesh.edge_face`), against its ``jj``-th shape (found at
    ``at``, ``at + 1``): the closest point
    to a shape inside the edge (bisection, as a rod edge's) or the triangle (alternating closest points), kept where it
    is closer than the feature's nodes (else theirs: a node's pair holds a flat face or a convex body's near point).
    A curved body or a box's edge meets a sheet between its nodes. One gap per pass (one inlined copy each)."""
    m, b, q = sc.m, sc.b, sc.q
    a0 = m.edges[e][0]
    a1 = m.edges[e][1]
    xa = node(q, a0)
    xb = node(q, a1)
    f = m.edge_face[e]
    tri = wp.vec3i(a0, a1, a0)
    if f >= 0:
        tri = m.faces[f]
    x0 = node(q, tri[0])
    x1 = node(q, tri[1])
    x2 = node(q, tri[2])
    if jj < _shape_count(b, m.world[a0]):
        k = _shape_at(b, m.world[a0], jj)
        X = shape_pose(b, k, b.body_q)
        c = wp.transform_get_translation(X)
        th = 0.5 * (m.radius[a0] + m.radius[a1]) + b.margin[k]
        reach = REACH * th + _sweep(b, k)
        size = wp.max(wp.length(x0 - x1), wp.max(wp.length(x1 - x2), wp.length(x2 - x0)))
        if wp.length(c - x0) > b.radius[k] + reach + size and wp.length(c - xa) > b.radius[k] + reach + size:
            return
        if b.kh[k] > 0.0:  # a pad: its triangle's points (the edge holding it), none along the edge
            if f >= 0:
                _pad_points(sc, sc.num_nodes + e, k, X, tri, th, reach + size, reach + size)
            return
        # the edge: BISECT bisections of the gap's slope, then its point and both ends
        t0 = float(0.0)
        t1 = float(1.0)
        d = float(0.0)
        n = wp.vec3(0.0)
        ends = float(1.0e30)
        for it in range(BISECT + 3):
            u = 0.5 * (t0 + t1)
            if it == BISECT + 1:
                u = 0.0
            elif it == BISECT + 2:
                u = 1.0
            dj, nj = _gap(b, k, X, wp.lerp(xa, xb, u), reach)
            if it < BISECT:
                if wp.dot(nj, xb - xa) < 0.0:
                    t0 = u
                else:
                    t1 = u
            elif it == BISECT:
                d = dj
                n = nj
            else:
                ends = wp.min(ends, dj)
        t = 0.5 * (t0 + t1)
        if t > OPEN and t < 1.0 - OPEN and d <= reach and d < ends - NEARER * th:
            x = wp.lerp(xa, xb, t)
            _found(sc, at, k, wp.vec4i(a0, a1, -1, -1), wp.vec4(1.0 - t, t, 0.0, 0.0), x - d * n, n, th)
        if f < 0:
            return
        # the triangle: FACE_PASSES alternations from its centroid, then its point and its three nodes
        thf = (m.radius[tri[0]] + m.radius[tri[1]] + m.radius[tri[2]]) / 3.0 + b.margin[k]
        r = wp.where(b.kind[k] == GeoType.SPHERE or b.kind[k] == GeoType.CAPSULE, b.scale[k][0], 0.0)
        w = wp.vec3(1.0 / 3.0)
        p = (x0 + x1 + x2) / 3.0
        d = float(0.0)
        n = wp.vec3(0.0)
        ends = float(1.0e30)
        for it in range(FACE_PASSES + 4):
            if it == FACE_PASSES + 1:
                p = x0
            elif it == FACE_PASSES + 2:
                p = x1
            elif it == FACE_PASSES + 3:
                p = x2
            dj, nj = _gap(b, k, X, p, reach)
            if it < FACE_PASSES:
                w = G.triangle_closest_point_barycentric(x0, x1, x2, p - (dj + r) * nj)
                p = w[0] * x0 + w[1] * x1 + w[2] * x2
            elif it == FACE_PASSES:
                d = dj
                n = nj
            else:
                ends = wp.min(ends, dj)
        if wp.min(w) > INSIDE and d <= REACH * thf + _sweep(b, k) and d < ends - NEARER * thf:
            pw = w[0] * x0 + w[1] * x1 + w[2] * x2
            _found(sc, at + 1, k, wp.vec4i(tri[0], tri[1], tri[2], -1), wp.vec4(w[0], w[1], w[2], 0.0), pw - d * n, n,
                   thf)


@wp.func
def _edge_shape(sc: Scene, e: int, jj: int, at: int):
    """Rod edge ``e``'s interior against its ``jj``-th shape (found at ``at``, ``at + 1``)."""
    m, b, q = sc.m, sc.b, sc.q
    a0 = m.edges[e][0]
    a1 = m.edges[e][1]
    xa = node(q, a0)
    xb = node(q, a1)
    if jj < _shape_count(b, m.world[a0]):
        k = _shape_at(b, m.world[a0], jj)
        X = shape_pose(b, k, b.body_q)
        th = 0.5 * (m.radius[a0] + m.radius[a1]) + b.margin[k]
        reach = REACH * th
        _ya, da = _anchor(b, k, xa)
        _yb, db = _anchor(b, k, xb)
        reach += wp.max(wp.length(da), wp.length(db)) + _sweep(b, k)  # the body's motion over the step
        c = wp.transform_get_translation(X)
        t = wp.clamp(_ahead(c, xa, xb) / wp.length_sq(xb - xa), 0.0, 1.0)
        if wp.length(c - wp.lerp(xa, xb, t)) > b.radius[k] + reach:
            return
        far = reach + wp.length(xb - xa)  # a mesh's query radius
        if b.kh[k] > 0.0:  # a pad: its points
            _pad_points(sc, sc.num_nodes + e, k, X, wp.vec3i(a0, a1, -1), th, reach, far)
            return
        # The closest point; lying along a face, the edge touches it on a segment, and the segment's other end too, so
        # it holds when the edge tilts either way (else the frozen point is the wrong end and the other sinks in). A
        # point at a node is the node's pair.
        r = _edge_scan(b, k, X, xa, xb, far)
        other = wp.where(r[0] - r[1] > r[2] - r[0], r[1], r[2])
        ends = wp.where(wp.abs(other - r[0]) > OPEN, 2, 1)  # (a runtime bound: one inlined gap)
        for side in range(ends):
            u = wp.where(side == 0, r[0], other)
            if u <= OPEN or u >= 1.0 - OPEN:
                continue
            x = wp.lerp(xa, xb, u)
            d, n = _gap(b, k, X, x, far)
            if d <= reach:
                _found(sc, at + side, k, wp.vec4i(a0, a1, -1, -1), wp.vec4(1.0 - u, u, 0.0, 0.0), x - d * n, n, th)


@wp.func
def _edge_contacts(sc: Scene, bvh: wp.uint64, lo: wp.vec3, hi: wp.vec3, e: int):
    """Edge ``e``'s contacts: a rod's interior against shapes (:func:`_edge_shape`), and against higher edges'
    interiors."""
    m, q = sc.m, sc.q
    a0 = m.edges[e][0]
    a1 = m.edges[e][1]
    if m.rod[a0] >= 0:
        _keep_found(sc, sc.num_nodes + e)
    query = wp.bvh_query_aabb(bvh, lo, hi)
    f = int(0)
    while wp.bvh_query_next(query, f):
        b0 = m.edges[f][0]
        b1 = m.edges[f][1]
        if f > e or m.edge_on[f] == 0 or b0 == a0 or b0 == a1 or b1 == a0 or b1 == a1 or _other_world(m, a0, b0):
            continue  # the higher edge's pair (a shell's, not a long rod edge's); or neighbours
        if (m.rod[a0] >= 0) == (m.rod[b0] >= 0) and sc.self_contact == 0:
            continue
        st = _closest_st(q, a0, a1, b0, b1)
        if wp.min(st) <= OPEN or wp.max(st) >= 1.0 - OPEN:
            continue  # an end: a node's pair (found exactly on a node, rounding would duplicate it)
        th = 0.5 * (m.radius[a0] + m.radius[a1] + m.radius[b0] + m.radius[b1])
        if not _excluded(q, m, e, -1, st[0], f, -1, st[1], EXCLUSION * th):
            _keep_pair(sc, sc.num_nodes + e, sc.num_nodes + f, wp.vec4i(a0, a1, b0, b1),
                       wp.vec4(1.0 - st[0], st[0], st[1] - 1.0, -st[1]), th, 0.5 * (m.mu[a0] + m.mu[b0]))


@wp.func
def _node_shape(sc: Scene, i: int, jj: int, at: int):
    """Node ``i`` against its ``jj``-th shape, the shape's surface offset outward (Sec. 3.9; found at ``at``)."""
    m, b, q = sc.m, sc.b, sc.q
    xv = node(q, i)
    if jj < _shape_count(b, m.world[i]):
        k = _shape_at(b, m.world[i], jj)
        X = shape_pose(b, k, b.body_q)
        th = m.radius[i] + b.margin[k]
        y, dy = _anchor(b, k, xv)
        reach = REACH * th + wp.length(dy) + _sweep(b, k)  # and the body's motion
        if wp.length(xv - wp.transform_get_translation(X)) > b.radius[k] + reach:
            return
        d, n = _gap(b, k, X, xv, reach)
        if d <= reach and not (m.rod[i] >= 0 and _edge_holds(q, m, b, k, X, i, n, reach)):
            _found(sc, at, k, wp.vec4i(i, -1, -1, -1), wp.vec4(1.0, 0.0, 0.0, 0.0), xv - d * n, n, th)


@wp.kernel
def _shapes_kernel(sc: Scene):
    """Feature ``i`` (a node, then the edges) against its ``jj``-th shape, into :class:`Found`."""
    i, jj = wp.tid()
    r = 2 * (sc.found.width * i + jj)
    sc.found.shape[r] = -1
    sc.found.shape[r + 1] = -1
    if i >= sc.num_nodes:
        e = i - sc.num_nodes
        if sc.m.edge_on[e] != 0:
            if sc.m.rod[sc.m.edges[e][0]] >= 0:
                _edge_shape(sc, e, jj, r)
            else:  # a shell's: between its nodes
                _shell_shape(sc, e, jj, r)
    elif sc.m.adj_ptr[i + 1] > sc.m.adj_ptr[i]:  # on a colliding rod or a shell
        _node_shape(sc, i, jj, r)


@wp.func
def _node_contacts(sc: Scene, bvh: wp.uint64, face_bvh: wp.uint64, i: int):
    """Node ``i``'s contacts: against rods (node-node, node-edge), shapes and triangles."""
    m, q = sc.m, sc.q
    xv = node(q, i)
    on_rod = m.rod[i] >= 0
    p = xv + m.shift[i]  # (in its world's place in the trees)
    query = wp.bvh_query_aabb(bvh, p, p)
    f = int(0)
    while wp.bvh_query_next(query, f):  # against rods
        b0 = m.edges[f][0]
        b1 = m.edges[f][1]
        if m.edge_on[f] == 0 or m.rod[b0] < 0 or b0 == i or b1 == i or (on_rod and sc.self_contact == 0) or                 _other_world(m, i, b0):
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
        ef = f
        if other < sc.num_nodes:
            if (on_rod and other < i) or not _in_node_region(q, m, other, xv):
                continue  # the lower node's pair; or xv lies past it into its other edge
            pair = wp.vec4i(i, other, -1, -1)
            w = wp.vec4(1.0, -1.0, 0.0, 0.0)
            th = m.radius[i] + m.radius[other]
            c0 = other
            ef = -1
        if on_rod:
            # Not a local minimum along this rod; within OPEN of it, the edge-edge pair is left to it (else a node
            # just over a still rod, rounded past itself, is neither's).
            if not _in_node_region(q, m, i, wp.lerp(node(q, b0), node(q, b1), t), OPEN):
                continue
            if _excluded(q, m, -1, i, 0.0, ef, c0, t, EXCLUSION * th):
                continue
        _keep_pair(sc, i, other, pair, w, th, 0.5 * (m.mu[i] + m.mu[c0]))
    _keep_found(sc, i)  # against shapes (:func:`_node_shape`)
    if m.faces.shape[0] == 0:
        return
    query = wp.bvh_query_aabb(face_bvh, p, p)
    while wp.bvh_query_next(query, f):  # against triangles: their closest feature (Algorithm 1)
        tri = m.faces[f]
        if tri[0] == i or tri[1] == i or tri[2] == i or (not on_rod and sc.self_contact == 0) or                 _other_world(m, i, tri[0]):
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
        _keep_pair(sc, i, key, pair, w, th, 0.5 * (m.mu[i] + m.mu[tri[0]]))


@wp.kernel
def _offset_kernel(bvh: wp.uint64, face_bvh: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], sc: Scene):
    """Thread ``i < num_nodes``: node ``i``'s contacts, else edge ``i - num_nodes``'s, appended to the list. Query:
    the node's point, or the edge's box (the trees' boxes carry the pad). A rod node's pair with another rod must lie
    in its own node's block; with a shape, unless a rod edge holds it (:func:`_edge_holds`); a shell vertex's need not
    (vertex-facet is one-sided)."""
    i = wp.tid()
    if i >= sc.num_nodes:
        e = i - sc.num_nodes
        if sc.m.edge_on[e] != 0:
            _edge_contacts(sc, bvh, lo[e], hi[e], e)
            if sc.m.rod[sc.m.edges[e][0]] < 0:  # a shell's: between its nodes
                _keep_found(sc, i)
    elif sc.m.adj_ptr[i + 1] > sc.m.adj_ptr[i]:  # on a colliding rod or a shell
        _node_contacts(sc, bvh, face_bvh, i)
