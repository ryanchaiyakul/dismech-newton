"""Contact and Coulomb friction as an ADMM term, on Newton's contacts mapped from the proxies to the rod.

A rod-rod contact is a local minimum of the distance between the centerlines. Each minimum lies in exactly one
half-open edge-pair cell ``[0, 1)^2`` (closed at free ends), so Newton's capsule pairs are only candidates: a
pair keeps the closest points of its segments iff it owns them (:func:`_owns`), and the overlapping capsules
around one minimum yield one contact.

A contact's local variable is ``p = n (n . Cq - thickness) + (I - n n^T)(Cq - c0) + u`` (``n``, ``c0``
frozen per step); it stays out of the global matrix, adding only ``-rho C^T u`` to the right-hand side.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp
from newton import Contacts, Model, State

from .dofs import node, scatter_node
from .strains import skew


@dataclass
class ContactSnapshot:
    """One step's contacts, frozen for its adjoint; ``smoothing`` in [m/s]."""

    count: int
    active: wp.array
    pairs: wp.array
    bary: wp.array
    normal: wp.array
    anchor: wp.array
    shift: wp.array
    thickness: wp.array
    force: wp.array
    rho: wp.array
    friction: wp.array
    smoothing: float


class ContactTerm:
    """Rod contacts and friction; per-contact arrays are indexed like the :class:`~newton.Contacts` buffer.

    Args:
        friction: Coulomb ``mu``, in the one-element array :attr:`friction` (differentiable).
        rho_scale: Penalty over the inverse Gershgorin bound; converges below 2.
    """

    def __init__(self, model: Model, fixed: wp.array, *, friction: float, self_contact: bool, rho_scale: float = 1.8):
        self.model = model
        self.der = der = model.dismech
        self.fixed = fixed
        self.device = dev = wp.get_device(model.device)
        self.friction = wp.array([friction], dtype=float, device=dev)
        self.self_contact = self_contact
        self.rho_scale = rho_scale
        bodies = max(model.body_count, 1)
        edge_body = der.edge_body.numpy()
        body_edge = np.full(bodies, -1, dtype=np.int32)
        body_edge[edge_body[edge_body >= 0]] = np.nonzero(edge_body >= 0)[0]
        self._body_edge = wp.array(body_edge, dtype=wp.int32, device=dev)
        # Per edge its nodes and its neighbours along the rod (previous, next; -1 at a free end).
        node0, node1 = der.edge_node0.numpy(), der.edge_node1.numpy()
        ending, starting = np.full((2, model.particle_count), -1, dtype=np.int32)
        ending[node1] = np.arange(len(node1))
        starting[node0] = np.arange(len(node0))
        self._edges = wp.array(np.stack([node0, node1], 1), dtype=wp.vec2i, device=dev)
        self._edge_adj = wp.array(np.stack([ending[node0], starting[node1]], 1), dtype=wp.vec2i, device=dev)
        self._load = wp.zeros(fixed.shape[0], dtype=float, device=dev)
        self._reach = wp.zeros(fixed.shape[0], dtype=float, device=dev)
        self._no_match = wp.zeros(0, dtype=wp.int32, device=dev)
        self._no_velocity = wp.zeros(0, dtype=wp.spatial_vector, device=dev)
        self._no_pose = wp.zeros(0, dtype=wp.transform, device=dev)
        self._mobility = wp.zeros(0, dtype=wp.spatial_matrix, device=dev)
        self._body_count = wp.zeros(bodies, dtype=float, device=dev)
        self._body_wrench = [wp.zeros(bodies, dtype=wp.spatial_vector, device=dev) for _ in range(2)]
        self._h = 0.0
        self._seen_generation = wp.full(1, -1, dtype=wp.int32, device=dev)
        self._buffer = None  # the Contacts the per-contact arrays are sized for
        self.count = 0  # slots in use: that buffer's capacity, 0 without contacts
        self._alloc(1)

    def _alloc(self, n: int) -> None:
        def z(dtype):
            return wp.zeros(n, dtype=dtype, device=self.device)

        self.active, self.pairs, self.thickness, self.bary = z(wp.int32), z(wp.vec4i), z(float), z(wp.vec2)
        self.normal, self.anchor, self.shift, self.c0, self.u = (z(wp.vec3) for _ in range(5))
        self.rho_i, self.arm = z(float), z(wp.vec3)
        self.body = wp.full(n, -1, dtype=wp.int32, device=self.device)
        self._u_prev, self._rho_prev = z(wp.vec3), z(float)

    def reset(self) -> None:
        """Drop the warm start: the next step starts as on a new contact buffer."""
        self._buffer = None

    def begin_step(self, state_in: State, contacts: Contacts | None, solve, h: float,
                   body_q_end: wp.array | None = None, end_weight: float = 1.0) -> None:
        """Map ``contacts`` onto the rod for the step; obstacles move with ``body_qd`` or toward ``body_q_end``."""
        if contacts is None:
            self.count = 0
            return
        if contacts is not self._buffer:  # a new buffer starts without warm start
            n = contacts.rigid_contact_max
            if n != self.active.shape[0]:
                self._alloc(max(n, 1))
            else:
                self._u_prev.zero_()
            self._seen_generation.fill_(-1)
            self._buffer = contacts
        self.count = contacts.rigid_contact_max
        self._h = h
        if not self.count:
            return
        c, dev = contacts, self.device
        wp.launch(
            _convert_contacts_kernel,
            dim=self.count,
            inputs=[
                c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, c.rigid_contact_point0,
                c.rigid_contact_point1, c.rigid_contact_normal, c.rigid_contact_margin0, c.rigid_contact_margin1,
                c.rigid_contact_match_index if c.rigid_contact_match_index is not None else self._no_match,
                c.contact_generation, self._seen_generation, self.model.shape_body, state_in.body_q,
                state_in.body_qd if state_in.body_qd is not None else self._no_velocity, self.model.body_com, h,
                body_q_end if body_q_end is not None else self._no_pose, float(end_weight), self._body_edge,
                self._edges, self._edge_adj, state_in.dismech.q,
                self.fixed, self._u_prev, self._rho_prev, int(self.self_contact),
            ],
            outputs=[self.active, self.pairs, self.thickness, self.bary, self.normal, self.anchor, self.shift, self.c0,
                     self.u, self.rho_i, self.body, self.arm],
            device=dev,
        )
        if self._mobility.shape[0]:
            self._body_count.zero_()
            wp.launch(_body_count_kernel, dim=self.count,
                      inputs=[self.active, self.body, self.normal, self.c0, self.thickness, self.u],
                      outputs=[self._body_count], device=dev)
        # H^-1 >= 0 entrywise (an M-matrix), so sum_j |C_i H^-1 C_j^T| <= C_i H^-1 load: one solve.
        self._load.zero_()
        wp.launch(_load_kernel, dim=self.count, inputs=[self.active, self.pairs, self.bary, self.fixed],
                  outputs=[self._load], device=dev)
        solve(self._load, self._reach)
        wp.launch(_penalty_kernel, dim=self.count,
                  inputs=[self.active, self.pairs, self.bary, self.fixed, self._reach, self.body, self.arm,
                          self._mobility, self._body_count, h * h, self.rho_scale],
                  outputs=[self.rho_i, self.u], device=dev)
        wp.copy(self._seen_generation, c.contact_generation)

    def local(self, q: wp.array, rhs: wp.array, stats: wp.array, update: int) -> None:
        """With ``update``: project and step the duals. Always: ``rhs -= rho C^T u``."""
        if not self.count:
            return
        wp.launch(
            _contact_local_kernel,
            dim=self.count,
            inputs=[q, self.pairs, self.thickness, self.active, self.bary, self.normal, self.anchor, self.c0,
                    self.friction, self.rho_i, self.fixed, update, self.body, self.arm, self._mobility,
                    self._h * self._h, self._body_wrench[0]],
            outputs=[self.u, rhs, stats, self._body_wrench[1]],
            device=self.device,
        )
        if self._mobility.shape[0]:
            wp.copy(self._body_wrench[0], self._body_wrench[1])
            self._body_wrench[1].zero_()

    def set_body_mobility(self, mobility: wp.array | None) -> None:
        """Per body 6x6 mobility ``W`` about its center of mass, world frame; ``None``: bodies do not yield."""
        self._mobility = mobility if mobility is not None else wp.zeros(0, dtype=wp.spatial_matrix, device=self.device)
        for f in self._body_wrench:
            f.zero_()

    def add_reactions(self, body_f: wp.array) -> None:
        """Add the last step's contact wrenches (about the COM) to ``body_f``."""
        if self.count:
            wp.launch(_reaction_kernel, dim=self.count, inputs=[self.active, self.body, self.arm, self.rho_i, self.u],
                      outputs=[body_f], device=self.device)

    def end_step(self) -> None:
        if self.count:
            wp.copy(self._u_prev, self.u)  # indexed like this step's contacts, which the next step matches
            wp.copy(self._rho_prev, self.rho_i)

    def snapshot(self, smoothing: float) -> ContactSnapshot | None:
        """This step's active contacts, gathered for its adjoint (sized by them, not by the buffer's capacity:
        one count read)."""
        n, dev = self.count, self.device
        if not n:
            return None
        offset = wp.empty(n, dtype=wp.int32, device=dev)
        wp.utils.array_scan(self.active[:n], offset, inclusive=False)
        m = int(offset[n - 1 :].numpy()[0] + self.active[n - 1 : n].numpy()[0])
        if not m:
            return None
        slot = wp.empty(m, dtype=wp.int32, device=dev)
        wp.launch(_active_slots_kernel, dim=n, inputs=[self.active, offset], outputs=[slot], device=dev)
        out = [wp.empty(m, dtype=t, device=dev)
               for t in (wp.int32, wp.vec4i, wp.vec2, wp.vec3, wp.vec3, wp.vec3, float, wp.vec3, float)]
        wp.launch(_snapshot_kernel, dim=m,
                  inputs=[slot, self.pairs, self.bary, self.normal, self.anchor, self.shift, self.thickness,
                          self.rho_i, self.u, self.rho_scale],
                  outputs=out, device=dev)
        return ContactSnapshot(m, *out, self.friction, smoothing)

    def forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Host ``(pairs, force)``: nodes ``(a0, a1, b0, b1)`` (``b0 = -1``: non-rod), force on ``a``."""
        active = self.active.numpy()[: self.count] != 0
        force = -self.rho_i.numpy()[: self.count, None] * self.u.numpy()[: self.count]
        return self.pairs.numpy()[: self.count][active], force[active]


def contact_c_pattern(pairs: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, ...]:
    """``C``'s entries (``3 m x n``, over the free DOFs): per entry its contact, node slot in the pair (0-3, the
    weight :func:`_weight`), component, and DOF. Row ``3 contact + component``."""
    dof = 3 * pairs.astype(np.int64)[:, :, None] + np.arange(3)  # (contact, node, component); negative: no node
    keep = (dof >= 0) & ~fixed[np.maximum(dof, 0)]
    contact, node, comp = np.nonzero(keep)
    return contact, node, comp, dof[keep]


# -- geometry -----------------------------------------------------------------------------


@wp.func
def _weight(k: int, st: wp.vec2) -> float:
    """Weight of node ``k`` in ``C q = x_a - x_b``."""
    if k == 0:
        return 1.0 - st[0]
    if k == 1:
        return st[0]
    if k == 2:
        return st[1] - 1.0
    return -st[1]


@wp.func
def _edge_point(q: wp.array[float], e: wp.vec2i, s: float) -> wp.vec3:
    return (1.0 - s) * node(q, e[0]) + s * node(q, e[1])


@wp.func
def _contact_point(q: wp.array[float], pair: wp.vec4i, st: wp.vec2, anchor: wp.vec3) -> wp.vec3:
    """``C q``; ``b`` is ``anchor`` when not a rod edge."""
    xa = _edge_point(q, wp.vec2i(pair[0], pair[1]), st[0])
    if pair[2] < 0:
        return xa - anchor
    return xa - _edge_point(q, wp.vec2i(pair[2], pair[3]), st[1])


@wp.func
def _node_fixed(fixed: wp.array[wp.int32], n: int) -> int:
    if n < 0:
        return 1
    return fixed[3 * n] * fixed[3 * n + 1] * fixed[3 * n + 2]


@wp.func
def _edge_bary(q: wp.array[float], e: wp.vec2i, p_body: wp.vec3) -> float:
    """Barycentric of a point in the proxy's frame (origin at the midpoint, +Z along the edge)."""
    return wp.clamp(0.5 + p_body[2] / wp.length(node(q, e[1]) - node(q, e[0])), 0.0, 1.0)


@wp.func
def _closest_st(q: wp.array[float], pair: wp.vec4i) -> wp.vec2:
    """Closest points of the segments ``a`` and ``b`` of a rod-rod pair."""
    p0 = node(q, pair[0])
    q0 = node(q, pair[2])
    d1 = node(q, pair[1]) - p0
    d2 = node(q, pair[3]) - q0
    r = p0 - q0
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    b = wp.dot(d1, d2)
    c = wp.dot(d1, r)
    f = wp.dot(d2, r)
    den = a * e - b * b
    s = float(0.0)
    if den > 1.0e-12 * a * e:
        s = wp.clamp((b * f - c * e) / den, 0.0, 1.0)
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = wp.clamp(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = wp.clamp((b - c) / a, 0.0, 1.0)
    return wp.vec2(s, t)


@wp.func
def _owns(q: wp.array[float], edges: wp.array[wp.vec2i], edge_adj: wp.array[wp.vec2i], e: int, s: float,
          d: wp.vec3) -> bool:
    """Whether edge ``e`` owns its closest point at ``s``, ``d`` from the other rod's: an interior ``s``; at
    node 0 a local minimum (the distance not decreasing into the previous edge, ties included: the previous
    edge never owns its node 1); node 1 only at a free end."""
    if s >= 1.0 - 1.0e-5:
        return edge_adj[e][1] < 0
    p = edge_adj[e][0]
    if s > 1.0e-5 or p < 0:
        return True
    into = node(q, edges[p][0]) - node(q, edges[e][0])
    return wp.dot(d, into) >= -1.0e-4 * wp.length(d) * wp.length(into)


@wp.func
def _row(v: wp.array[float], fixed: wp.array[wp.int32], pair: wp.vec4i, st: wp.vec2) -> float:
    """``sum_n |w_n| v[3 n]`` over free nodes."""
    out = float(0.0)
    for k in range(4):
        n = pair[k]
        if n >= 0 and fixed[3 * n] == 0:
            out = out + wp.abs(_weight(k, st)) * v[3 * n]
    return out


@wp.func
def _local_point(c: wp.vec3, c0: wp.vec3, n: wp.vec3, thickness: float) -> wp.vec3:
    """``n (n . c - thickness) + (I - n n^T)(c - c0)``: the gap along ``n``, the slip across it."""
    slip = c - c0
    return (wp.dot(n, c) - thickness) * n + (slip - wp.dot(n, slip) * n)


@wp.func
def _wrench(f: wp.vec3, r: wp.vec3) -> wp.spatial_vector:
    return wp.spatial_vector(f, wp.cross(r, f))


@wp.func
def _point_velocity(W6: wp.spatial_matrix, wrench: wp.spatial_vector, r: wp.vec3) -> wp.vec3:
    t = W6 * wrench
    return wp.spatial_top(t) + wp.cross(wp.spatial_bottom(t), r)


@wp.func
def _point_mobility(W6: wp.spatial_matrix, r: wp.vec3) -> wp.mat33:
    """``J_r W6 J_r^T``, ``J_r = J_p - [r] J_w``."""
    A = wp.mat33()
    B = wp.mat33()
    C = wp.mat33()
    for i in range(3):
        for j in range(3):
            A[i, j] = W6[i, j]
            B[i, j] = W6[i, 3 + j]
            C[i, j] = W6[3 + i, 3 + j]
    S = skew(r)
    return A + B * S - S * wp.transpose(B) - S * C * S


# -- forward kernels ----------------------------------------------------------------------


@wp.kernel
def _convert_contacts_kernel(
    contact_count: wp.array[wp.int32], shape0: wp.array[wp.int32], shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3], point1: wp.array[wp.vec3], normal_ab: wp.array[wp.vec3], margin0: wp.array[float],
    margin1: wp.array[float], match_index: wp.array[wp.int32], generation: wp.array[wp.int32],
    seen_generation: wp.array[wp.int32], shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], h: float, body_q_end: wp.array[wp.transform],
    end_weight: float, body_edge: wp.array[wp.int32], edges: wp.array[wp.vec2i], edge_adj: wp.array[wp.vec2i],
    q: wp.array[float], dof_fixed: wp.array[wp.int32],
    u_prev: wp.array[wp.vec3], rho_prev: wp.array[float], self_contact: int,
    # outputs
    active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], thickness: wp.array[float], bary: wp.array[wp.vec2],
    normal: wp.array[wp.vec3], anchor: wp.array[wp.vec3], shift: wp.array[wp.vec3], c0: wp.array[wp.vec3],
    u: wp.array[wp.vec3], rho: wp.array[float], body: wp.array[wp.int32], arm: wp.array[wp.vec3],
):
    """Map Newton's contact onto the rod; ``a`` is a rod edge, the normal points from ``b`` to ``a``. A rod-rod
    contact is dropped unless its cell owns it (another reported pair does, or a filtered one)."""
    i = wp.tid()
    active[i] = 0
    u[i] = wp.vec3(0.0, 0.0, 0.0)
    rho[i] = 0.0
    body[i] = -1
    if i >= wp.min(contact_count[0], active.shape[0]):
        return
    b0 = shape_body[shape0[i]]
    b1 = shape_body[shape1[i]]
    e0 = int(-1)
    e1 = int(-1)
    if b0 >= 0:
        e0 = body_edge[b0]
    if b1 >= 0:
        e1 = body_edge[b1]
    n = normal_ab[i]
    ea = e1
    eb = e0
    pa = point1[i]
    pb = point0[i]
    bb = b0
    if e1 < 0:
        ea = e0
        eb = e1
        pa = point0[i]
        pb = point1[i]
        bb = b1
        n = -n
    if ea < 0 or (eb >= 0 and self_contact == 0):
        return

    pair = wp.vec4i(edges[ea][0], edges[ea][1], -1, -1)
    st = wp.vec2(0.0, 0.0)
    xb = wp.vec3(0.0, 0.0, 0.0)
    db = wp.vec3(0.0, 0.0, 0.0)  # the anchor's displacement over the step
    if eb >= 0:
        # Rod-rod: the closest points of the segments, if this cell owns them. Newton's second point on
        # near-parallel capsules is the same cell, in the next slot (contacts are sorted by shape pair).
        if i > 0 and shape0[i - 1] == shape0[i] and shape1[i - 1] == shape1[i]:
            return
        pair[2] = edges[eb][0]
        pair[3] = edges[eb][1]
        st = _closest_st(q, pair)
        d = _contact_point(q, pair, st, xb)
        if not (_owns(q, edges, edge_adj, ea, st[0], d) and _owns(q, edges, edge_adj, eb, st[1], -d)):
            return
        if wp.length(d) > 1.0e-9:
            n = wp.normalize(d)
    elif bb >= 0:
        xb = wp.transform_point(body_q[bb], pb)
        arm[i] = xb - wp.transform_point(body_q[bb], body_com[bb])
        if body_q_end.shape[0] > 0:
            db = end_weight * (wp.transform_point(body_q_end[bb], pb) - xb)
        elif body_qd.shape[0] > 0:
            v = body_qd[bb]
            db = h * (wp.spatial_top(v) + wp.cross(wp.spatial_bottom(v), arm[i]))
        xb = xb + db
    else:
        xb = pb
    if eb < 0:
        st[0] = _edge_bary(q, edges[ea], pa)
    fixed = int(1)
    for k in range(4):
        fixed = fixed * _node_fixed(dof_fixed, pair[k])
    if fixed != 0:  # nothing can move: the dual would grow without bound
        return

    active[i] = 1
    if eb < 0:
        body[i] = bb
    pairs[i] = pair
    thickness[i] = margin0[i] + margin1[i]
    bary[i] = st
    normal[i] = n
    anchor[i] = xb  # at the end of the step
    shift[i] = db
    c0[i] = _contact_point(q, pair, st, xb) + db
    m = int(-1)
    if generation[0] == seen_generation[0]:
        m = i
    elif match_index.shape[0] > 0:
        m = match_index[i]
    if m >= 0 and m < u_prev.shape[0]:
        u[i] = u_prev[m]
        rho[i] = rho_prev[m]  # the warm start's penalty, so its force can be kept


@wp.kernel
def _load_kernel(active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2],
                 fixed: wp.array[wp.int32], load: wp.array[float]):
    """``load[3 n]``: summed contact weight on free node ``n``."""
    i = wp.tid()
    if active[i] == 0:
        return
    pair = pairs[i]
    for k in range(4):
        n = pair[k]
        if n >= 0 and fixed[3 * n] == 0:
            wp.atomic_add(load, 3 * n, wp.abs(_weight(k, bary[i])))


@wp.kernel
def _penalty_kernel(active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2],
                    fixed: wp.array[wp.int32], reach: wp.array[float], body: wp.array[wp.int32],
                    arm: wp.array[wp.vec3], mobility: wp.array[wp.spatial_matrix], count: wp.array[float], h2: float,
                    scale: float, rho: wp.array[float], u: wp.array[wp.vec3]):
    """``rho_i = scale / bound``: the Gershgorin bound ``C_i reach``, plus a yielding body's coupling
    ``h2 |W_p| count``; a warm-started dual keeps its force."""
    i = wp.tid()
    if active[i] == 0:
        return
    bound = wp.max(_row(reach, fixed, pairs[i], bary[i]), 1.0e-12)
    b = body[i]
    if b >= 0 and mobility.shape[0] > 0:
        W = _point_mobility(mobility[b], arm[i])
        w = float(0.0)
        for k in range(3):
            w = wp.max(w, wp.abs(W[k, 0]) + wp.abs(W[k, 1]) + wp.abs(W[k, 2]))
        bound = bound + h2 * w * wp.max(count[b], 1.0)
    r = scale / bound
    if rho[i] > 0.0:
        u[i] = u[i] * (rho[i] / r)
    rho[i] = r


@wp.kernel
def _body_count_kernel(active: wp.array[wp.int32], body: wp.array[wp.int32], normal: wp.array[wp.vec3],
                       c0: wp.array[wp.vec3], thickness: wp.array[float], u: wp.array[wp.vec3],
                       count: wp.array[float]):
    """``count[b]``: contacts on body ``b`` that can carry force."""
    i = wp.tid()
    b = body[i]
    if active[i] != 0 and b >= 0 and (wp.dot(normal[i], c0[i]) - thickness[i] < thickness[i] or wp.length(u[i]) > 0.0):
        wp.atomic_add(count, b, 1.0)


@wp.kernel
def _contact_local_kernel(
    q: wp.array[float], pairs: wp.array[wp.vec4i], thickness: wp.array[float], active: wp.array[wp.int32],
    bary: wp.array[wp.vec2], normal: wp.array[wp.vec3], anchor: wp.array[wp.vec3], c0: wp.array[wp.vec3],
    friction: wp.array[float], rho: wp.array[float], dof_fixed: wp.array[wp.int32], update: int,
    body: wp.array[wp.int32], arm: wp.array[wp.vec3], body_mobility: wp.array[wp.spatial_matrix], h2: float,
    body_wrench: wp.array[wp.spatial_vector],
    # outputs
    u: wp.array[wp.vec3], rhs: wp.array[float], stats: wp.array[float], body_wrench_next: wp.array[wp.spatial_vector],
):
    """With ``update``: Coulomb projection and dual step. Always: ``rhs -= rho_i C^T u``.
    On a yielding body, other contacts' wrenches lag one iteration; its own is implicit."""
    i = wp.tid()
    if active[i] == 0:
        return
    pair = pairs[i]
    st = bary[i]
    n = normal[i]
    ui = u[i]
    b = body[i]
    compliant = b >= 0 and body_mobility.shape[0] > 0
    r = wp.vec3()
    if compliant:
        r = arm[i]
    if update != 0:
        xb = anchor[i]
        W = wp.mat33()
        if compliant:
            W6 = body_mobility[b]
            W = _point_mobility(W6, r)
            xb = xb + h2 * _point_velocity(W6, body_wrench[b] - _wrench(rho[i] * ui, r), r)
        p = _local_point(_contact_point(q, pair, st, xb), c0[i], n, thickness[i]) + ui
        pn = wp.dot(p, n)
        pt = p - pn * n
        lam = wp.max(0.0, -pn)  # normal force / rho_i
        pt_len = wp.length(pt)
        zt = wp.vec3(0.0, 0.0, 0.0)
        if pt_len > 0.0:
            zt = wp.max(0.0, 1.0 - friction[0] * lam / pt_len) * pt
        u_new = p - (wp.max(0.0, pn) * n + zt)
        if compliant:
            u_new = wp.inverse(wp.identity(3, dtype=float) + (h2 * rho[i]) * W) * u_new
        wp.atomic_max(stats, 0, wp.length(u_new - ui) / wp.max(thickness[i], 1.0e-6))
        ui = u_new
        u[i] = ui
    if compliant:
        wp.atomic_add(body_wrench_next, b, _wrench(rho[i] * ui, r))
    force = -rho[i] * ui
    for k in range(4):
        if pair[k] >= 0:
            scatter_node(rhs, dof_fixed, pair[k], _weight(k, st) * force)


@wp.kernel
def _reaction_kernel(active: wp.array[wp.int32], body: wp.array[wp.int32], arm: wp.array[wp.vec3],
                     rho: wp.array[float], u: wp.array[wp.vec3], body_f: wp.array[wp.spatial_vector]):
    """``body_f += (f, arm x f)``, ``f = rho_i u``."""
    i = wp.tid()
    b = body[i]
    if active[i] == 0 or b < 0:
        return
    wp.atomic_add(body_f, b, _wrench(rho[i] * u[i], arm[i]))


# -- adjoint ------------------------------------------------------------------------------


@wp.kernel
def _active_slots_kernel(active: wp.array[wp.int32], offset: wp.array[wp.int32], slot: wp.array[wp.int32]):
    """``slot[offset[i]] = i`` for the active slots (``offset``: the exclusive scan of ``active``)."""
    i = wp.tid()
    if active[i] != 0:
        slot[offset[i]] = i


@wp.kernel
def _snapshot_kernel(
    slot: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], normal: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3], shift: wp.array[wp.vec3], thickness: wp.array[float], rho_i: wp.array[float],
    u: wp.array[wp.vec3], rho_scale: float,
    # outputs
    active_out: wp.array[wp.int32], pairs_out: wp.array[wp.vec4i], bary_out: wp.array[wp.vec2],
    normal_out: wp.array[wp.vec3], anchor_out: wp.array[wp.vec3], shift_out: wp.array[wp.vec3],
    thickness_out: wp.array[float], force: wp.array[wp.vec3], rho: wp.array[float],
):
    """Active contact ``k`` (slot ``slot[k]``): its frozen data, force and penalty."""
    k = wp.tid()
    i = slot[k]
    active_out[k] = 1
    pairs_out[k] = pairs[i]
    bary_out[k] = bary[i]
    normal_out[k] = normal[i]
    anchor_out[k] = anchor[i]
    shift_out[k] = shift[i]
    thickness_out[k] = thickness[i]
    force[k] = -rho_i[i] * u[i]
    rho[k] = rho_i[i] / rho_scale


@wp.func
def _ramp(x: float, eps: float) -> float:
    """``max(0, x)``, smoothed for ``eps > 0`` (log-barrier prox)."""
    if eps > 0.0:
        return 0.5 * (x + wp.sqrt(x * x + 4.0 * eps * eps))
    return wp.max(x, 0.0)


@wp.func
def _ramp_slope(x: float, eps: float) -> float:
    if eps > 0.0:
        return 0.5 * (1.0 + x / wp.sqrt(x * x + 4.0 * eps * eps))
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return 0.0
    return 0.5


@wp.func
def _apply_c(x: wp.array[float], fixed: wp.array[wp.int32], pair: wp.vec4i, st: wp.vec2) -> wp.vec3:
    out = wp.vec3(0.0, 0.0, 0.0)
    for k in range(4):
        nk = pair[k]
        if nk >= 0:
            w = _weight(k, st)
            for c in range(3):
                if fixed[3 * nk + c] == 0:
                    out[c] = out[c] + w * x[3 * nk + c]
    return out


@wp.kernel
def contact_linearize_kernel(
    idx: wp.array[wp.int32], q: wp.array[float], q_in: wp.array[float], pairs: wp.array[wp.vec4i],
    bary: wp.array[wp.vec2], normal: wp.array[wp.vec3], anchor: wp.array[wp.vec3], shift: wp.array[wp.vec3],
    thickness: wp.array[float], force: wp.array[wp.vec3], rho: wp.array[float], friction: wp.array[float], eps: float,
    # outputs
    jac: wp.array[wp.mat33], jac_mu: wp.array[wp.vec3], point: wp.array[wp.vec3],
):
    """``D = dPi/dp`` and ``dPi/dmu`` at ``p = d(q) - force / rho`` (into ``point``)."""
    k = wp.tid()
    i = idx[k]
    pair = pairs[i]
    st = bary[i]
    n = normal[i]
    c0 = _contact_point(q_in, pair, st, anchor[i]) + shift[i]
    p = _local_point(_contact_point(q, pair, st, anchor[i]), c0, n, thickness[i]) - force[i] / rho[i]
    point[k] = p
    mu = friction[0]
    pn = wp.dot(p, n)
    pt = p - pn * n
    r = wp.length(pt)
    nn = wp.outer(n, n)
    P = wp.identity(3, dtype=float) - nn
    a = mu * _ramp(-pn, eps)  # the slip threshold
    D = _ramp_slope(pn, eps) * nn
    g_mu = wp.vec3(0.0, 0.0, 0.0)
    if r > 0.0:
        # z_t = g(r) t, g = ramp(r - a) - ramp(-r - a): odd in r, so smooth through p_t = 0.
        t = pt / r
        tt = wp.outer(t, t)
        g = _ramp(r - a, eps) - _ramp(-r - a, eps)
        g_r = _ramp_slope(r - a, eps) + _ramp_slope(-r - a, eps)
        g_a = _ramp_slope(-r - a, eps) - _ramp_slope(r - a, eps)
        D = D + g_r * tt + (g / r) * (P - tt) - (g_a * mu * _ramp_slope(-pn, eps)) * wp.outer(t, n)
        g_mu = (g_a * _ramp(-pn, eps)) * t
    else:
        D = D + (2.0 * _ramp_slope(-a, eps)) * P
    jac[k] = D
    jac_mu[k] = g_mu


@wp.kernel
def contact_matrix_values_kernel(
    entry_contact: wp.array[wp.int32], entry_node: wp.array[wp.int32], entry_comp: wp.array[wp.int32],
    pos_ct: wp.array[wp.int32], pos_lower: wp.array[wp.int32], idx: wp.array[wp.int32], bary: wp.array[wp.vec2],
    rho: wp.array[float], jac: wp.array[wp.mat33],
    # outputs
    vals: wp.array[wp.float64],
):
    """Per entry of ``C`` (:func:`contact_c_pattern`): its weight at ``pos_ct`` (``C^T``), and
    ``rho ((I - D)^T C)`` in its contact's three rows at ``pos_lower`` (``M^T``'s lower-left block)."""
    e = wp.tid()
    k = entry_contact[e]
    i = idx[k]
    c = entry_comp[e]
    w = _weight(entry_node[e], bary[i])
    vals[pos_ct[e]] = wp.float64(w)
    D = jac[k]
    for a in range(3):
        delta = float(0.0)
        if a == c:
            delta = 1.0
        vals[pos_lower[3 * e + a]] = wp.float64(rho[i] * (delta - D[c, a]) * w)


@wp.kernel
def contact_transpose_residual_kernel(
    idx: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], fixed: wp.array[wp.int32],
    jac: wp.array[wp.mat33], rho: wp.array[float], w_q: wp.array[float], w_p: wp.array[float],
    # outputs
    res_q: wp.array[float], res_p: wp.array[float],
):
    """The contacts' part of ``rhs - M^T w``."""
    k = wp.tid()
    i = idx[k]
    pair = pairs[i]
    st = bary[i]
    D = jac[k]
    wk = node(w_p, k)
    for j in range(4):
        if pair[j] >= 0:
            scatter_node(res_q, fixed, pair[j], -_weight(j, st) * wk)
    v = wp.transpose(D) * wk - rho[i] * (wp.transpose(wp.identity(3, dtype=float) - D) * _apply_c(w_q, fixed, pair, st))
    for c in range(3):
        res_p[3 * k + c] = v[c]


@wp.func
def _project(p: wp.vec3, n: wp.vec3, mu: float, eps: float) -> wp.vec3:
    """``Pi(p)``: onto the gap's half-space along ``n`` and the Coulomb cone, smoothed by ``eps`` (as
    :func:`contact_linearize_kernel` differentiates it)."""
    pn = wp.dot(p, n)
    pt = p - pn * n
    r = wp.length(pt)
    z = _ramp(pn, eps) * n
    if r > 0.0:
        a = mu * _ramp(-pn, eps)
        z = z + (_ramp(r - a, eps) - _ramp(-r - a, eps)) / r * pt
    return z


@wp.func
def _step_node(q: wp.array[float], q_in: wp.array[float], fixed: wp.array[wp.int32], n: int) -> wp.vec3:
    """Node ``n`` of the step's ``q``: fixed components are ``q_in``'s."""
    out = wp.vec3()
    for k in range(3):
        if fixed[3 * n + k] != 0:
            out[k] = q_in[3 * n + k]
        else:
            out[k] = q[3 * n + k]
    return out


@wp.kernel
def contact_frozen_inputs_kernel(
    idx: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], normal: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3], shift: wp.array[wp.vec3], thickness: wp.array[float], fixed: wp.array[wp.int32],
    q: wp.array[float], q_in: wp.array[float], point: wp.array[wp.vec3], rho: wp.array[float],
    friction: wp.array[float], eps: float,
    # outputs
    e_q: wp.array[float], e_p: wp.array[wp.vec3],
):
    """The contacts' equations at the solution as functions of ``q_in`` through what a step freezes from it,
    differentiated by Warp: ``e_q = rho C^T (p - Pi(p))`` and ``e_p = d(q) - Pi(p)``, ``d(q) = local(C q)``.
    A rod-rod contact's closest points ``st`` and normal are recomputed from ``q_in`` (as the step found them);
    every contact's reference point ``c0 = C q_in + shift`` too. ``q`` and ``p`` are held."""
    k = wp.tid()
    i = idx[k]
    pair = pairs[i]
    st = bary[i]
    n = normal[i]
    if pair[2] >= 0:
        st = _closest_st(q_in, pair)
        d = _contact_point(q_in, pair, st, anchor[i])
        if wp.length(d) > 1.0e-9:
            n = wp.normalize(d)
    c0 = _contact_point(q_in, pair, st, anchor[i]) + shift[i]
    c = wp.vec3()
    for j in range(4):
        nj = pair[j]
        if nj >= 0:
            c = c + _weight(j, st) * _step_node(q, q_in, fixed, nj)
    if pair[2] < 0:
        c = c - anchor[i]
    p = point[k]
    z = _project(p, n, friction[0], eps)
    e_p[k] = _local_point(c, c0, n, thickness[i]) - z
    f = rho[i] * (p - z)
    for j in range(4):
        if pair[j] >= 0:
            scatter_node(e_q, fixed, pair[j], _weight(j, st) * f)


@wp.kernel
def contact_friction_adjoint_kernel(
    idx: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], fixed: wp.array[wp.int32],
    rho: wp.array[float], jac_mu: wp.array[wp.vec3], w_q: wp.array[float], w_p: wp.array[float],
    # outputs
    friction_grad: wp.array[float],
):
    """``-w^T dF / d mu`` into the friction coefficient's adjoint."""
    k = wp.tid()
    i = idx[k]
    wp.atomic_add(friction_grad, 0, wp.dot(jac_mu[k], rho[i] * _apply_c(w_q, fixed, pairs[i], bary[i]) + node(w_p, k)))
