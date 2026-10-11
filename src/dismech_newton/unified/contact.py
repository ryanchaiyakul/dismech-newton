"""Contact and Coulomb friction as a strain-space ADMM block, on Newton's contacts mapped onto primitive pairs.

A pair is ``(vec4i nodes, st)``: ``C q = x_a - x_b``, side ``a`` a rod edge (or a shell vertex), side ``b`` a rod edge,
a vertex or an anchor on a body. A rod-rod contact is a local minimum of the distance between the centerlines; each
lies in exactly one half-open edge-pair cell, so Newton's capsule pairs are only candidates (:func:`_owns`).
"""

import numpy as np
import warp as wp
from newton import Contacts, Model, State

from ..dofs import node
from .state import row_slot, scatter_node


class ContactBlock:
    """The contacts as a strain-ADMM block: ``d_i(q) = local(C_i q)``, ``z_i`` in the friction set, penalty
    ``rho_i = beta / (C_i (alpha M)^-1 C_i^T)``; a contact takes part in a step only if the prediction brings it within
    ``engage_gap`` thicknesses of touching or it carries force. Per-contact arrays are indexed like the
    :class:`~newton.Contacts` buffer (rigid, then soft); a contact the buffer matches to the last step's warm-starts
    its dual."""

    def __init__(self, model: Model, fixed: wp.array, trials: int, *, friction: float, self_contact: bool, beta: float,
                 engage_gap: float):
        self.model, self.der, self.fixed = model, model.dismech, fixed
        self.device = dev = wp.get_device(model.device)
        self.trials_count, self.beta, self.engage_gap = trials, beta, engage_gap
        self.friction = wp.array([friction], dtype=float, device=dev)
        self.self_contact = self_contact
        der = self.der
        edge_body = der.edge_body.numpy()
        body_edge = np.full(max(model.body_count, 1), -1, dtype=np.int32)
        body_edge[edge_body[edge_body >= 0]] = np.nonzero(edge_body >= 0)[0]
        self._body_edge = wp.array(body_edge, dtype=wp.int32, device=dev)
        # Per edge its nodes and its neighbours along the rod (previous, next; -1 at a free end).
        node0, node1 = der.edge_node0.numpy(), der.edge_node1.numpy()
        ending, starting = np.full((2, model.particle_count), -1, dtype=np.int32)
        ending[node1] = np.arange(len(node1))
        starting[node0] = np.arange(len(node0))
        self._edges = wp.array(np.stack([node0, node1], 1), dtype=wp.vec2i, device=dev)
        self._edge_adj = wp.array(np.stack([ending[node0], starting[node1]], 1), dtype=wp.vec2i, device=dev)
        self._no_match = wp.zeros(0, dtype=wp.int32, device=dev)
        self._no_velocity = wp.zeros(0, dtype=wp.spatial_vector, device=dev)
        self._seen_generation = wp.full(1, -1, dtype=wp.int32, device=dev)
        # Soft contacts: the shell vertices take them.
        soft_node = np.zeros(max(model.particle_count, 1), dtype=np.int32)
        if getattr(der, "face_area", None) is not None and der.face_area.shape[0]:
            soft_node[np.concatenate([getattr(der, f"face_node{k}").numpy() for k in range(3)])] = 1
        self.has_soft = bool(soft_node.any())
        self._soft_node = wp.array(soft_node, dtype=wp.int32, device=dev)
        self.misses = wp.zeros(1, dtype=wp.int32, device=dev)  # assembled entries outside A's pattern
        self._buffer = None  # the Contacts the per-contact arrays are sized for
        self.count = 0  # slots in use: that buffer's capacity (rigid, then soft), 0 without contacts
        self._alloc(1)

    def _alloc(self, n: int) -> None:
        def z(dtype):
            return wp.zeros(n, dtype=dtype, device=self.device)

        self.active, self.pairs, self.thickness, self.bary = z(wp.int32), z(wp.vec4i), z(float), z(wp.vec2)
        self.normal, self.anchor, self.c0, self.u, self.z = (z(wp.vec3) for _ in range(5))
        self.rho = z(float)
        self._u_prev, self._rho_prev = z(wp.vec3), z(float)
        self.d_trial = wp.zeros((self.trials_count, n), dtype=wp.vec3, device=self.device)

    def reset(self) -> None:
        """Drop the warm start: the next step starts as on a new contact buffer."""
        self._buffer = None

    def begin_step(self, state_in: State, contacts: Contacts | None, dt: float, q: wp.array, mass: wp.array,
                   alpha: float) -> None:
        """Map ``contacts`` onto the model for the step (obstacles move with ``body_qd``), keep those the prediction
        ``q`` reaches, and set their penalties."""
        if contacts is None:
            self.count = 0
            return
        rigid = contacts.rigid_contact_max
        soft = contacts.soft_contact_max if self.has_soft else 0
        if contacts is not self._buffer:  # a new buffer starts without warm start
            if rigid + soft != self.active.shape[0]:
                self._alloc(max(rigid + soft, 1))
            else:
                self._u_prev.zero_()
            self._seen_generation.fill_(-1)
            self._buffer = contacts
        self.count = rigid + soft
        if not self.count:
            return
        c, m, dev = contacts, self.model, self.device
        body_qd = state_in.body_qd if state_in.body_qd is not None else self._no_velocity
        geometry = [self._body_edge, self._edges, self._edge_adj, state_in.dismech.q, self.fixed]
        outputs = [self.active, self.pairs, self.thickness, self.bary, self.normal, self.anchor, self.c0, self.u,
                   self.rho]
        wp.launch(
            _convert_contacts_kernel, dim=rigid,
            inputs=[c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, c.rigid_contact_point0,
                    c.rigid_contact_point1, c.rigid_contact_normal, c.rigid_contact_margin0, c.rigid_contact_margin1,
                    c.rigid_contact_match_index if c.rigid_contact_match_index is not None else self._no_match,
                    c.contact_generation, self._seen_generation, m.shape_body, state_in.body_q, body_qd, m.body_com,
                    dt, *geometry, self._u_prev, self._rho_prev, int(self.self_contact)],
            outputs=outputs, device=dev,
        )
        if soft:
            wp.launch(
                _convert_soft_contacts_kernel, dim=soft,
                inputs=[c.soft_contact_count, c.soft_contact_particle, c.soft_contact_shape, c.soft_contact_body_pos,
                        c.soft_contact_normal, rigid, self._soft_node, m.particle_radius, m.shape_body, m.shape_scale,
                        m.shape_margin, state_in.body_q, body_qd, m.body_com, dt, *geometry],
                outputs=outputs, device=dev,
            )
        wp.copy(self._seen_generation, c.contact_generation)
        # In A, a separated contact's penalty drags q toward the last iterate (its z lags d by one iteration).
        wp.launch(_engage_kernel, dim=self.count,
                  inputs=[q, self.pairs, self.bary, self.anchor, self.normal, self.thickness, self.engage_gap],
                  outputs=[self.active, self.u, self.rho], device=dev)
        wp.launch(_rho_kernel, dim=self.count, inputs=[self.active, self.pairs, self.bary, self.fixed, mass, alpha,
                                                       self.beta],
                  outputs=[self.rho, self.u], device=dev)

    def end_step(self) -> None:
        if self.count:
            wp.copy(self._u_prev, self.u)  # indexed like this step's contacts, which the next step matches
            wp.copy(self._rho_prev, self.rho)

    def _geom(self) -> list:
        return [self.active, self.pairs, self.bary, self.anchor, self.c0, self.normal, self.thickness]

    def assemble(self, q: wp.array, A) -> None:
        if self.count:
            wp.launch(_assemble_kernel, dim=self.count,
                      inputs=[self.active, self.pairs, self.bary, self.rho, self.fixed, A.indptr, A.indices],
                      outputs=[A.vals, self.misses], device=self.device)

    def grad(self, q: wp.array, b: wp.array) -> None:
        """``z = Pi(d(q) + u)`` from the warm-started duals, and the gradient."""
        if self.count:
            wp.launch(_start_kernel, dim=self.count,
                      inputs=[q, *self._geom(), self.friction, self.rho, self.u, self.fixed], outputs=[self.z, b],
                      device=self.device)

    def trials(self, qt: wp.array, obj: wp.array) -> None:
        if self.count:
            wp.launch(_trials_kernel, dim=(qt.shape[0], self.count),
                      inputs=[qt, *self._geom(), self.rho, self.z, self.u], outputs=[self.d_trial, obj],
                      device=self.device)

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        if self.count:
            wp.launch(_commit_kernel, dim=self.count,
                      inputs=[self.active, self.pairs, self.bary, self.normal, self.friction, self.rho, pick,
                              self.d_trial, self.fixed],
                      outputs=[self.z, self.u, b], device=self.device)


# -- the contact law -------------------------------------------------------------------------


@wp.func
def _contact_d(q: wp.array[float], pair: wp.vec4i, st: wp.vec2, anchor: wp.vec3, c0: wp.vec3, n: wp.vec3,
               thickness: float) -> wp.vec3:
    """``d(q) = local(C q)``: the gap along ``n``, the slip across it, ``n (n . c - thickness) + (I - n n^T)(c - c0)``."""
    c = _contact_point(q, pair, st, anchor)
    slip = c - c0
    return (wp.dot(n, c) - thickness) * n + (slip - wp.dot(n, slip) * n)


@wp.func
def _friction_projection(p: wp.vec3, n: wp.vec3, mu: float) -> wp.vec3:
    """``Pi(p)``: onto the gap's half-space along ``n`` and the Coulomb cone."""
    pn = wp.dot(p, n)
    pt = p - pn * n
    r = wp.length(pt)
    z = wp.max(pn, 0.0) * n
    if r > 0.0:
        a = mu * wp.max(-pn, 0.0)
        z = z + (wp.max(r - a, 0.0) - wp.max(-r - a, 0.0)) / r * pt
    return z


# -- kernels ---------------------------------------------------------------------------------


@wp.kernel
def _engage_kernel(
    q: wp.array[float], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], anchor: wp.array[wp.vec3],
    normal: wp.array[wp.vec3], thickness: wp.array[float], engage_gap: float,
    # outputs
    active: wp.array[wp.int32], u: wp.array[wp.vec3], rho: wp.array[float],
):
    """Deactivate a contact for the step unless the prediction ``q`` closes it to within ``engage_gap``
    thicknesses or it carries a warm-started force."""
    i = wp.tid()
    if active[i] == 0:
        return
    gap = wp.dot(normal[i], _contact_point(q, pairs[i], bary[i], anchor[i])) - thickness[i]
    if gap > engage_gap * thickness[i] and rho[i] * wp.length(u[i]) == 0.0:
        active[i] = 0
        u[i] = wp.vec3(0.0, 0.0, 0.0)
        rho[i] = 0.0


@wp.kernel
def _rho_kernel(
    active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], fixed: wp.array[wp.int32],
    mass: wp.array[float], alpha: float, beta: float,
    # outputs
    rho: wp.array[float], u: wp.array[wp.vec3],
):
    """``rho_i = beta / (C_i (alpha M)^-1 C_i^T)`` (x-y-z mean); a warm-started dual keeps its force ``rho u``."""
    i = wp.tid()
    if active[i] == 0:
        return
    pair = pairs[i]
    st = bary[i]
    flex = float(0.0)
    for k in range(4):
        nk = pair[k]
        if nk < 0:
            continue
        w = _weight(k, st)
        for c in range(3):
            d = 3 * nk + c
            if fixed[d] == 0:
                flex = flex + w * w / (3.0 * alpha * mass[d])
    r = beta / wp.max(flex, 1.0e-30)
    if rho[i] > 0.0:
        u[i] = u[i] * (rho[i] / r)
    rho[i] = r


@wp.kernel
def _assemble_kernel(
    active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], rho: wp.array[float],
    fixed: wp.array[wp.int32], indptr: wp.array[wp.int32], indices: wp.array[wp.int32],
    # outputs
    vals: wp.array[wp.float64], misses: wp.array[wp.int32],
):
    """``rho C^T C`` into the upper CSR: exact within one side, ``2 rho`` per side when both are rod edges."""
    i = wp.tid()
    if active[i] == 0:
        return
    pair = pairs[i]
    st = bary[i]
    f = rho[i]
    if pair[2] >= 0:
        f = 2.0 * rho[i]
    for side in range(2):
        for a in range(2):
            na = pair[2 * side + a]
            if na < 0:
                continue
            wa = _weight(2 * side + a, st)
            for bb in range(2):
                nb = pair[2 * side + bb]
                if nb < 0 or nb < na or (nb == na and bb < a):
                    continue
                v = f * wa * _weight(2 * side + bb, st)
                if nb == na and bb != a:
                    v = 2.0 * v  # both orders of a repeated node land on its diagonal
                for c in range(3):
                    da = 3 * na + c
                    db = 3 * nb + c
                    if fixed[da] != 0 or fixed[db] != 0:
                        continue
                    s = row_slot(indptr, indices, da, db)
                    if s < 0:
                        wp.atomic_add(misses, 0, 1)
                    else:
                        wp.atomic_add(vals, s, wp.float64(v))


@wp.kernel
def _start_kernel(
    q: wp.array[float], active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2],
    anchor: wp.array[wp.vec3], c0: wp.array[wp.vec3], normal: wp.array[wp.vec3], thickness: wp.array[float],
    friction: wp.array[float], rho: wp.array[float], u: wp.array[wp.vec3], fixed: wp.array[wp.int32],
    # outputs
    z: wp.array[wp.vec3], b: wp.array[wp.float64],
):
    """``z = Pi(d(q) + u)`` from the warm-started dual, and its gradient into ``b``."""
    i = wp.tid()
    if active[i] == 0:
        return
    d = _contact_d(q, pairs[i], bary[i], anchor[i], c0[i], normal[i], thickness[i])
    zi = _friction_projection(d + u[i], normal[i], friction[0])
    z[i] = zi
    scatter_ct(b, fixed, pairs[i], bary[i], rho[i] * (d - zi + u[i]))


@wp.kernel
def _trials_kernel(
    qt: wp.array2d[float], active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2],
    anchor: wp.array[wp.vec3], c0: wp.array[wp.vec3], normal: wp.array[wp.vec3], thickness: wp.array[float],
    rho: wp.array[float], z: wp.array[wp.vec3], u: wp.array[wp.vec3],
    # outputs
    d_trial: wp.array2d[wp.vec3], obj: wp.array[wp.float64],
):
    k, i = wp.tid()
    if active[i] == 0:
        return
    d = _contact_d(qt[k], pairs[i], bary[i], anchor[i], c0[i], normal[i], thickness[i])
    d_trial[k, i] = d
    r = d - z[i] + u[i]
    wp.atomic_add(obj, k, wp.float64(0.5 * rho[i] * wp.dot(r, r)))


@wp.kernel
def _commit_kernel(
    active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2], normal: wp.array[wp.vec3],
    friction: wp.array[float], rho: wp.array[float], pick: wp.array[wp.int32], d_trial: wp.array2d[wp.vec3],
    fixed: wp.array[wp.int32],
    # outputs
    z: wp.array[wp.vec3], u: wp.array[wp.vec3], b: wp.array[wp.float64],
):
    """``p = d + u``, ``z = Pi(p)``, ``u = p - z``, then the gradient at the new ``q``."""
    i = wp.tid()
    if active[i] == 0:
        return
    d = d_trial[pick[0], i]
    p = d + u[i]
    zi = _friction_projection(p, normal[i], friction[0])
    ui = p - zi
    z[i] = zi
    u[i] = ui
    scatter_ct(b, fixed, pairs[i], bary[i], rho[i] * (d - zi + ui))


# -- primitive pairs ---------------------------------------------------------------------------


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
def scatter_ct(rhs: wp.array[wp.float64], fixed: wp.array[wp.int32], pair: wp.vec4i, st: wp.vec2, f: wp.vec3):
    """``rhs += C^T f`` on the free DOFs."""
    for k in range(4):
        if pair[k] >= 0:
            scatter_node(rhs, fixed, pair[k], _weight(k, st) * f)


# -- Newton's contacts ------------------------------------------------------------------------


@wp.func
def _body_anchor(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 h: float, b: int, p: wp.vec3):
    """Body ``b``'s point ``p`` (body frame) at the end of the step, moving with ``body_qd``, and its displacement
    over the step."""
    x = wp.transform_point(body_q[b], p)
    dx = wp.vec3(0.0, 0.0, 0.0)
    if body_qd.shape[0] > 0:
        v = body_qd[b]
        dx = h * (wp.spatial_top(v) + wp.cross(wp.spatial_bottom(v), x - wp.transform_point(body_q[b], body_com[b])))
    return x + dx, dx


@wp.func
def _pair_fixed(dof_fixed: wp.array[wp.int32], pair: wp.vec4i) -> bool:
    fixed = int(1)
    for k in range(4):
        fixed = fixed * _node_fixed(dof_fixed, pair[k])
    return fixed != 0


@wp.func
def _emit(i: int, q: wp.array[float], pair: wp.vec4i, st: wp.vec2, n: wp.vec3, xb: wp.vec3, db: wp.vec3,
          th: float, active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], thickness: wp.array[float],
          bary: wp.array[wp.vec2], normal: wp.array[wp.vec3], anchor: wp.array[wp.vec3], c0: wp.array[wp.vec3]):
    """Activate slot ``i``; ``xb``: the anchor at the end of the step, ``db``: its displacement over it."""
    active[i] = 1
    pairs[i] = pair
    thickness[i] = th
    bary[i] = st
    normal[i] = n
    anchor[i] = xb
    c0[i] = _contact_point(q, pair, st, xb) + db


@wp.kernel
def _convert_contacts_kernel(
    contact_count: wp.array[wp.int32], shape0: wp.array[wp.int32], shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3], point1: wp.array[wp.vec3], normal_ab: wp.array[wp.vec3], margin0: wp.array[float],
    margin1: wp.array[float], match_index: wp.array[wp.int32], generation: wp.array[wp.int32],
    seen_generation: wp.array[wp.int32], shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], h: float, body_edge: wp.array[wp.int32],
    edges: wp.array[wp.vec2i], edge_adj: wp.array[wp.vec2i], q: wp.array[float], dof_fixed: wp.array[wp.int32],
    u_prev: wp.array[wp.vec3], rho_prev: wp.array[float], self_contact: int,
    # outputs
    active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], thickness: wp.array[float], bary: wp.array[wp.vec2],
    normal: wp.array[wp.vec3], anchor: wp.array[wp.vec3], c0: wp.array[wp.vec3], u: wp.array[wp.vec3],
    rho: wp.array[float],
):
    """Map Newton's contact onto the rod; ``a`` is a rod edge, the normal points from ``b`` to ``a``. A rod-rod
    contact is dropped unless its cell owns it (another reported pair does, or a filtered one)."""
    i = wp.tid()
    active[i] = 0
    u[i] = wp.vec3(0.0, 0.0, 0.0)
    rho[i] = 0.0
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
        xb, db = _body_anchor(body_q, body_qd, body_com, h, bb, pb)
    else:
        xb = pb
    if eb < 0:
        st[0] = _edge_bary(q, edges[ea], pa)
    if _pair_fixed(dof_fixed, pair):  # nothing can move: the dual would grow without bound
        return
    _emit(i, q, pair, st, n, xb, db, margin0[i] + margin1[i], active, pairs, thickness, bary, normal, anchor, c0)
    m = int(-1)
    if generation[0] == seen_generation[0]:
        m = i
    elif match_index.shape[0] > 0:
        m = match_index[i]
    if m >= 0 and m < u_prev.shape[0]:
        u[i] = u_prev[m]
        rho[i] = rho_prev[m]  # the warm start's penalty, so its force can be kept


@wp.kernel
def _convert_soft_contacts_kernel(
    contact_count: wp.array[wp.int32], particle: wp.array[wp.int32], shape: wp.array[wp.int32],
    body_pos: wp.array[wp.vec3], normal_in: wp.array[wp.vec3], offset: int, soft_node: wp.array[wp.int32],
    particle_radius: wp.array[float],
    shape_body: wp.array[wp.int32], shape_scale: wp.array[wp.vec3], shape_margin: wp.array[float],
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], h: float,
    body_edge: wp.array[wp.int32], edges: wp.array[wp.vec2i], edge_adj: wp.array[wp.vec2i], q: wp.array[float],
    dof_fixed: wp.array[wp.int32],
    # outputs
    active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], thickness: wp.array[float], bary: wp.array[wp.vec2],
    normal: wp.array[wp.vec3], anchor: wp.array[wp.vec3], c0: wp.array[wp.vec3], u: wp.array[wp.vec3],
    rho: wp.array[float],
):
    """A shell vertex's soft contact into slot ``offset + i``: against a rod's capsule, the rod edge (side ``a``)
    and the vertex; else the vertex (side ``a``) and the shape's surface point. No warm start: a light vertex's
    penalty changes its force slowly, so a warm-started impulse lingers and pumps energy (a heavy rod landing on a
    sheet bounces back)."""
    k = wp.tid()
    i = offset + k
    active[i] = 0
    u[i] = wp.vec3(0.0, 0.0, 0.0)
    rho[i] = 0.0
    if k >= wp.min(contact_count[0], active.shape[0] - offset):
        return
    p = particle[k]
    if soft_node[p] == 0:
        return
    sh = shape[k]
    bb = shape_body[sh]
    e = int(-1)
    if bb >= 0:
        e = body_edge[bb]
    r = particle_radius[p]
    margin = float(0.0)
    if shape_margin.shape[0] > 0:
        margin = shape_margin[sh]
    pair = wp.vec4i(p, p, -1, -1)
    st = wp.vec2(0.0, 0.0)
    n = normal_in[k]
    xb = wp.vec3(0.0, 0.0, 0.0)
    db = wp.vec3(0.0, 0.0, 0.0)
    th = r + margin
    if e >= 0:
        # The rod's own nodes and its neighbours' sit inside its capsules (a shared rod-shell node).
        ed = edges[e]
        if ed[0] == p or ed[1] == p:
            return
        adj = edge_adj[e]
        for a in range(2):
            if adj[a] >= 0:
                if edges[adj[a]][0] == p or edges[adj[a]][1] == p:
                    return
        pair = wp.vec4i(ed[0], ed[1], p, p)
        st = wp.vec2(_edge_bary(q, ed, body_pos[k]), 0.0)
        d = _contact_point(q, pair, st, xb)
        if wp.length(d) > 1.0e-9:
            n = wp.normalize(d)  # from the vertex to the rod
        else:
            n = -n
        th = th + shape_scale[sh][0]
    else:
        if bb >= 0:
            xb, db = _body_anchor(body_q, body_qd, body_com, h, bb, body_pos[k])
        else:
            xb = body_pos[k]
    if _pair_fixed(dof_fixed, pair):
        return
    _emit(i, q, pair, st, n, xb, db, th, active, pairs, thickness, bary, normal, anchor, c0)
