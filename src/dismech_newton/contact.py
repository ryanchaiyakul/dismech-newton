"""Contact and Coulomb friction as an ADMM term (:class:`ContactTerm`).

Contacts are Newton's, detected on the rod's capsule proxies. Each one is mapped back to the rod:
the body-frame point on a capsule's axis gives the barycentric along its edge, Newton's normal is
kept and the thickness is the sum of the two contact margins. A contact against a shape that is
not a rod edge (ground, obstacle) is anchored at its world point on that shape. A rod-rod contact
whose closest point lies on an end cap is moved to the edge of the chain that owns it (see
:func:`_rehome`): the capsule chain is smooth at its joints, and a cap's tilted normal would brake
anything sliding along the rod.

Each active contact fixes, for the step, its barycentrics, the normal ``n`` and the start-of-step
relative position ``c0``; its local variable is ``p = n (n . Cq - thickness) + (I - n n^T)(Cq - c0) + u``,
and its prox the Coulomb projection ``z_n = max(0, p_n)``, ``z_t = max(0, 1 - mu max(0, -p_n) / |p_t|) p_t``.
Contacts do not enter the global matrix (primal constraint approximation, Daviet 2023): their
right-hand side is ``-rho C^T u``, so they can appear and vanish without refactorising.
"""

import numpy as np
import warp as wp
from newton import Contacts, Model, State

from .frames import node, scatter_node


@wp.func
def _contact_point(q: wp.array[float], pair: wp.vec4i, st: wp.vec2, anchor: wp.vec3) -> wp.vec3:
    """``C q``: the relative position of the two closest points (``anchor`` when ``b`` is not a rod edge)."""
    xa = (1.0 - st[0]) * node(q, pair[0]) + st[0] * node(q, pair[1])
    if pair[2] < 0:
        return xa - anchor
    return xa - ((1.0 - st[1]) * node(q, pair[2]) + st[1] * node(q, pair[3]))


@wp.func
def _node_fixed(fixed: wp.array[wp.int32], n: int) -> int:
    if n < 0:
        return 1
    return fixed[3 * n] * fixed[3 * n + 1] * fixed[3 * n + 2]


@wp.func
def _edge_bary(q: wp.array[float], n0: int, n1: int, p_body: wp.vec3) -> float:
    """Barycentric along edge ``n0 n1`` of a body-frame point of its proxy (origin at the midpoint, +Z along it)."""
    return wp.clamp(0.5 + p_body[2] / wp.length(node(q, n1) - node(q, n0)), 0.0, 1.0)


@wp.func
def _closest_st(p0: wp.vec3, p1: wp.vec3, q0: wp.vec3, q1: wp.vec3) -> wp.vec2:
    """Barycentrics of the closest points of segments ``p0 p1`` and ``q0 q1``."""
    d1 = p1 - p0
    d2 = q1 - q0
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
def _edge_st(q: wp.array[float], edge_node0: wp.array[wp.int32], edge_node1: wp.array[wp.int32], ea: int,
             eb: int) -> wp.vec2:
    return _closest_st(node(q, edge_node0[ea]), node(q, edge_node1[ea]), node(q, edge_node0[eb]),
                       node(q, edge_node1[eb]))


_REHOME_REACH = 4  # edges searched along the chain on each side of an end-cap contact


@wp.func
def _seg_dist(q: wp.array[float], edge_node0: wp.array[wp.int32], edge_node1: wp.array[wp.int32], e: int,
              other: int, st: wp.vec2) -> float:
    xa = (1.0 - st[0]) * node(q, edge_node0[e]) + st[0] * node(q, edge_node1[e])
    xb = (1.0 - st[1]) * node(q, edge_node0[other]) + st[1] * node(q, edge_node1[other])
    return wp.length(xa - xb)


@wp.func
def _rehome(q: wp.array[float], edge_node0: wp.array[wp.int32], edge_node1: wp.array[wp.int32],
            edge_prev: wp.array[wp.int32], edge_next: wp.array[wp.int32], e: int, s: float, other: int) -> int:
    """The edge that owns a contact found on edge ``e`` at barycentric ``s`` against edge ``other``.

    A closest point on an end cap may belong to the smooth capsule chain elsewhere: the nearest edge
    within reach whose own closest point is strictly interior owns it. Kept on the cap, its normal,
    tilted along the chain and frozen for the step, reads as a bump to anything sliding past (a drag
    that grows with the step's displacement). A convex corner, or ``other`` lying along the chain,
    has no such edge and keeps the cap.
    """
    if s > 1.0e-4 and s < 1.0 - 1.0e-4:
        return e
    best = e
    best_dist = _seg_dist(q, edge_node0, edge_node1, e, other, _edge_st(q, edge_node0, edge_node1, e, other))
    for side in range(2):
        n = e
        for _k in range(_REHOME_REACH):
            if side == 0:
                n = edge_prev[n]
            else:
                n = edge_next[n]
            if n < 0 or n == other:
                break
            st = _edge_st(q, edge_node0, edge_node1, n, other)
            if st[0] > 1.0e-6 and st[0] < 1.0 - 1.0e-6:
                dist = _seg_dist(q, edge_node0, edge_node1, n, other, st)
                if dist < best_dist:
                    best = n
                    best_dist = dist
    return best


@wp.kernel
def _convert_contacts_kernel(
    contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal_ab: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    match_index: wp.array[wp.int32],
    generation: wp.array[wp.int32],
    seen_generation: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_edge: wp.array[wp.int32],
    edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32],
    edge_prev: wp.array[wp.int32],
    edge_next: wp.array[wp.int32],
    q: wp.array[float],
    dof_fixed: wp.array[wp.int32],
    u_prev: wp.array[wp.vec3],
    rho_prev: wp.array[float],
    self_contact: int,
    # outputs
    active: wp.array[wp.int32],
    pairs: wp.array[wp.vec4i],
    thickness: wp.array[float],
    bary: wp.array[wp.vec2],
    normal: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    c0: wp.array[wp.vec3],
    u: wp.array[wp.vec3],
    rho: wp.array[float],
):
    """Map Newton's contact ``i`` onto the rod and freeze it for the step.

    Side ``a`` is always a rod edge; Newton's normal points from shape 0 to shape 1, ours from
    ``b`` to ``a``, so ``a`` is shape 1 unless only shape 0 is a rod edge. The dual is
    warm-started from ``u_prev`` (last step's duals): at the same slot while the contact set is
    the one the last step saw (``generation`` unchanged), else through ``match_index``.
    """
    i = wp.tid()
    active[i] = 0
    u[i] = wp.vec3(0.0, 0.0, 0.0)
    rho[i] = 0.0
    if i >= wp.min(contact_count[0], active.shape[0]):
        return
    s0 = shape0[i]
    s1 = shape1[i]
    b0 = shape_body[s0]
    b1 = shape_body[s1]
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

    st = wp.vec2(_edge_bary(q, edge_node0[ea], edge_node1[ea], pa), 0.0)
    xb = wp.vec3(0.0, 0.0, 0.0)
    if eb >= 0:
        st[1] = _edge_bary(q, edge_node0[eb], edge_node1[eb], pb)
        # Rod-rod: move a contact off an end cap onto the edge that owns it, then take the
        # closest points and the normal of the edges themselves.
        ea2 = _rehome(q, edge_node0, edge_node1, edge_prev, edge_next, ea, st[0], eb)
        eb2 = _rehome(q, edge_node0, edge_node1, edge_prev, edge_next, eb, st[1], ea2)
        if ea2 != ea or eb2 != eb:
            ea = ea2
            eb = eb2
            st = _edge_st(q, edge_node0, edge_node1, ea, eb)
            xa = (1.0 - st[0]) * node(q, edge_node0[ea]) + st[0] * node(q, edge_node1[ea])
            xb = (1.0 - st[1]) * node(q, edge_node0[eb]) + st[1] * node(q, edge_node1[eb])
            d = xa - xb
            if wp.length(d) > 1.0e-9:
                n = wp.normalize(d)
            xb = wp.vec3(0.0, 0.0, 0.0)
    elif bb >= 0:
        xb = wp.transform_point(body_q[bb], pb)
    else:
        xb = pb
    pair = wp.vec4i(edge_node0[ea], edge_node1[ea], -1, -1)
    if eb >= 0:
        pair[2] = edge_node0[eb]
        pair[3] = edge_node1[eb]
    fixed = _node_fixed(dof_fixed, pair[0]) * _node_fixed(dof_fixed, pair[1])
    fixed = fixed * _node_fixed(dof_fixed, pair[2]) * _node_fixed(dof_fixed, pair[3])
    if fixed != 0:  # nothing can move: the dual would grow without bound
        return

    active[i] = 1
    pairs[i] = pair
    thickness[i] = margin0[i] + margin1[i]
    bary[i] = st
    normal[i] = n
    anchor[i] = xb
    c0[i] = _contact_point(q, pair, st, xb)
    m = int(-1)
    if generation[0] == seen_generation[0]:
        m = i
    elif match_index.shape[0] > 0:
        m = match_index[i]
    if m >= 0 and m < u_prev.shape[0]:
        u[i] = u_prev[m]
        rho[i] = rho_prev[m]  # the warm start's penalty, so its force can be kept


@wp.func
def _add_load(load: wp.array[float], fixed: wp.array[wp.int32], n: int, w: float):
    if n >= 0 and fixed[3 * n] == 0:
        wp.atomic_add(load, 3 * n, w)


@wp.kernel
def _load_kernel(active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2],
                 fixed: wp.array[wp.int32], load: wp.array[float]):
    """``load[3 n]``: the summed weight of the contacts on free node ``n``."""
    i = wp.tid()
    if active[i] == 0:
        return
    pair = pairs[i]
    st = bary[i]
    _add_load(load, fixed, pair[0], 1.0 - st[0])
    _add_load(load, fixed, pair[1], st[0])
    if pair[2] >= 0:
        _add_load(load, fixed, pair[2], 1.0 - st[1])
        _add_load(load, fixed, pair[3], st[1])


@wp.func
def _row(v: wp.array[float], fixed: wp.array[wp.int32], pair: wp.vec4i, st: wp.vec2) -> float:
    """``sum_n |w_n| v[3 n]`` over the contact's free nodes."""
    out = float(0.0)
    for k in range(4):
        n = pair[k]
        if n >= 0 and fixed[3 * n] == 0:
            w = st[0]
            if k == 0:
                w = 1.0 - st[0]
            elif k == 2:
                w = 1.0 - st[1]
            elif k == 3:
                w = st[1]
            out = out + w * v[3 * n]
    return out


@wp.kernel
def _penalty_kernel(active: wp.array[wp.int32], pairs: wp.array[wp.vec4i], bary: wp.array[wp.vec2],
                    fixed: wp.array[wp.int32], reach: wp.array[float], scale: float,
                    rho: wp.array[float], u: wp.array[wp.vec3]):
    """``rho_i = scale / sum_j |C_i H^-1 C_j^T|``, the Gershgorin bound on the contacts' coupling
    (``reach = H^-1 load``). The warm-started dual is rescaled so its force ``rho_i u_i`` is kept."""
    i = wp.tid()
    if active[i] == 0:
        return
    pair = pairs[i]
    st = bary[i]
    r = scale / wp.max(_row(reach, fixed, pair, st), 1.0e-12)
    if rho[i] > 0.0:
        u[i] = u[i] * (rho[i] / r)
    rho[i] = r


@wp.kernel
def _copy_generation_kernel(generation: wp.array[wp.int32], seen_generation: wp.array[wp.int32]):
    seen_generation[0] = generation[0]


@wp.kernel
def _contact_local_kernel(
    q: wp.array[float],
    pairs: wp.array[wp.vec4i],
    thickness: wp.array[float],
    active: wp.array[wp.int32],
    bary: wp.array[wp.vec2],
    normal: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    c0: wp.array[wp.vec3],
    mu: float,
    rho: wp.array[float],
    dof_fixed: wp.array[wp.int32],
    update: int,
    # outputs
    u: wp.array[wp.vec3],
    rhs: wp.array[float],
    stats: wp.array[float],
):
    """With ``update``: project onto the friction cone and step the dual. Always: ``rhs -= rho_i C^T u``."""
    i = wp.tid()
    if active[i] == 0:
        return
    pair = pairs[i]
    st = bary[i]
    n = normal[i]
    ui = u[i]
    if update != 0:
        c = _contact_point(q, pair, st, anchor[i])
        slip = c - c0[i]
        p = (wp.dot(n, c) - thickness[i]) * n + (slip - wp.dot(n, slip) * n) + ui
        pn = wp.dot(p, n)
        pt = p - pn * n
        lam = wp.max(0.0, -pn)  # normal force / rho_i
        pt_len = wp.length(pt)
        zt = wp.vec3(0.0, 0.0, 0.0)
        if pt_len > 0.0:
            zt = wp.max(0.0, 1.0 - mu * lam / pt_len) * pt
        z = wp.max(0.0, pn) * n + zt
        u_new = p - z
        wp.atomic_max(stats, 0, wp.length(u_new - ui) / wp.max(thickness[i], 1.0e-6))
        ui = u_new
        u[i] = ui
    force = -rho[i] * ui
    scatter_node(rhs, dof_fixed, pair[0], (1.0 - st[0]) * force)
    scatter_node(rhs, dof_fixed, pair[1], st[0] * force)
    if pair[2] >= 0:
        scatter_node(rhs, dof_fixed, pair[2], -(1.0 - st[1]) * force)
        scatter_node(rhs, dof_fixed, pair[3], -st[1] * force)


class ContactTerm:
    """Contact and friction of the rod against itself and the model's other shapes (see the module docstring).

    The per-contact arrays have one slot per slot of the :class:`~newton.Contacts` buffer
    passed to :meth:`begin_step`, sized on first sight of that buffer. Contact duals are kept
    as they are when a step sees the same contact set again (contacts refreshed every few
    substeps), else warm-started through ``contact_matching`` when the pipeline has it on.

    Args:
        model: Model built with the rod's capsule proxies.
        fixed: Per-DOF Dirichlet flags.
        friction: Coulomb coefficient ``mu``.
        self_contact: Keep rod-rod contacts (rod-obstacle contacts are always kept).
        rho_scale: Every contact's penalty is ``rho_scale / sum_j |C_i H^-1 C_j^T|``, the inverse of
            the Gershgorin bound on the contacts' coupling through the global matrix ``H``, so the dual
            iteration converges for ``rho_scale < 2``; an isolated contact converges in one iteration at 1.
    """

    def __init__(self, model: Model, fixed: wp.array, *, friction: float, self_contact: bool, rho_scale: float = 1.8):
        self.model = model
        self.der = model.dismech
        self.fixed = fixed
        self.device = wp.get_device(model.device)
        self.friction = friction
        self.self_contact = self_contact
        self.rho_scale = rho_scale
        edge_body = self.der.edge_body.numpy()
        body_edge = np.full(max(model.body_count, 1), -1, dtype=np.int32)
        body_edge[edge_body[edge_body >= 0]] = np.nonzero(edge_body >= 0)[0]
        self._body_edge = wp.array(body_edge, dtype=wp.int32, device=self.device)
        # Neighbouring edges of each edge along its rod (-1 at a free end).
        node0, node1 = self.der.edge_node0.numpy(), self.der.edge_node1.numpy()
        edge_ending = np.full(model.particle_count, -1, dtype=np.int32)
        edge_ending[node1] = np.arange(len(node1))
        edge_starting = np.full(model.particle_count, -1, dtype=np.int32)
        edge_starting[node0] = np.arange(len(node0))
        self._edge_prev = wp.array(edge_ending[node0], dtype=wp.int32, device=self.device)
        self._edge_next = wp.array(edge_starting[node1], dtype=wp.int32, device=self.device)
        n_dofs = fixed.shape[0]
        self._load = wp.zeros(n_dofs, dtype=float, device=self.device)
        self._reach = wp.zeros(n_dofs, dtype=float, device=self.device)
        self._no_match = wp.zeros(0, dtype=wp.int32, device=self.device)
        self._seen_generation = wp.full(1, -1, dtype=wp.int32, device=self.device)
        self._buffer = None  # the Contacts the per-contact arrays are sized for
        self.count = 0  # slots in use: the capacity of that buffer, 0 without contacts
        self._alloc(1)

    def _alloc(self, n: int) -> None:
        dev = self.device
        self.active = wp.zeros(n, dtype=wp.int32, device=dev)
        self.pairs = wp.zeros(n, dtype=wp.vec4i, device=dev)
        self.thickness = wp.zeros(n, dtype=float, device=dev)
        self.bary = wp.zeros(n, dtype=wp.vec2, device=dev)
        self.normal = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.anchor = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.c0 = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.u = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.rho_i = wp.zeros(n, dtype=float, device=dev)
        self._u_prev = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._rho_prev = wp.zeros(n, dtype=float, device=dev)

    def begin_step(self, state_in: State, contacts: Contacts | None, solve) -> None:
        """Size the per-contact arrays for ``contacts`` (a new buffer starts without warm start),
        then map ``contacts`` (detected on ``state_in``) onto the rod and freeze them for the step."""
        if contacts is None:
            self.count = 0
            return
        if contacts is not self._buffer:
            n = contacts.rigid_contact_max
            if n != self.active.shape[0]:
                self._alloc(max(n, 1))
            else:
                self._u_prev.zero_()
            self._seen_generation.fill_(-1)
            self._buffer = contacts
        self.count = contacts.rigid_contact_max
        if not self.count:
            return
        c = contacts
        match = c.rigid_contact_match_index if c.rigid_contact_match_index is not None else self._no_match
        wp.launch(
            _convert_contacts_kernel,
            dim=self.count,
            inputs=[
                c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, c.rigid_contact_point0,
                c.rigid_contact_point1, c.rigid_contact_normal, c.rigid_contact_margin0, c.rigid_contact_margin1,
                match, c.contact_generation, self._seen_generation, self.model.shape_body, state_in.body_q,
                self._body_edge, self.der.edge_node0, self.der.edge_node1, self._edge_prev, self._edge_next,
                state_in.dismech.q, self.fixed,
                self._u_prev, self._rho_prev, int(self.self_contact),
            ],
            outputs=[self.active, self.pairs, self.thickness, self.bary, self.normal, self.anchor, self.c0, self.u,
                     self.rho_i],
            device=self.device,
        )
        # Penalties from the contacts' coupling through H: H^-1 >= 0 entrywise (an M-matrix), so
        # sum_j |C_i H^-1 C_j^T| <= C_i |H^-1| load = C_i H^-1 load, one solve (``solve(b, x)``).
        self._load.zero_()
        wp.launch(_load_kernel, dim=self.count, inputs=[self.active, self.pairs, self.bary, self.fixed],
                  outputs=[self._load], device=self.device)
        solve(self._load, self._reach)
        wp.launch(_penalty_kernel, dim=self.count,
                  inputs=[self.active, self.pairs, self.bary, self.fixed, self._reach, self.rho_scale],
                  outputs=[self.rho_i, self.u], device=self.device)
        wp.launch(
            _copy_generation_kernel, dim=1, inputs=[c.contact_generation], outputs=[self._seen_generation],
            device=self.device,
        )

    def local(self, q: wp.array, rhs: wp.array, stats: wp.array, update: int) -> None:
        """With ``update``: project and step the duals at ``q``. Always: add ``-rho C^T u`` to ``rhs``."""
        if not self.count:
            return
        wp.launch(
            _contact_local_kernel,
            dim=self.count,
            inputs=[
                q, self.pairs, self.thickness, self.active, self.bary, self.normal, self.anchor, self.c0,
                self.friction, self.rho_i, self.fixed, update,
            ],
            outputs=[self.u, rhs, stats],
            device=self.device,
        )

    def end_step(self) -> None:
        if self.count:
            wp.copy(self._u_prev, self.u)  # indexed like this step's contacts, which the next step matches
            wp.copy(self._rho_prev, self.rho_i)

    def forces(self) -> tuple[np.ndarray, np.ndarray]:
        """``(pairs, force)`` of the active contacts of the last step: node quadruples
        ``(a0, a1, b0, b1)`` (``b0 = -1`` for a shape that is not a rod edge) and the force on
        ``a`` (host arrays)."""
        active = self.active.numpy()[: self.count] != 0
        force = -self.rho_i.numpy()[: self.count, None] * self.u.numpy()[: self.count]
        return self.pairs.numpy()[: self.count][active], force[active]
