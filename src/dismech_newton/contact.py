"""Contact and Coulomb friction as a strain-space ADMM block, on primitive pairs.

A contact is a pair ``(vec4i nodes, vec4 w)``: ``C q = sum_k w_k x_k - anchor = x_a - x_b``; side ``a`` holds the
nodes of positive weight, side ``b`` those of negative weight (a point on an edge or a triangle, or a node) or the
anchor alone. ``-1`` marks an unused node. The block owns the contact law (``d(q)``, the friction set, the penalties)
and the ADMM kernels; where contacts come from is :mod:`.collide`'s, whose source fills the :class:`Slots`.
"""

from typing import Any

import warp as wp
from newton import Model, State

from .linear import row_slot
from .state import block_sum, node, padded, scatter_node

RECOVERY = 0.5  # a contact undoes at most this many thicknesses of penetration per step


@wp.struct
class Slots:
    """Per contact: ``pair`` with weights ``w``, the normal from ``b`` to ``a``, the anchor, the start ``c0`` of
    ``C q`` (minus the anchor's motion), the friction set, penalty and dual; ``key``: what a source matches the warm
    start by."""

    active: wp.array[wp.int32]
    pair: wp.array[wp.vec4i]
    w: wp.array[wp.vec4]
    normal: wp.array[wp.vec3]
    anchor: wp.array[wp.vec3]
    c0: wp.array[wp.vec3]
    thickness: wp.array[float]
    mu: wp.array[float]
    rho: wp.array[float]
    u: wp.array[wp.vec3]
    z: wp.array[wp.vec3]
    key: wp.array[wp.int32]


def _slots(n: int, device) -> Slots:
    s = Slots()
    for name, var in Slots.vars.items():
        setattr(s, name, wp.zeros(n, dtype=var.type.dtype, device=device))
    s.key.fill_(-1)
    return s


class ContactBlock:
    """The contacts as a strain-ADMM block: ``d_i(q) = local(C_i q)``, ``z_i`` in the friction set, penalty
    ``rho_i = beta / (C_i (alpha M)^-1 C_i^T)``; a contact takes part in a step only if the prediction brings it within
    ``engage_gap`` thicknesses of touching or it carries force (the source decides). ``source`` fills the slots
    (see :mod:`.collide`)."""

    def __init__(self, model: Model, fixed: wp.array, num_candidates: int, source, *, beta: float):
        self.fixed, self.device, self.source, self.beta = fixed, wp.get_device(model.device), source, beta
        self._no_pick = wp.zeros(0, dtype=wp.int32, device=self.device)
        self.count = source.slots()
        size = max(self.count, 1)
        self.s, self.prev = _slots(size, self.device), _slots(size, self.device)
        self.d_cand = wp.zeros((num_candidates, size), dtype=wp.vec3, device=self.device)

    def _launch(self, kernel, inputs, dim=None):
        if self.count:
            wp.launch(kernel, dim=dim or self.count, inputs=inputs, device=self.device)

    def reset(self) -> None:
        """Drop the warm start."""
        self.prev.u.zero_()
        self.prev.key.fill_(-1)

    def begin_step(self, state_in: State, dt: float, q: wp.array, mass: wp.array, alpha: float) -> None:
        """Let the source fill the slots for the step to the prediction ``q``, and set the penalties."""
        if self.count:
            self.source.fill(state_in, dt, q, self.s, self.prev)
        self._launch(_penalty_kernel, [self.fixed, mass, alpha, self.beta, self.s])

    def snapshot(self) -> Slots | None:
        """This step's active contacts, gathered for its adjoint (one count read)."""
        if not self.count:
            return None
        offset = wp.empty(self.count, dtype=wp.int32, device=self.device)
        wp.utils.array_scan(self.s.active, offset, inclusive=False)
        m = int(offset[-1:].numpy()[0] + self.s.active[-1:].numpy()[0])
        if not m:
            return None
        out = _slots(m, self.device)
        for name in Slots.vars:
            wp.launch(_compact_kernel, dim=self.count, inputs=[self.s.active, offset, getattr(self.s, name)],
                      outputs=[getattr(out, name)], device=self.device)
        return out

    def end_step(self) -> None:
        if self.count:  # indexed like this step's contacts, which the next step's source matches
            for name in ("u", "rho", "key"):
                wp.copy(getattr(self.prev, name), getattr(self.s, name))

    def assemble(self, q: wp.array, A) -> None:
        self._launch(_assemble_kernel, [self.s, self.fixed, A.indptr, A.indices, A.vals])

    def grad(self, q: wp.array, b: wp.array) -> None:
        """``z = Pi(d(q) + u)`` from the warm-started duals, and the gradient."""
        self._launch(_local_kernel, [q, self.d_cand, self._no_pick, self.fixed, self.s, b])

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(_energy_kernel, [q_cand, self.s, self.d_cand, energy], dim=(q_cand.shape[0], padded(self.count)))

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self._launch(_local_kernel, [q, self.d_cand, pick, self.fixed, self.s, b])


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
def pair_fixed(fixed: wp.array[wp.int32], pair: wp.vec4i) -> bool:
    """Nothing in ``pair`` can move: its dual would grow without bound."""
    for k in range(4):
        n = pair[k]
        if n >= 0 and fixed[3 * n] * fixed[3 * n + 1] * fixed[3 * n + 2] == 0:
            return False
    return True


@wp.func
def engaged(
    q0: wp.array[float],
    q: wp.array[float],
    pair: wp.vec4i,
    w: wp.vec4,
    n: wp.vec3,
    anchor: wp.vec3,
    th: float,
    engage_gap: float,
    moving: bool,
) -> bool:
    """The prediction ``q`` brings the pair within ``engage_gap`` thicknesses of touching; ``moving``: less the
    contact point's motion from ``q0`` (what the gap may lose if a contact stops one side: two sheets falling together
    onto the ground, say)."""
    reach = (1.0 + engage_gap) * th
    for k in range(4):
        if pair[k] >= 0 and moving:
            reach += wp.abs(w[k]) * wp.length(node(q, pair[k]) - node(q0, pair[k]))
    return wp.dot(n, contact_point(q, pair, w, anchor)) <= reach


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
):
    """Activate slot ``i``; ``xb``: the anchor at the end of the step, ``db``: its displacement over it."""
    s.active[i] = 1
    s.pair[i] = pair
    s.w[i] = w
    s.normal[i] = n
    s.anchor[i] = xb
    s.c0[i] = contact_point(q, pair, w, xb) + db
    s.thickness[i] = th
    s.mu[i] = mu


@wp.func
def clear(s: Slots, i: int):
    s.active[i] = 0
    s.u[i] = wp.vec3(0.0)
    s.rho[i] = 0.0
    s.key[i] = -1


# -- the contact law -------------------------------------------------------------------------------


@wp.func
def gap_slip(c: wp.vec3, n: wp.vec3, c0: wp.vec3, thickness: float) -> wp.vec3:
    """``local(c)``, ``c = C q``: the gap along ``n`` and the slip across it,
    ``n (n . c - thickness) + (I - n n^T)(c - c0)``."""
    slip = c - c0
    return (wp.dot(n, c) - thickness) * n + (slip - wp.dot(n, slip) * n)


@wp.func
def contact_d(q: wp.array[float], s: Slots, i: int) -> wp.vec3:
    """``d(q) = local(C q)``."""
    return gap_slip(contact_point(q, s.pair[i], s.w[i], s.anchor[i]), s.normal[i], s.c0[i], s.thickness[i])


@wp.func
def friction_projection(p: wp.vec3, n: wp.vec3, mu: float) -> wp.vec3:
    """``Pi(p)``: onto the gap's half-space along ``n`` and the Coulomb cone."""
    pn = wp.dot(p, n)
    pt = p - pn * n
    r = wp.length(pt)
    z = wp.max(pn, 0.0) * n
    if r > 0.0:
        a = mu * wp.max(-pn, 0.0)
        z = z + (wp.max(r - a, 0.0) - wp.max(-r - a, 0.0)) / r * pt
    return z


@wp.kernel
def _penalty_kernel(fixed: wp.array[wp.int32], mass: wp.array[float], alpha: float, beta: float, s: Slots):
    """Drop a contact whose free nodes carry under 1% of its weight. Else cap its thickness at the start gap plus
    :data:`RECOVERY` thicknesses (pushing a deep penetration out in one step, along a normal taken near crossing,
    throws sheets through each other), and set ``rho = beta / (C (alpha M)^-1 C^T)`` (x-y-z mean), a warm-started
    dual keeping its force ``rho u``."""
    i = wp.tid()
    if s.active[i] == 0:
        return
    pair = s.pair[i]
    w = s.w[i]
    th = s.thickness[i]
    s.thickness[i] = wp.min(th, wp.max(wp.dot(s.normal[i], s.c0[i]), 0.0) + RECOVERY * th)
    flex = float(0.0)
    free = float(0.0)  # the free share of |w|^2
    for k in range(4):
        if pair[k] >= 0:
            for c in range(3):
                if fixed[3 * pair[k] + c] == 0:
                    flex += w[k] * w[k] / (3.0 * alpha * mass[3 * pair[k] + c])
                    free += w[k] * w[k] / (3.0 * wp.length_sq(w))
    if free <= 1.0e-4:  # held by fixed nodes: rho would blow up, and its rounding swamp the energy
        clear(s, i)
        return
    r = beta / flex
    if s.rho[i] > 0.0:
        s.u[i] = s.u[i] * (s.rho[i] / r)
    s.rho[i] = r


@wp.kernel
def _assemble_kernel(
    s: Slots,
    fixed: wp.array[wp.int32],
    indptr: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    """``rho C^T C`` into the upper CSR: exact within one side, ``2 rho`` per side when both have nodes (majorized,
    so ``A``'s pattern stays each edge's or triangle's nodes)."""
    i = wp.tid()
    if s.active[i] == 0:
        return
    pair = s.pair[i]
    w = s.w[i]
    f = s.rho[i]
    for k in range(4):
        if pair[k] >= 0 and w[k] < 0.0:
            f = 2.0 * s.rho[i]
    for a in range(4):
        na = pair[a]
        for bb in range(4):
            nb = pair[bb]
            if na >= 0 and nb >= na and w[a] * w[bb] > 0.0:  # every order of a node pair, once in the upper part
                v = f * w[a] * w[bb]
                for c in range(3):
                    slot = row_slot(indptr, indices, 3 * na + c, 3 * nb + c)
                    if fixed[3 * na + c] == 0 and fixed[3 * nb + c] == 0 and slot >= 0:
                        wp.atomic_add(vals, slot, wp.float64(v))


@wp.kernel
def _local_kernel(
    q: wp.array[float],
    d_cand: wp.array2d[wp.vec3],
    pick: wp.array[wp.int32],
    fixed: wp.array[wp.int32],
    s: Slots,
    b: wp.array[wp.float64],
):
    """``z = Pi(d + u)`` and the gradient ``rho C^T (d - z + u)`` into ``b``: at ``q`` from the warm-started dual
    (no ``pick``), or at the picked candidate after the dual update ``u = d + u - z``."""
    i = wp.tid()
    if s.active[i] == 0:
        return
    u = s.u[i]
    d = wp.vec3(0.0)
    if pick.shape[0] > 0:
        d = d_cand[pick[0], i]
    else:
        d = contact_d(q, s, i)
    z = friction_projection(d + u, s.normal[i], s.mu[i])
    if pick.shape[0] > 0:
        u = d + u - z
        s.u[i] = u
    s.z[i] = z
    for k in range(4):
        if s.pair[i][k] >= 0:
            scatter_node(b, fixed, s.pair[i][k], s.w[i][k] * s.rho[i] * (d - z + u))


@wp.kernel
def _compact_kernel(
    active: wp.array[wp.int32], offset: wp.array[wp.int32], src: wp.array(dtype=Any), dst: wp.array(dtype=Any)
):
    i = wp.tid()
    if active[i] != 0:
        dst[offset[i]] = src[i]


@wp.kernel
def _energy_kernel(q_cand: wp.array2d[float], s: Slots, d_cand: wp.array2d[wp.vec3], energy: wp.array[wp.float64]):
    k, i = wp.tid()
    e = wp.float64(0.0)
    if i < s.active.shape[0] and s.active[i] != 0:
        d = contact_d(q_cand[k], s, i)
        d_cand[k, i] = d
        r = d - s.z[i] + s.u[i]
        e = wp.float64(0.5 * s.rho[i] * wp.dot(r, r))
    block_sum(energy, k, e)
