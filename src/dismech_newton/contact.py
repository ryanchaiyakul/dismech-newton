"""Contact and Coulomb friction as energies in the step's objective, on primitive pairs.

A contact is a pair ``(vec4i nodes, vec4 w)``: ``C q = sum_k w_k x_k - anchor = x_a - x_b``; side ``a`` holds the
nodes of positive weight, side ``b`` those of negative weight (a point on an edge or a triangle, or a node) or the
anchor alone. ``-1`` marks an unused node. :class:`BarrierBlock` owns the contact law (a barrier on the gap, lagged
friction, each pair's stiffness); where contacts come from is :mod:`.collide`'s, whose source fills the
:class:`Slots`.
"""

from typing import Any

import warp as wp
from newton import Model, State

from .linear import row_slot
from .state import block_sum, node, padded, scatter_node

SMOOTH = wp.constant(0.02)  # the barrier's activation margin above the thickness (in thicknesses): a pair resting at
# its thickness already has curvature, so A sees it
LOAD = wp.constant(0.25)  # a node's pairs are stiffened until they hold its load at most this many thicknesses deep
GROW = wp.constant(2.0)  # a node's load scale grows at most this factor per step
DECAY = wp.constant(0.99)  # and relaxes by this factor per step, toward 1
IMPACT = wp.constant(0.8)  # the line search's longest step: this fraction of the first pair's impact (IPC's)
MAX_SCALE = wp.constant(1.0e5)  # and stays below this (unbounded, it spreads pair to pair and overflows k)
LIFT = wp.constant(0.25)  # a shape pair's least gap at a step's start, from the barrier's zero (in thicknesses)


@wp.struct
class Slots:
    """Per contact: ``pair`` with weights ``w``, the normal from ``b`` to ``a``, the anchor, the start ``c0`` of
    ``C q`` (minus the anchor's motion), the thickness, friction coefficient, lagged normal force ``rho``,
    stiffness and multiplier ``lam``; ``key``: the other feature, which a source keeps each contact once by."""

    active: wp.array[wp.int32]
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
def pair_fixed(fixed: wp.array[wp.int32], pair: wp.vec4i) -> bool:
    """Nothing in ``pair`` can move."""
    for k in range(4):
        n = pair[k]
        if n >= 0 and fixed[3 * n] * fixed[3 * n + 1] * fixed[3 * n + 2] == 0:
            return False
    return True


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
    s.rho[i] = 0.0
    s.key[i] = -1


# -- contact as an energy: a barrier and lagged friction ------------------------------------------------------------


class BarrierBlock:
    """The contacts as energies in the step's objective, each on its pair's gap ``d = n . C q`` along the normal at
    its detection: :func:`barrier` (infinite at ``d = 0``: the line search never takes a candidate through a
    contact), and lagged smoothed Coulomb friction (IPC), ``mu lam f0(|slip|)`` with the normal force ``lam`` at the
    step's start and ``f0`` smoothed over ``slip_speed dt``. Each pair's ``k`` is ``stiffness`` times its inertia
    ``1 / (dt^2 C M^-1 C^T)`` (a light sheet vertex and a rod node alike: Newton sees the contact coming), times its
    nodes' load scale (:func:`_lag_kernel`). The source finds every pair within its reach (no prediction); the solver
    caps each node's move per detection so no pair beyond it can close."""

    def __init__(self, model: Model, fixed: wp.array, source, *, stiffness: float, slip_speed: float):
        self.fixed, self.device, self.source = fixed, wp.get_device(model.device), source
        self.stiffness, self.slip_speed, self.slip = stiffness, slip_speed, 0.0
        self.count = source.slots()
        size = max(self.count, 1)
        self.s = _slots(size, self.device)
        self.touching = wp.zeros(1, dtype=wp.int32, device=self.device)  # a pair has curvature at the last grad's q
        self.scale = wp.ones(max(model.particle_count, 1), dtype=float, device=self.device)  # nodes' load scales
        self.block = source.scene.slots if self.count else 1  # a feature's slots
        self.lam = wp.zeros(size, dtype=float, device=self.device)  # the last step's multipliers, by slot and key
        self.lam_key = wp.full(size, -1, dtype=wp.int32, device=self.device)
        self.moved = wp.zeros(1, dtype=wp.int32, device=self.device)  # a multiplier moved the pair past tolerance
        self.left = wp.zeros(1, dtype=wp.int32, device=self.device)  # the step's updates left

    def _launch(self, kernel, inputs, dim=None):
        if self.count:
            wp.launch(kernel, dim=dim or self.count, inputs=inputs, device=self.device)

    def reset(self) -> None:
        self.lam_key.fill_(-1)

    def begin_step(self, state_in: State, dt: float, mass: wp.array) -> None:
        """The pairs near the step's start and their lagged normal forces."""
        self.slip = self.slip_speed * dt
        self.dt, self.mass = dt, mass
        self.left.fill_(1)
        wp.launch(_decay_kernel, dim=self.scale.shape[0], inputs=[self.scale], device=self.device)
        self.detect(state_in, state_in.dismech.q, float(LIFT), start=True)

    def detect(self, state_in: State, q: wp.array, lift: float = 0.01, start: bool = False) -> None:
        """The pairs near ``q`` (the step's start or an iterate), their normals there and lagged normal forces; a
        shape's pair starts at least ``lift`` thicknesses from its barrier's zero (an iterate's only if undetected).
        At the step's ``start``, loaded pairs raise their nodes' load scales."""
        if self.count:
            self.source.fill(state_in, self.dt, q, self.s)
        self._launch(_rebase_kernel, [q, state_in.dismech.q, self.s])
        self._launch(_carry_kernel, [self.block, self.lam, self.lam_key, self.s])
        self._launch(_lift_kernel, [q, lift, self.s])
        self._launch(_lag_kernel, [self.stiffness / (self.dt * self.dt), self.mass, self.fixed, self.scale,
                                   int(start), self.s])

    def snapshot(self) -> Slots | None:
        """This step's active pairs, gathered for its adjoint if any carries force at its end (one host read; two if
        so)."""
        if not self.count or not int(self.touching.numpy()[0]):
            return None
        offset = wp.empty(self.count, dtype=wp.int32, device=self.device)
        wp.utils.array_scan(self.s.active, offset, inclusive=False)
        out = _slots(int(offset[-1:].numpy()[0] + self.s.active[-1:].numpy()[0]), self.device)
        for name in Slots.vars:
            wp.launch(_compact_kernel, dim=self.count, inputs=[self.s.active, offset, getattr(self.s, name)],
                      outputs=[getattr(out, name)], device=self.device)
        return out

    def update(self, q: wp.array, tol: float, dq_max: wp.array, pick: wp.array, go: wp.array,
               converged: bool = True) -> None:
        """At a converged ``q``: each pair's multiplier, the force its barrier holds there beyond what it holds at
        its thickness (kept by key for the next detection and step); if one moves its pair's rest by more than
        ``tol``, the solve converged (its last Newton step ``dq_max`` within ``tol`` and taken: not stalled) and the
        step has updates left, the pairs take them and ``go`` is set: the solve goes on. Not ``converged``: at the
        step's end, whether its solve converged or not."""
        self.moved.zero_()
        self._launch(_uzawa_kernel, [q, self.s, tol, self.lam, self.lam_key, self.moved])
        wp.launch(_again_kernel, dim=1, inputs=[dq_max, pick, tol, int(converged), self.left, self.moved, go],
                  device=self.device)
        self._launch(_apply_kernel, [self.moved, self.lam, self.s])

    def end_step(self) -> None:
        """The multipliers the step solved with, kept for the next (an update it did not solve with would kick)."""
        self._launch(_keep_kernel, [self.s, self.lam, self.lam_key])

    def assemble(self, q: wp.array, A) -> None:
        self._launch(_barrier_assemble_kernel, [q, self.s, self.slip, self.fixed, A.indptr, A.indices,
                                                A.vals])

    def impact(self, q: wp.array, dq: wp.array, t: wp.array) -> None:
        """``t = min(t, ...)`` the line search's longest step along ``-dq`` (each pair's gap ``d`` is linear in it, on
        the frozen normal): :data:`IMPACT` of the length at which a touching pair's gap would reach the barrier's
        zero (IPC's filter), or the length at which an untouched pair reaches its barrier's full curvature (the Newton
        step, built without the pair's curvature, holds only that far)."""
        self._launch(_impact_kernel, [q, dq, self.s, t])

    def newton_size(self, q: wp.array, dq: wp.array, cap: float, size: wp.array) -> None:
        """``size = max(size, e''(d) |n . C dq| h^2 / m)`` over the pairs: the Newton step's change of each pair's
        normal force as the move it would give the pair's mass ``m = 1 / (C M^-1 C^T)`` in a step (its force error,
        comparable to the nodes' moves: the stopping test), relative to ``cap`` plus the force the pair holds (as a
        move too): a loaded pair's force converges to a relative tolerance."""
        self._launch(_newton_size_kernel, [q, dq, self.s, self.mass, self.fixed, self.dt * self.dt, cap, size])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self.touching.zero_()
        self._launch(_barrier_grad_kernel, [q, self.s, self.slip, self.fixed, b, self.touching])

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(_barrier_energy_kernel, [q_cand, self.s, self.slip, energy],
                     dim=(q_cand.shape[0], padded(self.count)))

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self.grad(q, b)


@wp.func
def barrier(d: float, th: float, k: float, lam: float) -> wp.vec3:
    """``(e, e', e'')`` of the barrier at the gap ``d``: OGC's (Chen et al. 2025, Eq. 18), quadratic with curvature
    ``k`` down to half the thickness, logarithmic below (``C^1`` there, infinite at ``d <= 0``), and its onset moved
    above ``th`` by a cubic ramp over :data:`SMOOTH` thicknesses (``C^2``: a pair resting at ``th`` already has
    curvature ``k``, so ``A`` and the energy agree there), and by ``lam / k`` more: the quadratic holds ``lam`` more
    at each gap (an augmented Lagrangian's ``(lam + k x)^2 / 2k``)."""
    dl = SMOOTH * th
    o = th + dl + lam / k
    if d >= o:
        return wp.vec3(0.0)
    if d <= 0.0:
        return wp.vec3(wp.inf, 0.0, 0.0)
    x = o - d
    tau = 0.5 * th
    if x < dl:
        return wp.vec3(k * x * x * x / (6.0 * dl), -k * x * x / (2.0 * dl), k * x / dl)
    if d >= tau:
        return wp.vec3(0.5 * k * (x - 0.5 * dl) * (x - 0.5 * dl) + k * dl * dl / 24.0, -k * (x - 0.5 * dl), k)
    xt = o - tau - 0.5 * dl
    b = k * tau * xt  # the log below tau, C^1 there
    a = 0.5 * k * xt * xt + k * dl * dl / 24.0
    return wp.vec3(a - b * wp.log(d / tau), -b / d, b / (d * d))


@wp.func
def slip_energy(y: float, eps: float) -> wp.vec2:
    """IPC's ``f0(y)`` and ``f1(y) / y`` (``f1 = f0'``), smoothed below ``eps``."""
    if y >= eps:
        return wp.vec2(y, 1.0 / y)
    return wp.vec2(y * y * (1.0 / eps - y / (3.0 * eps * eps)) + eps / 3.0, 2.0 / eps - y / (eps * eps))


@wp.kernel
def _rebase_kernel(q: wp.array[float], q0: wp.array[float], s: Slots):
    """``c0`` at the step's start ``q0``, not at the iterate ``q`` the pair was found at: friction resists the step's
    slip, and its normal force is the start's (an overshooting iterate's would push along the motion)."""
    i = wp.tid()
    if s.active[i] != 0:
        s.c0[i] = s.c0[i] - contact_point(q, s.pair[i], s.w[i], wp.vec3(0.0)) + contact_point(q0, s.pair[i], s.w[i], wp.vec3(0.0))


@wp.kernel
def _lift_kernel(q: wp.array[float], lift: float, s: Slots):
    """A shape's pair (all weights positive: against its anchor) starting closer than ``lift`` thicknesses, or
    inside (a node in a body, or a body moving onto it over the step): its anchor sinks along the normal so the
    barrier starts that close (finite), and the body pushes it out over the next steps."""
    i = wp.tid()
    if s.active[i] == 0 or wp.min(s.w[i]) < 0.0:
        return
    n = s.normal[i]
    o = wp.dot(n, contact_point(q, s.pair[i], s.w[i], s.anchor[i])) - lift * s.thickness[i]
    if o < 0.0:
        s.anchor[i] = s.anchor[i] + o * n
        s.c0[i] = s.c0[i] - o * n


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


@wp.kernel
def _decay_kernel(scale: wp.array[float]):
    i = wp.tid()
    scale[i] = wp.max(1.0, scale[i] * DECAY)


@wp.kernel
def _lag_kernel(c: float, mass: wp.array[float], fixed: wp.array[wp.int32], scale: wp.array[float], grow: int,
                s: Slots):
    """The pair's stiffness ``c / (C M^-1 C^T)`` (``c``: stiffness / dt^2) times its
    nodes' largest load scale, and its normal force at the step's start (in ``rho``): friction's ``lam``. At a step's
    start (``grow``), a pair compressed deeper than :data:`LOAD` thicknesses raises its nodes' scales (at most
    :data:`GROW` times), so a node's pairs come to hold its load that deep."""
    i = wp.tid()
    if s.active[i] != 0:
        sc = float(1.0)  # the pair's nodes' largest load scale
        for j in range(4):
            if s.pair[i][j] >= 0:
                sc = wp.max(sc, scale[s.pair[i][j]])
        k = c / wp.max(flexibility(s.pair[i], s.w[i], mass, fixed), 1.0e-30) * sc
        x = s.thickness[i] - wp.dot(s.normal[i], s.c0[i]) + s.lam[i] / k  # its compression at the step's start
        # (the multiplier's force as the barrier's compression)
        if grow != 0 and x > LOAD * s.thickness[i]:  # held deeper than LOAD: the next steps' pairs stiffer
            up = wp.min(sc * wp.min(GROW, x / (LOAD * s.thickness[i])), MAX_SCALE)
            for j in range(4):
                if s.pair[i][j] >= 0:
                    wp.atomic_max(scale, s.pair[i][j], up)
        s.stiffness[i] = k
        s.rho[i] = -barrier(wp.dot(s.normal[i], s.c0[i]), s.thickness[i], s.stiffness[i], s.lam[i])[1]


@wp.func
def _slip(c: wp.vec3, c0: wp.vec3, n: wp.vec3) -> wp.vec3:
    """The pair's tangential motion since the step's start."""
    slip = c - c0
    return slip - wp.dot(n, slip) * n


@wp.func
def contact_force(c: wp.vec3, n: wp.vec3, c0: wp.vec3, th: float, k: float, lam: float, mu_rho: float, eps: float):
    """A pair's gradient with respect to ``C q = c`` (normal ``n``, start ``c0``, multiplier ``lam``, friction
    ``mu rho = mu_rho``), and its barrier's and friction's curvatures."""
    e = barrier(wp.dot(n, c), th, k, lam)
    slip = _slip(c, c0, n)
    friction = mu_rho * slip_energy(wp.length(slip), eps)[1]
    return e[1] * n + friction * slip, e[2], friction


@wp.func
def _barrier_force(c: wp.vec3, s: Slots, i: int, eps: float):
    return contact_force(c, s.normal[i], s.c0[i], s.thickness[i], s.stiffness[i], s.lam[i], s.mu[i] * s.rho[i], eps)


@wp.func
def contact_hessian(c: wp.vec3, s: Slots, i: int, eps: float) -> wp.mat33:
    """The pair's exact Hessian with respect to ``C q = c`` (:func:`_barrier_assemble_kernel` majorizes its
    friction's along the slip)."""
    n = s.normal[i]
    nn = wp.outer(n, n)
    g, h, fr = _barrier_force(c, s, i, eps)
    H = h * nn + fr * (wp.identity(3, float) - nn)
    slip = _slip(c, s.c0[i], n)
    y = wp.length(slip)
    if y > 0.0:  # f1' - f1 / y along the slip
        t = slip / y
        H = H - s.mu[i] * s.rho[i] * wp.where(y < eps, y / (eps * eps), 1.0 / y) * wp.outer(t, t)
    return H


@wp.kernel
def _barrier_grad_kernel(q: wp.array[float], s: Slots, eps: float, fixed: wp.array[wp.int32],
                         b: wp.array[wp.float64], touching: wp.array[wp.int32]):
    i = wp.tid()
    if s.active[i] == 0:
        return
    g, h, fr = _barrier_force(contact_point(q, s.pair[i], s.w[i], s.anchor[i]), s, i, eps)
    if h > 0.0 or fr > 0.0:
        touching[0] = 1
    for j in range(4):
        if s.pair[i][j] >= 0:
            scatter_node(b, fixed, s.pair[i][j], s.w[i][j] * g)


@wp.kernel
def _barrier_assemble_kernel(
    q: wp.array[float],
    s: Slots,
    eps: float,
    fixed: wp.array[wp.int32],
    indptr: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    """``C^T (e'' n n^T + mu lam f1/y (I - n n^T)) C`` into the upper CSR, majorized as :func:`_assemble_kernel`'s."""
    i = wp.tid()
    if s.active[i] == 0:
        return
    pair = s.pair[i]
    w = s.w[i]
    n = s.normal[i]
    g, h, fr = _barrier_force(contact_point(q, pair, w, s.anchor[i]), s, i, eps)
    H = h * wp.outer(n, n) + fr * (wp.identity(3, float) - wp.outer(n, n))
    f = float(1.0)
    for j in range(4):
        if pair[j] >= 0 and w[j] < 0.0:
            f = 2.0
    for a in range(4):
        na = pair[a]
        for bb in range(4):
            nb = pair[bb]
            if na >= 0 and nb >= na and w[a] * w[bb] > 0.0:
                v = f * w[a] * w[bb]
                for c in range(3):
                    for c2 in range(3):
                        if (nb > na or c2 >= c) and fixed[3 * na + c] == 0 and fixed[3 * nb + c2] == 0:
                            slot = row_slot(indptr, indices, 3 * na + c, 3 * nb + c2)
                            if slot >= 0:
                                wp.atomic_add(vals, slot, wp.float64(v * H[c, c2]))


@wp.kernel
def _impact_kernel(q: wp.array[float], dq: wp.array[float], s: Slots, t: wp.array[float]):
    i = wp.tid()
    if s.active[i] == 0:
        return
    n = s.normal[i]
    approach = wp.dot(n, contact_point(dq, s.pair[i], s.w[i], wp.vec3(0.0)))  # d(t) = d - t approach
    d = wp.dot(n, contact_point(q, s.pair[i], s.w[i], s.anchor[i]))
    if approach > 0.0 and d > 0.0:
        th = s.thickness[i]
        k = s.stiffness[i]
        if barrier(d, th, k, s.lam[i])[2] == 0.0:  # untouched: Newton did not see it, so its step holds only to
            # the ramp's end, where the barrier is at full curvature (th + lam / k: past it, the multiplier's force)
            wp.atomic_min(t, 0, (d - th - s.lam[i] / k) / approach)
        else:
            wp.atomic_min(t, 0, IMPACT * d / approach)


@wp.kernel
def _newton_size_kernel(q: wp.array[float], dq: wp.array[float], s: Slots, mass: wp.array[float],
                        fixed: wp.array[wp.int32], h2: float, cap: float, size: wp.array[float]):
    i = wp.tid()
    if s.active[i] == 0:
        return
    pair = s.pair[i]
    w = s.w[i]
    n = s.normal[i]
    e = barrier(wp.dot(n, contact_point(q, pair, w, s.anchor[i])), s.thickness[i], s.stiffness[i], s.lam[i])
    if e[2] == 0.0:
        return
    m = flexibility(pair, w, mass, fixed) * h2  # a force as the move it gives the pair's mass in a step
    df = e[2] * wp.abs(wp.dot(n, contact_point(dq, pair, w, wp.vec3(0.0)))) * m
    f = -e[1] * m
    wp.atomic_max(size, 0, df * cap / (cap + f))  # within tol of the inertial scale (cap) plus the force it holds


@wp.kernel
def _barrier_energy_kernel(q_cand: wp.array2d[float], s: Slots, eps: float, energy: wp.array[wp.float64]):
    j, i = wp.tid()
    e = wp.float64(0.0)
    if i < s.active.shape[0] and s.active[i] != 0:
        c = contact_point(q_cand[j], s.pair[i], s.w[i], s.anchor[i])
        n = s.normal[i]
        e = wp.float64(barrier(wp.dot(n, c), s.thickness[i], s.stiffness[i], s.lam[i])[0])
        e += wp.float64(s.mu[i] * s.rho[i] * slip_energy(wp.length(_slip(c, s.c0[i], n)), eps)[0])
    block_sum(energy, j, e)


@wp.kernel
def _carry_kernel(block: int, lam: wp.array[float], lam_key: wp.array[wp.int32], s: Slots):
    """A pair's multiplier from the last step's, kept by its feature's slots under its key; zero if new."""
    i = wp.tid()
    s.lam[i] = 0.0
    if s.active[i] == 0:
        return
    base = (i // block) * block
    for j in range(base, base + block):
        if lam_key[j] == s.key[i] and lam_key[j] >= 0:
            s.lam[i] = lam[j]


@wp.kernel
def _uzawa_kernel(q: wp.array[float], s: Slots, tol: float, lam: wp.array[float], lam_key: wp.array[wp.int32],
                  moved: wp.array[wp.int32]):
    """``lam <- max(0, f(d) - k dl / 2)`` into the store: ``f(d)`` the pair's force at ``q``, ``k dl / 2`` the force
    its barrier alone holds at the thickness (the ramp's middle), so the pair holds ``f`` there; ``moved`` if that
    moves its rest by more than ``tol``."""
    i = wp.tid()
    lam_key[i] = -1
    lam[i] = 0.0
    if s.active[i] == 0:
        return
    th, k = s.thickness[i], s.stiffness[i]
    d = wp.dot(s.normal[i], contact_point(q, s.pair[i], s.w[i], s.anchor[i]))
    f = -barrier(d, th, k, s.lam[i])[1]
    lam_key[i] = s.key[i]
    lam[i] = wp.max(0.0, f - 0.5 * k * SMOOTH * th)
    if wp.abs(lam[i] - s.lam[i]) > k * tol:
        moved[0] = 1


@wp.kernel
def _keep_kernel(s: Slots, lam: wp.array[float], lam_key: wp.array[wp.int32]):
    i = wp.tid()
    lam_key[i] = wp.where(s.active[i] != 0, s.key[i], -1)
    lam[i] = s.lam[i]


@wp.kernel
def _again_kernel(dq_max: wp.array[float], pick: wp.array[wp.int32], tol: float, converged: int,
                  left: wp.array[wp.int32], moved: wp.array[wp.int32], go: wp.array[wp.int32]):
    """``moved`` kept only if the solve converged (or need not) and the step has updates left (one spent); the solve
    goes on if so."""
    if moved[0] != 0 and (converged == 0 or (dq_max[0] <= tol and pick[0] != 0)) and left[0] > 0:
        left[0] = left[0] - 1
        go[0] = 1
    else:
        moved[0] = 0


@wp.kernel
def _apply_kernel(moved: wp.array[wp.int32], lam: wp.array[float], s: Slots):
    i = wp.tid()
    if moved[0] != 0 and s.active[i] != 0:
        s.lam[i] = lam[i]


@wp.kernel
def _compact_kernel(
    active: wp.array[wp.int32], offset: wp.array[wp.int32], src: wp.array(dtype=Any), dst: wp.array(dtype=Any)
):
    i = wp.tid()
    if active[i] != 0:
        dst[offset[i]] = src[i]
