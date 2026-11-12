"""Contact and Coulomb friction as energies in the step's objective, on primitive pairs.

A contact is a pair ``(vec4i nodes, vec4 w)``: ``C q = sum_k w_k x_k - anchor = x_a - x_b``; side ``a`` holds the
nodes of positive weight, side ``b`` those of negative weight (a point on an edge or a triangle, or a node) or the
anchor alone. ``-1`` marks an unused node. :class:`BarrierBlock` owns the contact law (a barrier on the gap, lagged
friction, each pair's stiffness) and the pair list; where contacts come from is :mod:`.collide`'s, whose source
appends them to the :class:`Slots`.
"""

from typing import Any

import numpy as np
import warp as wp
from newton import Model, State

from .linear import row_slot
from .state import node, padded, scatter_node, world_sum

wp.set_module_options({"enable_backward": False})

SMOOTH = wp.constant(0.02)  # the barrier's activation margin above the thickness (in thicknesses): a pair resting at
# its thickness already has curvature, so A sees it
LOAD = wp.constant(0.25)  # a node's pairs are stiffened until they hold its load at most this many thicknesses deep
GROW = wp.constant(2.0)  # a node's load scale grows at most this factor per step
DECAY = wp.constant(0.99)  # and relaxes by this factor per step, toward 1
MAX_SCALE = wp.constant(1.0e5)  # and stays below this (unbounded, it spreads pair to pair and overflows k)
CURVATURE = wp.constant(1.0e2)  # A holds a pair's normal curvature to at most this many k (its log's: d >= tau/10)
STIFFNESS = 1.0e3  # each pair's barrier stiffness, in units of its inertia 1 / (dt^2 C M^-1 C^T)
SIDES = 16  # a shape's pair keys per feature: a line contact's ends, or a pad's points (collide)
PAD_FLOOR = wp.constant(0.05)  # A holds a pad pair at least at this fraction of its depth (:func:`_majorizer`)
SLIP_SPEED = 0.01  # friction is smoothed below this slip speed [m/s]
LIFT = wp.constant(0.25)  # a shape pair's least gap at a step's start, from the barrier's zero (in thicknesses)
IMPACT = wp.constant(0.8)  # the line search's longest step: this fraction of the first pair's impact (IPC's)
NONE = wp.constant(wp.int64(9223372036854775807))  # an empty place's sort key


@wp.struct
class Slots:
    """Per contact: ``pair`` with weights ``w``, the normal from ``b`` to ``a``, the anchor, the start ``c0`` of
    ``C q`` (minus the anchor's motion), the thickness, friction coefficient, lagged normal force ``rho``,
    stiffness and multiplier ``lam``; ``own``, ``key``: its feature and the other one, which the list is sorted and
    its multipliers carried by; ``roll``, ``ta``, ``tb``: its rod edges' surface turn (:func:`roll_point`); ``world``:
    its world."""

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
def roll_point(q: wp.array[float], pair: wp.vec4i, w: wp.vec4, roll: wp.vec4, n: wp.vec3, ta: wp.vec3,
               tb: wp.vec3) -> wp.vec3:
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
def arm_assemble(vals: wp.array[wp.float64], indptr: wp.array[wp.int32], indices: wp.array[wp.int32],
                 fixed: wp.array[wp.int32], arm: Arm, body: int, r: wp.vec3, H: wp.mat33, n: int, Jn: wp.mat33):
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
                wp.atomic_add(vals, row_slot(indptr, indices, zm, arm.base[body] + m2),
                              wp.float64(wp.dot(arm_jacobian(m2, r), HP)))


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
    step's start and ``f0`` smoothed over :data:`SLIP_SPEED` ``dt``. Each pair's ``k`` is :data:`STIFFNESS` times its
    inertia ``1 / (dt^2 C M^-1 C^T)`` (a light sheet vertex and a rod node alike: Newton sees the contact coming),
    times its nodes' load scale (:func:`_lag_kernel`). The source appends every pair within its reach (no
    prediction) to one list; the solver caps each node's move per detection so no pair beyond it can close (Divide
    and Truncate's isotropic region: a pair's frozen gap is linear in ``q``, so the line search's energy keeps every
    found pair from passing). The list is sorted by its pairs' features after each detection (a pair found twice kept
    once), and each pair's multiplier carried by them into the next detection."""

    def __init__(self, model: Model, fixed: wp.array, source, dof_world: np.ndarray, stride: int):
        self.fixed, self.device, self.source = fixed, wp.get_device(model.device), source
        self.stiffness, self.slip_speed, self.slip = STIFFNESS, SLIP_SPEED, 0.0
        self.count = source.capacity()
        size = max(self.count, 1)
        self.s = _slots(size, self.device)
        nw, self.stride = int(dof_world.max(initial=0)) + 1, stride  # the worlds; per-world sums a[w stride + k]
        self.num_worlds = nw
        self.touching = wp.zeros(1, dtype=wp.int32, device=self.device)  # a pair has curvature at the last grad's q
        self.two_sided = wp.zeros(1, dtype=wp.int32, device=self.device)  # a two-sided pair has curvature there
        self.scale = wp.ones(max(model.particle_count, 1), dtype=float, device=self.device)  # nodes' load scales
        self.lam = wp.zeros(size, dtype=float, device=self.device)  # per pair, the multiplier kept for the next
        # detection and step; the list's sort (keys, places; twice the size: the radix sort's) and the last
        # detection's sorted keys and kept multipliers, which the next one's pairs look theirs up in
        self.sort_key = wp.full(2 * size, NONE, dtype=wp.int64, device=self.device)
        self.sort_at = wp.zeros(2 * size, dtype=wp.int32, device=self.device)
        self.held_key = wp.full(size, NONE, dtype=wp.int64, device=self.device)
        self.held_lam = wp.zeros(size, dtype=float, device=self.device)
        self.prims = getattr(source, "num_prims", 1)
        # per world: a multiplier moved its pair past tolerance; moved[nw]: in any world
        self.moved = wp.zeros(nw + 1, dtype=wp.int32, device=self.device)
        self.any_moved = self.moved[nw:]
        self.left = wp.zeros(nw, dtype=wp.int32, device=self.device)  # per world, the step's updates left
        self.unsettled = wp.zeros(nw, dtype=wp.int32, device=self.device)  # begin_step's
        self.arm = make_arm(model, getattr(source, "shape_key0", 0), 0, 0)  # no yielding bodies (the solver's to set)

    def _launch(self, kernel, inputs, dim=None):
        if self.count:
            wp.launch(kernel, dim=dim or self.count, inputs=inputs, device=self.device)

    def reset(self, mask: wp.array | None = None, node_world: wp.array | None = None) -> None:
        """No multipliers or load scales carried into the next step (in the worlds of ``mask``, ``node_world`` per
        DOF: a node ``n``'s at ``3 n``)."""
        if mask is None:
            self.lam.zero_()
            self.scale.fill_(1.0)
            return
        self._launch(_forget_kernel, [self.s.world, mask, self.lam])
        wp.launch(_unscale_kernel, dim=self.scale.shape[0], inputs=[node_world, mask, self.scale], device=self.device)

    def begin_step(self, state_in: State, dt: float, mass: wp.array, unsettled: wp.array) -> None:
        """The pairs near the step's start and their lagged normal forces; ``unsettled[w]``: world ``w``'s last step
        ended unconverged (its pairs' depths are its solve's error, not their load: no load scale grows)."""
        self.unsettled = unsettled
        self.slip = self.slip_speed * dt
        self.dt, self.mass = dt, mass
        self.arm.inv_h2 = 1.0 / (dt * dt)
        self.left.fill_(1)
        wp.launch(_decay_kernel, dim=self.scale.shape[0], inputs=[self.scale], device=self.device)
        self.detect(state_in, state_in.dismech.q, float(LIFT), start=True)

    def detect(self, state_in: State, q: wp.array, lift: float = 0.01, start: bool = False) -> None:
        """The pairs near ``q`` (the step's start or an iterate), their normals there and lagged normal forces; a
        shape's pair starts at least ``lift`` thicknesses from its barrier's zero (an iterate's only if undetected).
        At the step's ``start``, loaded pairs raise their nodes' load scales."""
        if not self.count:
            return
        self._launch(_hold_kernel, [self.sort_key, self.sort_at, self.lam, self.held_key, self.held_lam])
        self.source.fill(state_in, self.dt, q, self.s)
        self._launch(_sort_key_kernel, [self.s, self.prims, self.sort_key, self.sort_at])
        wp.utils.radix_sort_pairs(self.sort_key, self.sort_at, self.count)
        self._launch(_carry_kernel, [self.sort_key, self.sort_at, self.held_key, self.held_lam, self.s, self.lam])
        self._launch(_rebase_kernel, [q, state_in.dismech.q, self.s])
        self._launch(_lift_kernel, [q, lift, self.s, self.arm])
        self._launch(_lag_kernel, [self.stiffness / (self.dt * self.dt), self.dt, self.mass, self.fixed, self.scale,
                                   int(start), self.unsettled, self.s, self.arm])

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

    def update(self, q: wp.array, tol: float, dq_max: wp.array, pick: wp.array, go: wp.array, mask: wp.array) -> None:
        """At a converged ``q``: each pair's multiplier, the force its barrier holds there beyond what it holds at
        its thickness (kept by key for the next detection and step); if one moves its pair's rest by more than
        ``tol``, the solve converged (its last Newton step ``dq_max`` within ``tol`` and taken: not stalled) and the
        step has updates left, the pairs take them and ``go`` is set: the solve goes on. Per world, in the worlds of
        ``mask``; ``go[num_worlds]``: any world goes on."""
        self.moved.zero_()
        self._launch(_uzawa_kernel, [q, self.s, self.arm, mask, tol, self.lam, self.moved])
        wp.launch(_again_kernel, dim=self.num_worlds, inputs=[dq_max, pick, tol, self.num_worlds,
                  mask, self.left, self.moved, go], device=self.device)
        self._launch(_apply_kernel, [self.moved, self.lam, self.s])

    def end_step(self) -> None:
        """The multipliers the step solved with, kept for the next (an update it did not solve with would kick)."""
        self._launch(_keep_kernel, [self.s, self.lam])

    def assemble(self, q: wp.array, A) -> None:
        self._launch(_barrier_assemble_kernel, [q, self.s, self.arm, self.slip, self.fixed, 0, A.indptr, A.indices,
                                                A.vals])
        if self.arm.k:
            wp.launch(_arm_assemble_kernel, dim=(self.arm.M.shape[0], self.arm.k), inputs=[self.arm, A.indptr,
                      A.indices, A.vals], device=self.device)

    def two_sided_mul(self, q: wp.array, p: wp.array, y: wp.array) -> None:
        """``y += delta_A_2side p``: per two-sided pair, ``-(C_a + C_b)^T H (C_a p_a + C_b p_b)`` so ``A`` plus this
        is the exact majorized two-sided Hessian ``(C_a - C_b)^T H (C_a - C_b)``."""
        self._launch(_two_sided_kernel, [q, self.s, self.arm, self.slip, self.fixed, p, y])

    def impact(self, q: wp.array, dq: wp.array, t: wp.array) -> None:
        """``t = min(t, ...)`` the line search's longest step along ``-dq`` (each pair's gap ``d`` is linear in it, on
        the frozen normal): :data:`IMPACT` of the length at which a touching pair's gap would reach the barrier's
        zero (IPC's filter), or the length at which an untouched pair reaches its barrier's full curvature (or a
        sliding pair rewinding reaches its sticking ramp: the Newton step, built without that curvature, holds only
        that far)."""
        self._launch(_impact_kernel, [q, dq, self.s, self.arm, self.slip, t])

    def newton_size(self, q: wp.array, dq: wp.array, cap: float, size: wp.array) -> None:
        """``size = max(size, e''(d) |n . C dq| h^2 / m)`` over the pairs: the Newton step's change of each pair's
        normal force as the move it would give the pair's mass ``m = 1 / (C M^-1 C^T)`` in a step (its force error,
        comparable to the nodes' moves: the stopping test), relative to ``cap`` plus the force the pair holds (as a
        move too): a loaded pair's force converges to a relative tolerance."""
        self._launch(_newton_size_kernel, [q, dq, self.s, self.arm, self.mass, self.fixed, self.dt * self.dt, cap,
                                           size])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self.touching.zero_()
        self.two_sided.zero_()
        self._launch(_barrier_grad_kernel, [q, self.s, self.arm, self.slip, self.fixed, b, self.touching,
                                            self.two_sided])
        if self.arm.k:
            wp.launch(_arm_grad_kernel, dim=(self.arm.M.shape[0], self.arm.k), inputs=[q, self.arm, b],
                      device=self.device)

    def load(self, q: wp.array, load: wp.array, floor: wp.array, ulp: float) -> None:
        """The loads the stopping test's force balance is relative to, as magnitudes (frame-invariant): each pair's
        force (barrier and friction, at ``q``) added into ``load[3 n]`` on its nodes (opposed pairs add, as a
        pinch's), and on a yielding body its force and torque into ``load[k]``, ``load[k + 3]`` (``k`` its first DOF),
        with its inertia's; ``floor[k]``, ``floor[k + 3]``: the body's force and torque below float32's reach (``ulp``
        of a move)."""
        self._launch(_barrier_load_kernel, [q, self.s, self.arm, self.slip, load])
        if self.arm.k:
            wp.launch(_arm_load_kernel, dim=(self.arm.M.shape[0], self.arm.k // 6),
                      inputs=[q, self.arm, ulp, load, floor], device=self.device)

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(_barrier_energy_kernel, [q_cand, self.s, self.arm, self.stride, self.slip, energy],
                     dim=(q_cand.shape[0], padded(self.count)))
        if self.arm.k:
            wp.launch(_arm_energy_kernel, dim=(q_cand.shape[0], padded(self.arm.M.shape[0] * self.arm.k)),
                      inputs=[q_cand, self.arm, self.stride, energy], device=self.device)

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
def pad_energy(d: float, th: float, pad: wp.vec4) -> wp.vec3:
    """``(e, e', e'')`` at the gap ``d`` of a node in a hydroelastic pad (Elandt et al. 2019 with a linear pressure
    field: an elastic foundation, pressure ``kh`` times depth) ``h`` deep over its rigid core: ``c delta^p``, ``delta =
    th + h - d`` the node's surface's depth in the pad; per node its pressure integrated over its share of the partner
    (a rod's cylinder across its length: ``p = 5/2``; a shell's face: ``p = 2``)."""
    x = th + pad[2] - d
    if pad[0] == 0.0 or x <= 0.0:
        return wp.vec3(0.0)
    c, q = pad[0], pad[1]
    return wp.vec3(c * wp.pow(x, q), -q * c * wp.pow(x, q - 1.0), q * (q - 1.0) * c * wp.pow(x, q - 2.0))


@wp.func
def pad_damping(d: float, d0: float, th: float, pad: wp.vec4, damp: float) -> wp.vec3:
    """Hunt-Crossley's dissipation in the pad (Elandt et al. 2019, Eq. 3: ``p0 d v_n``), lagged as Drake's: the
    pressure's force at the step's start ``f0`` times ``d`` times the approach ``(d0 - d) / dt``, the potential
    ``damp (d0 - d)^2 / 2`` (``damp = d f0 / dt``) while in the pad."""
    if damp == 0.0 or th + pad[2] - d <= 0.0:
        return wp.vec3(0.0)
    y = d0 - d
    return wp.vec3(0.5 * damp * y * y, -damp * y, damp)


@wp.func
def pair_energy(d: float, s: Slots, i: int) -> wp.vec3:
    """The pair's barrier (its pad's core's, on a pad) plus its pad's energy and damping."""
    th = s.thickness[i]
    return barrier(d, th, s.stiffness[i], s.lam[i]) + pad_energy(d, th, s.pad[i]) + pad_damping(
        d, wp.dot(s.normal[i], s.c0[i]), th, s.pad[i], s.damp[i])


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
        s.c0[i] = (s.c0[i] - contact_point(q, s.pair[i], s.w[i], wp.vec3(0.0)) - slot_roll(q, s, i)
                   + contact_point(q0, s.pair[i], s.w[i], wp.vec3(0.0)) + slot_roll(q0, s, i))


@wp.kernel
def _lift_kernel(q: wp.array[float], lift: float, s: Slots, arm: Arm):
    """A shape's pair (all weights positive: against its anchor) starting closer than ``lift`` thicknesses, or
    inside (a node in a body, or a body moving onto it over the step): its anchor sinks along the normal so the
    barrier starts that close (finite), and the body pushes it out over the next steps. At an iterate a yielding body
    is posed at ``q`` for detection, then its anchor is pulled back to the step's start."""
    i = wp.tid()
    if s.active[i] == 0 or wp.min(s.w[i]) < 0.0:
        return
    n = s.normal[i]
    c = contact_point(q, s.pair[i], s.w[i], s.anchor[i])
    o = wp.dot(n, c) - lift * s.thickness[i]
    if o < 0.0:
        s.anchor[i] = s.anchor[i] + o * n
        s.c0[i] = s.c0[i] - o * n
    b = arm_body(arm, s, i)
    if b >= 0 and q.shape[0] > arm.start:  # (the step's start: the state's DOFs only, the bodies not yet moved)
        j = arm.base[b]
        dx = wp.vec3(q[j], q[j + 1], q[j + 2])
        dth = wp.vec3(q[j + 3], q[j + 4], q[j + 5])
        y = s.anchor[i] - arm.com[b] - dx
        r0 = (y - wp.cross(dth, y) + dth * wp.dot(dth, y)) / (1.0 + wp.dot(dth, dth))
        move = dx + wp.cross(dth, r0)
        s.anchor[i] = s.anchor[i] - move
        s.c0[i] = s.c0[i] + move


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
def _lag_kernel(c: float, dt: float, mass: wp.array[float], fixed: wp.array[wp.int32], scale: wp.array[float],
                grow: int, unsettled: wp.array[wp.int32], s: Slots, arm: Arm):
    """The pair's stiffness ``c / (C M^-1 C^T)`` (``c``: stiffness / dt^2) times its
    nodes' largest load scale, and its normal force at the step's start (in ``rho``): friction's ``lam``. At a step's
    start (``grow``), a pair compressed deeper than :data:`LOAD` thicknesses raises its nodes' scales (at most
    :data:`GROW` times), so a node's pairs come to hold its load that deep: if the last step converged."""
    i = wp.tid()
    if s.active[i] != 0:
        sc = float(1.0)  # the pair's nodes' largest load scale
        for j in range(4):
            if s.pair[i][j] >= 0:
                sc = wp.max(sc, scale[s.pair[i][j]])
        k = c / wp.max(flexibility(s.pair[i], s.w[i], mass, fixed) + arm_flexibility(arm, s, i), 1.0e-30) * sc
        x = s.thickness[i] - wp.dot(s.normal[i], s.c0[i]) + s.lam[i] / k  # its compression at the step's start
        # (the multiplier's force as the barrier's compression)
        if grow != 0 and unsettled[s.world[i]] == 0 and x > LOAD * s.thickness[i]:  # deeper than LOAD: stiffer pairs
            up = wp.min(sc * wp.min(GROW, x / (LOAD * s.thickness[i])), MAX_SCALE)
            for j in range(4):
                if s.pair[i][j] >= 0:
                    wp.atomic_max(scale, s.pair[i][j], up)
        s.stiffness[i] = k
        d0 = wp.dot(s.normal[i], s.c0[i])
        s.damp[i] = s.pad[i][3] * -pad_energy(d0, s.thickness[i], s.pad[i])[1] / dt
        s.rho[i] = -pair_energy(d0, s, i)[1]


@wp.func
def _slip(c: wp.vec3, c0: wp.vec3, n: wp.vec3) -> wp.vec3:
    """The pair's tangential motion since the step's start."""
    slip = c - c0
    return slip - wp.dot(n, slip) * n


@wp.func
def contact_force(c: wp.vec3, n: wp.vec3, c0: wp.vec3, th: float, k: float, lam: float, mu_rho: float, eps: float,
                  pad: wp.vec4, damp: float):
    """A pair's gradient with respect to ``C q = c`` (normal ``n``, start ``c0``, multiplier ``lam``, friction
    ``mu rho = mu_rho``), and its barrier's and friction's curvatures."""
    e = barrier(wp.dot(n, c), th, k, lam) + pad_energy(wp.dot(n, c), th, pad) + pad_damping(
        wp.dot(n, c), wp.dot(n, c0), th, pad, damp)
    slip = _slip(c, c0, n)
    friction = mu_rho * slip_energy(wp.length(slip), eps)[1]
    return e[1] * n + friction * slip, e[2], friction


@wp.func
def _barrier_force(c: wp.vec3, s: Slots, i: int, eps: float):
    return contact_force(c, s.normal[i], s.c0[i], s.thickness[i], s.stiffness[i], s.lam[i], s.mu[i] * s.rho[i], eps,
                         s.pad[i], s.damp[i])


@wp.func
def contact_hessian(c: wp.vec3, s: Slots, i: int, eps: float) -> wp.mat33:
    """The pair's exact Hessian with respect to ``C q = c`` (:func:`_barrier_assemble_kernel` majorizes its
    friction's along the slip)."""
    n = s.normal[i]
    nn = wp.outer(n, n)
    g, h, fr = _barrier_force(c, s, i, eps)
    return h * nn + fr * (wp.identity(3, float) - nn) + _along_slip(c, s, i, eps)


@wp.func
def _along_slip(c: wp.vec3, s: Slots, i: int, eps: float) -> wp.mat33:
    """Friction's exact curvature along the slip less its majorizer's: ``mu rho (f1' - f1 / y) t t^T``."""
    slip = _slip(c, s.c0[i], s.normal[i])
    y = wp.length(slip)
    if y == 0.0:
        return wp.mat33()
    t = slip / y
    return -s.mu[i] * s.rho[i] * wp.where(y < eps, y / (eps * eps), 1.0 / y) * wp.outer(t, t)


@wp.kernel
def _barrier_grad_kernel(q: wp.array[float], s: Slots, arm: Arm, eps: float, fixed: wp.array[wp.int32],
                         b: wp.array[wp.float64], touching: wp.array[wp.int32], two_sided: wp.array[wp.int32]):
    i = wp.tid()
    if s.active[i] == 0:
        return
    g, h, fr = _barrier_force(pair_point(q, s, i, arm, s.anchor[i]), s, i, eps)
    if h > 0.0 or fr > 0.0:
        touching[0] = 1
        if wp.min(s.w[i]) < 0.0:
            two_sided[0] = 1
    for j in range(4):
        if s.pair[i][j] >= 0:
            scatter_node(b, fixed, s.pair[i][j], wp.transpose(slot_map(s, i, j)) * g)
    body = arm_body(arm, s, i)
    if body >= 0:
        arm_scatter(b, arm, body, s.anchor[i] - arm.com[body], g)


@wp.kernel
def _barrier_load_kernel(q: wp.array[float], s: Slots, arm: Arm, eps: float, load: wp.array[float]):
    i = wp.tid()
    if s.active[i] == 0:
        return
    g, h, fr = _barrier_force(pair_point(q, s, i, arm, s.anchor[i]), s, i, eps)
    for j in range(4):
        n = s.pair[i][j]
        if n >= 0:
            wp.atomic_add(load, 3 * n, wp.length(wp.transpose(slot_map(s, i, j)) * g))
    body = arm_body(arm, s, i)
    if body >= 0:
        k = arm.base[body]
        wp.atomic_add(load, k, wp.length(g))
        wp.atomic_add(load, k + 3, wp.length(wp.cross(s.anchor[i] - arm.com[body], g)))


@wp.func
def _majorizer(c: wp.vec3, s: Slots, i: int, eps: float) -> wp.mat33:
    """The pair's Hessian as ``A`` holds it (with respect to ``C q = c``): the barrier's curvature capped, friction's
    majorized in the sticking ramp (``f1 / y``) and exact when sliding (``0`` along the slip, ``f1 / y`` across it)."""
    n = s.normal[i]
    g, h, fr = _barrier_force(c, s, i, eps)
    h = wp.min(h, CURVATURE * s.stiffness[i])  # the log's 1/d^2 bounded (float32 cuDSS); the energy stays exact
    pad = s.pad[i]
    x = s.thickness[i] + pad[2] - wp.dot(n, c)  # in its pad: a rod's curvature (delta^1/2) is 0 at the surface,
    if pad[0] > 0.0 and x >= 0.0:  # so A holds at least the pad's at PAD_FLOOR of it deep
        h = wp.max(h, pad_energy(s.thickness[i] + pad[2] - wp.max(x, PAD_FLOOR * pad[2]), s.thickness[i], pad)[2])
    slip = _slip(c, s.c0[i], n)
    y = wp.length(slip)
    if y >= eps:
        b = wp.cross(n, slip / y)
        return h * wp.outer(n, n) + fr * wp.outer(b, b)
    return h * wp.outer(n, n) + fr * (wp.identity(3, float) - wp.outer(n, n))


@wp.func
def _mat33d(M: wp.mat33) -> wp.mat33d:
    return wp.mat33d(
        wp.float64(M[0, 0]), wp.float64(M[0, 1]), wp.float64(M[0, 2]),
        wp.float64(M[1, 0]), wp.float64(M[1, 1]), wp.float64(M[1, 2]),
        wp.float64(M[2, 0]), wp.float64(M[2, 1]), wp.float64(M[2, 2]),
    )


@wp.kernel
def _two_sided_kernel(q: wp.array[float], s: Slots, arm: Arm, eps: float, fixed: wp.array[wp.int32],
                      p: wp.array[wp.float64], y: wp.array[wp.float64]):
    i = wp.tid()
    if s.active[i] == 0 or wp.min(s.w[i]) >= 0.0:
        return
    H = _majorizer(pair_point(q, s, i, arm, s.anchor[i]), s, i, eps)
    if H[0, 0] == 0.0 and H[1, 1] == 0.0 and H[2, 2] == 0.0:
        return
    pair = s.pair[i]
    w = s.w[i]
    u = wp.vec3d(wp.float64(0.0))
    for a in range(4):
        na = pair[a]
        if na >= 0:
            pa = wp.vec3d(wp.float64(0.0))
            for c in range(3):
                if fixed[3 * na + c] == 0:
                    pa[c] = p[3 * na + c]
            u += wp.where(w[a] < 0.0, wp.float64(-1.0), wp.float64(1.0)) * (_mat33d(slot_map(s, i, a)) * pa)
    g = -(_mat33d(H) * u)
    for a in range(4):
        na = pair[a]
        if na >= 0:
            v = wp.where(w[a] < 0.0, wp.float64(-1.0), wp.float64(1.0)) * (wp.transpose(_mat33d(slot_map(s, i, a))) * g)
            for c in range(3):
                if fixed[3 * na + c] == 0:
                    wp.atomic_add(y, 3 * na + c, v[c])


@wp.kernel
def _barrier_assemble_kernel(
    q: wp.array[float],
    s: Slots,
    arm: Arm,
    eps: float,
    fixed: wp.array[wp.int32],
    exact: int,
    indptr: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    """``C^T (e'' n n^T + mu lam f1/y (I - n n^T)) C`` into the upper CSR, majorized as :func:`_assemble_kernel`'s;
    ``exact``: a one-sided pair's exact Hessian (:func:`contact_hessian`, uncapped: a float64 factorization's)."""
    i = wp.tid()
    if s.active[i] == 0:
        return
    pair = s.pair[i]
    w = s.w[i]
    H = _majorizer(pair_point(q, s, i, arm, s.anchor[i]), s, i, eps)
    f = float(1.0)
    for j in range(4):
        if pair[j] >= 0 and w[j] < 0.0:
            f = 2.0
    if exact != 0 and f == 1.0:
        H = contact_hessian(pair_point(q, s, i, arm, s.anchor[i]), s, i, eps)
    Hd = _mat33d(H)
    for a in range(4):
        na = pair[a]
        for bb in range(4):
            nb = pair[bb]
            if na >= 0 and nb >= na and w[a] * w[bb] > 0.0:
                B = wp.float64(f) * wp.transpose(_mat33d(slot_map(s, i, a))) * Hd * _mat33d(slot_map(s, i, bb))
                for c in range(3):
                    for c2 in range(3):
                        if (nb > na or c2 >= c) and fixed[3 * na + c] == 0 and fixed[3 * nb + c2] == 0:
                            slot = row_slot(indptr, indices, 3 * na + c, 3 * nb + c2)
                            if slot >= 0:
                                wp.atomic_add(vals, slot, B[c, c2])
    body = arm_body(arm, s, i)
    if body >= 0:  # the anchor's DOFs
        r = s.anchor[i] - arm.com[body]
        for a in range(4):
            if pair[a] >= 0:
                arm_assemble(vals, indptr, indices, fixed, arm, body, r, H, pair[a], slot_map(s, i, a))
        arm_assemble(vals, indptr, indices, fixed, arm, body, r, H, -1, H)


@wp.kernel
def _impact_kernel(q: wp.array[float], dq: wp.array[float], s: Slots, arm: Arm, eps: float, t: wp.array[float]):
    i = wp.tid()
    if s.active[i] == 0:
        return
    n = s.normal[i]
    dq_c = pair_point(dq, s, i, arm, wp.vec3(0.0))
    approach = wp.dot(n, dq_c)  # d(t) = d - t approach
    c = pair_point(q, s, i, arm, s.anchor[i])
    d = wp.dot(n, c)
    if approach > 0.0 and d > 0.0:
        th = s.thickness[i]
        k = s.stiffness[i]
        if s.pad[i][0] > 0.0 and d > th + s.pad[i][2]:  # outside its pad: Newton did not see it, so its step holds
            # only to the depth A holds it from (PAD_FLOOR h: a point just outside does not stall the step)
            wp.atomic_min(t, s.world[i], (d - th - (1.0 - PAD_FLOOR) * s.pad[i][2]) / approach)
        elif barrier(d, th, k, s.lam[i])[2] == 0.0:  # untouched: Newton did not see it, so its step holds only to
            # the ramp's end, where the barrier is at full curvature (th + lam / k: past it, the multiplier's force)
            wp.atomic_min(t, s.world[i], (d - th - s.lam[i] / k) / approach)
        else:
            wp.atomic_min(t, s.world[i], IMPACT * d / approach)
    if s.mu[i] * s.rho[i] > 0.0:
        slip = _slip(c, s.c0[i], n)
        y = wp.length(slip)
        if y >= eps:  # sliding: Newton did not see the sticking ramp, so a step back holds only to its middle
            rewind = wp.dot(slip / y, _slip(dq_c, wp.vec3(0.0), n))
            if rewind > 0.0:
                wp.atomic_min(t, s.world[i], (y - 0.5 * eps) / rewind)


@wp.kernel
def _newton_size_kernel(q: wp.array[float], dq: wp.array[float], s: Slots, arm: Arm, mass: wp.array[float],
                        fixed: wp.array[wp.int32], h2: float, cap: float, size: wp.array[float]):
    i = wp.tid()
    if s.active[i] == 0:
        return
    pair = s.pair[i]
    w = s.w[i]
    n = s.normal[i]
    e = pair_energy(wp.dot(n, pair_point(q, s, i, arm, s.anchor[i])), s, i)
    if e[2] == 0.0:
        return
    m = (flexibility(pair, w, mass, fixed) + arm_flexibility(arm, s, i)) * h2  # a force as the move it gives the
    # pair's mass in a step
    df = e[2] * wp.abs(wp.dot(n, pair_point(dq, s, i, arm, wp.vec3(0.0)))) * m
    f = -e[1] * m
    wp.atomic_max(size, s.world[i], df * cap / (cap + f))  # within tol of the inertial scale (cap) plus its force


@wp.kernel
def _barrier_energy_kernel(q_cand: wp.array2d[float], s: Slots, arm: Arm, stride: int, eps: float,
                           energy: wp.array[wp.float64]):
    j, i = wp.tid()
    e = wp.float64(0.0)
    w = int(-1)
    if i < s.active.shape[0]:
        w = s.world[i]
    if i < s.active.shape[0] and s.active[i] != 0:
        c = pair_point(q_cand[j], s, i, arm, s.anchor[i])
        n = s.normal[i]
        e = wp.float64(pair_energy(wp.dot(n, c), s, i)[0])
        e += wp.float64(s.mu[i] * s.rho[i] * slip_energy(wp.length(_slip(c, s.c0[i], n)), eps)[0])
    world_sum(energy, w, j, stride, e)


@wp.kernel
def _hold_kernel(sort_key: wp.array[wp.int64], sort_at: wp.array[wp.int32], lam: wp.array[float],
                 held_key: wp.array[wp.int64], held_lam: wp.array[float]):
    """Before a detection: the list's sorted keys and its pairs' kept multipliers, for the next list to look up."""
    j = wp.tid()
    held_key[j] = sort_key[j]
    held_lam[j] = wp.where(sort_key[j] != NONE, lam[sort_at[j]], 0.0)


@wp.kernel
def _sort_key_kernel(s: Slots, prims: int, sort_key: wp.array[wp.int64], sort_at: wp.array[wp.int32]):
    """Pair ``i``'s key: its features ``own * prims + key`` (none: :data:`NONE`, sorted last)."""
    i = wp.tid()
    sort_key[i] = NONE
    if s.active[i] != 0:
        sort_key[i] = wp.int64(s.own[i]) * wp.int64(prims) + wp.int64(s.key[i])
    sort_at[i] = i


@wp.kernel
def _carry_kernel(sort_key: wp.array[wp.int64], sort_at: wp.array[wp.int32], held_key: wp.array[wp.int64],
                  held_lam: wp.array[float], s: Slots, lam: wp.array[float]):
    """Per sorted place: a pair found twice (through two triangles) kept once; its multiplier the last detection's
    pair's of the same features (binary search: the first, the kept one), else zero."""
    j = wp.tid()
    key = sort_key[j]
    if key == NONE:
        return
    i = sort_at[j]
    s.lam[i] = 0.0
    lam[i] = 0.0
    if j > 0:
        if sort_key[j - 1] == key:
            s.active[i] = 0
            return
    lo = int(0)
    hi = held_key.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if held_key[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < held_key.shape[0]:
        if held_key[lo] == key:
            s.lam[i] = held_lam[lo]
            lam[i] = held_lam[lo]


@wp.kernel
def _uzawa_kernel(q: wp.array[float], s: Slots, arm: Arm, mask: wp.array[wp.int32], tol: float,
                  lam: wp.array[float], moved: wp.array[wp.int32]):
    """``lam <- max(0, lam + k (th - d))`` into the store: the quadratic's force at ``q`` less the ``k dl / 2`` its
    barrier alone holds at the thickness (the ramp's middle), so the pair holds it there (the augmented Lagrangian's
    update); the log's force below ``th / 2`` is not taken (at an unconverged iterate it diverges, and through the
    load scale stiffens every later step). ``moved`` (its world's) if that moves its rest by more than ``tol``. Only
    in the worlds of ``mask``."""
    i = wp.tid()
    if mask[s.world[i]] == 0:
        return
    lam[i] = 0.0
    if s.active[i] == 0:
        return
    th, k = s.thickness[i], s.stiffness[i]
    d = wp.dot(s.normal[i], pair_point(q, s, i, arm, s.anchor[i]))
    lam[i] = wp.max(0.0, s.lam[i] + k * (th - d))
    if wp.abs(lam[i] - s.lam[i]) > k * tol:
        moved[s.world[i]] = 1


@wp.kernel
def _forget_kernel(world: wp.array[wp.int32], mask: wp.array[wp.bool], lam: wp.array[float]):
    i = wp.tid()
    if mask[world[i]]:
        lam[i] = 0.0


@wp.kernel
def _unscale_kernel(node_world: wp.array[wp.int32], mask: wp.array[wp.bool], scale: wp.array[float]):
    n = wp.tid()
    if 3 * n < node_world.shape[0] and mask[node_world[3 * n]]:
        scale[n] = 1.0


@wp.kernel
def _keep_kernel(s: Slots, lam: wp.array[float]):
    i = wp.tid()
    lam[i] = wp.where(s.active[i] != 0, s.lam[i], 0.0)


@wp.kernel
def _again_kernel(dq_max: wp.array[float], pick: wp.array[wp.int32], tol: float, count: int,
                  mask: wp.array[wp.int32], left: wp.array[wp.int32], moved: wp.array[wp.int32],
                  go: wp.array[wp.int32]):
    """Per world of ``mask``: ``moved`` kept only if its solve converged and its step has updates left (one spent);
    its solve goes on if so. ``moved[count]`` (zero before), ``go[count]``: in any world."""
    w = wp.tid()
    if mask[w] != 0 and moved[w] != 0 and dq_max[w] <= tol and pick[w] != 0 and left[w] > 0:
        left[w] = left[w] - 1
        go[w] = 1
        moved[count] = 1
        go[count] = 1
    else:
        moved[w] = 0


@wp.kernel
def _apply_kernel(moved: wp.array[wp.int32], lam: wp.array[float], s: Slots):
    i = wp.tid()
    if moved[s.world[i]] != 0 and s.active[i] != 0:
        s.lam[i] = lam[i]


@wp.kernel(module="unique")  # generic: an overload recompiles this, not the module
def _compact_kernel(
    active: wp.array[wp.int32], offset: wp.array[wp.int32], src: wp.array(dtype=Any), dst: wp.array(dtype=Any)
):
    i = wp.tid()
    if active[i] != 0:
        dst[offset[i]] = src[i]


@wp.func
def _arm_inertia(q: wp.array[float], arm: Arm, w: int, m: int) -> float:
    """``(M (z - pred))_m / h^2`` of world ``w``: ``M`` block-diagonal, a body's 6 DOFs (:class:`.coupling.ProxyArm`)."""
    v = float(0.0)
    for m2 in range(6 * (m // 6), 6 * (m // 6) + 6):
        j = w * arm.k + m2
        v += arm.M[w, m, m2] * (q[arm.start + j] - arm.pred[j])
    return v * arm.inv_h2


@wp.kernel
def _arm_grad_kernel(q: wp.array[float], arm: Arm, b: wp.array[wp.float64]):
    w, m = wp.tid()
    wp.atomic_add(b, arm.start + w * arm.k + m, wp.float64(_arm_inertia(q, arm, w, m)))


@wp.kernel
def _arm_load_kernel(q: wp.array[float], arm: Arm, ulp: float, load: wp.array[float], floor: wp.array[float]):
    w, p = wp.tid()
    m = 6 * p
    i = arm.start + w * arm.k + m
    f = wp.vec3(_arm_inertia(q, arm, w, m), _arm_inertia(q, arm, w, m + 1), _arm_inertia(q, arm, w, m + 2))
    t = wp.vec3(_arm_inertia(q, arm, w, m + 3), _arm_inertia(q, arm, w, m + 4), _arm_inertia(q, arm, w, m + 5))
    wp.atomic_add(load, i, wp.length(f))
    wp.atomic_add(load, i + 3, wp.length(t))
    floor[i] = arm.M[w, m, m] * arm.inv_h2 * ulp
    floor[i + 3] = (arm.M[w, m + 3, m + 3] + arm.M[w, m + 4, m + 4] + arm.M[w, m + 5, m + 5]) / 3.0 * arm.inv_h2 * ulp


@wp.kernel
def _arm_energy_kernel(q_cand: wp.array2d[float], arm: Arm, stride: int, energy: wp.array[wp.float64]):
    j, i = wp.tid()  # a thread per arm DOF
    w = i // wp.max(arm.k, 1)
    e = float(0.0)
    if w < arm.M.shape[0]:
        e = 0.5 * (q_cand[j, arm.start + i] - arm.pred[i]) * _arm_inertia(q_cand[j], arm, w, i - w * arm.k)
    world_sum(energy, wp.where(w < arm.M.shape[0], w, -1), j, stride, wp.float64(e))


@wp.kernel
def _arm_assemble_kernel(arm: Arm, indptr: wp.array[wp.int32], indices: wp.array[wp.int32],
                         vals: wp.array[wp.float64]):
    w, m = wp.tid()
    zm = arm.start + w * arm.k
    for m2 in range(m, 6 * (m // 6) + 6):  # its body's block
        wp.atomic_add(vals, row_slot(indptr, indices, zm + m, zm + m2), wp.float64(arm.M[w, m, m2] * arm.inv_h2))
