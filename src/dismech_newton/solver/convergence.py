"""Per-world net force balance, float32 ULP floors, and iteration control kernels."""

import warp as wp

from ..constants import EPS32
from ..dofs import Dofs, _external
from ..worlds import Worlds
from .linesearch import SOLVER_MODULE

_NET, _LOAD, _FLOOR, _ANCHOR2, _N = 0, 3, 4, 5, 6  # a world's net force (3), load, floor, anchor floor²


@wp.kernel(module=SOLVER_MODULE)
def _hinge_size_kernel(dq: wp.array[float], offset: int, thickness: wp.array[float], wd: Worlds,
                       dq_max: wp.array[float]):
    """A midedge angle's step as the move of its offset midpoint ``h/2 |d theta|`` (Chen et al. 2026, Sec. 7.3)."""
    h = wp.tid()
    wp.atomic_max(dq_max, wd.of[offset + h], 0.5 * thickness[h] * wp.abs(dq[offset + h]))


@wp.kernel(module=SOLVER_MODULE)
def _unbalanced_kernel(q: wp.array[float], q_pred: wp.array[float], f: Dofs, b: wp.array[wp.float64],
                       load: wp.array[float], anchor: wp.array[float], wd: Worlds, ulp: float,
                       net: wp.array[wp.float64]):
    """The nodes' force balance at ``q``, summed per world into ``net`` (net force, its loads, float32's floor):
    ``b``, the step's gradient, is the unbalanced force; the loads are magnitudes: its contacts' forces (``load``,
    opposed pairs adding), its inertia and external force; the elastic forces are internal: they carry the load, they
    are not it. Summed, not node by node: a stiff rod's nodes resolve their elastic forces only to its stiffness times
    float32's resolution of a coordinate, but those cancel in the sum (each force is added once with each sign); the
    net, the momentum's balance, does not, nor does a fixed DOF's reaction: it is resolved only to its coupling to a
    node (``anchor``, the assembly's) times the node's ulp, summed in quadrature (independent round-offs: 2x the
    net's change under random 1-ulp moves on a clamped woven sheet).
    Midedge angles and twists: internal, left to the step test."""
    n = wp.tid()
    i = 3 * n
    if f.fixed[i] != 0 and f.fixed[i + 1] != 0 and f.fixed[i + 2] != 0:
        return
    w = wd.of[i]
    r = wp.vec3(q[i] - q_pred[i], q[i + 1] - q_pred[i + 1], q[i + 2] - q_pred[i + 2])
    ext = wp.vec3(_external(f, i), _external(f, i + 1), _external(f, i + 2))
    k = f.mass[i] * f.inv_dt2
    o = _N * w
    for c in range(3):
        wp.atomic_add(net, o + _NET + c, b[i + c])
    x = wp.vec3(q[i], q[i + 1], q[i + 2])
    wp.atomic_add(net, o + _LOAD, wp.float64(load[i] + k * wp.length(r) + wp.length(ext)))
    a = wp.vec3(anchor[i], anchor[i + 1], anchor[i + 2])
    s = EPS32 * wp.length(a) * wp.length(x)
    wp.atomic_add(net, o + _ANCHOR2, wp.float64(s * s))
    wp.atomic_add(net, o + _FLOOR, wp.float64(k * ulp * (wp.length(x) + wp.length(r))))


@wp.kernel(module=SOLVER_MODULE)
def _arm_unbalanced_kernel(b: wp.array[wp.float64], load: wp.array[float], floor: wp.array[float], start: int,
                           k: int, tolerance: float, unbalanced: wp.array[float]):
    """A proxy body's force and torque balance: ``|b|`` within ``tolerance`` of its contacts' and inertia's, or below
    float32's reach."""
    w, p = wp.tid()
    i = start + w * k + 6 * p
    for o in range(0, 6, 3):
        u = wp.length(wp.vec3(float(b[i + o]), float(b[i + o + 1]), float(b[i + o + 2])))
        wp.atomic_max(unbalanced, w, u / (tolerance * load[i + o] + floor[i + o] + 1.0e-30))


@wp.kernel(module=SOLVER_MODULE)
def _net_kernel(net: wp.array[wp.float64], tolerance: float, unbalanced: wp.array[float]):
    """Per world, the nodes' net force against ``tolerance`` of their loads (plus float32's floor)."""
    o = _N * wp.tid()
    u = wp.sqrt(net[o] * net[o] + net[o + 1] * net[o + 1] + net[o + 2] * net[o + 2])
    tol = wp.float64(tolerance) * net[o + _LOAD] + net[o + _FLOOR] + wp.sqrt(net[o + _ANCHOR2]) + wp.float64(1.0e-30)
    unbalanced[wp.tid()] = wp.max(unbalanced[wp.tid()], float(u / tol))


@wp.kernel(module=SOLVER_MODULE)
def _go_kernel(dq_max: wp.array[float], unbalanced: wp.array[float], pick: wp.array[wp.int32],
               moved: wp.array[wp.int32], failed: wp.array[wp.int32], tol: float, wd: Worlds,
               go: wp.array[wp.int32], stalled: wp.array[wp.int32], settle: wp.array[wp.int32]):
    """Per iterating world: iterate on while the Newton step moves a node more than ``tol`` and the line search took a
    step, or the pairs were just found again (the objective changed); a step that finds no decrease short of that has
    stalled. ``settle``: the worlds that stopped at this iterate (``[count]``: any; ``go[count]``: any goes; both
    zeroed before)."""
    w = wp.tid()
    n = wd.count
    settle[w] = 0
    if wd.active[w] != 0:
        far = dq_max[w] > tol or unbalanced[w] > 1.0  # the step still moves, or the forces do not balance
        go[w] = wp.where(far and (pick[w] != 0 or moved[w] != 0) and failed[w] == 0, 1, 0)
        if far and pick[w] == 0 and moved[w] == 0:
            stalled[w] = 1
        if go[w] == 0:
            settle[w] = 1
            settle[n] = 1
    if go[w] != 0:
        go[n] = 1


@wp.kernel(module=SOLVER_MODULE)
def _count_kernel(go: wp.array[wp.int32], stalled: wp.array[wp.int32], failed: wp.array[wp.int32],
                  counts: wp.array[wp.int32]):
    """The step's end: the worlds unconverged (their iterations spent) or stalled, counted."""
    w = wp.tid()
    if go[w] != 0:
        wp.atomic_add(counts, 0, 1)
    if stalled[w] != 0:
        wp.atomic_add(counts, 1, 1)
    if failed[w] != 0:
        wp.atomic_add(counts, 2, 1)
    stalled[w] = 0
    failed[w] = 0


@wp.kernel(module=SOLVER_MODULE)
def _activate_kernel(go: wp.array[wp.int32], active: wp.array[wp.int32]):
    """The worlds that iterate: those not converged."""
    w = wp.tid()
    active[w] = go[w]


@wp.kernel(module=SOLVER_MODULE)
def _nonfinite_kernel(dq: wp.array[float], wd: Worlds, q: wp.array[float], failed: wp.array[wp.int32]):
    """A non-finite Newton step (a singular or overflowed solve) is taken, not skipped: the line search would pick no
    candidate and the step read as converged, the state silently frozen (its world's)."""
    i = wp.tid()
    w = wd.of[i]
    if wd.active[w] != 0 and not wp.isfinite(dq[i]):
        q[i] = dq[i]
        failed[w] = 1


@wp.kernel(module=SOLVER_MODULE)
def _refresh_kernel(go: wp.array[wp.int32], touching: wp.array[wp.int32], exact: int, refresh: wp.array[wp.int32]):
    refresh[0] = wp.where(go[0] != 0 and (touching[0] != 0 or exact != 0), 1, 0)
