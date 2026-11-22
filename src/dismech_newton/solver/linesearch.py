"""Line search, Armijo backtracking, candidate generation, and node/proxy move caps."""

from typing import Any

import warp as wp

from ..constants import ARMIJO_C1 as _C1
from ..constants import FWD_MODULE_OPTS
from ..coupling import _turn
from ..dofs import Dofs, inertia_energy, inertia_grad, node
from ..worlds import Worlds, world_sum

SOLVER_MODULE = wp.get_module("dismech_newton.solver._kernels")
SOLVER_MODULE.options.update(FWD_MODULE_OPTS)


@wp.func
def _step_length(k: int) -> float:
    """Candidate 0 is the current iterate, and ``k > 0`` the backtracking steps ``2^(1-k)``."""
    if k == 0:
        return 0.0
    return wp.pow(0.5, float(k - 1))


@wp.kernel(module="unique", module_options=FWD_MODULE_OPTS)  # generic: an overload recompiles this, not the module
def _copy_if_kernel(valid: wp.array[wp.int32], src: wp.array(dtype=Any), dst: wp.array(dtype=Any)):
    i = wp.tid()
    if valid[0] != 0:
        dst[i] = src[i]


@wp.func
def _armijo(energy: wp.array[wp.float64], slope: wp.array[wp.float64], w: int, num_candidates: int,
            noise_tol: float) -> int:
    """World ``w``'s: Newton's full step if its expected decrease is below the objective's float32 rounding (no
    candidate can be told apart). Else the longest step with sufficient decrease; else the candidate with the lowest
    objective if it rises by no more than that rounding, or ``0``: no step (a stall)."""
    o = w * num_candidates
    f0 = energy[o]
    s = slope[w]
    noise = wp.float64(noise_tol) * wp.abs(f0)
    if s >= wp.float64(0.0) and wp.float64(0.5) * s <= noise and energy[o + 1] <= f0 + noise and wp.isfinite(
            energy[o + 1]):
        return 1  # the Newton step's decrease is below the objective's rounding: its full step is taken
    best = int(0)
    for k in range(1, num_candidates):
        if wp.isfinite(energy[o + k]) and energy[o + k] <= f0 - wp.float64(_C1 * _step_length(k)) * s:
            return k
        if wp.isfinite(energy[o + k]) and (best == 0 or energy[o + k] < energy[o + best]):
            best = k
    if best > 0 and energy[o + best] <= f0 + noise:
        return best
    return 0


@wp.kernel(module=SOLVER_MODULE)
def _dof_kernel(
    q: wp.array[float],
    q_pred: wp.array[float],
    f: Dofs,
    b: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
):
    """``b += M (q - q_pred) / dt^2 - f_ext`` (``0`` on fixed DOFs); zero the line search's sums."""
    i = wp.tid()
    if i < energy.shape[0]:
        energy[i] = wp.float64(0.0)
    if i < slope.shape[0]:
        slope[i] = wp.float64(0.0)
    if i >= b.shape[0]:
        return
    if f.fixed[i] != 0:
        b[i] = wp.float64(0.0)
    else:
        b[i] = b[i] + wp.float64(inertia_grad(f, i, q[i] - q_pred[i]))


@wp.kernel(module=SOLVER_MODULE)
def _candidates_kernel(
    q: wp.array[float],
    dq: wp.array[float],
    t: wp.array[float],
    q_pred: wp.array[float],
    f: Dofs,
    q0: wp.array[float],
    cap: float,
    limit: wp.array[float],
    arm_start: int,
    arm_radius: wp.array[float],
    wd: Worlds,
    q_cand: wp.array2d[float],
    b: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
):
    """``q_cand[k] = q - t_k t dq`` (``t``: the longest step), the DOF part of each candidate's objective,
    ``slope = b . (q - q_cand[1])``, then ``b = 0``; a node's move from ``q0`` capped to ``limit`` (a midedge angle's
    past its barrier is the energy's to reject), a proxy body's so its farthest point moves at most ``cap``. Launched
    over ``(k, padded(n))``."""
    k, i = wp.tid()
    e = wp.float64(0.0)
    s = wp.float64(0.0)
    w = int(-1)
    if i < q.shape[0]:
        w = wd.of[i]
        tw = t[w]
        d = dq[i] * tw
        step = d
        qk = q[i]
        if k > 0:  # (candidate 0 the iterate itself, whatever dq is)
            qk = q[i] - _step_length(k) * d
        if cap > 0.0 and i < f.num_node_dofs and f.fixed[i] == 0:  # the node's move from the start, at most cap
            n = i // 3
            cap = wp.min(cap, limit[n])
            move = wp.vec3(0.0)
            for c in range(3):
                j = 3 * n + c
                move[c] = q[j] - _step_length(k) * (dq[j] * tw) - q0[j]
            length = wp.length(move)
            if length > cap:
                qk = q0[i] + move[i - 3 * n] * (cap / length)
                step = q[i] - qk
        elif cap > 0.0 and i >= arm_start and arm_radius.shape[0] > 0:
            j0 = arm_start + 6 * ((i - arm_start) // 6)
            dx, dth = wp.vec3(0.0), wp.vec3(0.0)
            for c in range(3):
                dx[c] = q[j0 + c] - _step_length(k) * (dq[j0 + c] * tw) - q0[j0 + c]
                dth[c] = q[j0 + 3 + c] - _step_length(k) * (dq[j0 + 3 + c] * tw) - q0[j0 + 3 + c]
            sweep = wp.length(dx) + wp.length(dth) * arm_radius[(i - arm_start) // 6]
            if sweep > cap:
                qk = q0[i] + (qk - q0[i]) * (cap / sweep)
                step = q[i] - qk
        q_cand[k, i] = qk
        if f.fixed[i] == 0:
            e = wp.float64(inertia_energy(f, i, qk - q_pred[i]))
            if k == 1:
                s = b[i] * wp.float64(step)
                b[i] = wp.float64(0.0)
    world_sum(energy, w, k, wd.stride, e)
    world_sum(slope, w, 0, 1, s)


@wp.kernel(module=SOLVER_MODULE)
def _moved_kernel(q: wp.array[float], q0: wp.array[float], dq: wp.array[float], limit: wp.array[float], wd: Worlds,
                  moved: wp.array[wp.int32], dq_max: wp.array[float]):
    n = wp.tid()
    w = wd.of[3 * n]
    if wp.length(node(q, n) - node(q0, n)) > 0.5 * limit[n]:
        moved[w] = 1
        moved[wd.count] = 1
    wp.atomic_max(dq_max, w, wp.length(node(dq, n)))


@wp.func
def _arm_sweep(q: wp.array[float], q0: wp.array[float], j0: int, radius: float) -> float:
    """How far a proxy body's farthest point moves from ``q0`` to ``q`` (its DOFs from ``j0``), at most."""
    dx, dth = wp.vec3(0.0), wp.vec3(0.0)
    for c in range(3):
        dx[c] = q[j0 + c] - q0[j0 + c]
        dth[c] = q[j0 + 3 + c] - q0[j0 + 3 + c]
    return wp.length(dx) + wp.length(dth) * radius


@wp.kernel(module=SOLVER_MODULE)
def _arm_cap_kernel(q: wp.array[float], q0: wp.array[float], start: int, radius: wp.array[float], cap: float,
                    out: wp.array[float]):
    b = wp.tid()
    j0 = start + 6 * b
    sweep = _arm_sweep(q, q0, j0, radius[b])
    if sweep > cap:
        for c in range(6):
            out[j0 + c] = q0[j0 + c] + (q[j0 + c] - q0[j0 + c]) * (cap / sweep)


@wp.kernel(module=SOLVER_MODULE)
def _arm_sweep_kernel(q: wp.array[float], body: wp.array[wp.int32], base: wp.array[wp.int32],
                      com: wp.array[wp.vec3], body_com: wp.array[wp.vec3], start_q: wp.array[wp.transform],
                      cap: float, moved: int, sweep: wp.array[float], body_q: wp.array[wp.transform]):
    """Pose each proxy body at ``q`` (``moved``) or back at its step start, and set its reach ``cap``."""
    p = wp.tid()
    b = body[p]
    j0 = base[b]
    sweep[b] = cap
    q0 = start_q[b]
    if moved != 0:
        dx = wp.vec3(q[j0], q[j0 + 1], q[j0 + 2])
        dth = wp.vec3(q[j0 + 3], q[j0 + 4], q[j0 + 5])
        rot = _turn(dth, wp.transform_get_rotation(q0))
        q0 = wp.transform(com[b] + dx - wp.quat_rotate(rot, body_com[b]), rot)
    body_q[b] = q0


@wp.kernel(module=SOLVER_MODULE)
def _arm_moved_kernel(q: wp.array[float], q0: wp.array[float], start: int, radius: wp.array[float], limit: float,
                      wd: Worlds, moved: wp.array[wp.int32]):
    """A proxy body moved more than ``limit`` since the pairs were found: find them again (as a node)."""
    b = wp.tid()
    if _arm_sweep(q, q0, start + 6 * b, radius[b]) > limit:
        moved[wd.of[start + 6 * b]] = 1
        moved[wd.count] = 1


@wp.kernel(module=SOLVER_MODULE)
def _guess_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], limit: wp.array[float], q: wp.array[float]):
    n = wp.tid()
    cap = limit[n]
    if fixed[3 * n] == 0:
        move = node(q, n) - node(q0, n)
        length = wp.length(move)
        if length > cap:
            for c in range(3):
                q[3 * n + c] = q0[3 * n + c] + move[c] * (cap / length)


@wp.kernel(module=SOLVER_MODULE)
def _fallback_kernel(energy: wp.array[wp.float64], wd: Worlds, q0: wp.array[float], q: wp.array[float]):
    i = wp.tid()
    if not wp.isfinite(energy[wd.of[i] * wd.stride]):
        q[i] = q0[i]


@wp.kernel(module=SOLVER_MODULE)
def _pick_kernel(
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
    wd: Worlds,
    q_cand: wp.array2d[float],
    noise_tol: float,
    q: wp.array[float],
    pick: wp.array[wp.int32],
):
    """Each iterating world's candidate (``pick[w]``, written only if at least one DOF moves in float32, else ``0``
    so a no-op candidate stalls instead of spinning); the others keep ``q``."""
    i = wp.tid()
    w = wd.of[i]
    if wd.active[w] == 0:
        return
    k = _armijo(energy, slope, w, wd.stride, noise_tol)
    qk = q_cand[k, i]
    if k > 0 and qk != q[i]:
        pick[w] = k
        q[i] = qk
