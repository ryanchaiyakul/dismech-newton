"""Newton ball-joint pins between a ``ProxyArm`` body and a rod."""

import numpy as np
import warp as wp
from newton import JointType, Model

from ..contact import STIFFNESS, Arm, arm_assemble, arm_move, arm_scatter
from ..dofs import node, padded, scatter_node
from ..solver.linear import row_slot
from ..worlds import world_sum
from .proxy import COUPLING_MODULE


@wp.struct
class Pins:
    """Per pin its node, body, point in the body's frame, that point at the step's start and stiffness."""

    node: wp.array[wp.int32]
    body: wp.array[wp.int32]
    local: wp.array[wp.vec3]
    point: wp.array[wp.vec3]
    k: wp.array[float]


class ProxyPins:
    """Newton's ``BALL`` joints in the entry between a :class:`ProxyArm` body and a rod's capsule (either as the
    parent): the capsule's nearer end node held to the joint's point on the body by ``k/2 |x - p - P z|^2`` in the
    step, ``p`` the point at the step's start and ``P z`` its move by the body's DOFs (as a contact's anchor), ``k``
    :data:`~.contact.STIFFNESS` times the node's inertia ``m / h^2`` (a gap ``f h^2 / (STIFFNESS m)`` under a force
    ``f``). The body takes the reaction in the step, and the coupler its momentum: a cable fixed to a rigid body."""

    def __init__(self, model: Model, arm: Arm, mass: wp.array, fixed: wp.array):
        self.arm, self.mass, self.fixed, self.device = arm, mass, fixed, model.device
        edge = {int(b): e for e, b in enumerate(model.dismech.edge_body.numpy()) if b >= 0}
        base = arm.base.numpy()
        rows = []
        if model.joint_count:
            kind, on = model.joint_type.numpy(), model.joint_enabled.numpy()
            parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
            X_p, X_c = model.joint_X_p.numpy(), model.joint_X_c.numpy()
            node0, node1 = model.dismech.edge_node0.numpy(), model.dismech.edge_node1.numpy()
            for j in np.flatnonzero((kind == int(JointType.BALL)) & on):
                (b, xb), (c, xc) = (parent[j], X_p[j]), (child[j], X_c[j])
                if b in edge:
                    (b, xb), (c, xc) = (c, xc), (b, xb)
                if c not in edge:
                    continue
                if b < 0 or base[b] < 0:
                    raise ValueError(f"joint {j} pins a rod to body {b}, not a proxy with shapes in this entry")
                e = edge[c]
                rows.append((node0[e] if xc[2] < 0.0 else node1[e], b, xb[:3]))  # the capsule's axis is its +Z
        self.count = len(rows)
        p, n = Pins(), max(self.count, 1)
        p.node = wp.array([r[0] for r in rows] or [0], dtype=wp.int32, device=self.device)
        p.body = wp.array([r[1] for r in rows] or [0], dtype=wp.int32, device=self.device)
        p.local = wp.array([r[2] for r in rows] or [np.zeros(3)], dtype=wp.vec3, device=self.device)
        p.point = wp.zeros(n, dtype=wp.vec3, device=self.device)
        p.k = wp.zeros(n, dtype=float, device=self.device)
        self.p = p

    def _launch(self, kernel, inputs, dim=None):
        wp.launch(kernel, dim=dim or self.count, inputs=inputs, device=self.device)

    def begin_step(self, body_q: wp.array, h: float) -> None:
        """The points at the step's start (``body_q``: the proxies' own start poses) and the stiffnesses."""
        self._launch(_pin_begin_kernel, [self.p, body_q, self.mass, STIFFNESS / (h * h)])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self._launch(_pin_grad_kernel, [q, self.p, self.arm, self.fixed, b])

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self.grad(q, b)

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(_pin_energy_kernel, [q_cand, self.p, self.arm, self.count, self.world, self.stride, energy],
                     dim=(q_cand.shape[0], padded(self.count)))

    def assemble(self, q: wp.array, A) -> None:
        self._launch(_pin_assemble_kernel, [self.p, self.arm, self.fixed, A.indptr, A.indices, A.vals])

    def reactions(self, q: wp.array, body_f: wp.array) -> None:
        """Add each pin's force at ``q`` on its body (and its torque about the centre of mass at the step's start)."""
        self._launch(_pin_reaction_kernel, [q, self.p, self.arm, body_f])


@wp.func
def _pin_gap(q: wp.array[float], p: Pins, i: int, arm: Arm) -> wp.vec3:
    """``x - p - P z``."""
    b = p.body[i]
    return node(q, p.node[i]) - p.point[i] - arm_move(q, arm, b, p.point[i] - arm.com[b])


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
def _pin_begin_kernel(p: Pins, body_q: wp.array[wp.transform], mass: wp.array[float], scale: float):
    i = wp.tid()
    p.point[i] = wp.transform_point(body_q[p.body[i]], p.local[i])
    p.k[i] = scale * mass[3 * p.node[i]]


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
def _pin_grad_kernel(q: wp.array[float], p: Pins, arm: Arm, fixed: wp.array[wp.int32], b: wp.array[wp.float64]):
    i = wp.tid()
    g = p.k[i] * _pin_gap(q, p, i, arm)
    scatter_node(b, fixed, p.node[i], g)
    arm_scatter(b, arm, p.body[i], p.point[i] - arm.com[p.body[i]], g)


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
def _pin_energy_kernel(q_cand: wp.array2d[float], p: Pins, arm: Arm, count: int, world: wp.array[wp.int32],
                       stride: int, energy: wp.array[wp.float64]):
    j, i = wp.tid()
    e = wp.float64(0.0)
    w = int(-1)
    if i < count:
        w = world[3 * p.node[i]]
        c = _pin_gap(q_cand[j], p, i, arm)
        e = wp.float64(0.5 * p.k[i] * wp.dot(c, c))
    world_sum(energy, w, j, stride, e)


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
def _pin_assemble_kernel(p: Pins, arm: Arm, fixed: wp.array[wp.int32], indptr: wp.array[wp.int32],
                         indices: wp.array[wp.int32], vals: wp.array[wp.float64]):
    """``k [I, -P; -P^T, P^T P]`` into the upper CSR (all in the pattern: a node and its world's arm)."""
    i = wp.tid()
    k, n, body = p.k[i], p.node[i], p.body[i]
    r = p.point[i] - arm.com[body]
    for c in range(3):
        if fixed[3 * n + c] == 0:
            wp.atomic_add(vals, row_slot(indptr, indices, 3 * n + c, 3 * n + c), wp.float64(k))
    eye = wp.identity(3, float)
    arm_assemble(vals, indptr, indices, fixed, arm, body, r, k * eye, n, eye)
    arm_assemble(vals, indptr, indices, fixed, arm, body, r, k * eye, -1, eye)


@wp.kernel(module=COUPLING_MODULE, enable_backward=False)
def _pin_reaction_kernel(q: wp.array[float], p: Pins, arm: Arm, body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    f = p.k[i] * _pin_gap(q, p, i, arm)  # pulls the body toward the node
    body = p.body[i]
    wp.atomic_add(body_f, body, wp.spatial_vector(f, wp.cross(p.point[i] - arm.com[body], f)))
