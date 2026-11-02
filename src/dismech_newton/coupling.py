"""Rods and shells coupled two-way with a Newton rigid solver's bodies through contact.

A step: the rigid solver steps the bodies alone; the rods and shells step against that motion (the pose at the step's
start, the rigid step's velocity over it), yielding bodies moving within their solve; the rigid solver then steps
again from the start with the converged contact reactions (:attr:`~.solver.DiSMechSolver.body_reaction`) in
``body_f``.

With MuJoCo, every body it moves yields: its joints' moves ``z`` over the step join the soft step's unknowns, from
the bodies' start pose (feasible, as the nodes'), with the inertia of MuJoCo's step about its own moves ``z0``,
``(z - z0)^T (M + h D + C) (z - z0) / 2 h^2``: ``M`` with armature, ``D`` joint damping and the
position servos' ``kv`` (implicit in ``implicitfast``), ``C = J^T efc_D J`` over its active constraint rows (a
supported body does not sink). MuJoCo's step is affine in an applied wrench while its constraints keep their state;
its exact response is ``(M + h D)^-1 M (M + C)^-1``, which this matches where constraints and damping act on separate
DOFs (a free body on the ground; a gripper's fingers). The second rigid step then moves the bodies as the soft solve
assumed: the coupled step is monolithic in one pass (a pinch holds at the servo's force). Other rigid solvers' bodies are prescribed in the soft
solve (explicit: stable while a body outweighs what it pushes in a step).
"""

from collections.abc import Callable

import mujoco
import numpy as np
import warp as wp
from newton import BodyFlags, Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo

from .contact import Arm
from .solver import DiSMechSolver

QUADRATIC = wp.constant(int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC))
CONE = wp.constant(int(mujoco.mjtConstraintState.mjCNSTRSTATE_CONE))


class CoupledSolver(SolverBase):
    """A :class:`~.solver.DiSMechSolver` (``soft``, built with ``soft_options``) and the rigid solver ``rigid(model)``
    builds, coupled two-way through contact. :meth:`step`'s ``contacts`` go to the rigid solver. No adjoint."""

    def __init__(self, model: Model, rigid: Callable[[Model], SolverBase], **soft_options):
        super().__init__(model)
        self.rigid = rigid(model)
        mobility = _Mobility(model, self.rigid) if isinstance(self.rigid, SolverMuJoCo) else None
        self.mobility = mobility if mobility and mobility.k else None
        self.soft = DiSMechSolver(model, arm_dofs=self.mobility.k if self.mobility else 0, **soft_options)
        if self.mobility:
            self.mobility.bind(self.soft.contact.arm)
        self._body_f = wp.zeros(max(model.body_count, 1), dtype=wp.spatial_vector, device=self.device)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        self.rigid.step(state_in, state_out, control, contacts, dt)  # the bodies' own step
        qd = state_out.body_qd  # the bodies' motion over the step, from their pose at its start
        if self.mobility:
            qd = self.mobility.update(state_in, state_out, dt)  # the yielding bodies' still: their moves are z
        start = state_in.body_qd
        state_in.body_qd = qd
        try:
            self.soft.step(state_in, state_out, None, None, dt)
        finally:
            state_in.body_qd = start
        body_f = state_in.body_f
        wp.launch(_add_kernel, dim=self.model.body_count, inputs=[body_f, self.soft.body_reaction],
                  outputs=[self._body_f], device=self.device)
        state_in.body_f = self._body_f
        try:
            self.rigid.step(state_in, state_out, control, contacts, dt)  # again, with the soft side's reactions
        finally:
            state_in.body_f = body_f
        self.soft.update_proxies(state_out)  # the rigid step moved them

    @property
    def graph_capturable(self) -> bool:
        return self.soft.graph_capturable and getattr(self.rigid, "graph_capturable", True)

    def notify_model_changed(self, flags: int) -> None:
        self.soft.notify_model_changed(flags)
        self.rigid.notify_model_changed(flags)


class _Mobility:
    """MuJoCo's yielding DOFs (those of the bodies it moves, the same in every world) as a soft step's :class:`Arm`:
    per world ``M + h D`` on them, per body its centre of mass Jacobian (rows: linear, angular; world frame) at the
    step's start, from MuJoCo Warp's data."""

    def __init__(self, model: Model, rigid: SolverMuJoCo):
        m, mw = rigid.mj_model, rigid.mjw_model
        self.rigid, self.device = rigid, model.device
        to_newton = rigid.mjc_body_to_newton.numpy()  # (worlds, nbody)
        kinematic = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
        newton = np.maximum(to_newton, 0)
        yielding = (to_newton >= 0) & ~kinematic[newton]  # (the rods' capsule bodies are kinematic)
        yielding[:, 0] = False  # the world body
        dofs = np.flatnonzero(yielding.any(axis=0)[m.dof_bodyid])
        self.k = k = len(dofs)
        damping = m.dof_damping.copy()
        for a in range(m.nu):  # a position servo's bias is -kp q - kv qdot: kv is implicit in implicitfast
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                damping[m.jnt_dofadr[m.actuator_trnid[a, 0]]] += max(0.0, -m.actuator_biasprm[a, 2]) * \
                    m.actuator_gear[a, 0] ** 2
        local = np.full(m.nv, -1)  # M[dofs, dofs]'s slots in MuJoCo's CSR (lower triangle), -1 where zero
        local[dofs] = np.arange(k)
        slot = np.full((max(k, 1), max(k, 1)), -1, dtype=np.int32)
        rowadr, rownnz, colind = mw.M_rowadr.numpy(), mw.M_rownnz.numpy(), mw.M_colind.numpy()
        for i in dofs:
            for a in range(rowadr[i], rowadr[i] + rownnz[i]):
                if local[colind[a]] >= 0:
                    slot[local[i], local[colind[a]]] = slot[local[colind[a]], local[i]] = a
        world, body = np.nonzero(yielding)
        self._world, self._body, self._newton = (wp.array(a, dtype=wp.int32, device=self.device)
                                                 for a in (world, body, to_newton[world, body]))
        self._dofs = wp.array(dofs, dtype=wp.int32, device=self.device)
        self._damping = wp.array(damping[dofs], dtype=float, device=self.device)
        self._slot = wp.array(slot, dtype=wp.int32, device=self.device)
        self._local = wp.array(local, dtype=wp.int32, device=self.device)
        self._qd = wp.zeros(max(model.body_count, 1), dtype=wp.spatial_vector, device=self.device)
        self._yield = (world, to_newton[world, body])
        self._arm: Arm | None = None

    def bind(self, arm: Arm) -> None:
        """Fill ``arm`` from now on; each yielding body's DOFs are its world's."""
        self._arm = arm
        world, newton = self._yield
        base = np.full(arm.base.shape[0], -1, dtype=np.int32)
        base[newton] = arm.start + world * self.k
        arm.base.assign(base)

    def update(self, state_in: State, state_pred: State, dt: float) -> wp.array:
        """The arm at the rigid step just taken from ``state_in`` to ``state_pred`` (MuJoCo's ``M``, constraints and
        Jacobians are its start's, ``qvel`` its end's); returns ``state_pred``'s body velocities with the yielding
        bodies' zeroed. Graph-capturable."""
        d, mw, a = self.rigid.mjw_data, self.rigid.mjw_model, self._arm
        e = d.efc
        wp.launch(_inertia_kernel, dim=a.M.shape[0], inputs=[d.M, self._slot, self._damping, self._dofs, self._local,
                  int(mw.is_sparse), d.nefc, e.J_rowadr, e.J_rownnz, e.J_colind, e.J, e.D, e.state, d.qvel, dt],
                  outputs=[a.M, a.Minv, a.pred], device=self.device)
        wp.copy(self._qd, state_pred.body_qd)
        wp.launch(_jacobian_kernel, dim=self._world.shape[0],
                  inputs=[self._world, self._body, self._newton, self._dofs, mw.body_rootid, mw.body_isdofancestor,
                          d.xipos, d.subtree_com, d.cdof, state_in.body_q, self.rigid.model.body_com],
                  outputs=[a.J, a.com, self._qd], device=self.device)
        return self._qd


@wp.kernel
def _add_kernel(a: wp.array[wp.spatial_vector], b: wp.array[wp.spatial_vector], out: wp.array[wp.spatial_vector]):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def _inertia_kernel(M: wp.array2d[float], slot: wp.array2d[wp.int32], damping: wp.array[float],
                    dofs: wp.array[wp.int32], local: wp.array[wp.int32], sparse: int, nefc: wp.array[int],
                    rowadr: wp.array2d[int], rownnz: wp.array2d[int], colind: wp.array3d[int], J: wp.array3d[float],
                    D: wp.array2d[float], state: wp.array2d[int], qvel: wp.array2d[float], h: float,
                    out: wp.array3d[float], inv: wp.array3d[float], pred: wp.array[float]):
    """``M + h D + J^T efc_D J`` on the dofs, of world ``w``: the constraint rows quadratic in MuJoCo's cost (an
    elliptic cone's taken as quadratic: sticking); the world's predicted moves ``h qvel``."""
    w = wp.tid()
    k = dofs.shape[0]
    for i in range(k):
        pred[w * k + i] = h * qvel[w, dofs[i]]
    for i in range(k):
        for j in range(k):
            v = float(0.0)
            if slot[i, j] >= 0:
                v = M[w, slot[i, j]]
            if i == j:
                v += h * damping[i]
            out[w, i, j] = v
    for r in range(nefc[w]):
        if state[w, r] != QUADRATIC and state[w, r] != CONE:
            continue
        if sparse == 0:
            for i in range(k):
                for j in range(k):
                    out[w, i, j] = out[w, i, j] + D[w, r] * J[w, r, dofs[i]] * J[w, r, dofs[j]]
        else:  # MuJoCo's sparse rows
            a0 = rowadr[w, r]
            for p in range(rownnz[w, r]):
                i = local[colind[w, 0, a0 + p]]
                if i >= 0:
                    for p2 in range(rownnz[w, r]):
                        j = local[colind[w, 0, a0 + p2]]
                        if j >= 0:
                            out[w, i, j] = out[w, i, j] + D[w, r] * J[w, 0, a0 + p] * J[w, 0, a0 + p2]
    for i in range(k):  # its inverse: Cholesky L L^T in place, L^-1, then L^-T L^-1
        for j in range(k):
            inv[w, i, j] = out[w, i, j]
    for j in range(k):
        s = inv[w, j, j]
        for p in range(j):
            s -= inv[w, j, p] * inv[w, j, p]
        ljj = wp.sqrt(s)
        inv[w, j, j] = ljj
        for i in range(j + 1, k):
            s = inv[w, i, j]
            for p in range(j):
                s -= inv[w, i, p] * inv[w, j, p]
            inv[w, i, j] = s / ljj
    for j in range(k):
        inv[w, j, j] = 1.0 / inv[w, j, j]
        for i in range(j + 1, k):
            s = float(0.0)
            for p in range(j, i):
                s -= inv[w, i, p] * inv[w, p, j]
            inv[w, i, j] = s / inv[w, i, i]
    for i in range(k):
        for j in range(i + 1):
            s = float(0.0)
            for p in range(i, k):
                s += inv[w, p, i] * inv[w, p, j]
            inv[w, i, j] = s
        for j in range(i):
            inv[w, j, i] = inv[w, i, j]


@wp.kernel
def _jacobian_kernel(
    world: wp.array[wp.int32], body: wp.array[wp.int32], newton_body: wp.array[wp.int32], dofs: wp.array[wp.int32],
    body_rootid: wp.array[int], body_isdofancestor: wp.array2d[int], xipos: wp.array2d[wp.vec3],
    subtree_com: wp.array2d[wp.vec3], cdof: wp.array2d[wp.spatial_vector], body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3], J: wp.array2d[wp.spatial_vector], com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
):
    """A yielding body's COM Jacobian on the dofs (MuJoCo's ``cdof``: rotation, then translation at the root's subtree
    COM) and its COM, at the step's start; its velocity zeroed (its move is the arm's)."""
    p = wp.tid()
    w, b, nb = world[p], body[p], newton_body[p]
    body_qd[nb] = wp.spatial_vector()
    offset = xipos[w, b] - subtree_com[w, body_rootid[b]]
    for i in range(dofs.shape[0]):
        J[nb, i] = wp.spatial_vector()
        if body_isdofancestor[b, dofs[i]] != 0:
            c = cdof[w, dofs[i]]
            ang = wp.spatial_top(c)
            J[nb, i] = wp.spatial_vector(wp.spatial_bottom(c) + wp.cross(ang, offset), ang)
    com[nb] = wp.transform_point(body_q[nb], body_com[nb])
