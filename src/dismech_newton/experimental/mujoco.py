"""Rods coupled two-way with MuJoCo rigid bodies.

A step: MuJoCo predicts the bodies without the rods; the rods step against those poses, bodies
yielding through their mobility ``W = J (M + dt D)^-1 J^T``, taken on the device from MuJoCo's ``M`` at the
step's start; MuJoCo steps again with the rods' reactions.
"""

import mujoco
import numpy as np
import warp as wp
from newton import BodyFlags, Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo

from ..admm import ADMMDiSMechSolver
from ..solver import active_tape


class MuJoCoCoupledSolver(SolverBase):
    """Rods (ADMM) coupled two-way with MuJoCo bodies; :meth:`step` takes the rods' contacts. No adjoint.

    Args:
        rod_options: Keyword arguments of :class:`~dismech_newton.ADMMDiSMechSolver`.
        mujoco_options: Keyword arguments of :class:`~newton.solvers.SolverMuJoCo`.
        bodies_yield: Bodies yield to the rod within the step (``False``: infinitely heavy, no pinch).
    """

    def __init__(self, model: Model, *, rod_options: dict | None = None, mujoco_options: dict | None = None,
                 bodies_yield: bool = True):
        super().__init__(model)
        self.rod = ADMMDiSMechSolver(model, **{**(rod_options or {}), "obstacle_motion": "pose"})
        # Collision group 0 collides with nothing: MuJoCo keeps the proxies' geoms (their inertia) inert.
        group = model.shape_collision_group.numpy()
        hidden = group.copy()
        hidden[_proxy_shapes(model)] = 0
        model.shape_collision_group.assign(hidden)
        try:
            self.rigid = SolverMuJoCo(model, **{"use_mujoco_contacts": True, **(mujoco_options or {})})
        finally:
            model.shape_collision_group.assign(group)
        self.mobility = BodyMobility(model, self.rigid) if bodies_yield else None
        if self.mobility is not None:
            self.rod.contact.set_body_mobility(self.mobility.W)
        self._body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self.device)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        if active_tape() is not None:
            raise RuntimeError("MuJoCoCoupledSolver has no adjoint: step it outside a wp.Tape")
        dt = float(dt)
        self.rigid.step(state_in, state_out, control, None, dt)  # the motion without the rods
        if self.mobility is not None:
            self.mobility.update(dt)  # MuJoCo's M and Jacobians, at state_in
        self.rod.step(state_in, state_out, None, contacts, dt)
        wp.copy(self._body_f, state_in.body_f)
        self.rod.add_contact_reactions(state_in.body_f)
        self.rigid.step(state_in, state_out, control, None, dt)
        wp.copy(state_in.body_f, self._body_f)
        self.rod.update_proxies(state_out)  # MuJoCo rewrote them

    @property
    def graph_capturable(self) -> bool:
        return self.rod.graph_capturable

    def notify_model_changed(self, flags: int) -> None:
        self.rigid.notify_model_changed(flags)


class BodyMobility:
    """Per Newton body ``W = J (M + dt D)^-1 J^T`` about its center of mass, world frame, from MuJoCo Warp's
    data (zero for kinematic and static bodies). ``D`` is the implicit damping: joint damping plus position
    servos' ``kv``. ``M`` is restricted to the yielding bodies' DOFs, inverted per world on the device."""

    def __init__(self, model: Model, rigid: SolverMuJoCo):
        m, mw = rigid.mj_model, rigid.mjw_model
        self.rigid = rigid
        self.device = model.device
        to_newton = rigid.mjc_body_to_newton.numpy()  # (worlds, nbody)
        kinematic = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
        yielding = (to_newton >= 0) & ~kinematic[np.maximum(to_newton, 0)]
        yielding[:, 0] = False  # the world body
        dofs = np.nonzero(yielding.any(axis=0)[m.dof_bodyid])[0]
        k = len(dofs)
        damping = m.dof_damping.copy()
        for a in range(m.nu):  # a position actuator's bias is -kp q - kv qdot
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                dof = m.jnt_dofadr[m.actuator_trnid[a, 0]]
                damping[dof] += max(0.0, -m.actuator_biasprm[a, 2]) * m.actuator_gear[a, 0] ** 2
        # Slots of M[dofs, dofs] in MuJoCo's CSR (lower triangle with diagonal), -1 where zero.
        local = np.full(m.nv, -1)
        local[dofs] = np.arange(k)
        slot = np.full((k, k), -1, dtype=np.int32)
        rowadr, rownnz, colind = mw.M_rowadr.numpy(), mw.M_rownnz.numpy(), mw.M_colind.numpy()
        for i in dofs:
            for a in range(rowadr[i], rowadr[i] + rownnz[i]):
                j = colind[a]
                if local[j] >= 0:
                    slot[local[i], local[j]] = slot[local[j], local[i]] = a
        world, body = np.nonzero(yielding)
        self.k = k
        self.W = wp.zeros(model.body_count, dtype=wp.spatial_matrix, device=self.device)
        self._dofs = wp.array(dofs, dtype=wp.int32, device=self.device)
        self._damping = wp.array(damping[dofs], dtype=float, device=self.device)
        self._slot = wp.array(slot, dtype=wp.int32, device=self.device)
        self._world = wp.array(world, dtype=wp.int32, device=self.device)
        self._body = wp.array(body, dtype=wp.int32, device=self.device)
        self._newton = wp.array(to_newton[world, body], dtype=wp.int32, device=self.device)
        worlds = to_newton.shape[0]
        self._A = wp.zeros((worlds, k, k), dtype=float, device=self.device)  # (M + dt D)^-1, Cholesky in place
        self._J = wp.zeros((len(world), k), dtype=wp.spatial_vector, device=self.device)

    def update(self, dt: float) -> None:
        """Retake ``W`` from the MuJoCo data's last forward pass; graph-capturable."""
        if not self.k or not self._world.shape[0]:
            return
        d, mw = self.rigid.mjw_data, self.rigid.mjw_model
        wp.launch(_invert_mass_kernel, dim=self._A.shape[0], inputs=[d.M, self._slot, self._damping, dt],
                  outputs=[self._A], device=self.device)
        wp.launch(
            _mobility_kernel,
            dim=self._world.shape[0],
            inputs=[self._world, self._body, self._newton, self._dofs, mw.body_rootid, mw.body_isdofancestor,
                    d.xipos, d.subtree_com, d.cdof, self._A],
            outputs=[self._J, self.W],
            device=self.device,
        )


def _proxy_shapes(model: Model) -> np.ndarray:
    edge_body = model.dismech.edge_body.numpy()
    is_proxy = np.zeros(max(model.body_count, 1), dtype=bool)
    is_proxy[edge_body[edge_body >= 0]] = True
    shape_body = model.shape_body.numpy()
    return np.nonzero((shape_body >= 0) & is_proxy[np.maximum(shape_body, 0)])[0]


# -- kernels ------------------------------------------------------------------------------


@wp.kernel
def _invert_mass_kernel(M: wp.array2d[float], slot: wp.array2d[wp.int32], damping: wp.array[float], dt: float,
                        A: wp.array3d[float]):
    """``A = (M[dofs, dofs] + dt D)^-1`` of one world: Cholesky ``L L^T``, then ``L^-T L^-1``, in place."""
    w = wp.tid()
    k = slot.shape[0]
    for i in range(k):
        for j in range(k):
            a = slot[i, j]
            v = float(0.0)
            if a >= 0:
                v = M[w, a]
            if i == j:
                v += dt * damping[i]
            A[w, i, j] = v
    for j in range(k):  # L in the lower triangle
        s = A[w, j, j]
        for p in range(j):
            s -= A[w, j, p] * A[w, j, p]
        ljj = wp.sqrt(s)
        A[w, j, j] = ljj
        for i in range(j + 1, k):
            s = A[w, i, j]
            for p in range(j):
                s -= A[w, i, p] * A[w, j, p]
            A[w, i, j] = s / ljj
    for j in range(k):  # L^-1 in the lower triangle
        A[w, j, j] = 1.0 / A[w, j, j]
        for i in range(j + 1, k):
            s = float(0.0)
            for p in range(j, i):
                s -= A[w, i, p] * A[w, p, j]
            A[w, i, j] = s / A[w, i, i]
    for i in range(k):  # L^-T L^-1, symmetric, row by row (row i reads only rows >= i)
        for j in range(i + 1):
            s = float(0.0)
            for p in range(i, k):
                s += A[w, p, i] * A[w, p, j]
            A[w, i, j] = s
        for j in range(i):
            A[w, j, i] = A[w, i, j]


@wp.kernel
def _mobility_kernel(
    world: wp.array[wp.int32], body: wp.array[wp.int32], newton_body: wp.array[wp.int32], dofs: wp.array[wp.int32],
    body_rootid: wp.array[int], body_isdofancestor: wp.array2d[int], xipos: wp.array2d[wp.vec3],
    subtree_com: wp.array2d[wp.vec3], cdof: wp.array2d[wp.spatial_vector], A: wp.array3d[float],
    # outputs
    J: wp.array2d[wp.spatial_vector], W: wp.array[wp.spatial_matrix],
):
    """``W = J A J^T`` of one yielding body, ``J`` its COM Jacobian, rows ``(linear, angular)``."""
    p = wp.tid()
    w = world[p]
    b = body[p]
    k = dofs.shape[0]
    offset = xipos[w, b] - subtree_com[w, body_rootid[b]]
    for i in range(k):
        dof = dofs[i]
        if body_isdofancestor[b, dof] != 0:
            c = cdof[w, dof]
            ang = wp.spatial_top(c)
            J[p, i] = wp.spatial_vector(wp.spatial_bottom(c) + wp.cross(ang, offset), ang)
        else:
            J[p, i] = wp.spatial_vector()
    out = wp.spatial_matrix()
    for i in range(k):
        Ji = J[p, i]
        AJ = wp.spatial_vector()
        for j in range(k):
            AJ += A[w, i, j] * J[p, j]
        out += wp.outer(Ji, AJ)
    W[newton_body[p]] = out
