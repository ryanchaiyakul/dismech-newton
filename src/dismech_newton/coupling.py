"""Rods held by MuJoCo-simulated rigid bodies (grippers, arms): :class:`MuJoCoCoupledSolver`.

One model holds the articulation and the rods; MuJoCo steps the rigid bodies (and their contacts
among themselves), :class:`~dismech_newton.ADMMDiSMechSolver` the rods (every contact involving a
rod). A step:

1. MuJoCo predicts the bodies' motion without the rods.
2. The rods step against the predicted poses (``obstacle_motion="pose"``). A pushed body yields
   within the solve through its mobility ``W = J M^-1 J^T`` (6x6 about its center of mass, see
   :meth:`~dismech_newton.contact.ContactTerm.set_body_mobility`), so a pinch is feasible.
3. MuJoCo steps again from ``state_in`` with the rods' reactions added to ``body_f``.

``W`` (with the implicit joint and actuator damping) is taken at ``qpos0`` on the first step. An
arm's changes with its configuration: call :meth:`MuJoCoCoupledSolver.update_effective_mass` now and
then (host, between frames). A stale ``W`` only shifts where a contact settles within the step.
"""

import numpy as np
import warp as wp
from newton import BodyFlags, Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo

from .admm import ADMMDiSMechSolver
from .frames import flatten_state


class MuJoCoCoupledSolver(SolverBase):
    """Rods (:class:`~dismech_newton.ADMMDiSMechSolver`) coupled two-way with MuJoCo rigid bodies.

    :meth:`step` takes the rods' contacts (a :class:`~newton.CollisionPipeline` on ``state_in``);
    MuJoCo finds its own. ``control`` drives MuJoCo; ``state_in.body_f`` is applied and kept.

    Args:
        model: Model with rods and rigid bodies (register :meth:`SolverMuJoCo.register_custom_attributes`).
        rod_options: Keyword arguments of :class:`~dismech_newton.ADMMDiSMechSolver`.
        mujoco_options: Keyword arguments of :class:`~newton.solvers.SolverMuJoCo`; it never sees the
            rod proxies' collision shapes.
        yield_: Let bodies yield to the rod within the step; ``False`` treats them as infinitely
            heavy for the rod (the force is still applied), which cannot hold a pinch.
    """

    def __init__(self, model: Model, *, rod_options: dict | None = None, mujoco_options: dict | None = None,
                 yield_: bool = True):
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
        self.yield_ = yield_
        self._dt = None  # the step W was taken for
        self._W = None
        self._body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self.device)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        dt = float(dt)
        if self.yield_ and dt != self._dt:
            self._dt = dt
            self.update_effective_mass(at_qpos0=True)
        flatten_state(state_in)
        flatten_state(state_out)
        self.rigid.step(state_in, state_out, control, None, dt)  # the motion without the rods
        self.rod.step(state_in, state_out, None, contacts, dt)
        wp.copy(self._body_f, state_in.body_f)
        self.rod.add_contact_reactions(state_in.body_f)
        self.rigid.step(state_in, state_out, control, None, dt)
        wp.copy(state_in.body_f, self._body_f)
        self.rod.update_proxies(state_out)  # MuJoCo rewrote them

    def update_effective_mass(self, at_qpos0: bool = False) -> None:
        """Take every body's ``W`` at MuJoCo's current configuration (or ``qpos0``); host only,
        outside graph capture. A no-op before the first step."""
        if not self.yield_ or self._dt is None:
            return
        qpos = None if at_qpos0 else self.rigid.mjw_data.qpos.numpy()
        W = effective_inverse_mass(self.model, self.rigid, self._dt, qpos)
        if self._W is None:
            self._W = wp.array(W, dtype=wp.spatial_matrix, device=self.device)
            self.rod.contact.set_body_mobility(self._W)
        else:
            self._W.assign(W)  # in place: a captured graph keeps reading it

    def notify_model_changed(self, flags: int) -> None:
        self.rigid.notify_model_changed(flags)


def effective_inverse_mass(model: Model, rigid: SolverMuJoCo, dt: float, qpos: np.ndarray | None = None) -> np.ndarray:
    """Per body, ``J (M + dt D)^-1 J^T`` (``(body_count, 6, 6)``, world frame): wrench ``(f, tau)``
    about its center of mass to ``(v, w)``, with ``D`` the joint and position-actuator damping MuJoCo
    integrates implicitly. At ``qpos`` (``(worlds, nq)``), else ``qpos0``; zero for kinematic and
    static bodies. Only yielding DOFs are inverted (the proxies' free joints are separate trees)."""
    import mujoco

    m = rigid.mj_model
    d = mujoco.MjData(m)
    damping = m.dof_damping.copy()
    for a in range(m.nu):  # a position actuator's bias is -kp q - kv qdot
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            dof = m.jnt_dofadr[m.actuator_trnid[a, 0]]
            damping[dof] += max(0.0, -m.actuator_biasprm[a, 2]) * m.actuator_gear[a, 0] ** 2
    to_newton = rigid.mjc_body_to_newton.numpy()  # (worlds, nbody)
    kinematic = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
    yielding = (to_newton >= 0) & ~kinematic[np.maximum(to_newton, 0)]
    yields = yielding.any(axis=0)
    bodies = [int(b) + 1 for b in np.nonzero(yields[1:])[0]]
    dofs = np.nonzero(yields[m.dof_bodyid])[0]
    out = np.zeros((model.body_count, 6, 6), dtype=np.float32)
    if not len(dofs):
        return out
    M = np.zeros((m.nv, m.nv))
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    W = None
    for world in range(to_newton.shape[0]):
        if W is None or qpos is not None:  # without qpos every world sits at qpos0
            d.qpos[:] = m.qpos0 if qpos is None else qpos[world]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_crb(m, d)  # M, with armature
            mujoco.mj_fullM(m, d, M)
            Minv = np.linalg.inv(M[np.ix_(dofs, dofs)] + dt * np.diag(damping[dofs]))
            W = {}
            for b in bodies:
                mujoco.mj_jacBodyCom(m, d, jacp, jacr, b)
                J = np.vstack((jacp[:, dofs], jacr[:, dofs]))
                W[b] = J @ Minv @ J.T
        for b in bodies:
            if yielding[world, b]:
                out[to_newton[world, b]] = W[b]
    return out


def _proxy_shapes(model: Model) -> np.ndarray:
    edge_body = model.dismech.edge_body.numpy()
    is_proxy = np.zeros(max(model.body_count, 1), dtype=bool)
    is_proxy[edge_body[edge_body >= 0]] = True
    shape_body = model.shape_body.numpy()
    return np.nonzero((shape_body >= 0) & is_proxy[np.maximum(shape_body, 0)])[0]
