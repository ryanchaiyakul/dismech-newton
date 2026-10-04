"""The MuJoCo coupling's body mobility against MuJoCo's own mass matrix and Jacobians (needs the 'mujoco' extra)."""

import numpy as np
import pytest
import warp as wp

mujoco = pytest.importorskip("mujoco")
mjw = pytest.importorskip("mujoco_warp")
newton = pytest.importorskip("newton")

from newton.solvers import SolverMuJoCo  # noqa: E402

from dismech_newton import ADMMDiSMechSolver  # noqa: E402
from dismech_newton.experimental.mujoco import MuJoCoCoupledSolver  # noqa: E402

DT = 1.0e-3


def _arm(builder: newton.ModelBuilder) -> None:
    """Three boxes on revolute joints (alternating axes, COMs off the axes), position servos on two."""
    parent, joints = -1, []
    for i, axis in enumerate((newton.Axis.Z, newton.Axis.Y, newton.Axis.X)):
        link = builder.add_link(mass=1.0 + 0.5 * i, com=wp.vec3(0.05, 0.02 * i, 0.1))
        builder.add_shape_box(link, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                              hx=0.03, hy=0.03, hz=0.1)
        joints.append(builder.add_joint_revolute(
            parent, link, axis=axis, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2 if i else 0.0),
                                                               wp.quat_identity()),
            target_ke=50.0 * (i < 2), target_kd=5.0 * (i < 2), damping=0.3, armature=0.01))
        parent = link
    builder.add_articulation(joints)


def _reference(solver: MuJoCoCoupledSolver, model) -> np.ndarray:
    """``J (M + dt D)^-1 J^T`` per body with CPU MuJoCo, at the GPU data's ``qpos``."""
    rigid = solver.rigid
    m = rigid.mj_model
    d = mujoco.MjData(m)
    damping = m.dof_damping.copy()
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            damping[m.jnt_dofadr[m.actuator_trnid[a, 0]]] += max(0.0, -m.actuator_biasprm[a, 2]) * m.actuator_gear[a, 0] ** 2
    to_newton = rigid.mjc_body_to_newton.numpy()
    kinematic = (model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)) != 0
    yielding = (to_newton >= 0) & ~kinematic[np.maximum(to_newton, 0)]
    yielding[:, 0] = False
    dofs = np.nonzero(yielding.any(axis=0)[m.dof_bodyid])[0]
    qpos = rigid.mjw_data.qpos.numpy()
    out = np.zeros((model.body_count, 6, 6))
    M = np.zeros((m.nv, m.nv))
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for world in range(to_newton.shape[0]):
        d.qpos[:] = qpos[world]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_fullM(m, d, M)
        Minv = np.linalg.inv(M[np.ix_(dofs, dofs)] + DT * np.diag(damping[dofs]))
        for b in np.nonzero(yielding[world])[0]:
            mujoco.mj_jacBodyCom(m, d, jacp, jacr, b)
            J = np.vstack((jacp[:, dofs], jacr[:, dofs]))
            out[to_newton[world, b]] = J @ Minv @ J.T
    return out


def test_body_mobility(device):
    template = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(template)
    _arm(template)
    points = np.column_stack((np.linspace(0.3, 0.6, 6), np.zeros(6), np.full(6, 0.01)))
    ADMMDiSMechSolver.add_rod(template, newton.Rod(points, radius=0.01), bend_stiffness=0.1)
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    revolute = np.nonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))[0]
    q[model.joint_q_start.numpy()[revolute]] = [0.3, -0.7, 1.1, -1.2, 0.4, 0.9]  # a different pose per world
    model.joint_q.assign(q)
    solver = MuJoCoCoupledSolver(model)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, model.control(), None, DT)
    mjw.forward(solver.rigid.mjw_model, solver.rigid.mjw_data)  # kinematics and M at the data's qpos
    solver.mobility.update(DT)
    W, ref = solver.mobility.W.numpy(), _reference(solver, model)
    assert np.abs(ref).max() > 0.0
    np.testing.assert_allclose(W, ref, rtol=1.0e-4, atol=1.0e-5 * np.abs(ref).max())
    assert not np.allclose(W[1], W[1 + model.body_count // 2]), "the two worlds' poses should differ"
