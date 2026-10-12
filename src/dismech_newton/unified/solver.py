import numpy as np
import warp as wp
from newton import Contacts, Control, Model, ModelFlags, State
from newton.solvers import SolverBase

from ..builder import fix_segment, register_custom_attributes
from ..dofs import dof_constants, external_force, flatten_state
from ..solver import advance_frames_kernel, pose_proxies_kernel, predict_kernel, proxy_joints
from ..triplet import Triplets, linear_energy
from .collide import Offset, add_colliding_rod
from .contact import ContactBlock
from .element import ElementBlock, element_kernels
from .rod import Rod, bind_rod, rod_dofs, rod_jacobian, rod_strain
from .shell import Shells, add_cloth_mesh
from .state import SymmetricCSR, linear_solver

_C1 = 1.0e-4  # Armijo's sufficient decrease
_NOISE = 1.0e-5  # a rise of the objective within its float32 rounding


class DiSMechSolver(SolverBase):
    """Implicit discrete elastic rods and midedge shells with contact and friction, by strain-space ADMM.

    Fixed DOFs are prescribed through ``state_in.dismech.q``. Rods and shells collide with Newton's shapes, each
    other and themselves by Offset Geometric Contact, found by the solver (see :mod:`.collide`): ``contacts`` is
    ignored. Coulomb ``mu`` is Newton's: the mean of the two sides' ``shape_material_mu`` (a rod's capsules',
    ``particle_mu`` for a shell vertex), so a rod's friction is its ``cfg.mu``. Obstacles move with
    ``state_in.body_qd``.

    Args:
        iterations: Iterations per step (a fixed count, graph-capturable).
        beta: Strain penalty over the energy's stiffness ``K + C/dt``.
        linesearch: Step lengths tried per iteration, halving from ``1`` (``1`` takes the full chord step).
        exact: Gauss-Newton on the step's energy, ``A`` refactorised every iteration (no elastic splitting).
        self_contact: Rod-rod and shell-shell contacts (rod-shell and shape contacts are always on).
        contact_beta: Contact penalty over the inertia-only inverse Delassus diagonal.
        hinge_step: Trust region of a shell's midedge angles per iteration [rad] (near ``psi + theta = pi/2`` their
            tangent stiffens without bound).
        engage_gap: A contact takes part in a step only if the inertial prediction brings it within that many
            thicknesses of touching, or it carried force in the last step (a separated contact's penalty in ``A``
            would slow the motion near obstacles).
    """

    register_custom_attributes = staticmethod(register_custom_attributes)
    add_rod = staticmethod(add_colliding_rod)
    add_cloth_mesh = staticmethod(add_cloth_mesh)
    fix_segment = staticmethod(fix_segment)

    def __init__(
        self,
        model: Model,
        *,
        iterations: int = 10,
        beta: float = 1.0,
        linesearch: int = 4,
        exact: bool = False,
        self_contact: bool = True,
        contact_beta: float = 16.0,
        engage_gap: float = 0.25,
        hinge_step: float = 0.1,
    ):
        super().__init__(model)
        if linesearch < 1:
            raise ValueError("linesearch must be >= 1")
        self.iterations, self.exact, self.hinge_step = int(iterations), bool(exact), float(hinge_step)
        self.num_candidates = nc = int(linesearch) + 1  # candidate 0: the current iterate
        dev = self.device
        self.der = d = model.dismech
        self.mass, self.fixed = dof_constants(model)
        self.num_dofs = n = self.mass.shape[0]
        self.num_node_dofs = 3 * model.particle_count
        self.q_pred = wp.zeros(n, dtype=float, device=dev)
        self._no_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=dev)
        self._no_velocity = wp.zeros(0, dtype=wp.spatial_vector, device=dev)
        self._no_joint = wp.zeros(0, dtype=float, device=dev)
        self._has_proxies = bool((d.edge_body.numpy() >= 0).any())
        self._edge_joint = wp.array(proxy_joints(model), dtype=wp.vec2i, device=dev)

        self.triplets = tr = Triplets(model, self.fixed, linear_energy)
        node0, node1 = d.edge_node0.numpy().astype(np.int64), d.edge_node1.numpy().astype(np.int64)
        edge_dofs = np.column_stack([3 * node0 + k for k in range(3)] + [3 * node1 + k for k in range(3)])
        self.shells = sh = Shells(model, self.num_node_dofs + d.edge_length.shape[0])
        self.A = SymmetricCSR(n, (tr.dofs(), edge_dofs, sh.dofs()), dev)  # an edge's nodes: its contacts
        self.rods = ElementBlock(
            element_kernels(Rod, rod_strain, rod_jacobian, rod_dofs, 5, 11),
            tr.count,
            self.A.slots(tr.dofs()),
            self.fixed,
            nc,
            beta,
            dev,
        )
        self.faces = ElementBlock(sh.kernels, sh.count, self.A.slots(sh.dofs()), self.fixed, nc, beta, dev)
        self.contact = ContactBlock(
            model,
            self.fixed,
            nc,
            Offset(model, self.fixed, self_contact=self_contact, engage_gap=engage_gap),
            beta=contact_beta,
        )
        self.blocks = (self.rods, self.faces, self.contact)
        self._linear = linear_solver(self.A)
        self._b = wp.zeros(n, dtype=wp.float64, device=dev)  # the objective's gradient
        self._dq = wp.zeros(n, dtype=float, device=dev)
        self._q_cand = wp.zeros((nc, n), dtype=float, device=dev)
        self._energy = wp.zeros(nc, dtype=wp.float64, device=dev)
        self._slope = wp.zeros(1, dtype=wp.float64, device=dev)
        self._pick = wp.zeros(1, dtype=wp.int32, device=dev)
        self._started = False

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """Advance ``state_in`` by ``dt``."""
        dt = float(dt)
        flatten_state(state_in)
        flatten_state(state_out)
        q_in, q, dev, d = state_in.dismech.q, state_out.dismech.q, self.device, self.der
        self.triplets.begin_step(state_in)
        wp.launch(
            predict_kernel,
            dim=self.num_dofs,
            inputs=[self.fixed, q_in, state_in.dismech.qd, dt],
            outputs=[self.q_pred, q],
            device=dev,
        )
        self._solve(state_in, q, dt)
        wp.launch(
            _velocity_kernel,
            dim=self.num_dofs,
            inputs=[self.fixed, q_in, q, dt],
            outputs=[state_out.dismech.qd],
            device=dev,
        )
        wp.launch(
            advance_frames_kernel,
            dim=d.edge_length.shape[0],
            inputs=[state_in.particle_q, state_out.particle_q, d.edge_node0, d.edge_node1, state_in.dismech.edge_d1_q],
            outputs=[state_out.dismech.edge_d1_q],
            device=dev,
        )
        self.triplets.end_step(state_in, state_out)
        self.shells.end_step(state_out)
        self.update_proxies(state_out)

    def reset(self, state: State, world_mask: wp.array | None = None, flags: int | None = None) -> None:
        """Start the next step from ``state`` without the last step's warm start: for a step that does not continue
        the last one, such as a restart from data."""
        if world_mask is not None:
            raise NotImplementedError(f"{type(self).__name__}.reset resets every world")
        flatten_state(state)
        self.rods.reset(bind_rod(self.triplets, state), state.dismech.q)
        self.faces.reset(self.shells.view, state.dismech.q)
        self.contact.reset()
        self._started = True

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & ModelFlags.MODEL_PROPERTIES:
            self.mass.assign(dof_constants(self.model)[0])

    @property
    def graph_capturable(self) -> bool:
        return self.device.is_cuda and self._linear.graph_capturable

    def update_proxies(self, state: State) -> None:
        """Pose the capsule proxies and set their velocities (and their free joints)."""
        if not self._has_proxies or state.body_q is None:
            return
        body_qd = state.body_qd if state.body_qd is not None else self._no_velocity
        joints = state.joint_q is not None and state.joint_qd is not None and state.joint_q.shape[0] > 0
        d = self.der
        wp.launch(
            pose_proxies_kernel,
            dim=d.edge_length.shape[0],
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.dismech.edge_q,
                state.dismech.edge_d1_q,
                d.edge_node0,
                d.edge_node1,
                d.edge_body,
                self._edge_joint,
                int(state.body_qd is not None),
            ],
            outputs=[
                state.body_q,
                body_qd,
                state.joint_q if joints else self._no_joint,
                state.joint_qd if joints else self._no_joint,
            ],
            device=self.device,
        )

    def _solve(self, state_in: State, q: wp.array, dt: float) -> None:
        dev, n, tr = self.device, self.num_dofs, self.triplets
        inv_dt2 = 1.0 / (dt * dt)
        particle_f = state_in.particle_f if state_in.particle_f is not None else self._no_force
        dof_args = [self.mass, inv_dt2, self.fixed, self.model.gravity, particle_f, self.num_node_dofs]
        b = self._b

        self.rods.begin_step(bind_rod(tr, state_in), tr.params, tr.rest, tr.strain_prev, dt, self.exact)
        sh = self.shells
        sh.begin_step(state_in)
        self.faces.begin_step(sh.view, sh.params, sh.rest, sh.strain_prev, dt, self.exact)
        if not self._started and not self.exact:
            self.reset(state_in)
        self.contact.begin_step(state_in, dt, q, self.mass, inv_dt2)
        # b, the objective's gradient at q: the blocks' part comes from grad() here, then from commit() at each
        # picked candidate; _dof_kernel adds inertia and external force, _candidates_kernel spends it on the slope.
        b.zero_()
        for blk in self.blocks:
            blk.grad(q, b)
        for it in range(self.iterations):
            if it == 0 or self.exact:  # the chord matrix at the prediction (every iterate: Gauss-Newton)
                self._assemble(q, inv_dt2)
            wp.launch(
                _dof_kernel,
                dim=n,
                inputs=[q, self.q_pred, *dof_args],
                outputs=[b, self._energy, self._slope],
                device=dev,
            )
            self._linear.solve(b, self._dq)
            wp.launch(
                _candidates_kernel,
                dim=(self.num_candidates, n),
                inputs=[q, self._dq, self.q_pred, *dof_args, sh.view.hinge_offset, self.hinge_step],
                outputs=[self._q_cand, b, self._energy, self._slope],
                device=dev,
            )
            for blk in self.blocks:
                blk.energy(self._q_cand, self._energy)
            wp.launch(
                _pick_kernel,
                dim=n,
                inputs=[self._energy, self._slope, self.num_candidates, _C1, self._q_cand],
                outputs=[q, self._pick],
                device=dev,
            )
            for blk in self.blocks:
                blk.commit(q, self._pick, b)
        self.contact.end_step()

    def _assemble(self, q: wp.array, inv_dt2: float) -> None:
        self.A.vals.zero_()
        for blk in self.blocks:
            blk.assemble(q, self.A)
        wp.launch(
            _inertia_kernel,
            dim=self.num_dofs,
            inputs=[self.mass, inv_dt2, self.fixed],
            outputs=[self.A.indptr, self.A.vals],
            device=self.device,
        )
        self._linear.invalidate()


# -- the global step and its line search ---------------------------------------------------


@wp.func
def _step_length(k: int) -> float:
    """Candidate 0 is the current iterate, candidate ``k >= 1`` the step ``2^(1-k)``."""
    if k == 0:
        return 0.0
    return wp.pow(0.5, float(k - 1))


@wp.func
def _ext(
    i: int, mass: wp.array[float], gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], num_node_dofs: int
) -> float:
    if i < num_node_dofs:
        return external_force(i, mass, gravity, particle_f)
    return 0.0


@wp.func
def _armijo(energy: wp.array[wp.float64], slope: wp.array[wp.float64], num_candidates: int, c1: float) -> int:
    """The longest step with sufficient decrease. Else the candidate with the lowest objective if it rises by no
    more than float32 noise (near convergence every candidate ties), or ``0``: no step."""
    f0 = energy[0]
    s = slope[0]
    best = int(0)
    for k in range(1, num_candidates):
        if energy[k] <= f0 - wp.float64(c1 * _step_length(k)) * s:
            return k
        if wp.isfinite(energy[k]) and (best == 0 or energy[k] < energy[best]):
            best = k
    if best > 0 and energy[best] <= f0 + wp.float64(_NOISE) * wp.abs(f0):
        return best
    return 0


@wp.kernel
def _dof_kernel(
    q: wp.array[float],
    q_pred: wp.array[float],
    mass: wp.array[float],
    inv_dt2: float,
    fixed: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    num_node_dofs: int,
    # outputs
    b: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
):
    """``b += M (q - q_pred) / dt^2 - f_ext`` (``0`` on fixed DOFs); zero the line search's sums."""
    i = wp.tid()
    if i < energy.shape[0]:
        energy[i] = wp.float64(0.0)
    if i == 0:
        slope[0] = wp.float64(0.0)
    if fixed[i] != 0:
        b[i] = wp.float64(0.0)
        return
    b[i] = b[i] + wp.float64(mass[i] * inv_dt2 * (q[i] - q_pred[i]) - _ext(i, mass, gravity, particle_f, num_node_dofs))


@wp.kernel
def _candidates_kernel(
    q: wp.array[float],
    dq: wp.array[float],
    q_pred: wp.array[float],
    mass: wp.array[float],
    inv_dt2: float,
    fixed: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    num_node_dofs: int,
    hinge_offset: int,
    hinge_step: float,
    # outputs
    q_cand: wp.array2d[float],
    b: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
):
    """``q_cand[k] = q - t_k dq``, the DOF part of each candidate's objective, ``slope = b . dq``, then ``b = 0``; a
    midedge angle's step is clamped to ``hinge_step``."""
    k, i = wp.tid()
    d = dq[i]
    if i >= hinge_offset:
        d = wp.clamp(d, -hinge_step, hinge_step)
    qk = q[i] - _step_length(k) * d
    q_cand[k, i] = qk
    if fixed[i] != 0:
        return
    if k == 0:
        wp.atomic_add(slope, 0, b[i] * wp.float64(d))
        b[i] = wp.float64(0.0)
    r = qk - q_pred[i]
    v = 0.5 * mass[i] * inv_dt2 * r * r - _ext(i, mass, gravity, particle_f, num_node_dofs) * r
    wp.atomic_add(energy, k, wp.float64(v))


@wp.kernel
def _pick_kernel(
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
    num_candidates: int,
    c1: float,
    q_cand: wp.array2d[float],
    # outputs
    q: wp.array[float],
    pick: wp.array[wp.int32],
):
    i = wp.tid()
    k = _armijo(energy, slope, num_candidates, c1)
    if i == 0:
        pick[0] = k
    q[i] = q_cand[k, i]


@wp.kernel
def _velocity_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], q: wp.array[float], dt: float, v: wp.array[float]):
    """Backward Euler's velocity; ``0`` on fixed DOFs."""
    i = wp.tid()
    if fixed[i] != 0:
        v[i] = 0.0
    else:
        v[i] = (q[i] - q0[i]) / dt


@wp.kernel
def _inertia_kernel(
    mass: wp.array[float],
    inv_dt2: float,
    fixed: wp.array[wp.int32],
    indptr: wp.array[wp.int32],
    vals: wp.array[wp.float64],
):
    """``A``'s diagonal: ``+= M / dt^2``, ``1`` on fixed DOFs."""
    i = wp.tid()
    slot = indptr[i]
    if fixed[i] != 0:
        vals[slot] = wp.float64(1.0)
    else:
        vals[slot] = vals[slot] + wp.float64(mass[i] * inv_dt2)
