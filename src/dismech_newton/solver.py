from functools import partial

import numpy as np
import warp as wp
from newton import Contacts, Control, Model, ModelFlags, State
from newton.solvers import SolverBase

from .builder import add_cloth_mesh, add_rod, fix_segment, register_custom_attributes
from .collide import Offset
from .contact import ContactBlock
from .element import ElementBlock
from .linear import SymmetricCSR, sparse_solver
from .rod import Rods, advance_frames_kernel, advance_ref_twist_kernel, pose_proxies_kernel, proxy_joints
from .shell import Shells
from .state import (
    Dofs,
    bind_dofs,
    block_sum,
    diagonal_kernel,
    dof_constants,
    dof_mass,
    end_kernel,
    flatten_state,
    inertia_energy,
    inertia_grad,
    padded,
    predict_kernel,
    suspended_tape,
)

_C1 = 1.0e-4  # Armijo's sufficient decrease
_NOISE = 1.0e-5  # a rise of the objective within its float32 rounding


class DiSMechSolver(SolverBase):
    """Implicit discrete elastic rods and midedge shells with contact and friction, by strain-space ADMM.

    Fixed DOFs are prescribed through ``state_in.dismech.q``. Under a ``wp.Tape`` a step is recorded as its adjoint
    (:meth:`vjp`; with contacts, recording reads their count on the host). Rods and shells collide with Newton's
    shapes, each other and themselves by Offset Geometric Contact, found by the solver (see :mod:`.collide`):
    ``contacts`` is ignored. Coulomb ``mu`` is Newton's: the mean of the two sides' ``shape_material_mu`` (a rod's
    capsules', ``particle_mu`` for a shell vertex), so a rod's friction is its ``cfg.mu``; ``friction`` sets every
    contact's. Obstacles move with ``state_in.body_qd``.

    Args:
        iterations: Iterations per step (a fixed count, graph-capturable).
        beta: Strain penalty over the energy's stiffness ``K + C/dt``.
        linesearch: Step lengths tried per iteration, halving from ``1`` (``1`` takes the full chord step).
        exact: Gauss-Newton on the step's energy, ``A`` refactorised every iteration (no elastic splitting).
        self_contact: Rod-rod and shell-shell contacts (rod-shell and shape contacts are always on).
        friction: Every contact's Coulomb ``mu`` (else the materials' means).
        contact_beta: Contact penalty over the inertia-only inverse Delassus diagonal.
        hinge_step: Trust region of a shell's midedge angles per iteration [rad] (near ``psi + theta = pi/2`` their
            tangent stiffens without bound).
        engage_gap: A contact takes part in a step only if the inertial prediction brings it within that many
            thicknesses of touching, or it carried force in the last step (a separated contact's penalty in ``A``
            would slow the motion near obstacles).
        theta: The theta method, in ``[1/2, 1]``: ``1`` implicit Euler, ``1/2`` implicit midpoint (the implicit
            Euler step ``theta dt``, extrapolated). Below ``1`` for contact-free scenes: the extrapolation can step
            into contact; undamped, ADMM's leftover residual can pump the stiff modes (``exact`` does not).
    """

    register_custom_attributes = staticmethod(register_custom_attributes)
    add_rod = staticmethod(partial(add_rod, collide=True))  # colliding by default
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
        friction: float | None = None,
        contact_beta: float = 16.0,
        engage_gap: float = 0.25,
        hinge_step: float = 0.1,
        theta: float = 1.0,
    ):
        super().__init__(model)
        if linesearch < 1:
            raise ValueError("linesearch must be >= 1")
        if not 0.5 <= theta <= 1.0:
            raise ValueError(f"theta must be in [1/2, 1], got {theta}")
        self.theta = float(theta)
        self.iterations, self.exact, self.hinge_step = int(iterations), bool(exact), float(hinge_step)
        self.num_candidates = nc = int(linesearch) + 1  # candidate 0: the current iterate
        dev = self.device
        self.der = d = model.dismech
        self.mass, self.fixed = dof_constants(model)
        self.num_dofs = n = self.mass.shape[0]
        self.num_node_dofs = 3 * model.particle_count
        self._no_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=dev)
        self.dofs = Dofs()  # set per step (bind_dofs)
        self.q_pred = wp.zeros(n, dtype=float, device=dev)
        self._no_velocity = wp.zeros(0, dtype=wp.spatial_vector, device=dev)
        self._no_joint = wp.zeros(0, dtype=float, device=dev)
        self._has_proxies = bool((d.edge_body.numpy() >= 0).any())
        self._edge_joint = wp.array(proxy_joints(model), dtype=wp.vec2i, device=dev)

        types = self.triplets, self.shells = Rods(model), Shells(model, self.num_node_dofs + d.edge_length.shape[0])
        node0, node1 = d.edge_node0.numpy().astype(np.int64), d.edge_node1.numpy().astype(np.int64)
        edge_dofs = np.column_stack([3 * node0 + k for k in range(3)] + [3 * node1 + k for k in range(3)])
        self.A = SymmetricCSR(n, (edge_dofs, *(t.dofs() for t in types)), dev)  # an edge's nodes: its contacts
        self.elements = tuple(
            ElementBlock(t, slots=self.A.slots(t.dofs()), fixed=self.fixed, num_candidates=nc, beta=beta) for t in types
        )
        source = Offset(model, self.fixed, self_contact=self_contact, engage_gap=engage_gap, friction=friction)
        self.contact = ContactBlock(model, self.fixed, nc, source, beta=contact_beta)
        self.blocks = (*self.elements, self.contact)
        self._adjoint = None  # StepAdjoint, built on the first vjp
        # float32 cuDSS: A is the step's chord matrix, an approximation the line search corrects
        self._linear = sparse_solver(self.A, single=True)
        self._b = wp.zeros(n, dtype=wp.float64, device=dev)  # the objective's gradient
        self._dq = wp.zeros(n, dtype=float, device=dev)
        self._q_cand = wp.zeros((nc, n), dtype=float, device=dev)
        self._energy = wp.zeros(nc, dtype=wp.float64, device=dev)
        self._slope = wp.zeros(1, dtype=wp.float64, device=dev)
        self._pick = wp.zeros(1, dtype=wp.int32, device=dev)
        self._started = False

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """Advance ``state_in`` by ``dt``; under a ``wp.Tape`` the step is recorded as its adjoint (:meth:`vjp`)."""
        dt = float(dt)
        with suspended_tape() as tape:
            self._step(state_in, state_out, dt)
            if tape is not None:
                self._record(tape, state_in, state_out, dt)

    def vjp(self, state_in: State, state_out: State, dt: float, contacts=None) -> None:
        """Backpropagate the step from ``state_out``'s ``.grad``; ``contacts``: the step's
        :meth:`~.contact.ContactBlock.snapshot`. Gradients reach ``state_in`` and, where ``requires_grad``, the
        elements' params and rest strains, ``edge_length``, ``particle_mass`` and ``edge_inertia``."""
        if self._adjoint is None:
            from .adjoint import StepAdjoint  # noqa: PLC0415 (imports this module's kernels)

            self._adjoint = StepAdjoint(self)
        self._adjoint.vjp(state_in, state_out, float(dt), contacts)

    def _record(self, tape: wp.Tape, state_in: State, state_out: State, dt: float) -> None:
        s_in, s_out = state_in.dismech, state_out.dismech
        names = ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", *(e.stored for e in self.elements))
        outputs = [getattr(s_out, k) for k in names]
        if not any(a.requires_grad for a in outputs):
            return
        contacts = self.contact.snapshot()
        inputs = [
            *(getattr(s_in, k) for k in names),
            state_in.particle_f,
            self.der.edge_length,
            self.model.particle_mass,
            self.der.edge_inertia,
            *(a for e in self.elements for a in (e.params, e.rest)),
        ]
        arrays = [a for a in (*inputs, *outputs) if a is not None and a.requires_grad]
        tape.record_func(lambda: self.vjp(state_in, state_out, dt, contacts), arrays)

    def _step(self, state_in: State, state_out: State, dt: float) -> None:
        flatten_state(state_in)
        flatten_state(state_out)
        s_in, s_out, n = state_in.dismech, state_out.dismech, self.num_dofs
        h = self.theta * dt  # the implicit Euler step solved
        wp.launch(predict_kernel, dim=n, inputs=[self.fixed, s_in.q, s_in.qd, h], outputs=[self.q_pred, s_out.q],
                  device=self.device)
        self._solve(state_in, s_out.q, h)
        wp.launch(end_kernel, dim=n, inputs=[self.fixed, s_in.q, s_in.qd, s_out.q, self.theta, dt],
                  outputs=[s_out.q, s_out.qd], device=self.device)
        self._carry(state_in.particle_q, state_in.dismech, state_out.particle_q, state_out.dismech)
        self.update_proxies(state_out)

    def _carry(self, prev_nodes: wp.array, p, nodes: wp.array, s) -> None:
        """``s``'s edge frames, reference twists and stored strains as a step from ``p`` leaves them (each a state's
        ``dismech``, or the adjoint's stand-in; ``nodes`` its ``q``'s positions)."""
        d, tr, dev = self.der, self.triplets, self.device
        wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0], inputs=[prev_nodes, nodes,
                  d.edge_node0, d.edge_node1, p.edge_d1_q], outputs=[s.edge_d1_q], device=dev)
        wp.launch(advance_ref_twist_kernel, dim=tr.count, inputs=[s.q, s.edge_d1_q, tr.conn, p.triplet_ref_twist_q],
                  outputs=[s.triplet_ref_twist_q], device=dev)
        for e in self.elements:
            e.store(s)

    def restart(self, state: State, prev: State) -> None:
        """Start the next step from ``state``, its DOFs set (to data, say) and the rest of its state as a step from
        ``prev`` would leave it (``prev``: the state its DOFs moved from, or the model's rest state), without a warm
        start (:meth:`reset`)."""
        flatten_state(state)
        flatten_state(prev)
        self._carry(prev.particle_q, prev.dismech, state.particle_q, state.dismech)
        self.reset(state)

    def reset(self, state: State, world_mask: wp.array | None = None, flags: int | None = None) -> None:
        """Start the next step from ``state`` without the last step's warm start: for a step that does not continue
        the last one, such as a restart from data."""
        if world_mask is not None:
            raise NotImplementedError(f"{type(self).__name__}.reset resets every world")
        flatten_state(state)
        for e in self.elements:
            e.reset(state)
        self.contact.reset()
        self._started = True

    def refresh_mass(self) -> None:
        """Re-read the masses and twist inertias after they changed."""
        dof_mass(self.model, self.mass)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """``MODEL_PROPERTIES``: :meth:`refresh_mass`."""
        if flags & ModelFlags.MODEL_PROPERTIES:
            self.refresh_mass()

    @property
    def graph_capturable(self) -> bool:
        return self.device.is_cuda and self._linear.graph_capturable

    def update_proxies(self, state: State) -> None:
        """Pose the capsule proxies and set their velocities (and their free joints)."""
        if not self._has_proxies or state.body_q is None:
            return
        body_qd = state.body_qd if state.body_qd is not None else self._no_velocity
        joints = state.joint_q is not None and state.joint_qd is not None and state.joint_q.shape[0] > 0
        d, s = self.der, state.dismech
        wp.launch(pose_proxies_kernel, dim=d.edge_length.shape[0],
                  inputs=[state.particle_q, state.particle_qd, s.edge_q, s.edge_d1_q, d.edge_node0, d.edge_node1,
                          d.edge_body, self._edge_joint, int(state.body_qd is not None)],
                  outputs=[state.body_q, body_qd, state.joint_q if joints else self._no_joint,
                           state.joint_qd if joints else self._no_joint], device=self.device)

    def _solve(self, state_in: State, q: wp.array, dt: float) -> None:
        dev, n, b = self.device, self.num_dofs, self._b
        f = bind_dofs(self.dofs, self.model, self.mass, self.fixed, self._no_force, state_in, dt)
        for e in self.elements:
            e.begin_step(state_in, dt, self.exact)
        if not self._started and not self.exact:
            self.reset(state_in)
        self.contact.begin_step(state_in, dt, q, self.mass, f.inv_dt2)
        # b, the objective's gradient at q: the blocks' part comes from grad() here, then from commit() at each
        # picked candidate; _dof_kernel adds inertia and external force, _candidates_kernel spends it on the slope.
        b.zero_()
        for blk in self.blocks:
            blk.grad(q, b)
        for it in range(self.iterations):
            if it == 0 or self.exact:  # the chord matrix at the prediction (every iterate: Gauss-Newton)
                self.A.vals.zero_()
                for blk in self.blocks:
                    blk.assemble(q, self.A)
                wp.launch(diagonal_kernel, dim=n, inputs=[f], outputs=[self.A.indptr, self.A.vals], device=dev)
                self._linear.invalidate()
            wp.launch(_dof_kernel, dim=n, inputs=[q, self.q_pred, f], outputs=[b, self._energy, self._slope],
                      device=dev)
            self._linear.solve(b, self._dq)
            wp.launch(_candidates_kernel, dim=(self.num_candidates, padded(n)),
                      inputs=[q, self._dq, self.q_pred, f, self.shells.view.hinge_offset, self.hinge_step],
                      outputs=[self._q_cand, b, self._energy, self._slope], device=dev)
            for blk in self.blocks:
                blk.energy(self._q_cand, self._energy)
            wp.launch(_pick_kernel, dim=n, inputs=[self._energy, self._slope, self.num_candidates, self._q_cand],
                      outputs=[q, self._pick], device=dev)
            for blk in self.blocks:
                blk.commit(q, self._pick, b)
        self.contact.end_step()


# -- the global step and its line search ---------------------------------------------------


@wp.func
def _step_length(k: int) -> float:
    """Candidate 0 is the current iterate, candidate ``k >= 1`` the step ``2^(1-k)``."""
    if k == 0:
        return 0.0
    return wp.pow(0.5, float(k - 1))


@wp.func
def _armijo(energy: wp.array[wp.float64], slope: wp.array[wp.float64], num_candidates: int) -> int:
    """The longest step with sufficient decrease. Else the candidate with the lowest objective if it rises by no
    more than float32 noise (near convergence every candidate ties), or ``0``: no step."""
    f0 = energy[0]
    s = slope[0]
    best = int(0)
    for k in range(1, num_candidates):
        if energy[k] <= f0 - wp.float64(_C1 * _step_length(k)) * s:
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
    f: Dofs,
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
    if f.fixed[i] != 0:
        b[i] = wp.float64(0.0)
    else:
        b[i] = b[i] + wp.float64(inertia_grad(f, i, q[i] - q_pred[i]))


@wp.kernel
def _candidates_kernel(
    q: wp.array[float],
    dq: wp.array[float],
    q_pred: wp.array[float],
    f: Dofs,
    hinge_offset: int,
    hinge_step: float,
    q_cand: wp.array2d[float],
    b: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
):
    """``q_cand[k] = q - t_k dq``, the DOF part of each candidate's objective, ``slope = b . dq``, then ``b = 0``; a
    midedge angle's step is clamped to ``hinge_step``. Launched over ``(k, padded(n))``."""
    k, i = wp.tid()
    e = wp.float64(0.0)
    s = wp.float64(0.0)
    if i < q.shape[0]:
        d = dq[i]
        if i >= hinge_offset:
            d = wp.clamp(d, -hinge_step, hinge_step)
        qk = q[i] - _step_length(k) * d
        q_cand[k, i] = qk
        if f.fixed[i] == 0:
            e = wp.float64(inertia_energy(f, i, qk - q_pred[i]))
            if k == 0:
                s = b[i] * wp.float64(d)
                b[i] = wp.float64(0.0)
    block_sum(energy, k, e)
    block_sum(slope, 0, s)


@wp.kernel
def _pick_kernel(
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
    num_candidates: int,
    q_cand: wp.array2d[float],
    q: wp.array[float],
    pick: wp.array[wp.int32],
):
    i = wp.tid()
    k = _armijo(energy, slope, num_candidates)
    if i == 0:
        pick[0] = k
    q[i] = q_cand[k, i]
