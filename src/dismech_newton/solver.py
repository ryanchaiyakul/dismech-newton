from functools import partial

import numpy as np
import warp as wp
from newton import Contacts, Control, Model, ModelFlags, State, StateFlags
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import CouplingInterface

from .builder import add_cloth_mesh, add_rod, fix_segment, register_custom_attributes
from .collide import Offset
from .contact import BarrierBlock, make_arm
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
    node,
    padded,
    predict_kernel,
    suspended_tape,
)

_C1 = 1.0e-4  # Armijo's sufficient decrease
_NOISE = 1.0e-5  # a rise of the objective within its float32 rounding
LINESEARCH = 4  # step lengths tried per iteration, halving from 1 (1: the full chord step)
FINISH = 3  # a step's last iterations, after its multipliers are updated whether it has converged or not
# the barrier's step has converged when its Newton step moves no node more than this many radii, nor changes a pair's
# force by more than this fraction of (the force that moves its mass a radius in a step + the force it holds)
TOLERANCE = 0.01


class DiSMechSolver(SolverBase, CouplingInterface):
    """Implicit discrete elastic rods and midedge shells with contact and friction: the elastic elements by
    strain-space ADMM, contact and friction as barrier energies in the same step objective.

    Fixed DOFs are prescribed through ``state_in.dismech.q``. Under a ``wp.Tape`` a step is recorded as its adjoint
    (:meth:`vjp`; recording reads on the host whether a contact touches). Rods and shells collide with Newton's
    shapes, each other and themselves by Offset Geometric Contact, found by the solver (see :mod:`.collide`):
    ``contacts`` is ignored. Coulomb ``mu`` is Newton's: the mean of the two sides' ``shape_material_mu`` (a rod's
    capsules', ``particle_mu`` for a shell vertex), so a rod's friction is its ``cfg.mu``. Obstacles move with
    ``state_in.body_qd``.

    Args:
        iterations: Iterations per step (a fixed count, graph-capturable).
        exact: Gauss-Newton on the step's energy, ``A`` refactorised every iteration (no elastic splitting).
        self_contact: Rod-rod and shell-shell contacts (rod-shell and shape contacts are always on).
        theta: The theta method, in ``[1/2, 1]``: ``1`` implicit Euler, ``1/2`` implicit midpoint (the implicit
            Euler step ``theta dt``, extrapolated). Below ``1`` for contact-free scenes: the extrapolation can step
            into contact; undamped, ADMM's leftover residual can pump the stiff modes (``exact`` does not).
        refactor: With contacts, ``A`` assembled and factorised anew at each iterate while a contact touches; else
            once per step, at the prediction. Rods converge better with it (their factorizations are cheap); sheets,
            whose factorizations dominate a step, are as accurate without it given 10-20 iterations and 1.5-3.5x
            faster (shell_fold, cloth_twist).
        arm_dofs: DOFs per world of bodies that yield to the contacts within the step (``contact.arm``, set by
            :class:`~.coupling.CoupledSolver`), after the state's DOFs in the step's unknowns. No adjoint.

    An entry of Newton's ``SolverCoupled`` (node masses, forces through ``particle_f``; the coupler carries only
    nodes, so a step's input takes its twists, frames and stored strains from the last output), with the rods' proxy
    bodies in its entry and ``SolverCoupledADMM``'s ``gamma = 0`` (its proximal term on the massless proxies breaks
    momentum). The contacts with another entry's bodies are then the coupler's, by node spheres: for rods and sheets
    struck by or resting on bodies, not pinches (no ``rho`` holds a servoed finger against far lighter nodes); for
    those, :class:`~.coupling.CoupledSolver`.

    Contacts are barriers (:class:`~.contact.BarrierBlock`): no candidate passes through one, and each node moves at
    most the smallest particle radius per detection (the pairs are found again when a node has moved half of it).
    With contacts the step iterates until its Newton step moves no node (nor a pair's force, nor a midedge angle's
    offset midpoint) more than :data:`TOLERANCE` radii, or the line search stalls, at most ``iterations`` times.

    ``counts`` (device ``int32[2]``) sums the contact steps that spent their ``iterations`` unconverged, and those
    that stalled. A step is a fixed budget, not a solve to tolerance: compressed cloth (a buckling membrane, whose
    step objective is nonconvex) often ends unconverged, and its :meth:`vjp` (exact at a solution) is then
    approximate.
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
        exact: bool = False,
        self_contact: bool = True,
        theta: float = 1.0,
        refactor: bool = True,
        arm_dofs: int = 0,
    ):
        super().__init__(model)
        if not 0.5 <= theta <= 1.0:
            raise ValueError(f"theta must be in [1/2, 1], got {theta}")
        self.theta = float(theta)
        self.iterations, self.exact, self.refactor = int(iterations), bool(exact), bool(refactor)
        self.num_candidates = nc = LINESEARCH + 1  # candidate 0: the current iterate
        dev = self.device
        self.der = d = model.dismech
        self.mass, self.fixed = dof_constants(model)
        self.num_state_dofs = n0 = self.mass.shape[0]
        k = int(arm_dofs)
        if k:  # the arms' DOFs: no diagonal mass, free
            self.mass = wp.array(np.concatenate([self.mass.numpy(), np.zeros(k * model.world_count)]), dtype=float,
                                 device=dev)
            self.fixed = wp.array(np.concatenate([self.fixed.numpy(), np.zeros(k * model.world_count)]),
                                  dtype=wp.int32, device=dev)
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
        stencils = [edge_dofs, *(t.dofs() for t in types)]  # an edge's nodes: its contacts
        if k:  # a node and its world's arm: its contacts on the arm's bodies
            p = np.arange(model.particle_count)
            arm = n0 + k * np.maximum(model.particle_world.numpy(), 0)[:, None] + np.arange(k)
            stencils.append(np.column_stack([3 * p, 3 * p + 1, 3 * p + 2, arm]).astype(np.int64))
        self.A = SymmetricCSR(n, tuple(stencils), dev)
        self.elements = tuple(
            ElementBlock(t, slots=self.A.slots(t.dofs()), fixed=self.fixed, num_candidates=nc) for t in types
        )
        # every pair within twice its thickness; a node moves at most its smallest radius per detection
        source = Offset(model, self.fixed, self_contact=self_contact)
        self.contact = BarrierBlock(model, self.fixed, source)
        if k:
            self.contact.arm = make_arm(model, source.shape_key0, n0, k)
        self._qw = wp.zeros(n, dtype=float, device=dev) if k else None  # the step's unknowns: the state's, the arms'
        self.cap = float(model.particle_radius.numpy().min()) if self.contact.count else 0.0
        self._tol = TOLERANCE * self.cap
        source.cap = self.cap
        self.blocks = (*self.elements, self.contact)
        self._adjoint = None  # StepAdjoint, built on the first vjp
        # float32 cuDSS: A is the step's chord matrix, an approximation the line search corrects (positive definite:
        # majorized elements and contacts, plus inertia)
        self._linear = sparse_solver(self.A, single=True, spd=True)
        self._b = wp.zeros(n, dtype=wp.float64, device=dev)  # the objective's gradient
        self._dq = wp.zeros(n, dtype=float, device=dev)
        self._q0 = wp.zeros(n, dtype=float, device=dev)  # the move cap's anchor: the start, or the last detection
        self._moved = wp.zeros(1, dtype=wp.int32, device=dev)  # a node moved half its cap since: detect again
        self._q_cand = wp.zeros((nc, n), dtype=float, device=dev)
        self._energy = wp.zeros(nc, dtype=wp.float64, device=dev)
        self._slope = wp.zeros(1, dtype=wp.float64, device=dev)
        self._pick = wp.zeros(1, dtype=wp.int32, device=dev)
        self._go = wp.zeros(1, dtype=wp.int32, device=dev)  # the barrier's step iterates on: not converged
        self._refresh = wp.zeros(1, dtype=wp.int32, device=dev)  # A anew at this iterate
        self._dq_max = wp.zeros(1, dtype=float, device=dev)
        self._impact = wp.ones(1, dtype=float, device=dev)  # the line search's longest step: the first impact
        self._stalled = wp.zeros(1, dtype=wp.int32, device=dev)
        self.counts = wp.zeros(2, dtype=wp.int32, device=dev)  # steps ended unconverged (budget spent), stalled
        # the last step's contact forces on Newton's bodies (body_f layout): what a rigid solver couples back
        self.body_reaction = wp.zeros(max(model.body_count, 1), dtype=wp.spatial_vector, device=dev)
        self._started = False
        self._last_out = None  # the last step's output: what a coupler's next input carries on from (see below)

    def coupling_notify_input_state_update(self, state: State, flags, *, iteration_restart: bool = False, dt=0.0):
        """Newton's ``SolverCoupled`` hands a step only node, body and joint state: the new step's input takes the
        rest (twists, frames, reference twists, stored strains, Dahl stresses) from the last step's output."""
        last = self._last_out
        if last is None or last is state or iteration_restart or not int(flags) & int(StateFlags.PARTICLE_Q):
            return
        flatten_state(state)
        s, p, nd = state.dismech, last.dismech, 3 * self.model.particle_count
        if s.q.shape[0] > nd:
            for k in ("q", "qd"):
                wp.copy(getattr(s, k), getattr(p, k), dest_offset=nd, src_offset=nd, count=s.q.shape[0] - nd)
        for k in ("edge_d1_q", "triplet_ref_twist_q", "triplet_dahl_q", *(e.stored for e in self.elements)):
            if getattr(s, k).shape[0]:
                wp.copy(getattr(s, k), getattr(p, k))

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        """Advance ``state_in`` by ``dt``; under a ``wp.Tape`` the step is recorded as its adjoint (:meth:`vjp`)."""
        dt = float(dt)
        with suspended_tape() as tape:
            self._step(state_in, state_out, dt)
            if tape is not None:
                self._record(tape, state_in, state_out, dt)

    def vjp(self, state_in: State, state_out: State, dt: float, contacts=None) -> None:
        """Backpropagate the step from ``state_out``'s ``.grad``, ``contacts`` its touching contacts as recorded
        (:meth:`~.contact.BarrierBlock.snapshot`). Gradients reach ``state_in`` and, where
        ``requires_grad``, the elements' params and rest strains, ``edge_length``, ``particle_mass`` and
        ``edge_inertia``."""
        if self._qw is not None:
            raise NotImplementedError("no adjoint with yielding bodies (arm_dofs)")
        if self._adjoint is None:
            from .adjoint import StepAdjoint  # noqa: PLC0415 (imports this module's kernels)

            self._adjoint = StepAdjoint(self)
        self._adjoint.vjp(state_in, state_out, float(dt), contacts)

    def _record(self, tape: wp.Tape, state_in: State, state_out: State, dt: float) -> None:
        s_in, s_out = state_in.dismech, state_out.dismech
        names = ("q", "qd", "edge_d1_q", "triplet_ref_twist_q", "triplet_dahl_q", *(e.stored for e in self.elements))
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
        s_in, s_out, n = state_in.dismech, state_out.dismech, self.num_state_dofs
        h = self.theta * dt  # the implicit Euler step solved
        q = s_out.q if self._qw is None else self._qw
        if self._qw is not None:  # the arms' guess: their own step
            wp.copy(q, self.contact.arm.pred, dest_offset=n, count=self.num_dofs - n)
        wp.launch(predict_kernel, dim=n, inputs=[self.fixed, s_in.q, s_in.qd, h], outputs=[self.q_pred, q],
                  device=self.device)
        self._solve(state_in, q, h)
        self.body_reaction.zero_()
        if self.contact.count:
            self.contact.source.reactions(q, self.contact.s, self.contact.arm, self.contact.slip, self.body_reaction)
        wp.launch(end_kernel, dim=n, inputs=[self.fixed, s_in.q, s_in.qd, q, self.theta, dt],
                  outputs=[s_out.q, s_out.qd], device=self.device)
        self._carry(state_in.particle_q, state_in.dismech, state_out.particle_q, state_out.dismech)
        self.update_proxies(state_out)
        self._last_out = state_out

    def _carry(self, prev_nodes: wp.array, p, nodes: wp.array, s) -> None:
        """``s``'s edge frames, reference twists, stored strains and Dahl stresses as a step from ``p`` leaves them
        (each a state's ``dismech``, or the adjoint's stand-in; ``nodes`` its ``q``'s positions)."""
        d, tr, dev = self.der, self.triplets, self.device
        wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0], inputs=[prev_nodes, nodes,
                  d.edge_node0, d.edge_node1, p.edge_d1_q], outputs=[s.edge_d1_q], device=dev)
        wp.launch(advance_ref_twist_kernel, dim=tr.count, inputs=[s.q, s.edge_d1_q, tr.conn, p.triplet_ref_twist_q],
                  outputs=[s.triplet_ref_twist_q], device=dev)
        for e in self.elements:
            e.store(s)
        tr.carry(p, s)

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
        dof_mass(self.model, self.mass[: self.num_state_dofs])

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
        dev, n = self.device, self.num_dofs
        f = bind_dofs(self.dofs, self.model, self.mass, self.fixed, self._no_force, state_in, dt)
        for e in self.elements:
            e.begin_step(state_in, dt, self.exact)
        if not self._started and not self.exact:
            self.reset(state_in)
        self.contact.begin_step(state_in, dt, self.mass)
        q0 = self._q0
        q0.zero_()
        wp.copy(q0, state_in.dismech.q, count=self.num_state_dofs)
        if self.cap > 0.0:  # the prediction, each node's move capped
            wp.launch(_guess_kernel, dim=self.model.particle_count, inputs=[self.fixed, q0, self.contact.source.limit],
                      outputs=[q], device=dev)
        self._energy.zero_()  # the start instead if the prediction is through a contact or past an angle's barrier
        for blk in self.blocks:
            blk.energy(q.reshape((1, n)), self._energy)
        wp.launch(_fallback_kernel, dim=n, inputs=[self._energy, q0], outputs=[q], device=dev)
        # b, the objective's gradient at q: the blocks' part comes from grad() here, then from commit() at each
        # picked candidate; _dof_kernel adds inertia and external force, _candidates_kernel spends it on the slope.
        self._regrad(q)
        self._go.fill_(1)
        for it in range(self.iterations):
            self._room = it < self.iterations - 2  # a multiplier update leaves the solve an iteration
            if self.contact.count and it == self.iterations - FINISH and it > 0:  # the multipliers updated,
                # converged or not, if one moved: the last iterations solve from there
                self.contact.update(q, self._tol, self._dq_max, self._pick, self._go, converged=False)
                wp.capture_if(self.contact.moved, on_true=partial(self._regrad, q))
            if it == 0 or (self.exact and self.cap == 0.0):  # the chord matrix at the prediction (every iterate:
                self._assemble(q, f)  # Gauss-Newton)
                self._linear.invalidate()
            elif self.cap > 0.0 and self.refactor:  # the barrier's: at the iterate while a contact touches
                wp.launch(_refresh_kernel, dim=1, inputs=[self._go, self.contact.touching, int(self.exact)],
                          outputs=[self._refresh], device=dev)
                wp.capture_if(self._refresh, on_true=partial(self._refactor, q, f))
            if self.cap > 0.0 and it > 0:  # converged: the rest of the step skipped
                wp.capture_if(self._go, on_true=partial(self._iterate, state_in, q, f))
            else:
                self._iterate(state_in, q, f)
        self.contact.end_step()
        if self.cap > 0.0:
            wp.launch(_count_kernel, dim=1, inputs=[self._go, self._stalled, self.counts], device=dev)

    def _iterate(self, state_in: State, q: wp.array, f) -> None:
        self._dof(q, f)
        self._linear.solve(self._b, self._dq)
        self._search(state_in, q, f)

    def _regrad(self, q: wp.array) -> None:
        self._b.zero_()
        for blk in self.blocks:
            blk.grad(q, self._b)

    def _dof(self, q: wp.array, f) -> None:
        wp.launch(_dof_kernel, dim=self.num_dofs, inputs=[q, self.q_pred, f],
                  outputs=[self._b, self._energy, self._slope], device=self.device)

    def _search(self, state_in: State, q: wp.array, f) -> None:
        """The line search along the Newton step ``dq = A^-1 b``; with the barrier, the pairs found again if a node
        moved half its cap, and ``_go`` cleared once converged."""
        dev, n, b, q0 = self.device, self.num_dofs, self._b, self._q0
        if self.cap > 0.0:  # the longest step stops short of the first pair's impact
            self._impact.fill_(1.0)
            self.contact.impact(q, self._dq, self._impact)
        wp.launch(_candidates_kernel, dim=(self.num_candidates, padded(n)),
                  inputs=[q, self._dq, self._impact, self.q_pred, f, q0, self.cap, self.contact.source.limit],
                  outputs=[self._q_cand, b, self._energy, self._slope], device=dev)
        for blk in self.blocks:
            blk.energy(self._q_cand, self._energy)
        wp.launch(_pick_kernel, dim=n, inputs=[self._energy, self._slope, self.num_candidates, self._q_cand],
                  outputs=[q, self._pick], device=dev)
        if self.cap > 0.0:
            self._moved.zero_()
            self._dq_max.zero_()
            wp.launch(_moved_kernel, dim=self.model.particle_count, inputs=[q, q0, self._dq, 0.5 * self.cap],
                      outputs=[self._moved, self._dq_max], device=dev)
            h = self.der.hinge_thickness
            if h.shape[0]:
                wp.launch(_hinge_size_kernel, dim=h.shape[0], inputs=[self._dq, self.shells.view.hinge_offset, h],
                          outputs=[self._dq_max], device=dev)
            self.contact.newton_size(q, self._dq, self.cap, self._dq_max)
            wp.capture_if(self._moved, on_true=partial(self._detect, state_in, q))
            wp.launch(_go_kernel, dim=1, inputs=[self._dq_max, self._pick, self._moved, self._tol],
                      outputs=[self._go, self._stalled], device=dev)
            if self._room:  # converged: the multipliers updated, and on if one moved
                wp.capture_if(self._go, on_false=partial(self.contact.update, q, self._tol, self._dq_max, self._pick,
                                                         self._go))
        for blk in self.blocks:
            blk.commit(q, self._pick, b)

    def _assemble(self, q: wp.array, f) -> None:
        self.A.vals.zero_()
        for blk in self.blocks:
            blk.assemble(q, self.A)
        wp.launch(diagonal_kernel, dim=self.num_dofs, inputs=[f], outputs=[self.A.indptr, self.A.vals],
                  device=self.device)

    def _refactor(self, q: wp.array, f) -> None:
        self._assemble(q, f)
        self._linear.factor()

    def _detect(self, state_in: State, q: wp.array) -> None:
        wp.copy(self._q0, q)
        self.contact.detect(state_in, q)


# -- the global step and its line search ---------------------------------------------------


@wp.func
def _step_length(k: int) -> float:
    """Candidate 0 is the current iterate, candidate ``k >= 1`` the step ``2^(1-k)``."""
    if k == 0:
        return 0.0
    return wp.pow(0.5, float(k - 1))


@wp.func
def _armijo(energy: wp.array[wp.float64], slope: wp.array[wp.float64], num_candidates: int) -> int:
    """Newton's full step if its expected decrease is below the objective's float32 rounding (no candidate can be
    told apart). Else the longest step with sufficient decrease; else the candidate with the lowest objective if it
    rises by no more than that rounding, or ``0``: no step (a stall)."""
    f0 = energy[0]
    s = slope[0]
    noise = wp.float64(_NOISE) * wp.abs(f0)
    if s >= wp.float64(0.0) and wp.float64(0.5) * s <= noise and energy[1] <= f0 + noise and wp.isfinite(energy[1]):
        return 1  # the Newton step's decrease is below the objective's rounding: its full step is taken
    best = int(0)
    for k in range(1, num_candidates):
        if wp.isfinite(energy[k]) and energy[k] <= f0 - wp.float64(_C1 * _step_length(k)) * s:
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
    t: wp.array[float],
    q_pred: wp.array[float],
    f: Dofs,
    q0: wp.array[float],
    cap: float,
    limit: wp.array[float],
    q_cand: wp.array2d[float],
    b: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
):
    """``q_cand[k] = q - t_k t dq`` (``t``: the longest step), the DOF part of each candidate's objective,
    ``slope = b . t dq``, then ``b = 0``; a node's move from ``q0`` capped to ``limit`` (a midedge angle's past its
    barrier is the energy's to reject). Launched over ``(k, padded(n))``."""
    k, i = wp.tid()
    e = wp.float64(0.0)
    s = wp.float64(0.0)
    if i < q.shape[0]:
        d = dq[i] * t[0]
        qk = q[i]
        if k > 0:  # (candidate 0 the iterate itself, whatever dq is)
            qk = q[i] - _step_length(k) * d
        if cap > 0.0 and i < f.num_node_dofs and f.fixed[i] == 0:  # the node's move from the start, at most cap
            n = i // 3
            cap = wp.min(cap, limit[n])
            move = wp.vec3(0.0)
            for c in range(3):
                j = 3 * n + c
                move[c] = q[j] - _step_length(k) * (dq[j] * t[0]) - q0[j]
            length = wp.length(move)
            if length > cap:
                qk = q0[i] + move[i - 3 * n] * (cap / length)
        q_cand[k, i] = qk
        if f.fixed[i] == 0:
            e = wp.float64(inertia_energy(f, i, qk - q_pred[i]))
            if k == 0:
                s = b[i] * wp.float64(d)
                b[i] = wp.float64(0.0)
    block_sum(energy, k, e)
    block_sum(slope, 0, s)


@wp.kernel
def _moved_kernel(q: wp.array[float], q0: wp.array[float], dq: wp.array[float], limit: float,
                  moved: wp.array[wp.int32], dq_max: wp.array[float]):
    n = wp.tid()
    if wp.length(node(q, n) - node(q0, n)) > limit:
        moved[0] = 1
    wp.atomic_max(dq_max, 0, wp.length(node(dq, n)))


@wp.kernel
def _hinge_size_kernel(dq: wp.array[float], offset: int, thickness: wp.array[float], dq_max: wp.array[float]):
    """A midedge angle's step as the move of its offset midpoint ``h/2 |d theta|`` (Chen et al. 2026, Sec. 7.3)."""
    h = wp.tid()
    wp.atomic_max(dq_max, 0, 0.5 * thickness[h] * wp.abs(dq[offset + h]))


@wp.kernel
def _go_kernel(dq_max: wp.array[float], pick: wp.array[wp.int32], moved: wp.array[wp.int32], tol: float,
               go: wp.array[wp.int32], stalled: wp.array[wp.int32]):
    """Iterate on while the Newton step moves a node more than ``tol`` and the line search took a step, or the pairs
    were just found again (the objective changed); a step that finds no decrease short of that has stalled."""
    go[0] = wp.where(dq_max[0] > tol and (pick[0] != 0 or moved[0] != 0), 1, 0)
    if dq_max[0] > tol and pick[0] == 0 and moved[0] == 0:
        stalled[0] = 1


@wp.kernel
def _count_kernel(go: wp.array[wp.int32], stalled: wp.array[wp.int32], counts: wp.array[wp.int32]):
    """The step's end: unconverged (its iterations spent) or stalled, counted."""
    counts[0] += go[0]
    counts[1] += stalled[0]
    stalled[0] = 0


@wp.kernel
def _refresh_kernel(go: wp.array[wp.int32], touching: wp.array[wp.int32], exact: int, refresh: wp.array[wp.int32]):
    refresh[0] = wp.where(go[0] != 0 and (touching[0] != 0 or exact != 0), 1, 0)


@wp.kernel
def _guess_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], limit: wp.array[float], q: wp.array[float]):
    n = wp.tid()
    cap = limit[n]
    if fixed[3 * n] == 0:
        move = node(q, n) - node(q0, n)
        length = wp.length(move)
        if length > cap:
            for c in range(3):
                q[3 * n + c] = q0[3 * n + c] + move[c] * (cap / length)


@wp.kernel
def _fallback_kernel(energy: wp.array[wp.float64], q0: wp.array[float], q: wp.array[float]):
    i = wp.tid()
    if not wp.isfinite(energy[0]):
        q[i] = q0[i]


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
