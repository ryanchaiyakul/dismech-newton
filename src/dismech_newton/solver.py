import logging
import re
from functools import partial
from typing import Any

import numpy as np
import warp as wp
from newton import BodyFlags, Contacts, Control, Model, ModelFlags, State, StateFlags
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import CouplingInterface

from .builder import add_cloth_mesh, add_rod, fix_segment, register_custom_attributes
from .collide import Offset
from .contact import BarrierBlock, make_arm
from .coupling import ProxyArm, ProxyPins, _turn
from .element import ElementBlock
from .linear import SymmetricCSR, sparse_solver
from .rod import Rods, advance_frames_kernel, advance_ref_twist_kernel, pose_proxies_kernel, proxy_joints
from .shell import Shells
from .state import (
    Dofs,
    Worlds,
    _external,
    bind_dofs,
    diagonal_kernel,
    dof_constants,
    dof_mass,
    dof_worlds,
    end_kernel,
    flatten_state,
    inertia_energy,
    inertia_grad,
    masked_copy_kernel,
    node,
    padded,
    predict_kernel,
    suspended_tape,
    world_sum,
)

wp.set_module_options({"enable_backward": False})

_C1 = 1.0e-4  # Armijo's sufficient decrease
_NOISE = 1.0e-5  # a rise of the objective within its float32 rounding
CG = 2  # SPD PCG sweeps on two-sided contacts (m = 0 scales A^-1 b; m = 1..CG-1 add CG-1 back-solves)
LINESEARCH = 4  # step lengths tried per iteration, halving from 1
NEST = 8  # iterations nested per conditional chain (:meth:`SolverDiSMech.step`)
# the barrier's step has converged when its Newton step moves no node more than this many radii, nor changes a pair's
# force by more than this fraction of (the force that moves its mass a radius in a step + the force it holds)
TOLERANCE = 0.01
ULP = 1.0e-6  # float32's reach: a few of its epsilons of a coordinate (+ the cap): forces below it, in a
# DOF's inertia, are not resolved (the force balance's floor)
EPS32 = 1.1920929e-07  # float32's epsilon: a coordinate's ulp is at most this times it
CARRIED = ("edge_d1_q", "triplet_ref_twist_q", "triplet_dahl_q")  # state a step carries beyond the DOFs (and strains)


class _OwnResetState(logging.Filter):
    """Newton's ``SolverCoupled`` warns that its reset skips the ``dismech:*`` state attributes (custom frequencies):
    :meth:`SolverDiSMech.reset` resets them itself, so only a warning naming others passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "reset reconciliation skips STATE attributes" not in message:
            return True
        return not all(name.startswith("dismech:") for name in re.findall(r"'([^']+)' \(", message))


logging.getLogger("newton._src.solvers.coupled.solver_coupled").addFilter(_OwnResetState())


class SolverDiSMech(SolverBase, CouplingInterface):
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

    An entry of Newton's ``SolverCoupled`` (node masses, forces through ``particle_f``; the coupler carries only
    nodes, so a step's input takes its twists, frames and stored strains from the last output), with the rods' proxy
    bodies in its entry and ``SolverCoupledADMM``'s ``gamma = 0`` (its proximal term on the massless proxies breaks
    momentum). The contacts with another entry's bodies are then the coupler's, by node spheres: for rods and sheets
    struck by or resting on bodies, not pinches (no ``rho`` holds a servoed finger against far lighter nodes). As
    ``SolverCoupledProxy``'s destination (as VBD's), the view's ``BodyFlags.PROXY`` bodies with shapes yield to the
    contacts within the step, their DOFs after the state's in the step's unknowns (:class:`~.coupling.ProxyArm`; no
    adjoint): a pinch holds.

    Contacts are barriers (:class:`~.contact.BarrierBlock`) on pairs frozen per detection (each gap linear in the
    DOFs): no candidate passes through a found pair, and each node moves at most the smallest particle radius per
    detection, so no pair not yet found can close (Divide and Truncate's isotropic region; the pairs are found again
    when a node has moved half of it). The line search's longest step stops where a pair's barrier changes the
    Newton model (:meth:`~.contact.BarrierBlock.impact`): a touching pair's impact, an untouched pair's onset.
    With contacts the step iterates until its Newton step moves no node (nor a pair's force, nor a midedge angle's
    offset midpoint) more than :data:`TOLERANCE` radii and the nodes' net force is within it of their
    loads, or the line search stalls, at most ``iterations`` times.

    Each world's step is its own solve: its line search, convergence and failure (a world that has converged takes
    no step while others iterate on).

    ``counts`` (device ``int32[3]``) sums the worlds' contact steps that spent their ``iterations`` unconverged, those
    that stalled, and those whose Newton step was not finite (it is taken: the state goes non-finite, loudly). A step
    is a fixed budget, not a solve to tolerance: compressed cloth (a buckling membrane, whose step objective is
    nonconvex) often ends unconverged, and its :meth:`vjp` (exact at a solution) is then
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
    ):
        super().__init__(model)
        if not 0.5 <= theta <= 1.0:
            raise ValueError(f"theta must be in [1/2, 1], got {theta}")
        self.theta = float(theta)
        self.iterations, self.exact, self.refactor = int(iterations), bool(exact), bool(refactor)
        self.tolerance = TOLERANCE  # (read here: tests set the module's)
        self.num_candidates = nc = LINESEARCH + 1  # candidate 0: the current iterate
        dev = self.device
        self.der = d = model.dismech
        self.mass, self.fixed = dof_constants(model)
        self.num_state_dofs = n0 = self.mass.shape[0]
        proxies = model.body_count and (model.body_flags.numpy() & int(BodyFlags.PROXY)).any()
        self._proxy_arm = arm = ProxyArm(model) if proxies else None
        k = arm.k if arm is not None else 0
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
            dofs = n0 + k * np.maximum(model.particle_world.numpy(), 0)[:, None] + np.arange(k)
            stencils.append(np.column_stack([3 * p, 3 * p + 1, 3 * p + 2, dofs]).astype(np.int64))
        self.A = SymmetricCSR(n, tuple(stencils), dev)
        of = dof_worlds(model, n, stencils)
        self.num_worlds = W = int(of.max(initial=0)) + 1
        self.worlds = wd = Worlds()
        wd.of, wd.active = wp.array(of, dtype=wp.int32, device=dev), wp.ones(W, dtype=wp.int32, device=dev)
        wd.count, wd.stride = W, nc
        rows = {"edge_d1_q": of[3 * node0]}  # a masked reset's rows: per attribute, its rows' worlds
        for t in types:  # an element's: its first node's (a rod's carry its triplets' reference twists, Dahl)
            for name in (t.stored, *(CARRIED[1:] if t is self.triplets else ())):
                rows[name] = of[t.dofs()[:, 0]] if t.count else []
        self._rows = {k: wp.array(v, dtype=wp.int32, device=dev) for k, v in rows.items() if len(v)}
        self._rest = self._scratch = None  # a masked reset's: the model's state, and the reset worlds' carried
        self.elements = tuple(
            ElementBlock(t, slots=self.A.slots(t.dofs()), fixed=self.fixed, num_candidates=nc) for t in types
        )
        self._anchor = wp.zeros(n, dtype=float, device=dev)  # per DOF, its elements' coupling to fixed DOFs
        for e in self.elements:
            e.worlds, e.anchor = wd, self._anchor
        # every pair within twice its thickness; a node moves at most its smallest radius per detection
        source = Offset(model, self.fixed, self_contact=self_contact, world=of)
        self.contact = BarrierBlock(model, self.fixed, source, of, nc)
        if k:
            self.contact.arm = make_arm(model, source.shape_key0, n0, k)
            arm.bind(self.contact.arm)
        self._arm_radius = arm.radius if k else wp.zeros(0, dtype=float, device=dev)  # per proxy body, its 6 DOFs
        self.pins = ProxyPins(model, self.contact.arm, self.mass, self.fixed) if k else None  # rods on proxy bodies
        if self.pins:
            self.pins.world, self.pins.stride = wd.of, nc
        self._qw = wp.zeros(n, dtype=float, device=dev) if k else None  # the step's unknowns: the state's, the arms'
        self.cap = float(model.particle_radius.numpy().min()) if self.contact.count else 0.0
        self._tol = self.tolerance * self.cap
        source.cap = self.cap
        self.blocks = (*self.elements, self.contact, *((self.pins,) if self.pins and self.pins.count else ()))
        self._adjoint = None  # StepAdjoint, built on the first vjp
        # float32 cuDSS: A is the step's chord matrix, an approximation the line search corrects (positive definite:
        # majorized elements and contacts, plus inertia)
        self._linear = sparse_solver(self.A, single=True, spd=True)
        self._b = wp.zeros(n, dtype=wp.float64, device=dev)  # the objective's gradient
        self._dq = wp.zeros(n, dtype=float, device=dev)
        self._q0 = wp.zeros(n, dtype=float, device=dev)  # the move cap's anchor: the start, or the last detection
        # per world (a flag's [W]: in any world): a node moved half its cap since: detect again
        self._moved = wp.zeros(W + 1, dtype=wp.int32, device=dev)
        self._q_cand = wp.zeros((nc, n), dtype=float, device=dev)
        self._energy = wp.zeros(W * nc, dtype=wp.float64, device=dev)  # [w * nc + candidate]
        self._slope = wp.zeros(W, dtype=wp.float64, device=dev)
        self._pick = wp.zeros(W, dtype=wp.int32, device=dev)
        self._go = wp.zeros(W + 1, dtype=wp.int32, device=dev)  # the barrier's step iterates on: not converged
        self._settle = wp.zeros(W + 1, dtype=wp.int32, device=dev)  # converged at this iterate: its multipliers
        self._refresh = wp.zeros(1, dtype=wp.int32, device=dev)  # A anew at this iterate
        self._dq_max = wp.zeros(W, dtype=float, device=dev)
        self._unbalanced = wp.zeros(W, dtype=float, device=dev)  # per world, max |residual| / its tolerance
        self._load = wp.zeros(n, dtype=float, device=dev)  # per DOF, the loads it carries (the tolerance's scale)
        self._floor = wp.zeros(n, dtype=float, device=dev)
        self._net = wp.zeros(_N * W, dtype=wp.float64, device=dev)  # per world: net force, load, floors (_NET...)
        self._impact = wp.ones(W, dtype=float, device=dev)  # per world, the line search's longest step (impact)
        self._stalled = wp.zeros(W, dtype=wp.int32, device=dev)
        self._failed = wp.zeros(W, dtype=wp.int32, device=dev)  # the step's Newton step was not finite
        self._cg = [wp.zeros(n, dtype=wp.float64, device=dev) for _ in range(4)]  # PCG: x, r, p, w
        self._cg_z = wp.zeros(n, dtype=float, device=dev)
        self._cgs = wp.zeros(W * _S, dtype=wp.float64, device=dev)
        self._cg_go = wp.zeros(1, dtype=wp.int32, device=dev)
        self.counts = wp.zeros(3, dtype=wp.int32, device=dev)  # steps ended unconverged (budget spent), stalled, failed
        # the last step's contact forces on Newton's bodies (body_f layout): what a rigid solver couples back
        self.body_reaction = wp.zeros(max(model.body_count, 1), dtype=wp.spatial_vector, device=dev)
        # a coupler's next input carries on from these (see below): the last step's output, once there is one
        self._carried = model.state()
        flatten_state(self._carried)
        self._carry_valid = wp.zeros(1, dtype=wp.int32, device=dev)

    def coupling_notify_input_state_update(self, state: State, flags, *, iteration_restart: bool = False, dt=0.0):
        """Newton's ``SolverCoupled`` hands a step only node, body and joint state: the new step's input takes the
        rest (twists, frames, reference twists, stored strains, Dahl stresses) from the last step's output."""
        if iteration_restart and self._proxy_arm is not None:
            self._proxy_arm.restart = True
        if iteration_restart or not int(flags) & int(StateFlags.PARTICLE_Q):
            return
        flatten_state(state)
        for dst, src in self._carry_pairs(state.dismech, self._carried.dismech):
            wp.launch(_copy_if_kernel, dim=dst.shape[0], inputs=[self._carry_valid, src], outputs=[dst],
                      device=self.device)

    def _carry_pairs(self, s, p):
        """The arrays a step's input takes from the last output (``s``'s, ``p``'s): all but the nodes."""
        nd = 3 * self.model.particle_count
        if s.q.shape[0] > nd:
            for k in ("q", "qd"):
                yield getattr(s, k)[nd:], getattr(p, k)[nd:]
        for k in (*CARRIED, *(e.stored for e in self.elements)):
            if getattr(s, k).shape[0]:
                yield getattr(s, k), getattr(p, k)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float,
             *, observables: SolverBase.Observables | None = None) -> None:
        """Advance ``state_in`` by ``dt``; under a ``wp.Tape`` the step is recorded as its adjoint (:meth:`vjp`).
        No solver observables are supported (the contact forces on bodies: :attr:`body_reaction`)."""
        self.validate_observables(observables)
        dt = float(dt)
        with suspended_tape() as tape:
            self._step(state_in, state_out, dt)
            if tape is not None:
                self._record(tape, state_in, state_out, dt)

    def vjp(self, state_in: State, state_out: State, dt: float, contacts=None, arm=None) -> None:
        """Backpropagate the step from ``state_out``'s ``.grad``, ``contacts`` its touching contacts as recorded
        (:meth:`~.contact.BarrierBlock.snapshot`), ``arm`` its proxies' (:meth:`~.coupling.ProxyArm.snapshot`).
        Gradients reach ``state_in`` (with proxies: their synced ``body_q``, ``body_qd``) and, where
        ``requires_grad``, the elements' params and rest strains, ``edge_length``, ``particle_mass`` and
        ``edge_inertia``."""
        if self._adjoint is None:
            from .adjoint import StepAdjoint  # noqa: PLC0415 (imports this module's kernels)

            self._adjoint = StepAdjoint(self)
        self._adjoint.vjp(state_in, state_out, float(dt), contacts, arm)

    def _record(self, tape: wp.Tape, state_in: State, state_out: State, dt: float) -> None:
        s_in, s_out = state_in.dismech, state_out.dismech
        names = ("q", "qd", *CARRIED, *(e.stored for e in self.elements))
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
        arm = None
        if self._proxy_arm is not None:  # the proxies' step, held for its adjoint
            inputs += [state_in.body_q, state_in.body_qd]
            outputs += [state_out.body_q, state_out.body_qd]
            arm = self._proxy_arm.snapshot(self._qw[self.num_state_dofs:], state_in, self.pins)
        arrays = [a for a in (*inputs, *outputs) if a is not None and a.requires_grad]
        tape.record_func(lambda: self.vjp(state_in, state_out, dt, contacts, arm), arrays)

    def _step(self, state_in: State, state_out: State, dt: float) -> None:
        flatten_state(state_in)
        flatten_state(state_out)
        s_in, s_out, n = state_in.dismech, state_out.dismech, self.num_state_dofs
        h = self.theta * dt  # the implicit Euler step solved
        q = s_out.q if self._qw is None else self._qw
        body_q, body_qd = state_in.body_q, state_in.body_qd
        if self._proxy_arm is not None:  # the proxies yield from their own start: their moves are the arm's
            state_in.body_q, state_in.body_qd = self._proxy_arm.update(state_in, h)
        try:
            if self._qw is not None:  # the arms' guess: their own step
                wp.copy(q, self.contact.arm.pred, dest_offset=n, count=self.num_dofs - n)
            wp.launch(predict_kernel, dim=n, inputs=[self.fixed, s_in.q, s_in.qd, h], outputs=[self.q_pred, q],
                      device=self.device)
            self._solve(state_in, q, h)
            self.body_reaction.zero_()
            if self.contact.count:
                self.contact.source.reactions(q, self.contact.s, self.contact.arm, self.contact.slip,
                                              self.body_reaction)
            if self.pins and self.pins.count:
                self.pins.reactions(q, self.body_reaction)
        finally:
            state_in.body_q, state_in.body_qd = body_q, body_qd
        for src, dst in ((body_q, state_out.body_q), (body_qd, state_out.body_qd)):  # bodies we don't move: as given
            if src is not None and dst is not None and src.ptr != dst.ptr:
                wp.copy(dst, src)
        if self._proxy_arm is not None:
            self._proxy_arm.finish(q, state_in, state_out, dt, h)
        wp.launch(end_kernel, dim=n, inputs=[self.fixed, s_in.q, s_in.qd, q, self.theta, dt],
                  outputs=[s_out.q, s_out.qd], device=self.device)
        self._carry(state_in.particle_q, state_in.dismech, state_out.particle_q, state_out.dismech)
        self.update_proxies(state_out)
        for dst, src in self._carry_pairs(self._carried.dismech, state_out.dismech):
            wp.copy(dst, src)
        self._carry_valid.fill_(1)

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

    def reset(self, state: State, world_mask: wp.array[wp.bool] | None = None, flags: StateFlags | int | None = None
              ) -> None:
        """Start the next step from ``state`` without the last step's warm start: for a step that does not continue
        the last one, such as a restart from data.

        With ``world_mask`` (``bool[world_count + 1]``, Newton's): only those worlds, which start anew from their nodes
        in ``state`` (an environment reset; graph-capturable): their twists, rates and midedge angles the model's,
        their frames, reference twists, stored strains and Dahl stresses as a step from the model's state to these
        nodes leaves them (:meth:`restart`), in ``state`` and in what a coupler's next input carries."""
        flatten_state(state)
        if world_mask is not None:
            self._reset_worlds(state, world_mask)
            return
        self.contact.reset()
        if self._proxy_arm is not None:
            self._proxy_arm.reset()

    def _reset_worlds(self, state: State, mask: wp.array) -> None:
        if self.num_worlds != max(self.model.world_count, 1):
            raise NotImplementedError("global nodes in a multi-world model: reset every world")
        if self._rest is None:
            self._rest, self._scratch = self.model.state(), self.model.state()
            flatten_state(self._rest)
            flatten_state(self._scratch)
        r, t, of, dev = self._rest.dismech, self._scratch, self.worlds.of, self.device
        wp.copy(t.dismech.q, r.q)
        wp.copy(t.particle_q, state.particle_q)
        self._carry(self._rest.particle_q, r, t.particle_q, t.dismech)
        nd, n = self.num_node_dofs, self.num_state_dofs
        for s in (state.dismech, self._carried.dismech):
            for k, src in (("q", t.dismech.q), ("qd", r.qd)):
                if n > nd:
                    wp.launch(masked_copy_kernel, dim=n - nd, inputs=[nd, of, mask, src, getattr(s, k)], device=dev)
            for k, world in self._rows.items():
                wp.launch(masked_copy_kernel, dim=world.shape[0], inputs=[0, world, mask, getattr(t.dismech, k),
                          getattr(s, k)], device=dev)
        self.contact.reset(mask, of)
        if self._proxy_arm is not None:
            self._proxy_arm.reset(mask)

    def refresh_mass(self) -> None:
        """Re-read the masses and twist inertias after they changed."""
        dof_mass(self.model, self.mass[: self.num_state_dofs])

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """``MODEL_PROPERTIES``: :meth:`refresh_mass` and the rods' and sheets' friction (``shape_material_mu`` of
        their capsules, ``particle_mu``). The elements' stiffness and damping (``dismech.triplet_params``,
        ``face_params``) and the other side's friction (``shape_material_mu``) are read every step: changed in place,
        they need no notice (per world at a reset: domain randomization)."""
        if flags & ModelFlags.MODEL_PROPERTIES:
            self.refresh_mass()
            if self.contact.count:
                self.contact.source.refresh_mu()

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
        dev, n, W = self.device, self.num_dofs, self.num_worlds
        f = bind_dofs(self.dofs, self.model, self.mass, self.fixed, self._no_force, state_in, dt)
        for e in self.elements:
            e.begin_step(state_in, dt, self.exact)
        if self._arm_radius.shape[0]:  # the step's start: the proxies not yet moved
            self._arm_sweep(self._q0, 0)
        self.contact.begin_step(state_in, dt, self.mass, self._go)  # _go: the last step's end
        if self.pins and self.pins.count:
            self.pins.begin_step(state_in.body_q, dt)
        q0 = self._q0
        q0.zero_()
        wp.copy(q0, state_in.dismech.q, count=self.num_state_dofs)
        if self.cap > 0.0:  # the prediction, each node's move capped
            wp.launch(_guess_kernel, dim=self.model.particle_count, inputs=[self.fixed, q0, self.contact.source.limit],
                      outputs=[q], device=dev)
            self._arm_cap(q, q0, q)
        self._energy.zero_()  # the start instead if the prediction is through a contact or past an angle's barrier
        for blk in self.blocks:
            blk.energy(q.reshape((1, n)), self._energy)
        wp.launch(_fallback_kernel, dim=n, inputs=[self._energy, self.worlds, q0], outputs=[q], device=dev)
        # b, the objective's gradient at q: the blocks' part comes from grad() here, then from commit() at each
        # picked candidate; _dof_kernel adds inertia and external force, _candidates_kernel spends it on the slope.
        self._regrad(q)
        self._go.fill_(1)
        def iteration(it):
            self._room = it < self.iterations - 2  # a multiplier update leaves the solve an iteration
            if it == 0 or (self.exact and self.cap == 0.0):  # the chord matrix at the prediction (every iterate:
                self._assemble(q, f)  # Gauss-Newton)
                self._linear.invalidate()
            elif self.cap > 0.0 and self.refactor:  # the barrier's: at the iterate while a contact touches
                wp.launch(_refresh_kernel, dim=1, inputs=[self._go[W:], self.contact.touching, int(self.exact)],
                          outputs=[self._refresh], device=dev)
                wp.capture_if(self._refresh, on_true=partial(self._refactor, q, f))
            self._iterate(state_in, q, f, it == 0 or self.refactor)

        def chain(it, stop):  # iterations it..stop-1, each nested in the last's go: converged, one node skips the rest
            iteration(it)
            if it + 1 < stop:
                wp.capture_if(self._go[W:], on_true=partial(chain, it + 1, stop))

        def chains(start, stop, first):  # NEST at a time, each run behind go (shallow nesting at 3000 iterations)
            for a in range(start, stop, NEST):
                if a == start and first:
                    chain(a, min(a + NEST, stop))
                else:
                    wp.capture_if(self._go[W:], on_true=partial(chain, a, min(a + NEST, stop)))

        if self.cap == 0.0:
            for it in range(self.iterations):
                iteration(it)
        else:
            chains(0, self.iterations, True)
        self.contact.end_step()
        if self.cap > 0.0:
            wp.launch(_count_kernel, dim=W, inputs=[self._go, self._stalled, self._failed, self.counts], device=dev)

    def _iterate(self, state_in: State, q: wp.array, f, fresh_a: bool = True) -> None:
        wp.launch(_activate_kernel, dim=self.num_worlds, inputs=[self._go], outputs=[self.worlds.active],
                  device=self.device)
        self._dof(q, f)
        if self.cap > 0.0:  # the force balance at q: |b| within tolerance of the loads, or below float32's reach
            self._load.zero_()
            self._net.zero_()
            self._unbalanced.zero_()
            self.contact.load(q, self._load, self._floor, ULP)
            wp.launch(_unbalanced_kernel, dim=self.model.particle_count, inputs=[q, self.q_pred, f, self._b,
                      self._load, self._anchor, self.worlds], outputs=[self._net], device=self.device)
            a = self.contact.arm
            if a.k:
                wp.launch(_arm_unbalanced_kernel, dim=(a.M.shape[0], a.k // 6),
                          inputs=[self._b, self._load, self._floor, a.start, a.k, self.tolerance],
                          outputs=[self._unbalanced],
                          device=self.device)
            wp.launch(_net_kernel, dim=self.num_worlds, inputs=[self._net, self.tolerance], outputs=[self._unbalanced],
                      device=self.device)
        self._linear.solve(self._b, self._dq)
        if self.contact.count and CG and fresh_a:
            wp.capture_if(self.contact.two_sided, on_true=partial(self._pcg, q))
        self._search(state_in, q, f)

    def _pcg(self, q: wp.array) -> None:
        """Refine ``dq = A^-1 b`` by SPD PCG on ``(A + delta_A_2side) x = b`` preconditioned by ``A``."""
        n, dev, W, wd = self.num_dofs, self.device, self.num_worlds, self.worlds
        x, r, p, w = self._cg
        z, c, go = self._cg_z, self._cgs, self._cg_go
        c.zero_()
        wp.launch(_pcg_start_kernel, dim=n, inputs=[self._b, self._dq, wd], outputs=[x, r, p, c], device=dev)
        wp.launch(_pcg_init_kernel, dim=W, inputs=[wd], outputs=[c], device=dev)

        def step_m(m):
            if m > 0:
                self._linear.solve(r, z)
                wp.launch(_pcg_rz_kernel, dim=n, inputs=[r, z, wd], outputs=[c], device=dev)
                wp.launch(_pcg_beta_kernel, dim=W, outputs=[c], device=dev)
                wp.launch(_pcg_dir_kernel, dim=n, inputs=[c, z, wd], outputs=[p], device=dev)
            w.zero_()
            wp.launch(_spmv_kernel, dim=n, inputs=[self.A.indptr, self.A.indices, self.A.vals, p, c, wd], outputs=[w],
                      device=dev)
            self.contact.two_sided_mul(q, p, w)
            wp.launch(_pcg_pw_kernel, dim=n, inputs=[p, w, wd], outputs=[c], device=dev)
            go.zero_()
            wp.launch(_pcg_alpha_kernel, dim=W, inputs=[m], outputs=[c, go], device=dev)
            wp.launch(_pcg_update_kernel, dim=n, inputs=[c, p, w, wd], outputs=[x, r], device=dev)
            if m + 1 < CG:
                wp.capture_if(go, on_true=partial(step_m, m + 1))

        if CG > 0:
            step_m(0)
        wp.launch(_pcg_end_kernel, dim=n, inputs=[x, c, wd], outputs=[self._dq], device=dev)

    def _arm_cap(self, q: wp.array, q0: wp.array, out: wp.array) -> None:
        """``out``: the proxy bodies' moves in ``q`` from ``q0``, each capped so its farthest point moves at most
        ``cap`` (a node's)."""
        if self._arm_radius.shape[0]:
            wp.launch(_arm_cap_kernel, dim=self._arm_radius.shape[0],
                      inputs=[q, q0, self.num_state_dofs, self._arm_radius, self.cap], outputs=[out],
                      device=self.device)

    def _regrad(self, q: wp.array) -> None:
        self._b.zero_()
        for blk in self.blocks:
            blk.grad(q, self._b)

    def _dof(self, q: wp.array, f) -> None:
        wp.launch(_dof_kernel, dim=max(self.num_dofs, self._energy.shape[0]), inputs=[q, self.q_pred, f],
                  outputs=[self._b, self._energy, self._slope], device=self.device)

    def _search(self, state_in: State, q: wp.array, f) -> None:
        """The line search along the Newton step ``dq = A^-1 b``; with the barrier, the pairs found again if a node
        moved half its cap, and ``_go`` cleared once converged: per world, in the worlds iterating."""
        dev, n, b, q0, wd, W = self.device, self.num_dofs, self._b, self._q0, self.worlds, self.num_worlds
        if self.cap > 0.0:  # the longest step stops short of the first pair's impact
            self._impact.fill_(1.0)
            self.contact.impact(q, self._dq, self._impact)
        wp.launch(_candidates_kernel, dim=(self.num_candidates, padded(n)),
                  inputs=[q, self._dq, self._impact, self.q_pred, f, q0, self.cap, self.contact.source.limit,
                          self.num_state_dofs, self._arm_radius, wd],
                  outputs=[self._q_cand, b, self._energy, self._slope], device=dev)
        for blk in self.blocks:
            blk.energy(self._q_cand, self._energy)
        wp.launch(_pick_kernel, dim=n, inputs=[self._energy, self._slope, wd, self._q_cand],
                  outputs=[q, self._pick], device=dev)
        wp.launch(_nonfinite_kernel, dim=n, inputs=[self._dq, wd], outputs=[q, self._failed], device=dev)
        if self.cap > 0.0:
            self._moved.zero_()
            self._dq_max.zero_()
            wp.launch(_moved_kernel, dim=self.model.particle_count,
                      inputs=[q, q0, self._dq, self.contact.source.limit, wd],
                      outputs=[self._moved, self._dq_max], device=dev)
            if self._arm_radius.shape[0]:
                wp.launch(_arm_moved_kernel, dim=self._arm_radius.shape[0],
                          inputs=[q, q0, self.num_state_dofs, self._arm_radius, 0.5 * self.cap, wd],
                          outputs=[self._moved], device=dev)
            h = self.der.hinge_thickness
            if h.shape[0]:
                wp.launch(_hinge_size_kernel, dim=h.shape[0],
                          inputs=[self._dq, self.shells.view.hinge_offset, h, wd], outputs=[self._dq_max], device=dev)
            self.contact.newton_size(q, self._dq, self.cap, self._dq_max)
            wp.capture_if(self._moved[W:], on_true=partial(self._detect, state_in, q))
            self._go[W:].zero_()
            self._settle[W:].zero_()
            wp.launch(_go_kernel, dim=W, inputs=[self._dq_max, self._unbalanced, self._pick, self._moved, self._failed,
                                                 self._tol, wd],
                      outputs=[self._go, self._stalled, self._settle], device=dev)
            if self._room:  # converged: the multipliers updated, and on if one moved
                wp.capture_if(self._settle[W:], on_true=partial(self.contact.update, q, self._tol, self._dq_max,
                                                                self._pick, self._go, self._settle))
        for blk in self.blocks:
            blk.commit(q, self._pick, b)

    def _assemble(self, q: wp.array, f) -> None:
        self.A.vals.zero_()
        self._anchor.zero_()
        for blk in self.blocks:
            blk.assemble(q, self.A)
        wp.launch(diagonal_kernel, dim=self.num_dofs, inputs=[f], outputs=[self.A.indptr, self.A.vals],
                  device=self.device)

    def _refactor(self, q: wp.array, f) -> None:
        self._assemble(q, f)
        self._linear.factor()

    def _detect(self, state_in: State, q: wp.array) -> None:
        wp.copy(self._q0, q)
        if self._arm_radius.shape[0]:  # pose the proxy bodies at q so detection sees them where they are
            self._arm_sweep(q, 1)
        self.contact.detect(state_in, q)
        if self._arm_radius.shape[0]:
            self._arm_sweep(q, 0)

    def _arm_sweep(self, q: wp.array, moved: int) -> None:
        a = self._proxy_arm
        wp.launch(_arm_sweep_kernel, dim=a._body.shape[0],
                  inputs=[q, a._body, self.contact.arm.base, self.contact.arm.com, self.model.body_com, a._start,
                          self.cap, moved],
                  outputs=[self.contact.source.bodies.sweep, a._q], device=self.device)


# -- the global step and its line search ---------------------------------------------------


@wp.func
def _step_length(k: int) -> float:
    """Candidate 0 is the current iterate, and ``k > 0`` the backtracking steps ``2^(1-k)``."""
    if k == 0:
        return 0.0
    return wp.pow(0.5, float(k - 1))


@wp.kernel(module="unique")  # generic: an overload recompiles this, not the module
def _copy_if_kernel(valid: wp.array[wp.int32], src: wp.array(dtype=Any), dst: wp.array(dtype=Any)):
    i = wp.tid()
    if valid[0] != 0:
        dst[i] = src[i]


@wp.func
def _armijo(energy: wp.array[wp.float64], slope: wp.array[wp.float64], w: int, num_candidates: int) -> int:
    """World ``w``'s: Newton's full step if its expected decrease is below the objective's float32 rounding (no
    candidate can be told apart). Else the longest step with sufficient decrease; else the candidate with the lowest
    objective if it rises by no more than that rounding, or ``0``: no step (a stall)."""
    o = w * num_candidates
    f0 = energy[o]
    s = slope[w]
    noise = wp.float64(_NOISE) * wp.abs(f0)
    if s >= wp.float64(0.0) and wp.float64(0.5) * s <= noise and energy[o + 1] <= f0 + noise and wp.isfinite(
            energy[o + 1]):
        return 1  # the Newton step's decrease is below the objective's rounding: its full step is taken
    best = int(0)
    for k in range(1, num_candidates):
        if wp.isfinite(energy[o + k]) and energy[o + k] <= f0 - wp.float64(_C1 * _step_length(k)) * s:
            return k
        if wp.isfinite(energy[o + k]) and (best == 0 or energy[o + k] < energy[o + best]):
            best = k
    if best > 0 and energy[o + best] <= f0 + wp.float64(_NOISE) * wp.abs(f0):
        return best
    return 0


# -- SPD PCG on two-sided contacts: per world its scalars at c[w * _S + k] ------------------------
_RZ, _RZ0, _PW, _RZN, _ALPHA, _BETA, _STOP = range(7)
_S = 7


@wp.kernel
def _pcg_start_kernel(b: wp.array[wp.float64], dq: wp.array[float], wd: Worlds, x: wp.array[wp.float64],
                      r: wp.array[wp.float64], p: wp.array[wp.float64], c: wp.array[wp.float64]):
    i = wp.tid()
    w = wd.of[i]
    x[i] = wp.float64(0.0)
    r[i] = b[i]
    d = wp.float64(dq[i])
    p[i] = d
    if wd.active[w] != 0:
        wp.atomic_add(c, w * _S + _RZ0, b[i] * d)


@wp.kernel
def _pcg_init_kernel(wd: Worlds, c: wp.array[wp.float64]):
    w = wp.tid()
    o = w * _S
    rz0 = c[o + _RZ0]
    c[o + _RZ] = rz0
    c[o + _STOP] = wp.where(wd.active[w] != 0 and rz0 > wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))


@wp.kernel
def _spmv_kernel(indptr: wp.array[wp.int32], indices: wp.array[wp.int32], vals: wp.array[wp.float64],
                 p: wp.array[wp.float64], c: wp.array[wp.float64], wd: Worlds, y: wp.array[wp.float64]):
    """``y += A p`` for an upper-triangle symmetric CSR ``A`` in the running worlds."""
    i = wp.tid()
    if c[wd.of[i] * _S + _STOP] != wp.float64(0.0):
        return
    pi = p[i]
    yi = wp.float64(0.0)
    for slot in range(indptr[i], indptr[i + 1]):
        j = indices[slot]
        v = vals[slot]
        yi += v * p[j]
        if j > i:
            wp.atomic_add(y, j, v * pi)
    wp.atomic_add(y, i, yi)


@wp.kernel
def _pcg_rz_kernel(r: wp.array[wp.float64], z: wp.array[float], wd: Worlds, c: wp.array[wp.float64]):
    i = wp.tid()
    o = wd.of[i] * _S
    if c[o + _STOP] == wp.float64(0.0):
        wp.atomic_add(c, o + _RZN, r[i] * wp.float64(z[i]))


@wp.kernel
def _pcg_beta_kernel(c: wp.array[wp.float64]):
    w = wp.tid()
    o = w * _S
    if c[o + _STOP] != wp.float64(0.0):
        return
    rzn = c[o + _RZN]
    c[o + _RZN] = wp.float64(0.0)
    if rzn <= wp.float64(1.0e-4) * c[o + _RZ0]:
        c[o + _STOP] = wp.float64(1.0)
        return
    c[o + _BETA] = rzn / c[o + _RZ]
    c[o + _RZ] = rzn


@wp.kernel
def _pcg_dir_kernel(c: wp.array[wp.float64], z: wp.array[float], wd: Worlds, p: wp.array[wp.float64]):
    i = wp.tid()
    o = wd.of[i] * _S
    if c[o + _STOP] == wp.float64(0.0):
        p[i] = wp.float64(z[i]) + c[o + _BETA] * p[i]


@wp.kernel
def _pcg_pw_kernel(p: wp.array[wp.float64], w: wp.array[wp.float64], wd: Worlds, c: wp.array[wp.float64]):
    i = wp.tid()
    o = wd.of[i] * _S
    if c[o + _STOP] == wp.float64(0.0):
        wp.atomic_add(c, o + _PW, p[i] * w[i])


@wp.kernel
def _pcg_alpha_kernel(m: int, c: wp.array[wp.float64], go: wp.array[wp.int32]):
    w = wp.tid()
    o = w * _S
    c[o + _ALPHA] = wp.float64(0.0)
    if c[o + _STOP] != wp.float64(0.0):
        return
    pw = c[o + _PW]
    c[o + _PW] = wp.float64(0.0)
    rz = c[o + _RZ]
    if pw <= wp.float64(0.0):
        c[o + _STOP] = wp.float64(1.0)
        if m == 0:
            c[o + _ALPHA] = wp.float64(1.0)
        return
    c[o + _ALPHA] = rz / pw
    go[0] = 1


@wp.kernel
def _pcg_update_kernel(c: wp.array[wp.float64], p: wp.array[wp.float64], w: wp.array[wp.float64], wd: Worlds,
                       x: wp.array[wp.float64], r: wp.array[wp.float64]):
    i = wp.tid()
    a = c[wd.of[i] * _S + _ALPHA]
    if a != wp.float64(0.0):
        x[i] = x[i] + a * p[i]
        r[i] = r[i] - a * w[i]


@wp.kernel
def _pcg_end_kernel(x: wp.array[wp.float64], c: wp.array[wp.float64], wd: Worlds, dq: wp.array[float]):
    i = wp.tid()
    w = wd.of[i]
    if wd.active[w] != 0 and c[w * _S + _RZ0] > wp.float64(0.0):
        dq[i] = float(x[i])


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
    if i < slope.shape[0]:
        slope[i] = wp.float64(0.0)
    if i >= b.shape[0]:
        return
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


@wp.kernel
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


@wp.kernel
def _arm_cap_kernel(q: wp.array[float], q0: wp.array[float], start: int, radius: wp.array[float], cap: float,
                    out: wp.array[float]):
    b = wp.tid()
    j0 = start + 6 * b
    sweep = _arm_sweep(q, q0, j0, radius[b])
    if sweep > cap:
        for c in range(6):
            out[j0 + c] = q0[j0 + c] + (q[j0 + c] - q0[j0 + c]) * (cap / sweep)


@wp.kernel
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


@wp.kernel
def _arm_moved_kernel(q: wp.array[float], q0: wp.array[float], start: int, radius: wp.array[float], limit: float,
                      wd: Worlds, moved: wp.array[wp.int32]):
    """A proxy body moved more than ``limit`` since the pairs were found: find them again (as a node)."""
    b = wp.tid()
    if _arm_sweep(q, q0, start + 6 * b, radius[b]) > limit:
        moved[wd.of[start + 6 * b]] = 1
        moved[wd.count] = 1


@wp.kernel
def _hinge_size_kernel(dq: wp.array[float], offset: int, thickness: wp.array[float], wd: Worlds,
                       dq_max: wp.array[float]):
    """A midedge angle's step as the move of its offset midpoint ``h/2 |d theta|`` (Chen et al. 2026, Sec. 7.3)."""
    h = wp.tid()
    wp.atomic_max(dq_max, wd.of[offset + h], 0.5 * thickness[h] * wp.abs(dq[offset + h]))


_NET, _LOAD, _FLOOR, _ANCHOR2, _N = 0, 3, 4, 5, 6  # a world's net force (3), load, floor, anchor floor²


@wp.kernel
def _unbalanced_kernel(q: wp.array[float], q_pred: wp.array[float], f: Dofs, b: wp.array[wp.float64],
                       load: wp.array[float], anchor: wp.array[float], wd: Worlds, net: wp.array[wp.float64]):
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
    wp.atomic_add(net, o + _FLOOR, wp.float64(k * ULP * (wp.length(x) + wp.length(r))))


@wp.kernel
def _arm_unbalanced_kernel(b: wp.array[wp.float64], load: wp.array[float], floor: wp.array[float], start: int,
                           k: int, tolerance: float, unbalanced: wp.array[float]):
    """A proxy body's force and torque balance: ``|b|`` within ``tolerance`` of its contacts' and inertia's, or below
    float32's reach."""
    w, p = wp.tid()
    i = start + w * k + 6 * p
    for o in range(0, 6, 3):
        u = wp.length(wp.vec3(float(b[i + o]), float(b[i + o + 1]), float(b[i + o + 2])))
        wp.atomic_max(unbalanced, w, u / (tolerance * load[i + o] + floor[i + o] + 1.0e-30))


@wp.kernel
def _net_kernel(net: wp.array[wp.float64], tolerance: float, unbalanced: wp.array[float]):
    """Per world, the nodes' net force against ``tolerance`` of their loads (plus float32's floor)."""
    o = _N * wp.tid()
    u = wp.sqrt(net[o] * net[o] + net[o + 1] * net[o + 1] + net[o + 2] * net[o + 2])
    tol = wp.float64(tolerance) * net[o + _LOAD] + net[o + _FLOOR] + wp.sqrt(net[o + _ANCHOR2]) + wp.float64(1.0e-30)
    unbalanced[wp.tid()] = wp.max(unbalanced[wp.tid()], float(u / tol))


@wp.kernel
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


@wp.kernel
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


@wp.kernel
def _activate_kernel(go: wp.array[wp.int32], active: wp.array[wp.int32]):
    """The worlds that iterate: those not converged."""
    w = wp.tid()
    active[w] = go[w]


@wp.kernel
def _nonfinite_kernel(dq: wp.array[float], wd: Worlds, q: wp.array[float], failed: wp.array[wp.int32]):
    """A non-finite Newton step (a singular or overflowed solve) is taken, not skipped: the line search would pick no
    candidate and the step read as converged, the state silently frozen (its world's)."""
    i = wp.tid()
    w = wd.of[i]
    if wd.active[w] != 0 and not wp.isfinite(dq[i]):
        q[i] = dq[i]
        failed[w] = 1


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
def _fallback_kernel(energy: wp.array[wp.float64], wd: Worlds, q0: wp.array[float], q: wp.array[float]):
    i = wp.tid()
    if not wp.isfinite(energy[wd.of[i] * wd.stride]):
        q[i] = q0[i]


@wp.kernel
def _pick_kernel(
    energy: wp.array[wp.float64],
    slope: wp.array[wp.float64],
    wd: Worlds,
    q_cand: wp.array2d[float],
    q: wp.array[float],
    pick: wp.array[wp.int32],
):
    """Each iterating world's candidate (``pick[w]``, written alike by its DOFs' threads); the others keep ``q``."""
    i = wp.tid()
    w = wd.of[i]
    if wd.active[w] == 0:
        pick[w] = 0
        return
    k = _armijo(energy, slope, w, wd.stride)
    pick[w] = k
    q[i] = q_cand[k, i]
