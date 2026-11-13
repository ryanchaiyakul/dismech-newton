"""Rods and shells as the destination of Newton's ``SolverCoupledProxy``: the source's proxy bodies yield to the
contacts within the soft step (:class:`ProxyArm`), so a pinch holds at the source's servo force."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp
from newton import BodyFlags, JointType, Model, ParticleFlags, State, StateFlags
from newton._src.solvers.coupled import solver_coupled as _sc
from newton.solvers.experimental.coupled import SolverCoupledProxy

from .builder import vec5f, vec6f
from .contact import STIFFNESS, Arm, _arm_inertia, arm_assemble, arm_move, arm_scatter
from .linear import row_slot
from .state import flatten_state, node, padded, scatter_node, suspended_tape, world_sum

_orig_coupled_init = SolverCoupledProxy.__init__
_lagged_step_coupled = SolverCoupledProxy._step_coupled
_orig_coupled_step = SolverCoupledProxy.step
_mjw = None


def _mjw_mod():
    global _mjw
    if _mjw is None:
        import mujoco_warp as mjw  # noqa: PLC0415

        _mjw = mjw
    return _mjw


def _coupled_init(self: SolverCoupledProxy, *args, **kwargs) -> None:
    _orig_coupled_init(self, *args, **kwargs)
    for (src_name, dst_name), group in self._proxy_groups.items():
        src, dst = self._entries[src_name], self._entries[dst_name]
        pa = getattr(dst.solver, "_proxy_arm", None)
        if pa is not None and group["bodies"] and hasattr(src.solver, "mjw_model"):
            pa.bind_articulated(src, dst, group["bodies"])


def _rsr_step_coupled(self: SolverCoupledProxy, state_in: State, state_out: State, control, contacts, dt: float) -> None:
    """Follow the 1-pass proxy solve with a rigid (``R-S-R``) or soft (``S-M-S``) corrector step on harvested forces."""
    _lagged_step_coupled(self, state_in, state_out, control, contacts, dt)
    if int(self._coupling.iterations) != 1:
        return
    for (src_name, dst_name), group in self._proxy_groups.items():
        src, dst = self._entries[src_name], self._entries[dst_name]
        body_proxies, particle_proxies = group["bodies"], group["particles"]
        if getattr(dst.solver, "_proxy_arm", None) is not None:
            if not body_proxies or any(int(p.mode) != 0 for p in body_proxies):
                continue
            if src.has_body_force_input:
                self._clear_body_force_input(src)
                self._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
                for proxy in body_proxies:
                    self._add_body_force_input(src, proxy.source_local_to_proxy_global, proxy.coupling_forces)
                self._notify_input_state_update(src, StateFlags.BODY_F, dt=dt)
            self._step_entry(src, control, contacts, dt)
        elif hasattr(src.solver, "der") and (body_proxies or particle_proxies):
            if body_proxies and src.has_body_force_input:
                self._clear_body_force_input(src)
                self._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
                for proxy in body_proxies:
                    self._add_body_force_input(src, proxy.source_local_to_proxy_global, proxy.coupling_forces)
                self._notify_input_state_update(src, StateFlags.BODY_F, iteration_restart=True, dt=dt)
            if particle_proxies and src.has_particle_force_input:
                self._clear_particle_force_input(src)
                self._add_particle_force_input(src, src.particle_local_to_global, state_in.particle_f)
                for proxy in particle_proxies:
                    self._add_particle_force_input(src, proxy.source_local_to_proxy_global, proxy.coupling_forces)
                self._notify_input_state_update(src, StateFlags.PARTICLE_F, iteration_restart=True, dt=dt)
            self._step_entry(src, control, contacts, dt)


def _coupled_step(self: SolverCoupledProxy, state_in: State, state_out: State, control, contacts, dt: float,
                  *, observables=None) -> None:
    """Dispatch forward steps directly; when a ``wp.Tape`` is recording, preserve per-step states & MuJoCo VJP."""
    tape = wp._src.context.runtime.tape
    if tape is None:
        return _orig_coupled_step(self, state_in, state_out, control, contacts, dt, observables=observables)
    has_art_tape = any(
        getattr(self._entries[dst_n].solver, "_proxy_arm", None) is not None
        and group["bodies"]
        and hasattr(self._entries[src_n].solver, "mjw_model")
        for (src_n, dst_n), group in self._proxy_groups.items()
    )
    if not has_art_tape or int(self._coupling.iterations) != 1:
        return _orig_coupled_step(self, state_in, state_out, control, contacts, dt, observables=observables)
    return _taped_step_coupled(self, state_in, state_out, control, contacts, float(dt), tape, observables=observables)


def _grad_state_clone(state: State, dismech: bool = False) -> State:
    """Create a per-step clone of ``state`` with ``requires_grad=True`` on continuous arrays."""
    out = State()
    for name in ("body_q", "body_qd", "body_f", "particle_q", "particle_qd", "particle_f", "joint_q", "joint_qd"):
        arr = getattr(state, name, None)
        setattr(out, name, wp.clone(arr, requires_grad=True) if arr is not None else None)
    if dismech and hasattr(state, "dismech"):
        src_ns = state.dismech
        ns = SimpleNamespace()
        for k, v in vars(src_ns).items():
            if k in ("q", "qd"):
                setattr(ns, k, None)
            elif isinstance(v, wp.array):
                setattr(ns, k, wp.clone(v, requires_grad=v.dtype in (float, wp.float32, wp.float64)))
            else:
                setattr(ns, k, v)
        out.dismech = ns
        flatten_state(out)
    return out


@wp.kernel(module="unique")
def _gather_mapped_any_kernel(
    local_to_global: wp.array[wp.int32],
    src: wp.array(dtype=Any),
    dst: wp.array(dtype=Any),
):
    dst[wp.tid()] = src[local_to_global[wp.tid()]]


@wp.kernel(module="unique")
def _scatter_mapped_any_kernel(
    indices: wp.array[wp.int32],
    global_to_local: wp.array[wp.int32],
    src: wp.array(dtype=Any),
    dst: wp.array(dtype=Any),
):
    g = indices[wp.tid()]
    dst[g] = src[global_to_local[g]]


@wp.kernel(module="unique")
def _copy_flat_any_kernel(src: wp.array(dtype=Any), dst: wp.array(dtype=Any)):
    i = wp.tid()
    dst[i] = src[i]


def _sync_dismech_attrs(src: State, dst: State, device) -> None:
    if not (hasattr(src, "dismech") and hasattr(dst, "dismech")):
        return
    s_ns, d_ns = src.dismech, dst.dismech
    for name in ("edge_q", "edge_qd", "hinge_q", "hinge_qd", "edge_d1_q", "triplet_ref_twist_q",
                 "triplet_dahl_q", "edge_strain", "hinge_strain", "cst_strain"):
        sa, da = getattr(s_ns, name, None), getattr(d_ns, name, None)
        if isinstance(sa, wp.array) and isinstance(da, wp.array) and sa.shape[0] > 0 and sa.shape == da.shape:
            if sa.dtype in (float, wp.float32, wp.float64, wp.vec3, vec5f, vec6f):
                wp.launch(_copy_flat_any_kernel, dim=sa.shape[0], inputs=[sa, da], device=device)


def _distribute_state_tape(self: SolverCoupledProxy, state_in: State, dt: float) -> None:
    """Differentiable version of ``SolverCoupled._distribute_state`` for ``wp.Tape``."""
    dev = self.model.device
    for entry in self._entries.values():
        s0 = entry.state_0
        if entry.body_local_to_global.shape[0] > 0 and state_in.body_q is not None and s0.body_q is not None:
            wp.launch(_gather_mapped_any_kernel, dim=entry.body_local_to_global.shape[0],
                      inputs=[entry.body_local_to_global, state_in.body_q, s0.body_q], device=dev)
            wp.launch(_gather_mapped_any_kernel, dim=entry.body_local_to_global.shape[0],
                      inputs=[entry.body_local_to_global, state_in.body_qd, s0.body_qd], device=dev)
        if s0.body_f is not None:
            if state_in.body_f is None:
                with suspended_tape():
                    s0.body_f.zero_()
            elif entry.body_local_to_global.shape[0] > 0:
                wp.launch(_gather_mapped_any_kernel, dim=entry.body_local_to_global.shape[0],
                          inputs=[entry.body_local_to_global, state_in.body_f, s0.body_f], device=dev)
        if entry.particle_local_to_global.shape[0] > 0 and state_in.particle_q is not None and s0.particle_q is not None:
            wp.launch(_gather_mapped_any_kernel, dim=entry.particle_local_to_global.shape[0],
                      inputs=[entry.particle_local_to_global, state_in.particle_q, s0.particle_q], device=dev)
            wp.launch(_gather_mapped_any_kernel, dim=entry.particle_local_to_global.shape[0],
                      inputs=[entry.particle_local_to_global, state_in.particle_qd, s0.particle_qd], device=dev)
        if s0.particle_f is not None:
            if state_in.particle_f is None:
                with suspended_tape():
                    s0.particle_f.zero_()
            elif entry.particle_local_to_global.shape[0] > 0:
                wp.launch(_gather_mapped_any_kernel, dim=entry.particle_local_to_global.shape[0],
                          inputs=[entry.particle_local_to_global, state_in.particle_f, s0.particle_f], device=dev)
        if entry.joint_coord_local_to_global.shape[0] > 0 and state_in.joint_q is not None and s0.joint_q is not None:
            wp.launch(_gather_mapped_any_kernel, dim=entry.joint_coord_local_to_global.shape[0],
                      inputs=[entry.joint_coord_local_to_global, state_in.joint_q, s0.joint_q], device=dev)
        if entry.joint_dof_local_to_global.shape[0] > 0 and state_in.joint_qd is not None and s0.joint_qd is not None:
            wp.launch(_gather_mapped_any_kernel, dim=entry.joint_dof_local_to_global.shape[0],
                      inputs=[entry.joint_dof_local_to_global, state_in.joint_qd, s0.joint_qd], device=dev)
        if getattr(entry.solver, "_proxy_arm", None) is not None:
            _sync_dismech_attrs(state_in, s0, dev)
        with suspended_tape():
            flags = self._input_state_copy_flags(state_in, s0)
            self._notify_input_state_update(entry, flags, dt=dt, iteration_restart=False)


def _reconcile_state_tape(self: SolverCoupledProxy, state_out: State) -> None:
    """Differentiable version of ``SolverCoupled._reconcile_state`` for ``wp.Tape``."""
    dev = self.model.device
    for entry in self._entries.values():
        s1 = entry.state_1
        if s1 is None:
            continue
        if entry.body_indices.shape[0] > 0 and s1.body_q is not None and state_out.body_q is not None:
            wp.launch(_scatter_mapped_any_kernel, dim=entry.body_indices.shape[0],
                      inputs=[entry.body_indices, entry.body_global_to_local, s1.body_q, state_out.body_q],
                      device=dev)
            wp.launch(_scatter_mapped_any_kernel, dim=entry.body_indices.shape[0],
                      inputs=[entry.body_indices, entry.body_global_to_local, s1.body_qd, state_out.body_qd],
                      device=dev)
        if entry.particle_indices.shape[0] > 0 and s1.particle_q is not None and state_out.particle_q is not None:
            wp.launch(_scatter_mapped_any_kernel, dim=entry.particle_indices.shape[0],
                      inputs=[entry.particle_indices, entry.particle_global_to_local, s1.particle_q,
                              state_out.particle_q], device=dev)
            wp.launch(_scatter_mapped_any_kernel, dim=entry.particle_indices.shape[0],
                      inputs=[entry.particle_indices, entry.particle_global_to_local, s1.particle_qd,
                              state_out.particle_qd], device=dev)
        if entry.joint_q_indices.shape[0] > 0 and s1.joint_q is not None and state_out.joint_q is not None:
            wp.launch(_scatter_mapped_any_kernel, dim=entry.joint_q_indices.shape[0],
                      inputs=[entry.joint_q_indices, entry.joint_coord_global_to_local, s1.joint_q,
                              state_out.joint_q], device=dev)
        if entry.joint_qd_indices.shape[0] > 0 and s1.joint_qd is not None and state_out.joint_qd is not None:
            wp.launch(_scatter_mapped_any_kernel, dim=entry.joint_qd_indices.shape[0],
                      inputs=[entry.joint_qd_indices, entry.joint_dof_global_to_local, s1.joint_qd,
                              state_out.joint_qd], device=dev)
        if getattr(entry.solver, "_proxy_arm", None) is not None:
            _sync_dismech_attrs(s1, state_out, dev)


def _taped_step_coupled(self: SolverCoupledProxy, state_in: State, state_out: State, control, contacts, dt: float,
                        tape: wp.Tape, *, observables=None) -> None:
    """Execute one 1-pass ``SolverCoupledProxy`` step while keeping every sub-solver intermediate differentiable."""
    self.validate_observables(observables, contacts)
    saved = {}
    for name, entry in self._entries.items():
        saved[name] = (entry.state_0, entry.state_1)
        is_soft = getattr(entry.solver, "_proxy_arm", None) is not None
        entry.state_0 = _grad_state_clone(entry.state_0, dismech=is_soft)
        entry.state_1 = _grad_state_clone(entry.state_1, dismech=is_soft)
    try:
        _distribute_state_tape(self, state_in, dt=dt)
        self._active_observables = observables
        try:
            self._reset_aitken_iteration_state()
            for (src_name, dst_name), group in self._proxy_groups.items():
                src, dst = self._entries[src_name], self._entries[dst_name]
                body_proxies, joint_proxies = group["bodies"], group["joints"]
                pa = getattr(dst.solver, "_proxy_arm", None)
                if pa is None or not body_proxies or not hasattr(src.solver, "mjw_model"):
                    continue
                if self.model.shape_count and self.model.shape_material_mu.requires_grad:
                    dst_mu = dst.solver.model.shape_material_mu
                    if dst_mu.shape == self.model.shape_material_mu.shape:
                        dst_mu.requires_grad = True
                        wp.launch(_copy_flat_any_kernel, dim=dst_mu.shape[0],
                                  inputs=[self.model.shape_material_mu, dst_mu], device=self.model.device)
                pa.bind_articulated(src, dst, body_proxies)
                proxy_f_in = []
                for p in body_proxies:
                    # Link f_lag differentiably to the previous taped step's harvested F_soft (or zeros at step 0).
                    prev = getattr(p, "_tape_f", None)
                    if prev is None or getattr(p, "_tape_id", None) is not tape:
                        f_in = wp.clone(p.coupling_forces, requires_grad=True)
                    else:
                        f_in = prev
                    proxy_f_in.append((p, f_in))
                    self._stash_proxy_feedback(p)

                with suspended_tape():
                    if src.has_body_force_input:
                        self._clear_body_force_input(src)
                        self._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
                        for p in body_proxies:
                            self._add_body_force_input(src, p.source_local_to_proxy_global, p.coupling_forces)
                        self._notify_input_state_update(src, StateFlags.BODY_F, dt=dt)
                    self._step_entry(src, control, contacts, dt)
                    pa.update_articulated(dt)
                    snap1 = pa.snapshot_mujoco(control)

                mjc_pass1_q = src.state_0.body_q
                mjc_pass1_q1 = src.state_1.body_q
                mjc_pass1_qd = src.state_1.body_qd
                mjc_pass1_joint_q = src.state_0.joint_q
                mjc_pass1_joint_qd = src.state_0.joint_qd

                tq_ref = control.joint_target_q if control is not None else None

                def _pass1_vjp(s1=snap1, f_list=proxy_f_in, q0_mjc=mjc_pass1_q, q1_mjc=mjc_pass1_q1,
                               qd1_mjc=mjc_pass1_qd, jq0=mjc_pass1_joint_q, jqd0=mjc_pass1_joint_qd, tq_arr=tq_ref,
                               pa_ref=pa, dt_s=dt):
                    tq_g = tq_arr.grad if tq_arr is not None else None
                    for p_i, f_in_i in f_list:
                        pa_ref.mujoco_vjp(
                            s1, dt_s, p_i.source_local_to_proxy_global,
                            body_q0=q0_mjc, body_q1=q1_mjc,
                            body_q0_grad=None, body_qd1_grad=qd1_mjc.grad,
                            joint_q1_grad=None, joint_qd1_grad=None, body_q1_grad=None,
                            out_target_q_grad=tq_g, out_body_f_grad=f_in_i.grad,
                            out_joint_q_grad=jq0.grad if jq0 is not None else None,
                            out_joint_qd_grad=jqd0.grad if jqd0 is not None else None,
                        )

                arrays_p1 = [mjc_pass1_q, mjc_pass1_qd]
                if mjc_pass1_joint_q is not None:
                    arrays_p1 += [mjc_pass1_joint_q, mjc_pass1_joint_qd]
                if control is not None and control.joint_target_q is not None and control.joint_target_q.requires_grad:
                    arrays_p1.append(control.joint_target_q)
                arrays_p1 += [fi for _, fi in proxy_f_in]
                tape.record_func(_pass1_vjp, [a for a in arrays_p1 if a.requires_grad])

                # Differentiably hand mjc pose/vel -> dst.state_0 & -f_lag -> dst.state_0.body_f.
                for p, _ in proxy_f_in:
                    wp.launch(_tape_sync_proxy_kernel, dim=p.source_local_to_proxy_local.shape[0],
                              inputs=[src.state_0.body_q, src.state_1.body_qd, p.source_local_to_proxy_local],
                              outputs=[dst.state_0.body_q, dst.state_0.body_qd], device=self.model.device)
                if dst.state_0.body_f is not None:
                    dst.state_0.body_f.zero_()
                    for p, f_in in proxy_f_in:
                        wp.launch(_tape_rewind_force_kernel, dim=p.destination_local_to_proxy_global.shape[0],
                                  inputs=[dst.body_gravity_acceleration, f_in, p.destination_local_to_proxy_global,
                                          dst.solver.model.body_mass, dst.solver.model.body_inv_mass],
                                  outputs=[dst.state_0.body_f], device=self.model.device)

                with suspended_tape():
                    self._notify_input_state_update(dst, StateFlags.BODY_Q | StateFlags.BODY_QD | StateFlags.BODY_F,
                                                    dt=dt)
                    for p in body_proxies:
                        wp.copy(p.proxy_qd_before, dst.state_0.body_qd)
                    dst_contacts = contacts
                    if body_proxies:
                        dst_contacts = dst.solver.coupling_prepare_proxy_contacts(
                            dst.state_0, dst_contacts, contacts_freshly_detected=True)
                    control_callback = None
                    if joint_proxies:
                        def control_callback(dst_ctrl, jp=joint_proxies, sc=control):
                            self._sync_proxy_joint_targets(jp, sc, dst_ctrl)

                # Soft step runs with tape ACTIVE -> records StepAdjoint + Tactile onto outer tape.
                dst_contacts_used = self._step_entry(dst, control, dst_contacts, dt, filter_contacts=True,
                                                     control_callback=control_callback)

                f_out_list = []
                with suspended_tape():
                    for p, f_in in proxy_f_in:
                        dst.solver.coupling_harvest_proxy_wrenches(
                            p.destination_local_to_proxy_global, p.coupling_forces,
                            body_qd_before=p.proxy_qd_before, state=dst.state_0, state_out=dst.state_1,
                            contacts=dst_contacts_used, dt=dt)
                        self._blend_proxy_feedback(p)
                        f_out = wp.clone(p.coupling_forces, requires_grad=True)
                        p._tape_f = f_out
                        p._tape_id = tape
                        f_out_list.append((p, f_in, f_out))

                h_step = float(dst.solver.theta * dt)
                m_art_step = wp.clone(pa._arm.M)
                w_art_step = wp.clone(pa._arm.Minv)
                mjc_pass1_soft = dst.state_1.body_qd

                def _harvest_vjp(f_list=f_out_list, m_art=m_art_step, w_art=w_art_step, h_s=h_step, dt_s=dt,
                                 qd1_soft=mjc_pass1_soft, qd_mjc=mjc_pass1_qd, pa_ref=pa, a_ref=pa._art):
                    for p_i, f_in_i, f_out_i in f_list:
                        if f_out_i.grad is None:
                            continue
                        wp.launch(_art_harvest_vjp_kernel, dim=pa_ref._arm.M.shape[0] * (pa_ref.k // 6),
                                  inputs=[pa_ref.k, pa_ref._slot_body, a_ref.dst_to_src,
                                          p_i.destination_local_to_proxy_global, m_art, w_art, 1.0 / h_s,
                                          dt_s / h_s, f_out_i.grad],
                                  outputs=[qd1_soft.grad, qd_mjc.grad, f_in_i.grad], device=pa_ref.device)

                arrays_h = [x for _, fi, fo in f_out_list for x in (fi, fo)] + [mjc_pass1_soft, mjc_pass1_qd]
                tape.record_func(_harvest_vjp, [a for a in arrays_h if a.requires_grad])

                # R-S-R Pass 2: MuJoCo corrector step driven by harvested f_out -> a fresh src.state_1 so Pass 1
                # (mjc_pass1_qd) and Pass 2 (mjc2_qd) remain distinct nodes on the outer tape.
                with suspended_tape():
                    src.state_1 = _grad_state_clone(src.state_1, dismech=False)
                    if src.has_body_force_input:
                        self._clear_body_force_input(src)
                        self._add_body_force_input(src, src.body_local_to_global, state_in.body_f)
                        for p in body_proxies:
                            self._add_body_force_input(src, p.source_local_to_proxy_global, p.coupling_forces)
                        self._notify_input_state_update(src, StateFlags.BODY_F, dt=dt)
                    self._step_entry(src, control, contacts, dt)
                    pa.update_articulated(dt)
                    snap2 = pa.snapshot_mujoco(control)

                mjc2_q, mjc2_qd = src.state_1.body_q, src.state_1.body_qd
                mjc2_jq, mjc2_jqd = src.state_1.joint_q, src.state_1.joint_qd

                def _pass2_vjp(s2=snap2, f_list=f_out_list, q0=mjc_pass1_q, q1=mjc2_q, qd1=mjc2_qd, jq1=mjc2_jq,
                               jqd1=mjc2_jqd, jq0=mjc_pass1_joint_q, jqd0=mjc_pass1_joint_qd, tq_arr=tq_ref,
                               pa_ref=pa, dt_s=dt):
                    tq_g = tq_arr.grad if tq_arr is not None else None
                    for p_i, _, f_out_i in f_list:
                        pa_ref.mujoco_vjp(
                            s2, dt_s, p_i.source_local_to_proxy_global,
                            body_q0=q0, body_q1=q1,
                            body_q0_grad=None, body_qd1_grad=qd1.grad,
                            joint_q1_grad=jq1.grad if jq1 is not None else None,
                            joint_qd1_grad=jqd1.grad if jqd1 is not None else None, body_q1_grad=q1.grad,
                            out_target_q_grad=tq_g, out_body_f_grad=f_out_i.grad,
                            out_joint_q_grad=jq0.grad if jq0 is not None else None,
                            out_joint_qd_grad=jqd0.grad if jqd0 is not None else None,
                        )

                arrays_p2 = [mjc2_q, mjc2_qd]
                if mjc2_jq is not None:
                    arrays_p2 += [mjc2_jq, mjc2_jqd]
                if mjc_pass1_joint_q is not None:
                    arrays_p2 += [mjc_pass1_joint_q, mjc_pass1_joint_qd]
                if control is not None and control.joint_target_q is not None and control.joint_target_q.requires_grad:
                    arrays_p2.append(control.joint_target_q)
                arrays_p2 += [fo for _, _, fo in f_out_list]
                tape.record_func(_pass2_vjp, [a for a in arrays_p2 if a.requires_grad])
        finally:
            self._active_observables = None
        _copy_state_on_tape(state_in, state_out)
        _reconcile_state_tape(self, state_out)
        self._entry_output_state_valid = True
    finally:
        for name, (s0, s1) in saved.items():
            self._entries[name].state_0, self._entries[name].state_1 = s0, s1


def _copy_state_on_tape(src: State, dst: State) -> None:
    """Differentiable non-mutating copy fallback for unowned fields before ``_reconcile_state``."""
    with suspended_tape():
        _sc._copy_state(src, dst)


SolverCoupledProxy.__init__ = _coupled_init
SolverCoupledProxy._step_coupled = _rsr_step_coupled
SolverCoupledProxy.step = _coupled_step


def mpm_collider_inertia(model: Model, bodies=None) -> tuple[wp.array, wp.array]:
    """Authoritative ``(body_mass, body_inv_inertia)`` for DER ``edge_body`` capsules passed to
    ``SolverImplicitMPM.setup_collider(...)``: lumped pair mass ``particle_mass[n0] + particle_mass[n1]``,
    transverse inertia ``0.25 * m * L^2``, and axial twist inertia ``edge_inertia[e]`` (zero for fixed edges)."""
    mass = model.body_mass.numpy().astype(np.float32, copy=True)
    inv_inertia = model.body_inv_inertia.numpy().astype(np.float32, copy=True)
    der = getattr(model, "dismech", None)
    if der is not None and getattr(der, "edge_body", None) is not None and der.edge_body.shape[0]:
        eb = der.edge_body.numpy()
        n0, n1 = der.edge_node0.numpy(), der.edge_node1.numpy()
        length, ie, fixed = der.edge_length.numpy(), der.edge_inertia.numpy(), der.edge_fixed.numpy()
        pm = model.particle_mass.numpy()
        active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
        for e in np.flatnonzero(eb >= 0):
            b = int(eb[e])
            if fixed[e] != 0 or not (active[n0[e]] or active[n1[e]]):
                mass[b] = 0.0
                inv_inertia[b] = 0.0
            else:
                m = float(pm[n0[e]] + pm[n1[e]])
                i_t = 0.25 * m * float(length[e]) ** 2
                i_a = float(ie[e])
                mass[b] = m
                inv_inertia[b] = np.diag([
                    1.0 / i_t if i_t > 0.0 else 0.0,
                    1.0 / i_t if i_t > 0.0 else 0.0,
                    1.0 / i_a if i_a > 0.0 else 0.0,
                ]).astype(np.float32)
    if bodies is not None:
        idx = bodies.numpy() if isinstance(bodies, wp.array) else np.asarray(bodies, dtype=np.int32)
        mass, inv_inertia = mass[idx], inv_inertia[idx]
    return wp.array(mass, dtype=float, device=model.device), wp.array(inv_inertia, dtype=wp.mat33, device=model.device)


@wp.kernel(enable_backward=False)
def _edge_body_wrench_kernel(
    node_q: wp.array[wp.vec3],
    edge_node0: wp.array[wp.int32],
    edge_node1: wp.array[wp.int32],
    edge_body: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    node_f: wp.array[wp.vec3],
    edge_f: wp.array[float],
):
    """Project spatial wrench ``(F, tau)`` about capsule midpoint onto ``(node0, node1, theta_e)`` by virtual work."""
    e = wp.tid()
    b = edge_body[e]
    if b < 0:
        return
    w = body_f[b]
    F = wp.spatial_top(w)
    tau = wp.spatial_bottom(w)
    n0 = edge_node0[e]
    n1 = edge_node1[e]
    vec = node_q[n1] - node_q[n0]
    l2 = wp.dot(vec, vec)
    if l2 <= 1.0e-24:
        return
    d = wp.cross(tau, vec) / l2
    wp.atomic_add(node_f, n0, 0.5 * F - d)
    wp.atomic_add(node_f, n1, 0.5 * F + d)
    edge_f[e] = edge_f[e] + wp.dot(tau, vec) / wp.sqrt(l2)


class ProxyArm:
    """Newton ``SolverCoupledProxy``'s proxy bodies (``BodyFlags.PROXY`` in the solver's view) as a soft step's
    :class:`Arm`: 6 DOFs each, its centre of mass's move and a small rotation (world frame), with the inertia of its
    view mass and inertia (the source's effective ones), ``diag(m, R I R^T)``.

    A proxy starts each step where the last left it (feasible: the source's pose, a step behind on the soft side's
    reactions, may overlap the nodes) and its inertia is about the source's own predicted end pose: its synced pose
    moved by ``h qd + h^2 (f / m + g, (R I R^T)^-1 tau)`` (the coupler's synced velocity and rewound ``body_f``). Its
    velocity out is its move from the synced pose over ``h``: the coupler's momentum harvest is then its contact force
    in the step, ``M (z - pred) / h^2``. Proxies without shapes stay out: they touch nothing, and move by their own
    step. The solver builds it itself."""

    def __init__(self, model: Model):
        every = np.flatnonzero(model.body_flags.numpy() & int(BodyFlags.PROXY))
        proxy = np.intersect1d(every, model.shape_body.numpy())  # those with shapes (a frame's has no inertia)
        self._free = wp.array(np.setdiff1d(every, proxy), dtype=wp.int32, device=model.device)
        world = np.maximum(model.body_world.numpy()[proxy], 0)
        slot = np.zeros(len(proxy), dtype=np.int32)  # its place among its world's proxies
        for w in np.unique(world):
            slot[world == w] = np.arange((world == w).sum())
        self.k = 6 * (int(slot.max()) + 1 if len(proxy) else 0)
        # each body's reach from its centre of mass (its shapes' bounding spheres): its farthest point moves at most
        # |dx| + |dth| radius, capped like a node's move (a body is detected like the nodes: at its start pose)
        radius = np.zeros(max(max(model.world_count, 1) * self.k // 6, 1), dtype=np.float32)
        sb, st, sr, com = (a.numpy() for a in (model.shape_body, model.shape_transform, model.shape_collision_radius,
                                               model.body_com))
        for b, w, s in zip(proxy, world, slot, strict=True):
            for j in np.flatnonzero(sb == b):
                i = w * self.k // 6 + s
                radius[i] = max(radius[i], np.linalg.norm(st[j, :3] - com[b]) + sr[j])
        self.radius = wp.array(radius, dtype=float, device=model.device)
        self.model, self.device = model, model.device
        self._world, self._slot = world, slot
        nb = max(model.body_count, 1)
        self._body = wp.array(proxy, dtype=wp.int32, device=self.device)
        self._q = wp.zeros(nb, dtype=wp.transform, device=self.device)  # the step's body poses: the proxies' own
        self._qd = wp.zeros(nb, dtype=wp.spatial_vector, device=self.device)
        self._last = wp.zeros(nb, dtype=wp.transform, device=self.device)  # the proxies' poses the last step left
        self._offset = wp.zeros(nb, dtype=wp.spatial_vector, device=self.device)  # synced pose - start
        self._start = wp.zeros(nb, dtype=wp.transform, device=self.device)  # the proxies' poses the step began at
        self._valid = wp.zeros(nb, dtype=wp.int32, device=self.device)  # per body: it has a last pose
        self._was_valid = wp.zeros_like(self._valid)  # the step's: its start pose is the last step's end
        self._last_grad = None  # the last recorded step's adjoint of its end poses (snapshot)
        self.restart = False  # the next step repeats the last (a coupler's iteration restart): from its start
        self._arm: Arm | None = None
        w_n = max(model.world_count, 1)
        nb_slot = max(self.k // 6, 1)
        slot_body = np.full((w_n, nb_slot), -1, dtype=np.int32)
        if self.k:
            slot_body[world, slot] = proxy
        self._slot_body = wp.array(slot_body, dtype=wp.int32, device=self.device)
        self._art = None
        self._art_ready = False
        self._art_used = False
        self._art_dt = 0.0
        self._global_body = None

    def bind_articulated(self, src_entry, dst_entry, body_proxies) -> None:
        """Bind a ``SolverMuJoCo`` source entry once so each proxy step sees its closed-chain Delassus operator."""
        self._global_body = dst_entry.body_local_to_global
        if self._art is not None or self.k == 0:
            return
        mjc = src_entry.solver
        mm, md = mjc.mjw_model, mjc.mjw_data
        W = int(md.nworld)
        nv, nu = int(mm.nv), int(mm.nu)
        nb_slot = self.k // 6
        mj_map = np.full((W, nb_slot), -1, dtype=np.int32)
        mj_world = np.zeros((W, nb_slot), dtype=np.int32)
        dinv = np.ones((W, self.k), dtype=np.float64)
        m2n = mjc.mjc_body_to_newton.numpy()
        src_to_mj = {
            int(m2n[wm, mb]): (int(wm), int(mb))
            for wm in range(m2n.shape[0])
            for mb in range(m2n.shape[1])
            if m2n[wm, mb] >= 0
        }
        bm, bi = self.model.body_mass.numpy(), self.model.body_inertia.numpy()
        src_g2l = src_entry.body_global_to_local.numpy() if src_entry.body_global_to_local is not None else None
        proxy_to_src = {
            int(b_p): (int(src_g2l[int(b_s)]) if src_g2l is not None else int(b_s))
            for p in body_proxies
            for b_s, b_p in zip(p.src_ids.numpy(), p.proxy_ids_local.numpy(), strict=True)
        }
        dst_to_src = np.full(max(self.model.body_count, 1), -1, dtype=np.int32)
        for b_dst, w_dst, s in zip(self._body.numpy(), self._world, self._slot, strict=True):
            b_src = proxy_to_src.get(int(b_dst), -1)
            if b_src < 0 or b_src not in src_to_mj:
                continue
            dst_to_src[b_dst] = b_src
            wm, mb = src_to_mj[b_src]
            mj_map[w_dst, s] = mb
            mj_world[w_dst, s] = wm
            m_b = max(float(bm[b_dst]), 1.0e-6)
            I_b = max(float(np.trace(bi[b_dst])) / 3.0, 1.0e-8)
            dinv[w_dst, 6 * s : 6 * s + 3] = m_b / 1.0e-3
            dinv[w_dst, 6 * s + 3 : 6 * s + 6] = I_b / 1.0e-3

        # Map src-entry MuJoCo DOFs & coords <-> Newton global joint arrays (1-DOF hinge/slide, D6 per axis).
        src_view = src_entry.view
        jnt_count_w = int(src_view.joint_count) // max(W, 1)
        dof_per_w = int(src_view.joint_dof_count) // max(W, 1)
        coord_per_w = int(src_view.joint_coord_count) // max(W, 1)
        mj_qd_start = mjc.mj_qd_start.numpy()
        mj_q_start = mjc.mj_q_start.numpy()
        j_type = src_view.joint_type.numpy()
        j_q0 = src_view.joint_q_start.numpy()
        j_qd0 = src_view.joint_qd_start.numpy()
        j_dd = src_view.joint_dof_dim.numpy()
        src_qd_map = np.full((W, nv), -1, dtype=np.int32)
        src_q_map = np.full((W, int(mm.nq)), -1, dtype=np.int32)
        for w in range(W):
            for j in range(jnt_count_w):
                jj = w * jnt_count_w + j
                nd = int(j_dd[jj, 0]) + int(j_dd[jj, 1])
                mj_d0, mj_c0 = int(mj_qd_start[j]), int(mj_q_start[j])
                n_d0, n_c0 = int(j_qd0[jj]) - w * dof_per_w, int(j_q0[jj]) - w * coord_per_w
                if j_type[jj] in (int(JointType.REVOLUTE), int(JointType.PRISMATIC)) and nd == 1 and mj_d0 >= 0:
                    src_qd_map[w, mj_d0] = w * dof_per_w + n_d0
                    if mj_c0 >= 0:
                        src_q_map[w, mj_c0] = w * coord_per_w + n_c0
                elif j_type[jj] == int(JointType.D6) and mj_d0 >= 0:
                    for ax in range(nd):
                        src_qd_map[w, mj_d0 + ax] = w * dof_per_w + n_d0 + ax
                        if mj_c0 >= 0:
                            src_q_map[w, mj_c0 + ax] = w * coord_per_w + n_c0 + ax

        # Actuator -> (dof, target_q_idx, ctrl_source).
        act_dof = np.full(max(nu, 1), -1, dtype=np.int32)
        act_tq = np.full(max(nu, 1), -1, dtype=np.int32)
        act_src = np.full(max(nu, 1), -1, dtype=np.int32)
        if nu > 0:
            trntype = mm.actuator_trntype.numpy()
            trnid = mm.actuator_trnid.numpy()
            jnt_dof = mm.jnt_dofadr.numpy()
            axis_idx = mjc.mjc_actuator_to_target_q_axis_idx.numpy()
            tq_idx = mjc.mjc_actuator_to_newton_target_q_idx.numpy()
            src_coord_l2g = (
                src_entry.joint_coord_local_to_global.numpy()
                if src_entry.joint_coord_local_to_global is not None
                else None
            )
            c_src = mjc.mjc_actuator_ctrl_source.numpy()
            for a in range(nu):
                local_tq = int(tq_idx[a])
                act_tq[a] = (
                    int(src_coord_l2g[local_tq])
                    if src_coord_l2g is not None and 0 <= local_tq < src_coord_l2g.shape[0]
                    else local_tq
                )
                act_src[a] = int(c_src[a])
                if trntype[a] == 0 and trnid[a, 0] >= 0:
                    act_dof[a] = int(jnt_dof[trnid[a, 0]]) + max(int(axis_idx[a]), 0)

        dev = self.device
        self._art = SimpleNamespace(
            mjc=mjc,
            W=W,
            nv=nv,
            nq=int(mm.nq),
            nu=nu,
            mj_map=wp.array(mj_map, dtype=wp.int32, device=dev),
            mj_world=wp.array(mj_world, dtype=wp.int32, device=dev),
            dst_to_src=wp.array(dst_to_src, dtype=wp.int32, device=dev),
            dinv=wp.array(dinv, dtype=wp.float64, device=dev),
            deriv=wp.zeros((W, mm.M_colind.shape[0]), dtype=float, device=dev),
            L=wp.zeros((W, nv, nv), dtype=wp.float64, device=dev),
            J=wp.zeros((W, self.k, nv), dtype=wp.float64, device=dev),
            X=wp.zeros((W, nv, self.k), dtype=wp.float64, device=dev),
            K=wp.zeros((W, nv, nv), dtype=wp.float64, device=dev),
            Y=wp.zeros((W, nv, self.k), dtype=wp.float64, device=dev),
            act_dof=wp.array(act_dof, dtype=wp.int32, device=dev),
            act_tq=wp.array(act_tq, dtype=wp.int32, device=dev),
            act_src=wp.array(act_src, dtype=wp.int32, device=dev),
            act_gain=wp.zeros((W, max(nu, 1)), dtype=float, device=dev),
            src_qd_map=wp.array(src_qd_map, dtype=wp.int32, device=dev),
            src_q_map=wp.array(src_q_map, dtype=wp.int32, device=dev),
            adj_qvel=wp.zeros((W, nv), dtype=wp.float64, device=dev),
            adj_qpos=wp.zeros((W, max(int(mm.nq), nv)), dtype=wp.float64, device=dev),
            u_sol=wp.zeros((W, nv), dtype=wp.float64, device=dev),
        )

    def update_articulated(self, dt: float) -> None:
        """Compute batched closed-chain Delassus ``W_art = J H_eff^-1 J^T`` and ``M_art = (W_art + D)^-1``."""
        art = self._art
        if art is None or dt <= 0.0:
            self._art_ready = False
            return
        mjc, a = art.mjc, self._arm
        mm, md = mjc.mjw_model, mjc.mjw_data
        _mjw_mod().deriv_smooth_vel(mm, md, art.deriv)
        wp.launch(
            _art_factor_kernel,
            dim=art.W,
            inputs=[art.nv, self.k, art.mj_map, art.mj_world, mm.M_rowadr, mm.M_rownnz, mm.M_colind, art.deriv, md.ne,
                    md.efc.J, md.efc.D, mm.body_rootid, mm.body_isdofancestor, md.xipos, md.subtree_com, md.cdof,
                    art.dinv],
            outputs=[art.L, art.J, art.X, art.K, art.Y],
            device=self.device,
        )
        wp.launch(
            _art_mats_kernel,
            dim=(art.W, self.k),
            inputs=[art.nv, self.k, art.dinv, art.J, art.X, art.Y],
            outputs=[a.Minv, a.M],
            device=self.device,
        )
        self._art_dt = float(dt)
        self._art_ready = True

    def snapshot_mujoco(self, control) -> SimpleNamespace:
        """Capture MuJoCo step's factored ``L_H``, Jacobian ``J``, mass ``M`` and active actuator gains for ``wp.Tape``."""
        art = self._art
        mjc = art.mjc
        mm, md = mjc.mjw_model, mjc.mjw_data
        if art.nu > 0 and control is not None and control.joint_target_q is not None:
            wp.launch(
                _art_act_gain_kernel,
                dim=(art.W, art.nu),
                inputs=[art.act_src, art.act_tq, mm.actuator_ctrllimited, mm.actuator_forcelimited,
                        mm.actuator_ctrlrange, mm.actuator_forcerange, mm.actuator_gainprm, mm.actuator_gear,
                        md.ctrl, md.actuator_force],
                outputs=[art.act_gain],
                device=self.device,
            )
            tq_stride = control.joint_target_q.shape[0] // max(art.W, 1)
        else:
            art.act_gain.zero_()
            tq_stride = 0
        return SimpleNamespace(
            L=wp.clone(art.L),
            J=wp.clone(art.J),
            qM=wp.clone(md.M),
            xipos=wp.clone(md.xipos),
            act_gain=wp.clone(art.act_gain),
            tq_stride=tq_stride,
        )

    def mujoco_vjp(self, snap, dt: float, source_local_to_proxy_global: wp.array, *, body_q0, body_q1,
                   body_q0_grad, body_qd1_grad, joint_q1_grad, joint_qd1_grad, body_q1_grad, out_target_q_grad,
                   out_body_f_grad, out_joint_q_grad, out_joint_qd_grad) -> None:
        """Backpropagate adjoints across one ``SolverMuJoCo`` step via cached ``L_H`` and ``J``."""
        art = self._art
        mjc = art.mjc
        mm = mjc.mjw_model
        dev = self.device
        art.adj_qvel.zero_()
        art.adj_qpos.zero_()
        # 1. Seed joint velocity adjoint from body_qd1, joint_qd1, joint_q1 (q1 = q0 + dt qd1) & body_q1.
        if body_qd1_grad is not None or body_q1_grad is not None or body_q0_grad is not None:
            q0_ref = body_q0 if body_q0 is not None else mjc.state_0.body_q
            q1_ref = body_q1 if body_q1 is not None else mjc.state_1.body_q
            wp.launch(
                _mjc_seed_body_vjp_kernel,
                dim=(art.W, self.k // 6),
                inputs=[self.k, art.dst_to_src, self._slot_body, snap.xipos,
                        q0_ref, q1_ref, self.model.body_com,
                        body_q0_grad if body_q0_grad is not None else self._q,
                        int(body_q0_grad is not None),
                        body_qd1_grad if body_qd1_grad is not None else self._qd,
                        int(body_qd1_grad is not None),
                        body_q1_grad if body_q1_grad is not None else self._q,
                        int(body_q1_grad is not None),
                        snap.J, art.nv, float(dt)],
                outputs=[art.adj_qvel, art.adj_qpos],
                device=dev,
            )
        if joint_qd1_grad is not None or joint_q1_grad is not None:
            wp.launch(
                _mjc_seed_joint_vjp_kernel,
                dim=(art.W, art.nv),
                inputs=[art.src_qd_map, art.src_q_map,
                        joint_qd1_grad if joint_qd1_grad is not None else art.deriv,
                        int(joint_qd1_grad is not None),
                        joint_q1_grad if joint_q1_grad is not None else art.deriv,
                        int(joint_q1_grad is not None), float(dt)],
                outputs=[art.adj_qvel, art.adj_qpos],
                device=dev,
            )
        # 2. Solve H_eff @ u = adj_qvel.
        wp.launch(
            _mjc_solve_H_kernel,
            dim=art.W,
            inputs=[art.nv, snap.L, art.adj_qvel],
            outputs=[art.u_sol],
            device=dev,
        )
        # 3. Accumulate into target_q, body_f, joint_qd0, joint_q0.
        if out_target_q_grad is not None and art.nu > 0 and snap.tq_stride > 0:
            wp.launch(
                _mjc_target_q_vjp_kernel,
                dim=(art.W, art.nu),
                inputs=[snap.tq_stride, art.act_dof, art.act_tq, snap.act_gain, art.u_sol, float(dt)],
                outputs=[out_target_q_grad],
                device=dev,
            )
        if out_body_f_grad is not None:
            wp.launch(
                _mjc_body_f_vjp_kernel,
                dim=(art.W, self.k // 6),
                inputs=[self.k, art.nv, self._slot_body, art.dst_to_src, source_local_to_proxy_global, snap.J,
                        art.u_sol, float(dt)],
                outputs=[out_body_f_grad],
                device=dev,
            )
        if out_joint_qd_grad is not None or out_joint_q_grad is not None:
            wp.launch(
                _mjc_prev_state_vjp_kernel,
                dim=art.W,
                inputs=[art.nv, mm.M_rowadr, mm.M_rownnz, mm.M_colind, snap.qM, art.src_qd_map, art.src_q_map,
                        art.adj_qpos, art.u_sol, art.nu, art.act_dof, snap.act_gain, float(dt),
                        int(out_joint_qd_grad is not None), int(out_joint_q_grad is not None)],
                outputs=[out_joint_qd_grad if out_joint_qd_grad is not None else art.deriv,
                         out_joint_q_grad if out_joint_q_grad is not None else art.deriv],
                device=dev,
            )

    def harvest(self, q_full: wp.array, body_local_to_proxy_global: wp.array, out_body_f: wp.array,
                body_qd_before: wp.array, state: State, state_out: State, dt: float) -> None:
        """Harvest proxy feedback ``M (z - pred) / h^2 + (state.body_f + m g)`` into ``out_body_f``."""
        m, a = self.model, self._arm
        wp.launch(
            _proxy_harvest_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[float(dt), body_local_to_proxy_global, q_full, a, m.body_world, m.body_mass, m.body_inv_mass,
                    m.body_inertia, m.gravity, state.body_f, body_qd_before, state_out.body_qd, state_out.body_q],
            outputs=[out_body_f],
            device=self.device,
        )

    def bind(self, arm: Arm) -> None:
        self._arm = arm
        base = np.full(arm.base.shape[0], -1, dtype=np.int32)
        base[self._body.numpy()] = arm.start + self._world * self.k + 6 * self._slot  # its 6 in its world's
        arm.base.assign(base)
        eye = np.broadcast_to(np.eye(self.k, dtype=np.float32), arm.M.shape)  # the worlds' unused DOFs: still
        arm.M.assign(eye)
        arm.Minv.assign(eye)

    def reset(self, mask: wp.array | None = None) -> None:
        """Start the proxies from their synced poses next step (those of the worlds of ``mask``, else all)."""
        self.restart = False
        if mask is None:
            self._valid.zero_()
        elif self._body.shape[0]:
            wp.launch(_proxy_reset_kernel, dim=self._body.shape[0], inputs=[self._body, self.model.body_world, mask],
                      outputs=[self._valid], device=self.device)

    def update(self, state: State, h: float) -> tuple[wp.array, wp.array]:
        """The arm at the step from ``state`` (step ``h``); returns the body poses and velocities for the step: the
        proxies' at their own start, still."""
        m, a = self.model, self._arm
        wp.copy(self._was_valid, self._valid)
        wp.copy(self._q, state.body_q)
        wp.copy(self._qd, state.body_qd)
        if self._art_ready and self._art is not None:
            wp.launch(_proxy_update_art_kernel, dim=self._body.shape[0],
                      inputs=[self._body, a.base, m.body_com, state.body_q, state.body_qd, self._last,
                              self._valid, int(self.restart)],
                      outputs=[a.com, self._q, self._qd, self._offset, self._start], device=self.device)
            wp.launch(_proxy_pred_art_kernel, dim=(a.M.shape[0], self.k),
                      inputs=[self.k, self._slot_body, m.body_world, m.body_mass, m.gravity, state.body_qd,
                              state.body_f if state.body_f is not None else self._qd, int(state.body_f is not None),
                              self._offset, a.Minv, h, self._art_dt],
                      outputs=[a.pred], device=self.device)
            self._art_ready = False
            self._art_used = True
        else:
            wp.launch(_proxy_update_kernel, dim=self._body.shape[0],
                      inputs=[self._body, a.base, a.start, self.k, m.body_world, m.body_mass, m.body_inertia,
                              m.body_com, m.gravity, state.body_q, state.body_qd, state.body_f, self._last, self._valid,
                              int(self.restart), h],
                      outputs=[a.M, a.Minv, a.pred, a.com, self._q, self._qd, self._offset, self._start],
                      device=self.device)
            self._art_used = False
        self.restart = False
        return self._q, self._qd

    def snapshot(self, z: wp.array, state_in: State, pins) -> SimpleNamespace:
        """The step just taken, for its adjoint: the moves ``z``, ``Minv``, the centres of mass, the step's body poses
        (``poses``: a proxy's its start), ``body_f``, the pins' stiffnesses, whether a proxy started from the last
        step's end (``chained``) and the adjoint of that end (``prev``: the last snapshot's ``end_grad``)."""
        a = self._arm
        f = state_in.body_f if state_in.body_f is not None else wp.zeros_like(self._qd)
        out = SimpleNamespace(z=z, Minv=a.Minv, com=a.com, poses=self._q, body_f=f, chained=self._was_valid)
        if self._art_used:
            out.M = a.M
        if pins is not None:
            out.pin_k = pins.p.k
        out = SimpleNamespace(**{k: wp.clone(v) for k, v in vars(out).items()})
        out.art = bool(self._art_used)
        out.art_dt = float(self._art_dt)
        out.body_f_in = state_in.body_f
        out.poses.requires_grad = True
        out.end_grad = wp.zeros_like(self._q)
        out.prev = self._last_grad
        self._last_grad = out.end_grad
        return out

    def finish(self, z: wp.array, state_in: State, state_out: State, dt: float, h: float) -> None:
        """Pose the proxies in ``state_out`` from their moves ``z`` over ``h``, extrapolated to ``dt``."""
        a, m = self._arm, self.model
        if self._free.shape[0]:
            wp.launch(_proxy_free_kernel, dim=self._free.shape[0],
                      inputs=[self._free, m.body_world, m.body_mass, m.gravity, state_in.body_q, state_in.body_qd,
                              state_in.body_f, dt], outputs=[state_out.body_q, state_out.body_qd], device=self.device)
        wp.launch(_proxy_finish_kernel, dim=self._body.shape[0],
                  inputs=[self._body, a.base, z, a.com, m.body_com, self._q, self._offset, dt, h],
                  outputs=[state_out.body_q, state_out.body_qd, self._last], device=self.device)
        self._valid.fill_(1)


@wp.func
def _rotvec(q: wp.quat) -> wp.vec3:
    """``q``'s rotation vector (the short way)."""
    if q[3] < 0.0:
        q = -q
    v = wp.vec3(q[0], q[1], q[2])
    s = wp.length(v)
    if s < 1.0e-12:
        return 2.0 * v
    return (2.0 * wp.atan2(s, q[3]) / s) * v


@wp.func
def _turn(v: wp.vec3, q: wp.quat) -> wp.quat:
    """``q`` turned by the rotation vector ``v``."""
    angle = wp.length(v)
    if angle > 0.0:
        return wp.normalize(wp.quat_from_axis_angle(v / angle, angle) * q)
    return q


@wp.func
def proxy_step(q0: wp.transform, synced: wp.transform, qd: wp.spatial_vector, f: wp.spatial_vector, mass: float,
               inertia: wp.mat33, com: wp.vec3, g: wp.vec3, h: float, w: int, i: int, i0: int, pred: wp.array[float],
               M: wp.array3d[float], Minv: wp.array3d[float]):
    """A proxy's own move over the step from its start ``q0`` (its synced pose ``synced`` moved by
    ``h qd + h^2 (f / m + g, I^-1 tau)``) into ``pred[i:i+6]`` and its inertia ``diag(m, R I R^T)`` about ``q0`` into
    world ``w``'s ``M``, ``Minv`` at ``i0``; returns the synced pose's offset from ``q0`` (``dc``, ``dr``)."""
    R = wp.quat_to_matrix(wp.transform_get_rotation(q0))
    m = wp.max(mass, 1.0e-12)
    I = R * inertia * wp.transpose(R)
    Iinv = wp.mat33(0.0)
    if wp.determinant(inertia) > 0.0:  # (a shapeless proxy may have none)
        Iinv = R * wp.inverse(inertia) * wp.transpose(R)
    dc = wp.transform_point(synced, com) - wp.transform_point(q0, com)
    dr = _rotvec(wp.transform_get_rotation(synced) * wp.quat_inverse(wp.transform_get_rotation(q0)))
    dx = dc + h * wp.spatial_top(qd) + h * h * (wp.spatial_top(f) / m + g)
    dth = dr + h * wp.spatial_bottom(qd) + h * h * (Iinv * wp.spatial_bottom(f))
    for a in range(3):
        pred[i + a] = dx[a]
        pred[i + 3 + a] = dth[a]
        for c in range(3):
            M[w, i0 + a, i0 + c] = wp.where(a == c, m, 0.0)
            M[w, i0 + 3 + a, i0 + 3 + c] = I[a, c]
            Minv[w, i0 + a, i0 + c] = wp.where(a == c, 1.0 / m, 0.0)
            Minv[w, i0 + 3 + a, i0 + 3 + c] = Iinv[a, c]
    return dc, dr


@wp.kernel(enable_backward=False)
def _proxy_update_kernel(body: wp.array[wp.int32], base: wp.array[wp.int32], start: int, k: int,
                         body_world: wp.array[wp.int32], mass: wp.array[float], inertia: wp.array[wp.mat33],
                         body_com: wp.array[wp.vec3], gravity: wp.array[wp.vec3], body_q: wp.array[wp.transform],
                         body_qd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector],
                         last: wp.array[wp.transform], valid: wp.array[wp.int32], restart: int, h: float,
                         M: wp.array3d[float], Minv: wp.array3d[float], pred: wp.array[float], com: wp.array[wp.vec3],
                         q_step: wp.array[wp.transform], qd_step: wp.array[wp.spatial_vector],
                         offset: wp.array[wp.spatial_vector], start_q: wp.array[wp.transform]):
    p = wp.tid()
    b = body[p]
    i = base[b] - start
    w = i // k
    i0 = i - w * k
    synced = body_q[b]
    q0 = synced
    if restart != 0:
        q0 = start_q[b]
    elif valid[b] != 0:
        q0 = last[b]
    start_q[b] = q0
    dc, dr = proxy_step(q0, synced, body_qd[b], body_f[b], mass[b], inertia[b], body_com[b],
                        gravity[wp.max(body_world[b], 0)], h, w, i, i0, pred, M, Minv)
    com[b] = wp.transform_point(q0, body_com[b])
    q_step[b] = q0
    qd_step[b] = wp.spatial_vector()
    offset[b] = wp.spatial_vector(dc, dr)


@wp.kernel(enable_backward=False)
def _proxy_reset_kernel(body: wp.array[wp.int32], body_world: wp.array[wp.int32], mask: wp.array[wp.bool],
                        valid: wp.array[wp.int32]):
    b = body[wp.tid()]
    w = body_world[b]
    if mask[wp.where(w >= 0, w, mask.shape[0] - 1)]:
        valid[b] = 0


@wp.kernel
def _proxy_finish_kernel(body: wp.array[wp.int32], base: wp.array[wp.int32], z: wp.array[float],
                         com: wp.array[wp.vec3], body_com: wp.array[wp.vec3], q_step: wp.array[wp.transform],
                         offset: wp.array[wp.spatial_vector], dt: float, h: float, q_out: wp.array[wp.transform],
                         qd_out: wp.array[wp.spatial_vector], last: wp.array[wp.transform]):
    p = wp.tid()
    b = body[p]
    i = base[b]
    dx = wp.vec3(z[i], z[i + 1], z[i + 2])
    dth = wp.vec3(z[i + 3], z[i + 4], z[i + 5])
    s = dt / h
    rot = _turn(s * dth, wp.transform_get_rotation(q_step[b]))
    q = wp.transform(com[b] + s * dx - wp.quat_rotate(rot, body_com[b]), rot)
    q_out[b] = q
    last[b] = q
    qd_out[b] = wp.spatial_vector((dx - wp.spatial_top(offset[b])) / h, (dth - wp.spatial_bottom(offset[b])) / h)


@wp.kernel(enable_backward=False)
def _proxy_free_kernel(body: wp.array[wp.int32], body_world: wp.array[wp.int32], mass: wp.array[float],
                       gravity: wp.array[wp.vec3], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                       body_f: wp.array[wp.spatial_vector], dt: float, q_out: wp.array[wp.transform],
                       qd_out: wp.array[wp.spatial_vector]):
    """A proxy without shapes: its velocity by its force and gravity (its spin kept), symplectic Euler."""
    b = body[wp.tid()]
    v = wp.spatial_top(body_qd[b]) + dt * (gravity[wp.max(body_world[b], 0)])
    if mass[b] > 0.0:
        v += dt * wp.spatial_top(body_f[b]) / mass[b]
    om = wp.spatial_bottom(body_qd[b])
    rot = _turn(dt * om, wp.transform_get_rotation(body_q[b]))
    q_out[b] = wp.transform(wp.transform_get_translation(body_q[b]) + dt * v, rot)
    qd_out[b] = wp.spatial_vector(v, om)


# -- pins: Newton's ball joints between a proxy body and a rod ---------------------------------


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


@wp.kernel(enable_backward=False)
def _pin_begin_kernel(p: Pins, body_q: wp.array[wp.transform], mass: wp.array[float], scale: float):
    i = wp.tid()
    p.point[i] = wp.transform_point(body_q[p.body[i]], p.local[i])
    p.k[i] = scale * mass[3 * p.node[i]]


@wp.kernel(enable_backward=False)
def _pin_grad_kernel(q: wp.array[float], p: Pins, arm: Arm, fixed: wp.array[wp.int32], b: wp.array[wp.float64]):
    i = wp.tid()
    g = p.k[i] * _pin_gap(q, p, i, arm)
    scatter_node(b, fixed, p.node[i], g)
    arm_scatter(b, arm, p.body[i], p.point[i] - arm.com[p.body[i]], g)


@wp.kernel(enable_backward=False)
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


@wp.kernel(enable_backward=False)
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


@wp.kernel(enable_backward=False)
def _pin_reaction_kernel(q: wp.array[float], p: Pins, arm: Arm, body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    f = p.k[i] * _pin_gap(q, p, i, arm)  # pulls the body toward the node
    body = p.body[i]
    wp.atomic_add(body_f, body, wp.spatial_vector(f, wp.cross(p.point[i] - arm.com[body], f)))


# -- closed-chain articulated Delassus and MuJoCo tape kernels ---------------------------------


@wp.func
def _chol_factor64(A: wp.array3d[wp.float64], w: int, n: int):
    """In-place lower-triangular float64 Cholesky factorization of ``A[w, :n, :n]``."""
    for i in range(n):
        for j in range(i):
            s = A[w, i, j]
            for p in range(j):
                s -= A[w, i, p] * A[w, j, p]
            A[w, i, j] = s / A[w, j, j]
        d = A[w, i, i]
        for p in range(i):
            d -= A[w, i, p] * A[w, i, p]
        A[w, i, i] = wp.sqrt(wp.max(d, wp.float64(1.0e-12)))


@wp.kernel(enable_backward=False)
def _art_factor_kernel(
    nv: int,
    k: int,
    mj_map: wp.array2d[wp.int32],
    mj_world: wp.array2d[wp.int32],
    rowadr: wp.array[wp.int32],
    rownnz: wp.array[wp.int32],
    colind: wp.array[wp.int32],
    deriv: wp.array2d[float],
    ne_arr: wp.array[wp.int32],
    efc_J: wp.array3d[float],
    efc_D: wp.array2d[float],
    body_rootid: wp.array[wp.int32],
    body_anc: wp.array2d[wp.int32],
    xipos: wp.array2d[wp.vec3],
    subtree_com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    dinv: wp.array2d[wp.float64],
    L: wp.array3d[wp.float64],
    J: wp.array3d[wp.float64],
    X: wp.array3d[wp.float64],
    K: wp.array3d[wp.float64],
    Y: wp.array3d[wp.float64],
):
    w = wp.tid()
    for i in range(nv):
        for j in range(nv):
            L[w, i, j] = wp.float64(0.0)
            K[w, i, j] = wp.float64(0.0)
    # 1. M - dt * qDeriv (lower triangle stored by MuJoCoWarp -> symmetric).
    for i in range(nv):
        adr, nnz = rowadr[i], rownnz[i]
        for t in range(nnz):
            j = colind[adr + t]
            v = wp.float64(deriv[w, adr + t])
            L[w, i, j] = v
            L[w, j, i] = v
    # 2. Closed-chain equality constraints: H_eff += J_eq^T D_eq J_eq.
    ne = ne_arr[w]
    for e in range(ne):
        de = wp.float64(efc_D[w, e])
        for i in range(nv):
            ji = de * wp.float64(efc_J[w, e, i])
            for j in range(i, nv):
                v = ji * wp.float64(efc_J[w, e, j])
                L[w, i, j] = L[w, i, j] + v
                if i != j:
                    L[w, j, i] = L[w, j, i] + v
    for i in range(nv):
        L[w, i, i] = L[w, i, i] + wp.float64(1.0e-10)
        for j in range(nv):
            K[w, i, j] = L[w, i, j]
    # 3. Proxy body spatial Jacobian at body COM (xipos).
    nb = k // 6
    for s in range(nb):
        mb = mj_map[w, s]
        o = 6 * s
        if mb < 0:
            for a in range(6):
                for c in range(nv):
                    J[w, o + a, c] = wp.float64(0.0)
            continue
        wm = mj_world[w, s]
        dx = xipos[wm, mb] - subtree_com[wm, body_rootid[mb]]
        for c in range(nv):
            if body_anc[mb, c] != 0:
                cd = cdof[wm, c]
                w_ax = wp.spatial_top(cd)
                v_p = wp.spatial_bottom(cd) + wp.cross(w_ax, dx)
                for a in range(3):
                    J[w, o + a, c] = wp.float64(v_p[a])
                    J[w, o + 3 + a, c] = wp.float64(w_ax[a])
            else:
                for a in range(6):
                    J[w, o + a, c] = wp.float64(0.0)
    # 4. Cholesky factorize L L^T = H_eff and forward-solve L X = J^T.
    _chol_factor64(L, w, nv)
    for m in range(k):
        for i in range(nv):
            s_val = J[w, m, i]
            for p in range(i):
                s_val -= L[w, i, p] * X[w, p, m]
            X[w, i, m] = s_val / L[w, i, i]
    # 5. Woodbury core K = H_eff + J^T D^-1 J, factorize and solve K Y = D^-1 J.
    for i in range(nv):
        for j in range(i, nv):
            acc = wp.float64(0.0)
            for m in range(k):
                acc += J[w, m, i] * dinv[w, m] * J[w, m, j]
            K[w, i, j] = K[w, i, j] + acc
            if i != j:
                K[w, j, i] = K[w, i, j]
    _chol_factor64(K, w, nv)
    for m in range(k):
        di = dinv[w, m]
        for i in range(nv):
            s_val = di * J[w, m, i]
            for p in range(i):
                s_val -= K[w, i, p] * Y[w, p, m]
            Y[w, i, m] = s_val / K[w, i, i]
        for i_rev in range(nv):
            i = nv - 1 - i_rev
            s_val = Y[w, i, m]
            for p in range(i + 1, nv):
                s_val -= K[w, p, i] * Y[w, p, m]
            Y[w, i, m] = s_val / K[w, i, i]


@wp.kernel(enable_backward=False)
def _art_mats_kernel(nv: int, k: int, dinv: wp.array2d[wp.float64], J: wp.array3d[wp.float64],
                     X: wp.array3d[wp.float64], Y: wp.array3d[wp.float64],
                     W_out: wp.array3d[float], M_out: wp.array3d[float]):
    w, m = wp.tid()
    di = dinv[w, m]
    for m2 in range(m, k):
        w_val = wp.float64(0.0)
        jty = wp.float64(0.0)
        for d in range(nv):
            w_val += X[w, d, m] * X[w, d, m2]
            jty += J[w, m, d] * Y[w, d, m2]
        m_val = wp.where(m == m2, di, wp.float64(0.0)) - di * jty
        wf = float(w_val)
        mf = float(m_val)
        W_out[w, m, m2] = wf
        W_out[w, m2, m] = wf
        M_out[w, m, m2] = mf
        M_out[w, m2, m] = mf


@wp.kernel(enable_backward=False)
def _proxy_update_art_kernel(body: wp.array[wp.int32], base: wp.array[wp.int32], body_com: wp.array[wp.vec3],
                             body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                             last: wp.array[wp.transform], valid: wp.array[wp.int32], restart: int,
                             com: wp.array[wp.vec3], q_step: wp.array[wp.transform],
                             qd_step: wp.array[wp.spatial_vector], offset: wp.array[wp.spatial_vector],
                             start_q: wp.array[wp.transform]):
    p = wp.tid()
    b = body[p]
    synced = body_q[b]
    q0 = synced
    if restart != 0:
        q0 = start_q[b]
    elif valid[b] != 0:
        q0 = last[b]
    start_q[b] = q0
    dc = wp.transform_point(synced, body_com[b]) - wp.transform_point(q0, body_com[b])
    dr = _rotvec(wp.transform_get_rotation(synced) * wp.quat_inverse(wp.transform_get_rotation(q0)))
    com[b] = wp.transform_point(q0, body_com[b])
    q_step[b] = q0
    qd_step[b] = wp.spatial_vector()
    offset[b] = wp.spatial_vector(dc, dr)


@wp.kernel(enable_backward=False)
def _proxy_pred_art_kernel(k: int, slot_body: wp.array2d[wp.int32], body_world: wp.array[wp.int32],
                           mass: wp.array[float], gravity: wp.array[wp.vec3],
                           body_qd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector],
                           has_f: int, offset: wp.array[wp.spatial_vector], Minv: wp.array3d[float],
                           h: float, art_dt: float, pred: wp.array[float]):
    w, m = wp.tid()
    s = m // 6
    b = slot_body[w, s]
    if b < 0:
        pred[w * k + m] = 0.0
        return
    acc = float(0.0)
    nb = k // 6
    for s2 in range(nb):
        b2 = slot_body[w, s2]
        if b2 >= 0:
            w2 = wp.max(body_world[b2], 0)
            f2 = wp.spatial_vector(mass[b2] * gravity[w2], wp.vec3(0.0))
            if has_f != 0:
                f2 = body_f[b2] + f2
            o2 = 6 * s2
            for a2 in range(6):
                acc += Minv[w, m, o2 + a2] * f2[a2]
    c = m - 6 * s
    pred[w * k + m] = offset[b][c] + h * body_qd[b][c] + (h * art_dt) * acc


@wp.kernel(enable_backward=False)
def _proxy_harvest_kernel(dt: float, body_local_to_proxy_global: wp.array[wp.int32], q_full: wp.array[float],
                          arm: Arm, body_world: wp.array[wp.int32], body_mass: wp.array[float],
                          body_inv_mass: wp.array[float], body_inertia: wp.array[wp.mat33],
                          gravity: wp.array[wp.vec3], body_f: wp.array[wp.spatial_vector],
                          qd_before: wp.array[wp.spatial_vector], qd_after: wp.array[wp.spatial_vector],
                          body_q: wp.array[wp.transform], out_coupling_forces: wp.array[wp.spatial_vector]):
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return
    base = arm.base[local_id]
    if base >= 0:
        i = base - arm.start
        w = i // arm.k
        m0 = i - w * arm.k
        f_soft = wp.spatial_vector(
            wp.vec3(_arm_inertia(q_full, arm, w, m0), _arm_inertia(q_full, arm, w, m0 + 1),
                    _arm_inertia(q_full, arm, w, m0 + 2)),
            wp.vec3(_arm_inertia(q_full, arm, w, m0 + 3), _arm_inertia(q_full, arm, w, m0 + 4),
                    _arm_inertia(q_full, arm, w, m0 + 5)),
        )
        g = gravity[wp.max(body_world[local_id], 0)]
        f_grav = wp.vec3(0.0)
        if body_inv_mass[local_id] > 0.0:
            f_grav = body_mass[local_id] * g
        f_ext = body_f[local_id] + wp.spatial_vector(f_grav, wp.vec3(0.0))
        wp.atomic_add(out_coupling_forces, global_id, f_soft + f_ext)
        return
    dv = wp.spatial_top(qd_after[local_id]) - wp.spatial_top(qd_before[local_id])
    dw = wp.spatial_bottom(qd_after[local_id]) - wp.spatial_bottom(qd_before[local_id])
    r = wp.transform_get_rotation(body_q[local_id])
    wp.atomic_add(
        out_coupling_forces,
        global_id,
        wp.spatial_vector(body_mass[local_id] * dv / dt,
                          wp.quat_rotate(r, body_inertia[local_id] * wp.quat_rotate_inv(r, dw)) / dt),
    )


@wp.kernel
def _tape_sync_proxy_kernel(src_q: wp.array[wp.transform], src_qd: wp.array[wp.spatial_vector],
                            src_to_dst: wp.array[wp.int32], dst_q: wp.array[wp.transform],
                            dst_qd: wp.array[wp.spatial_vector]):
    b = wp.tid()
    d = src_to_dst[b]
    if d >= 0:
        dst_q[d] = src_q[b]
        dst_qd[d] = src_qd[b]


@wp.kernel
def _tape_rewind_force_kernel(body_g: wp.array[wp.vec3], coupling_forces: wp.array[wp.spatial_vector],
                              dst_to_proxy: wp.array[wp.int32], body_mass: wp.array[float],
                              body_inv_mass: wp.array[float], dst_body_f: wp.array[wp.spatial_vector]):
    b = wp.tid()
    gid = dst_to_proxy[b]
    if gid < 0:
        return
    f_grav = wp.vec3(0.0)
    if body_inv_mass[b] > 0.0:
        f_grav = body_mass[b] * body_g[b]
    dst_body_f[b] = -coupling_forces[gid] - wp.spatial_vector(f_grav, wp.vec3(0.0))


@wp.kernel(module="unique", enable_backward=False)
def _art_act_gain_kernel(
    act_src: wp.array[wp.int32],
    act_tq: wp.array[wp.int32],
    ctrllimited: wp.array[wp.bool],
    forcelimited: wp.array[wp.bool],
    ctrlrange: wp.array2d[wp.vec2],
    forcerange: wp.array2d[wp.vec2],
    gainprm: wp.array2d(dtype=Any),
    gear: wp.array2d[wp.spatial_vector],
    ctrl: wp.array2d[float],
    act_force: wp.array2d[float],
    act_gain: wp.array2d[float],
):
    w, a = wp.tid()
    if act_src[a] != 0 or act_tq[a] < 0:
        act_gain[w, a] = 0.0
        return
    active = float(1.0)
    if ctrllimited[a]:
        cr = ctrlrange[w, a]
        if ctrl[w, a] <= cr[0] + 1.0e-6 or ctrl[w, a] >= cr[1] - 1.0e-6:
            active = 0.0
    if forcelimited[a]:
        fr = forcerange[w, a]
        if act_force[w, a] <= fr[0] + 1.0e-6 or act_force[w, a] >= fr[1] - 1.0e-6:
            active = 0.0
    act_gain[w, a] = active * float(gainprm[w, a][0]) * float(gear[w, a][0])


@wp.kernel(enable_backward=False)
def _art_harvest_vjp_kernel(k: int, slot_body: wp.array2d[wp.int32], dst_to_src: wp.array[wp.int32],
                            dst_to_proxy: wp.array[wp.int32], M: wp.array3d[float], W: wp.array3d[float],
                            inv_h: float, dt_over_h: float, f_out_grad: wp.array[wp.spatial_vector],
                            qd_soft_grad: wp.array[wp.spatial_vector], qd_mjc_grad: wp.array[wp.spatial_vector],
                            f_in_grad: wp.array[wp.spatial_vector]):
    idx = wp.tid()
    nb = k // 6
    w = idx // nb
    s = idx - w * nb
    b_dst = slot_body[w, s]
    if b_dst < 0:
        return
    b_src = dst_to_src[b_dst]
    gid = dst_to_proxy[b_dst]
    if gid < 0:
        return
    # d F_soft / d qd_out = M / h -> g_qd = sum_{s2} M[w, s2, s]^T * f_out_grad[gid2] / h.
    g_v = wp.vec3(0.0)
    g_w = wp.vec3(0.0)
    o = 6 * s
    for s2 in range(nb):
        b2 = slot_body[w, s2]
        if b2 < 0:
            continue
        g2 = dst_to_proxy[b2]
        if g2 < 0:
            continue
        fg2 = f_out_grad[g2]
        o2 = 6 * s2
        for a in range(3):
            for c in range(6):
                g_v[a] += M[w, o2 + c, o + a] * fg2[c]
                g_w[a] += M[w, o2 + c, o + 3 + a] * fg2[c]
    sv = inv_h * wp.spatial_vector(g_v, g_w)
    wp.atomic_add(qd_soft_grad, b_dst, sv)
    if b_src >= 0:
        wp.atomic_add(qd_mjc_grad, b_src, -sv)
    if dt_over_h != 0.0:
        fv = wp.vec3(0.0)
        fw = wp.vec3(0.0)
        for s2 in range(nb):
            b2 = slot_body[w, s2]
            if b2 < 0 or dst_to_proxy[b2] < 0:
                continue
            g2 = dst_to_proxy[b2]
            fg2 = f_out_grad[g2]
            o2 = 6 * s2
            for m_mid in range(k):
                for c in range(6):
                    coeff = fg2[c] * M[w, o2 + c, m_mid]
                    for a in range(3):
                        fv[a] += W[w, m_mid, o + a] * coeff
                        fw[a] += W[w, m_mid, o + 3 + a] * coeff
        wp.atomic_add(f_in_grad, gid, dt_over_h * wp.spatial_vector(fv, fw))


@wp.func
def _quat_mul_vec(a: wp.vec4, b: wp.vec4) -> wp.vec3:
    return wp.vec3(
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    )


@wp.func
def _transform_grad_to_com_wrench(pose: wp.transform, body_com: wp.vec3, g: wp.transform) -> wp.spatial_vector:
    """Convert Warp ``body_q.grad`` (origin + quat) into world spatial wrench about ``COM``."""
    gp = wp.transform_get_translation(g)
    gq = wp.transform_get_rotation(g)
    q = wp.transform_get_rotation(pose)
    q_vec = wp.vec4(gq[0], gq[1], gq[2], gq[3])
    q_conj = wp.vec4(-q[0], -q[1], -q[2], q[3])
    r_com = wp.quat_rotate(q, body_com)
    tau = 0.5 * _quat_mul_vec(q_vec, q_conj) + wp.cross(gp, r_com)
    return wp.spatial_vector(gp, tau)


@wp.kernel(enable_backward=False)
def _mjc_seed_body_vjp_kernel(
    k: int,
    dst_to_src: wp.array[wp.int32],
    slot_body: wp.array2d[wp.int32],
    xipos: wp.array2d[wp.vec3],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_q0_grad: wp.array[wp.transform],
    has_q0: int,
    body_qd1_grad: wp.array[wp.spatial_vector],
    has_qd1: int,
    body_q1_grad: wp.array[wp.transform],
    has_q1: int,
    J: wp.array3d[wp.float64],
    nv: int,
    dt: float,
    adj_qvel: wp.array2d[wp.float64],
    adj_qpos: wp.array2d[wp.float64],
):
    w, s = wp.tid()
    b_dst = slot_body[w, s]
    if b_dst < 0:
        return
    b_src = dst_to_src[b_dst]
    if b_src < 0:
        return
    o = 6 * s
    w_qd = wp.spatial_vector()
    w_q0 = wp.spatial_vector()
    if has_qd1 != 0:
        w_qd = w_qd + body_qd1_grad[b_src]
    if has_q1 != 0:
        wq1 = _transform_grad_to_com_wrench(body_q1[b_src], body_com[b_dst], body_q1_grad[b_src])
        w_qd = w_qd + dt * wq1
        w_q0 = w_q0 + wq1
    if has_q0 != 0:
        w_q0 = w_q0 + _transform_grad_to_com_wrench(body_q0[b_src], body_com[b_dst], body_q0_grad[b_src])
    for d in range(nv):
        v_qd = wp.float64(0.0)
        v_q0 = wp.float64(0.0)
        for a in range(6):
            j_val = J[w, o + a, d]
            v_qd += j_val * wp.float64(w_qd[a])
            v_q0 += j_val * wp.float64(w_q0[a])
        if v_qd != wp.float64(0.0):
            wp.atomic_add(adj_qvel, w, d, v_qd)
        if v_q0 != wp.float64(0.0):
            wp.atomic_add(adj_qpos, w, d, v_q0)


@wp.kernel(enable_backward=False)
def _mjc_seed_joint_vjp_kernel(src_qd_map: wp.array2d[wp.int32], src_q_map: wp.array2d[wp.int32],
                               joint_qd1_grad: wp.array[float], has_qd1: int,
                               joint_q1_grad: wp.array[float], has_q1: int, dt: float,
                               adj_qvel: wp.array2d[wp.float64], adj_qpos: wp.array2d[wp.float64]):
    w, d = wp.tid()
    if has_qd1 != 0:
        idx = src_qd_map[w, d]
        if idx >= 0:
            wp.atomic_add(adj_qvel, w, d, wp.float64(joint_qd1_grad[idx]))
    if has_q1 != 0:
        idx_q = src_q_map[w, d]
        if idx_q >= 0:
            g = wp.float64(joint_q1_grad[idx_q])
            wp.atomic_add(adj_qvel, w, d, wp.float64(dt) * g)
            wp.atomic_add(adj_qpos, w, d, g)


@wp.kernel(enable_backward=False)
def _mjc_solve_H_kernel(nv: int, L: wp.array3d[wp.float64], adj_qvel: wp.array2d[wp.float64],
                        u_sol: wp.array2d[wp.float64]):
    w = wp.tid()
    for i in range(nv):
        s = adj_qvel[w, i]
        for p in range(i):
            s -= L[w, i, p] * u_sol[w, p]
        u_sol[w, i] = s / L[w, i, i]
    for i_rev in range(nv):
        i = nv - 1 - i_rev
        s = u_sol[w, i]
        for p in range(i + 1, nv):
            s -= L[w, p, i] * u_sol[w, p]
        u_sol[w, i] = s / L[w, i, i]


@wp.kernel(enable_backward=False)
def _mjc_target_q_vjp_kernel(tq_stride: int, act_dof: wp.array[wp.int32], act_tq: wp.array[wp.int32],
                             act_gain: wp.array2d[float], u_sol: wp.array2d[wp.float64], dt: float,
                             out_tq_grad: wp.array[float]):
    w, a = wp.tid()
    d = act_dof[a]
    tq = act_tq[a]
    if d >= 0 and tq >= 0:
        g = dt * act_gain[w, a] * float(u_sol[w, d])
        if g != 0.0:
            wp.atomic_add(out_tq_grad, w * tq_stride + tq, g)


@wp.kernel(enable_backward=False)
def _mjc_body_f_vjp_kernel(k: int, nv: int, slot_body: wp.array2d[wp.int32], dst_to_src: wp.array[wp.int32],
                           source_to_global: wp.array[wp.int32], J: wp.array3d[wp.float64],
                           u_sol: wp.array2d[wp.float64], dt: float, out_bf_grad: wp.array[wp.spatial_vector]):
    w, s = wp.tid()
    b_dst = slot_body[w, s]
    if b_dst < 0:
        return
    b_src = dst_to_src[b_dst]
    if b_src < 0:
        return
    gid = source_to_global[b_src]
    if gid < 0:
        return
    o = 6 * s
    fv = wp.vec3(0.0)
    fw = wp.vec3(0.0)
    for d in range(nv):
        u = u_sol[w, d]
        for a in range(3):
            fv[a] += dt * float(J[w, o + a, d] * u)
            fw[a] += dt * float(J[w, o + 3 + a, d] * u)
    wp.atomic_add(out_bf_grad, gid, wp.spatial_vector(fv, fw))


@wp.kernel(enable_backward=False)
def _mjc_prev_state_vjp_kernel(nv: int, rowadr: wp.array[wp.int32], rownnz: wp.array[wp.int32],
                               colind: wp.array[wp.int32], qM: wp.array2d[float],
                               src_qd_map: wp.array2d[wp.int32], src_q_map: wp.array2d[wp.int32],
                               adj_qpos: wp.array2d[wp.float64], u_sol: wp.array2d[wp.float64],
                               nu: int, act_dof: wp.array[wp.int32], act_gain: wp.array2d[float],
                               dt: float, do_qd: int, do_q: int,
                               out_qd0_grad: wp.array[float], out_q0_grad: wp.array[float]):
    w = wp.tid()
    if do_qd != 0:
        for i in range(nv):
            ui = u_sol[w, i]
            adr, nnz = rowadr[i], rownnz[i]
            for t in range(nnz):
                j = colind[adr + t]
                mij = wp.float64(qM[w, adr + t])
                ii = src_qd_map[w, i]
                ij = src_qd_map[w, j]
                if ii >= 0:
                    wp.atomic_add(out_qd0_grad, ii, float(mij * u_sol[w, j]))
                if i != j and ij >= 0:
                    wp.atomic_add(out_qd0_grad, ij, float(mij * ui))
    if do_q != 0:
        for d in range(nv):
            iq = src_q_map[w, d]
            if iq >= 0 and adj_qpos[w, d] != wp.float64(0.0):
                wp.atomic_add(out_q0_grad, iq, float(adj_qpos[w, d]))
        for a in range(nu):
            d = act_dof[a]
            if d >= 0:
                iq = src_q_map[w, d]
                if iq >= 0:
                    wp.atomic_add(out_q0_grad, iq, -dt * act_gain[w, a] * float(u_sol[w, d]))
