from functools import cache

import numpy as np
import warp as wp
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase

from ..builder import add_colliding_rod, fix_segment, register_custom_attributes
from ..dofs import dof_constants, external_force, flatten_state
from ..solver import advance_frames_kernel, pose_proxies_kernel, predict_kernel, proxy_joints
from ..triplet import Triplets, linear_energy
from .contact import ContactBlock
from .rod import RodView, rod_dofs, rod_jacobian, rod_strain, rod_view
from .state import SymmetricCSR, linear_solver

_C1 = 1.0e-4  # Armijo's sufficient decrease
_NOISE = 1.0e-5  # a rise of the objective within its float32 rounding


class DiSMechSolver(SolverBase):
    """Implicit discrete elastic rods with contact and friction, by strain-space ADMM.

    Fixed DOFs are prescribed through ``state_in.dismech.q``. Contacts are the caller's: pass ``Contacts`` from a
    :class:`~newton.CollisionPipeline`; rods take its rigid contacts (their capsule proxies, posed every step).
    Obstacles move with ``state_in.body_qd``.

    Args:
        iterations: Iterations per step (a fixed count, graph-capturable).
        beta: Strain penalty over the energy's stiffness ``K + C/dt``.
        linesearch: Trial steps per iteration (``1`` takes the full chord step).
        exact: Gauss-Newton on the step's energy, ``A`` refactorised every iteration (no elastic splitting).
        self_contact: Keep rod-rod contacts.
        friction: Coulomb ``mu``.
        contact_beta: Contact penalty over the inertia-only inverse Delassus diagonal.
        engage_gap: A contact takes part in a step only if the inertial prediction brings it within that many
            thicknesses of touching, or it carried force in the last step (a separated contact's penalty in ``A``
            would slow the motion near obstacles).
    """

    register_custom_attributes = staticmethod(register_custom_attributes)
    add_rod = staticmethod(add_colliding_rod)
    fix_segment = staticmethod(fix_segment)

    def __init__(self, model: Model, *, iterations: int = 10, beta: float = 1.0, linesearch: int = 4,
                 exact: bool = False, self_contact: bool = True, friction: float = 0.3, contact_beta: float = 16.0,
                 engage_gap: float = 0.25):
        super().__init__(model)
        if linesearch < 1:
            raise ValueError("linesearch must be >= 1")
        self.iterations, self.exact = int(iterations), bool(exact)
        self.trials = trials = int(linesearch) + 1  # trial 0: the current iterate
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
        self.A = SymmetricCSR(n, (tr.dofs(), edge_dofs), dev)  # an edge's nodes: its contacts
        self.rods = ElementBlock(element_kernels(RodView, rod_strain, rod_jacobian, rod_dofs, 5, 11), tr.count,
                                 self.A.slots[0], self.fixed, trials, beta, dev)
        self.contact = ContactBlock(model, self.fixed, trials, friction=friction, self_contact=self_contact,
                                     beta=contact_beta, engage_gap=engage_gap)
        self.blocks = (self.rods, self.contact)
        self._linear = linear_solver(self.A)
        self._b = wp.zeros(n, dtype=wp.float64, device=dev)  # the objective's gradient
        self._dq = wp.zeros(n, dtype=float, device=dev)
        self._qt = wp.zeros((trials, n), dtype=float, device=dev)
        self._obj = wp.zeros(trials, dtype=wp.float64, device=dev)
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
        wp.launch(predict_kernel, dim=self.num_dofs, inputs=[self.fixed, q_in, state_in.dismech.qd, dt],
                  outputs=[self.q_pred, q], device=dev)
        self._solve(state_in, q, contacts, dt)
        wp.launch(_velocity_kernel, dim=self.num_dofs, inputs=[self.fixed, q_in, q, dt],
                  outputs=[state_out.dismech.qd], device=dev)
        wp.launch(advance_frames_kernel, dim=d.edge_length.shape[0],
                  inputs=[state_in.particle_q, state_out.particle_q, d.edge_node0, d.edge_node1,
                          state_in.dismech.edge_d1_q],
                  outputs=[state_out.dismech.edge_d1_q], device=dev)
        self.triplets.end_step(state_in, state_out)
        self.update_proxies(state_out)

    def reset(self, state: State, world_mask: wp.array | None = None, flags: int | None = None) -> None:
        """Start the next step from ``state`` without the last step's warm start: for a step that does not continue
        the last one, such as a restart from data."""
        if world_mask is not None:
            raise NotImplementedError(f"{type(self).__name__}.reset resets every world")
        flatten_state(state)
        self.rods.reset(rod_view(self.triplets, state), state.dismech.q)
        self.contact.reset()
        self._started = True

    def refresh_mass(self) -> None:
        """Re-read the masses and twist inertias after they changed."""
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
            pose_proxies_kernel, dim=d.edge_length.shape[0],
            inputs=[state.particle_q, state.particle_qd, state.dismech.edge_q, state.dismech.edge_d1_q, d.edge_node0,
                    d.edge_node1, d.edge_body, self._edge_joint, int(state.body_qd is not None)],
            outputs=[state.body_q, body_qd, state.joint_q if joints else self._no_joint,
                     state.joint_qd if joints else self._no_joint],
            device=self.device,
        )

    def _solve(self, state_in: State, q: wp.array, contacts: Contacts | None, dt: float) -> None:
        dev, n, tr = self.device, self.num_dofs, self.triplets
        alpha = 1.0 / (dt * dt)
        particle_f = state_in.particle_f if state_in.particle_f is not None else self._no_force
        ext = [self.mass, alpha, self.fixed, self.model.gravity, particle_f, self.num_node_dofs]
        b = self._b

        self.rods.begin_step(rod_view(tr, state_in), tr.params, tr.rest, tr.strain_prev, dt, self.exact)
        if not self._started and not self.exact:
            self.reset(state_in)
        self.contact.begin_step(state_in, contacts, dt, q, self.mass, alpha)
        b.zero_()
        for blk in self.blocks:
            blk.grad(q, b)
        for it in range(self.iterations):
            if it == 0 or self.exact:  # the chord matrix at the prediction (every iterate: Gauss-Newton)
                self._assemble(q, alpha)
            wp.launch(_dof_kernel, dim=n, inputs=[q, self.q_pred, *ext], outputs=[b, self._obj, self._slope],
                      device=dev)
            self._linear.solve(b, self._dq)
            wp.launch(_trials_dof_kernel, dim=(self.trials, n), inputs=[q, self._dq, self.q_pred, *ext],
                      outputs=[self._qt, b, self._obj, self._slope], device=dev)
            for blk in self.blocks:
                blk.trials(self._qt, self._obj)
            wp.launch(_pick_kernel, dim=n, inputs=[self._obj, self._slope, self.trials, _C1, self._qt],
                      outputs=[q, self._pick], device=dev)
            for blk in self.blocks:
                blk.commit(q, self._pick, b)
        self.contact.end_step()

    def _assemble(self, q: wp.array, alpha: float) -> None:
        self.A.vals.zero_()
        for blk in self.blocks:
            blk.assemble(q, self.A)
        wp.launch(_inertia_kernel, dim=self.num_dofs, inputs=[self.mass, alpha, self.fixed],
                  outputs=[self.A.indptr, self.A.vals], device=self.device)
        self._linear.invalidate()


class ElementBlock:
    """One element type's strain-ADMM block: penalty ``P``, local variables ``z``, duals ``u``, trial strains."""

    def __init__(self, kernels, count: int, slots: wp.array, fixed: wp.array, trials: int, beta: float, device):
        self.k, self.count, self.slots, self.fixed = kernels, count, slots, fixed
        self.beta, self.device = beta, device
        self.P, self.z, self.u = (wp.zeros(count, dtype=kernels.vec, device=device) for _ in range(3))
        self.eps_trial = wp.zeros((trials, count), dtype=kernels.vec, device=device)

    def _launch(self, kernel, dim, inputs, outputs):
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=self.device)

    def reset(self, view, q: wp.array) -> None:
        """``z = eps(q)``, ``u = 0``."""
        self._launch(self.k.reset, self.count, [q, view], [self.z, self.u])

    def begin_step(self, view, params: wp.array, rest: wp.array, eps_prev: wp.array, dt: float, exact: bool) -> None:
        self.view, self.params, self.rest, self.eps_prev, self.dt, self.exact = view, params, rest, eps_prev, dt, exact
        self._launch(self.k.penalty, self.count, [params, rest, eps_prev, 1.0 if exact else self.beta, dt, int(exact)],
                     [self.P, self.z, self.u])

    def assemble(self, q: wp.array, A: SymmetricCSR) -> None:
        self._launch(self.k.assemble, self.count, [q, self.view, self.P, self.fixed], [self.slots, A.vals])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self._launch(self.k.grad, self.count, [q, self.view, self.P, self.z, self.u, self.fixed], [b])

    def trials(self, qt: wp.array, obj: wp.array) -> None:
        self._launch(self.k.trials, (qt.shape[0], self.count), [qt, self.view, self.P, self.z, self.u],
                     [self.eps_trial, obj])

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self._launch(self.k.commit, self.count,
                     [q, self.view, pick, self.eps_trial, self.params, self.rest, self.eps_prev, self.P, self.dt,
                      int(self.exact), self.fixed],
                     [self.z, self.u, b])


@cache
def element_kernels(View, strain, jacobian, dofs, N: int, D: int):
    """The block kernels of an element (module docstring); ``params`` per element are ``[K, C]``."""
    vec = wp.types.vector(N, float)
    params_vec = wp.types.vector(2 * N, float)
    dofs_vec = wp.types.vector(D, wp.int32)

    @wp.func
    def grad_into(q: wp.array[float], v: View, i: int, P: vec, z: vec, u: vec, fixed: wp.array[wp.int32],
                  b: wp.array[wp.float64]):
        """``b += J^T P (eps(q) - z + u)`` on the free DOFs."""
        eps, J = jacobian(q, v, i)
        g = wp.transpose(J) * wp.cw_mul(P, eps - z + u)
        d = dofs(v, i)
        for k in range(D):
            if fixed[d[k]] == 0:
                wp.atomic_add(b, d[k], wp.float64(g[k]))

    @wp.kernel(module="unique")
    def reset(q: wp.array[float], v: View, z: wp.array[vec], u: wp.array[vec]):
        i = wp.tid()
        z[i] = strain(q, v, i)
        u[i] = vec()

    @wp.kernel(module="unique")
    def penalty(params: wp.array[params_vec], rest: wp.array[vec], eps_prev: wp.array[vec], beta: float, dt: float,
                exact: int, P: wp.array[vec], z: wp.array[vec], u: wp.array[vec]):
        """``P = beta (K + C/dt)``; ``exact``: ``z`` the energy's minimiser, ``u = 0``."""
        i = wp.tid()
        p = params[i]
        out = vec()
        zt = vec()
        for j in range(N):
            w = p[N + j] / dt
            out[j] = beta * (p[j] + w)
            if p[j] + w > 0.0:
                zt[j] = (p[j] * rest[i][j] + w * eps_prev[i][j]) / (p[j] + w)
        P[i] = out
        if exact != 0:
            z[i] = zt
            u[i] = vec()

    @wp.kernel(module="unique")
    def assemble(q: wp.array[float], v: View, P: wp.array[vec], fixed: wp.array[wp.int32],
                 slots: wp.array2d[wp.int32], vals: wp.array[wp.float64]):
        """``J^T P J`` (Gauss-Newton) into the upper CSR; ``slots``: :attr:`SymmetricCSR.pair_slots` of the DOFs."""
        i = wp.tid()
        eps, J = jacobian(q, v, i)
        d = dofs(v, i)
        p = P[i]
        free = dofs_vec()
        for a in range(D):
            free[a] = int(fixed[d[a]] == 0)
        for a in range(D):
            for c in range(a, D):
                val = float(0.0)
                for r in range(N):
                    val = val + J[r, a] * p[r] * J[r, c]
                if val != 0.0 and free[a] != 0 and free[c] != 0:
                    wp.atomic_add(vals, slots[a * D - (a * (a - 1)) // 2 + (c - a), i], wp.float64(val))

    @wp.kernel(module="unique")
    def grad(q: wp.array[float], v: View, P: wp.array[vec], z: wp.array[vec], u: wp.array[vec],
             fixed: wp.array[wp.int32], b: wp.array[wp.float64]):
        i = wp.tid()
        grad_into(q, v, i, P[i], z[i], u[i], fixed, b)

    @wp.kernel(module="unique")
    def trials(qt: wp.array2d[float], v: View, P: wp.array[vec], z: wp.array[vec], u: wp.array[vec],
               eps_trial: wp.array2d[vec], obj: wp.array[wp.float64]):
        k, i = wp.tid()
        eps = strain(qt[k], v, i)
        eps_trial[k, i] = eps
        r = eps - z[i] + u[i]
        wp.atomic_add(obj, k, wp.float64(0.5 * wp.dot(wp.cw_mul(P[i], r), r)))

    @wp.kernel(module="unique")
    def commit(q: wp.array[float], v: View, pick: wp.array[wp.int32], eps_trial: wp.array2d[vec],
               params: wp.array[params_vec], rest: wp.array[vec], eps_prev: wp.array[vec], P: wp.array[vec], dt: float,
               exact: int, fixed: wp.array[wp.int32], z: wp.array[vec], u: wp.array[vec], b: wp.array[wp.float64]):
        """The closed-form prox and dual step from the picked trial's strains (not with ``exact``), then the
        gradient at the new ``q``."""
        i = wp.tid()
        zi = z[i]
        ui = u[i]
        if exact == 0:
            p = params[i]
            Pi = P[i]
            d = eps_trial[pick[0], i] + ui
            for j in range(N):
                w = p[N + j] / dt
                den = p[j] + w + Pi[j]
                if den > 0.0:
                    zi[j] = (p[j] * rest[i][j] + w * eps_prev[i][j] + Pi[j] * d[j]) / den
                else:
                    zi[j] = d[j]
            ui = d - zi
            z[i] = zi
            u[i] = ui
        grad_into(q, v, i, P[i], zi, ui, fixed, b)

    return type("ElementKernels", (), dict(vec=vec, reset=reset, penalty=penalty, assemble=assemble, grad=grad,
                                           trials=trials, commit=commit))


# -- the global step and its line search ---------------------------------------------------


@wp.func
def _trial_step(k: int) -> float:
    """Trial 0 is the current iterate, trial ``k >= 1`` the step ``2^(1-k)``."""
    if k == 0:
        return 0.0
    return wp.pow(0.5, float(k - 1))


@wp.func
def _ext(i: int, mass: wp.array[float], gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3],
         num_node_dofs: int) -> float:
    if i < num_node_dofs:
        return external_force(i, mass, gravity, particle_f)
    return 0.0


@wp.func
def _pick(obj: wp.array[wp.float64], slope: wp.array[wp.float64], trials: int, c1: float) -> int:
    """Armijo: the largest trial step with sufficient decrease. Else (``-k``) the trial with the lowest objective
    if it rises by no more than float32 noise (near convergence every trial ties), or ``0``: no step."""
    f0 = obj[0]
    s = slope[0]
    best = int(-1)
    for k in range(1, trials):
        if obj[k] <= f0 - wp.float64(c1 * _trial_step(k)) * s:
            return k
        if wp.isfinite(obj[k]) and (best < 0 or obj[k] < obj[best]):
            best = k
    if best > 0 and obj[best] <= f0 + wp.float64(_NOISE) * wp.abs(f0):
        return -best
    return 0


@wp.kernel
def _dof_kernel(
    q: wp.array[float], q_pred: wp.array[float], mass: wp.array[float], alpha: float, fixed: wp.array[wp.int32],
    gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], num_node_dofs: int,
    # outputs
    b: wp.array[wp.float64], obj: wp.array[wp.float64], slope: wp.array[wp.float64],
):
    """``b += M alpha (q - q_pred) - f_ext`` (``0`` on fixed DOFs); zero the line search's sums."""
    i = wp.tid()
    if i < obj.shape[0]:
        obj[i] = wp.float64(0.0)
    if i == 0:
        slope[0] = wp.float64(0.0)
    if fixed[i] != 0:
        b[i] = wp.float64(0.0)
        return
    b[i] = b[i] + wp.float64(mass[i] * alpha * (q[i] - q_pred[i]) - _ext(i, mass, gravity, particle_f, num_node_dofs))


@wp.kernel
def _trials_dof_kernel(
    q: wp.array[float], dq: wp.array[float], q_pred: wp.array[float], mass: wp.array[float], alpha: float,
    fixed: wp.array[wp.int32], gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3], num_node_dofs: int,
    # outputs
    qt: wp.array2d[float], b: wp.array[wp.float64], obj: wp.array[wp.float64], slope: wp.array[wp.float64],
):
    """``qt[k] = q - t_k dq``, the DOF part of each trial's objective, ``slope = b . dq``, then ``b = 0``."""
    k, i = wp.tid()
    d = dq[i]
    qk = q[i] - _trial_step(k) * d
    qt[k, i] = qk
    if fixed[i] != 0:
        return
    if k == 0:
        wp.atomic_add(slope, 0, b[i] * wp.float64(d))
        b[i] = wp.float64(0.0)
    r = qk - q_pred[i]
    v = 0.5 * mass[i] * alpha * r * r - _ext(i, mass, gravity, particle_f, num_node_dofs) * r
    wp.atomic_add(obj, k, wp.float64(v))


@wp.kernel
def _pick_kernel(
    obj: wp.array[wp.float64], slope: wp.array[wp.float64], trials: int, c1: float, qt: wp.array2d[float],
    # outputs
    q: wp.array[float], pick: wp.array[wp.int32],
):
    i = wp.tid()
    k = wp.abs(_pick(obj, slope, trials, c1))
    if i == 0:
        pick[0] = k
    q[i] = qt[k, i]


@wp.kernel
def _velocity_kernel(fixed: wp.array[wp.int32], q0: wp.array[float], q: wp.array[float], dt: float,
                     v: wp.array[float]):
    """Backward Euler's velocity; ``0`` on fixed DOFs."""
    i = wp.tid()
    if fixed[i] != 0:
        v[i] = 0.0
    else:
        v[i] = (q[i] - q0[i]) / dt


@wp.kernel
def _inertia_kernel(mass: wp.array[float], alpha: float, fixed: wp.array[wp.int32], indptr: wp.array[wp.int32],
                    vals: wp.array[wp.float64]):
    """``A``'s diagonal: ``+= M alpha``, ``1`` on fixed DOFs."""
    i = wp.tid()
    slot = indptr[i]
    if fixed[i] != 0:
        vals[slot] = wp.float64(1.0)
    else:
        vals[slot] = vals[slot] + wp.float64(mass[i] * alpha)

