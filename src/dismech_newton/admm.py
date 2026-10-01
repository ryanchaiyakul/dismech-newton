"""Discrete elastic rods with contact and friction, solved with ADMM (``admm.md``).

Per step the solver minimises ``1/2 |q - y|^2_{M alpha} + sum_t W_t(S_t q)`` subject to the contact
cones, with ``y`` the implicit-Euler prediction. Every triplet and every contact has a local
variable ``z = S q``, a scaled dual ``u`` and a prox; one iteration is

    global:   (M alpha + S^T P S) q = M alpha y + f_ext + sum S^T P (z - u)
    local:    z = prox(S q + u)          (every triplet and contact, in parallel)
    duals:    u += S q - z

``H = M alpha + S^T P S`` depends only on masses, ``dt``, the penalties and the topology, so it is
factorised once. Contacts do not enter it (:mod:`~dismech_newton.contact`). Fixed DOFs have identity
rows of ``H``; their coupling moves to the right-hand side, so ``H`` stays constant as they move.

A triplet's local variable is ``z = [x1 - x0, theta_e, x2 - x1, theta_f - theta_e]`` (8 numbers).
The energy is translation invariant, so ``W(z)`` is the triplet energy at ``x0 = 0``. Splitting the
twist difference puts the twist Laplacian into ``H``. The penalty is
``diag(rho_x I3, rho_abs, rho_x I3, rho_twist)``, so ``H`` does not couple the x, y and z components.
"""

from functools import cache

import numpy as np
import scipy.sparse as sp
import warp as wp
from newton import Contacts, Model, State

from .builder import add_colliding_rod
from .contact import ContactTerm
from .frames import fixed_node, flatten_state, node, scatter_dof, scatter_node
from .linear import BlockInverseSolver, CudssSolver, SymmetricCSR
from .solver import DiSMechSolver, external_force
from .strains import (
    TripletGeometry,
    edge_direction,
    mat55f,
    rest_strain,
    strain_derivatives,
    triplet_geometry,
    unpack_conn,
    vec5f,
    vec5i,
    vec10f,
)
from .triplet import Triplets, linear_energy

vec8f = wp.types.vector(8, float)
mat88f = wp.types.matrix((8, 8), float)
mat11_8f = wp.types.matrix((11, 8), float)

# -- triplet local step -------------------------------------------------------------------


@wp.func
def _reduction() -> mat11_8f:
    """``d q_triplet / d z``: ``x0 = 0, theta_e = z3, x1 = e, theta_f = z3 + z7, x2 = e + f``."""
    T = mat11_8f()
    for k in range(3):
        T[4 + k, k] = 1.0
        T[8 + k, k] = 1.0
        T[8 + k, 4 + k] = 1.0
    T[3, 3] = 1.0
    T[7, 3] = 1.0
    T[7, 7] = 1.0
    return T


@wp.func
def _local_geometry(
    z: vec8f, d1e_old: wp.vec3, te_old: wp.vec3, d1f_old: wp.vec3, tf_old: wp.vec3, ref_twist_old: float,
    l0e: float, l0f: float,
):
    e = wp.vec3(z[0], z[1], z[2])
    f = wp.vec3(z[4], z[5], z[6])
    return triplet_geometry(
        wp.vec3(0.0, 0.0, 0.0), e, e + f, z[3], z[3] + z[7], d1e_old, te_old, d1f_old, tf_old, ref_twist_old, l0e, l0f
    )


@wp.func
def _local_derivatives(geom: TripletGeometry, sigma: vec5f, C: mat55f, l0e: float, l0f: float):
    """Gradient, Hessian and Gauss-Newton Hessian of ``W`` with respect to ``z``."""
    J, K_geo = strain_derivatives(geom, sigma, l0e, l0f)
    T = _reduction()
    JT = J * T  # (5, 8)
    K_gn = wp.transpose(JT) * C * JT
    K = K_gn + wp.transpose(T) * K_geo * T
    g = wp.transpose(JT) * sigma
    return g, K, K_gn


@wp.func
def _cholesky_solve(A: mat88f, b: vec8f):
    """``A^{-1} b`` and whether ``A`` was positive definite."""
    L = mat88f()
    ok = int(1)
    for j in range(8):
        s = A[j, j]
        for k in range(j):
            s = s - L[j, k] * L[j, k]
        if s <= 1.0e-12 * wp.abs(A[j, j]) or s <= 0.0:
            ok = 0
            s = wp.max(wp.abs(A[j, j]), 1.0e-12)
        L[j, j] = wp.sqrt(s)
        for i in range(j + 1, 8):
            v = A[i, j]
            for k in range(j):
                v = v - L[i, k] * L[j, k]
            L[i, j] = v / L[j, j]
    y = vec8f()
    for i in range(8):
        v = b[i]
        for k in range(i):
            v = v - L[i, k] * y[k]
        y[i] = v / L[i, i]
    x = vec8f()
    for ii in range(8):
        i = 7 - ii
        v = y[i]
        for k in range(i + 1, 8):
            v = v - L[k, i] * x[k]
        x[i] = v / L[i, i]
    return x, ok


@wp.func
def _penalty(rho: wp.vec3) -> vec8f:
    """``(rho_x, rho_abs, rho_twist)`` spread over ``[e, theta_e, f, dtheta]``."""
    return vec8f(rho[0], rho[0], rho[0], rho[1], rho[0], rho[0], rho[0], rho[2])


@cache
def make_admm_kernels(energy):
    """Local-step and rest-stiffness kernels with ``energy`` compiled in."""

    @wp.kernel(module="unique", module_options={"fast_math": True})
    def local_kernel(
        q: wp.array[float],
        node_q_old: wp.array[wp.vec3],
        edge_d1_old: wp.array[wp.vec3],
        triplet_ref_twist_old: wp.array[float],
        triplet_conn: wp.array[vec5i],
        edge_length: wp.array[float],
        triplet_params: wp.array[vec10f],
        triplet_rest: wp.array[wp.vec3],
        strain_prev: wp.array[vec5f],
        dt: float,
        theta_dof_offset: int,
        rho: wp.array[wp.vec3],
        dof_fixed: wp.array[wp.int32],
        local_iterations: int,
        update: int,
        # outputs
        z: wp.array[vec8f],
        u: wp.array[vec8f],
        rhs: wp.array[float],
        stats: wp.array[float],
    ):
        """With ``update``: the prox of ``W`` (Newton from the previous ``z``) and the dual step.
        Always: ``rhs += S_f^T P (z - u - S q_fixed)`` for the next global step."""
        t = wp.tid()
        e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
        ie = theta_dof_offset + e
        i_f = theta_dof_offset + f
        P = _penalty(rho[t])
        zt = z[t]
        ut = u[t]

        if update != 0:
            x0 = node(q, n0)
            x1 = node(q, n1)
            x2 = node(q, n2)
            ee = x1 - x0
            ef = x2 - x1
            Sq = vec8f(ee[0], ee[1], ee[2], q[ie], ef[0], ef[1], ef[2], q[i_f] - q[ie])
            d = Sq + ut

            d1e = edge_d1_old[e]
            d1f = edge_d1_old[f]
            te_old = edge_direction(node_q_old, n0, n1)
            tf_old = edge_direction(node_q_old, n1, n2)
            l0e = edge_length[e]
            l0f = edge_length[f]
            rest = rest_strain(triplet_rest[t])
            z_old = zt
            for _it in range(local_iterations):
                geom = _local_geometry(zt, d1e, te_old, d1f, tf_old, triplet_ref_twist_old[t], l0e, l0f)
                sigma, C = energy(geom.strain, strain_prev[t], rest, triplet_params[t], dt)
                g, K, K_gn = _local_derivatives(geom, sigma, C, l0e, l0f)
                G = g + wp.cw_mul(P, zt - d)
                step, ok = _cholesky_solve(K + wp.diag(P), G)
                if ok == 0:  # indefinite: fall back to Gauss-Newton, positive definite
                    step, ok = _cholesky_solve(K_gn + wp.diag(P), G)
                # Trust region: far from the solution (fast motion) a full step can collapse an edge.
                s_e = wp.length(wp.vec3(step[0], step[1], step[2])) / (0.25 * l0e)
                s_f = wp.length(wp.vec3(step[4], step[5], step[6])) / (0.25 * l0f)
                s_t = wp.max(wp.abs(step[3]), wp.abs(step[7])) / 0.5
                zt = zt - step / wp.max(1.0, wp.max(s_t, wp.max(s_e, s_f)))
            ut = d - zt
            z[t] = zt
            u[t] = ut

            # Primal residual S q - z and the change of z, relative to the rest lengths for edges.
            res = Sq - zt
            m = wp.max(wp.abs(res[3]), wp.abs(res[7]))
            m = wp.max(m, wp.length(wp.vec3(res[0], res[1], res[2])) / l0e)
            m = wp.max(m, wp.length(wp.vec3(res[4], res[5], res[6])) / l0f)
            dz = zt - z_old
            m = wp.max(m, wp.max(wp.abs(dz[3]), wp.abs(dz[7])))
            m = wp.max(m, wp.length(wp.vec3(dz[0], dz[1], dz[2])) / l0e)
            m = wp.max(m, wp.length(wp.vec3(dz[4], dz[5], dz[6])) / l0f)
            wp.atomic_max(stats, 0, m)

        # rhs += S_f^T P (z - u - S q_c): the fixed DOFs' coupling moves to the right-hand side.
        c0 = fixed_node(q, dof_fixed, n0)
        c1 = fixed_node(q, dof_fixed, n1)
        c2 = fixed_node(q, dof_fixed, n2)
        ce = c1 - c0
        cf = c2 - c1
        th_e = float(0.0)
        th_f = float(0.0)
        if dof_fixed[ie] != 0:
            th_e = q[ie]
        if dof_fixed[i_f] != 0:
            th_f = q[i_f]
        w = wp.cw_mul(P, zt - ut - vec8f(ce[0], ce[1], ce[2], th_e, cf[0], cf[1], cf[2], th_f - th_e))
        we = wp.vec3(w[0], w[1], w[2])
        wf = wp.vec3(w[4], w[5], w[6])
        scatter_node(rhs, dof_fixed, n0, -we)
        scatter_node(rhs, dof_fixed, n1, we - wf)
        scatter_node(rhs, dof_fixed, n2, wf)
        scatter_dof(rhs, dof_fixed, ie, w[3] - w[7])
        scatter_dof(rhs, dof_fixed, i_f, w[7])

    @wp.kernel(module="unique")
    def rest_stiffness_kernel(
        q: wp.array[float],
        node_q: wp.array[wp.vec3],
        edge_d1: wp.array[wp.vec3],
        triplet_ref_twist: wp.array[float],
        triplet_conn: wp.array[vec5i],
        edge_length: wp.array[float],
        triplet_params: wp.array[vec10f],
        triplet_rest: wp.array[wp.vec3],
        theta_dof_offset: int,
        # outputs
        stiffness: wp.array[wp.vec3],
    ):
        """Per triplet ``(axial, transverse, twist)`` stiffness of ``W`` in ``z`` at rest."""
        t = wp.tid()
        e, f, n0, n1, n2 = unpack_conn(triplet_conn[t])
        ee = node_q[n1] - node_q[n0]
        ef = node_q[n2] - node_q[n1]
        th_e = q[theta_dof_offset + e]
        z0 = vec8f(ee[0], ee[1], ee[2], th_e, ef[0], ef[1], ef[2], q[theta_dof_offset + f] - th_e)
        te = wp.normalize(ee)
        tf = wp.normalize(ef)
        l0e = edge_length[e]
        l0f = edge_length[f]
        geom = _local_geometry(z0, edge_d1[e], te, edge_d1[f], tf, triplet_ref_twist[t], l0e, l0f)
        sigma, C = energy(geom.strain, geom.strain, rest_strain(triplet_rest[t]), triplet_params[t], 1.0)
        g, K, K_gn = _local_derivatives(geom, sigma, C, l0e, l0f)
        Kee = wp.mat33()
        Kff = wp.mat33()
        for i in range(3):
            for j in range(3):
                Kee[i, j] = K_gn[i, j]
                Kff[i, j] = K_gn[4 + i, 4 + j]
        ae = wp.dot(te, Kee * te)
        af = wp.dot(tf, Kff * tf)
        be = 0.5 * (wp.trace(Kee) - ae)
        bf = 0.5 * (wp.trace(Kff) - af)
        stiffness[t] = wp.vec3(0.5 * (ae + af), 0.5 * (be + bf), K_gn[7, 7])

    return local_kernel, rest_stiffness_kernel


class TripletTerm:
    """Local variables, duals and prox of every triplet.

    Args:
        triplets: The bound triplets; read in place.
        rho_scale: ``rho_x`` is ``rho_scale`` times the geometric mean of the triplet's axial and
            transverse stiffness in edge-vector space at rest, ``rho_twist`` ``rho_scale`` times its
            twist stiffness.
        rho_abs_ratio: ``rho_abs / rho_twist``, the weight of the absolute twist angle.
        local_iterations: Newton iterations of every prox (warm-started).
    """

    def __init__(self, triplets: Triplets, *, rho_scale: float, rho_abs_ratio: float, local_iterations: int):
        self.triplets = triplets
        self.local_iterations = local_iterations
        self.device = triplets.device
        self._local_kernel, stiffness_kernel = make_admm_kernels(triplets.energy)
        tr, der = triplets, triplets.der

        rest_state = tr.model.state()
        flatten_state(rest_state)
        stiffness = wp.zeros(tr.count, dtype=wp.vec3, device=self.device)
        wp.launch(
            stiffness_kernel,
            dim=tr.count,
            inputs=[
                rest_state.dismech.q, rest_state.particle_q, rest_state.dismech.edge_d1_q,
                rest_state.dismech.triplet_ref_twist_q, tr.conn, der.edge_length, tr.params, tr.rest,
                tr.num_node_dofs,
            ],
            outputs=[stiffness],
            device=self.device,
        )
        axial, transverse, twist = stiffness.numpy().astype(np.float64).T
        rho_x = np.where(transverse > 0.0, np.sqrt(axial * np.maximum(transverse, 0.0)), axial)
        rho_twist = np.where(twist > 0.0, twist, 1.0e-3 * rho_x * np.mean(der.edge_length.numpy()) ** 2)
        # The absolute angle only enters through the bend/twist coupling, so it gets a small weight.
        rho = rho_scale * np.column_stack((rho_x, rho_abs_ratio * rho_twist, rho_twist))
        self.rho = wp.array(rho.astype(np.float32), dtype=wp.vec3, device=self.device)

        self.z = wp.zeros(tr.count, dtype=vec8f, device=self.device)
        self.u = wp.zeros(tr.count, dtype=vec8f, device=self.device)
        self._initialized = False

    def penalty(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """COO ``(rows, cols, vals)`` of ``S^T P S``, both triangles."""
        dofs = self.triplets.dofs()  # [x0, theta_e, x1, theta_f, x2]
        r = self.rho.numpy().astype(np.float64)
        rows, cols, vals = [], [], []
        for k in range(3):
            for a, b in ((0, 4), (4, 8)):  # edge e = x1 - x0, edge f = x2 - x1
                i, j = dofs[:, a + k], dofs[:, b + k]
                rows += [i, j, i, j]
                cols += [i, j, j, i]
                vals += [r[:, 0], r[:, 0], -r[:, 0], -r[:, 0]]
        i, j = dofs[:, 3], dofs[:, 7]  # theta_e (weight rho_abs) and theta_f - theta_e (rho_twist)
        rows += [i, i, j, i, j]
        cols += [i, i, j, j, i]
        vals += [r[:, 1], r[:, 2], r[:, 2], -r[:, 2], -r[:, 2]]
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def begin_step(self, state_in: State) -> None:
        """On the first step: ``z = S q``, ``u = 0`` (the rod starts at rest); later steps warm-start."""
        if self._initialized:
            return
        tr = self.triplets
        e, f, n0, n1, n2 = tr.conn.numpy().T
        q = state_in.dismech.q.numpy()
        x = q[: tr.num_node_dofs].reshape(-1, 3)
        th = q[tr.num_node_dofs :]
        ee, ef = x[n1] - x[n0], x[n2] - x[n1]
        self.z.assign(np.column_stack((ee, th[e], ef, th[f] - th[e])).astype(np.float32))
        self._initialized = True

    def local(self, state_in: State, q: wp.array, dt: float, rhs: wp.array, stats: wp.array, update: int) -> None:
        """With ``update``: prox and dual step at ``q``, residual into ``stats[0]``. Always: add ``S^T P (z - u)`` to ``rhs``."""
        tr, s_in = self.triplets, state_in.dismech
        wp.launch(
            self._local_kernel,
            dim=tr.count,
            inputs=[
                q, state_in.particle_q, s_in.edge_d1_q, s_in.triplet_ref_twist_q, tr.conn, tr.der.edge_length,
                tr.params, tr.rest, tr.strain_prev, dt, tr.num_node_dofs, self.rho, tr.dof_fixed,
                self.local_iterations, update,
            ],
            outputs=[self.z, self.u, rhs, stats],
            device=self.device,
        )


# -- solver -------------------------------------------------------------------------------


@wp.kernel
def _inertia_rhs_kernel(
    q_pred: wp.array[float],
    q_in: wp.array[float],
    mass: wp.array[float],
    alpha: float,
    dof_fixed: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    num_node_dofs: int,
    # outputs
    b: wp.array[float],
):
    """``b = M alpha q_pred + f_ext`` on free DOFs, the prescribed value on fixed ones (identity rows)."""
    i = wp.tid()
    if dof_fixed[i] != 0:
        b[i] = q_in[i]
        return
    v = mass[i] * alpha * q_pred[i]
    if i < num_node_dofs:
        v = v + external_force(i, mass, gravity, particle_f)
    b[i] = v


@wp.kernel
def _step_change_kernel(q: wp.array[float], q_prev: wp.array[float], scale: float, stats: wp.array[float]):
    i = wp.tid()
    wp.atomic_max(stats, 1, wp.abs(q[i] - q_prev[i]) / scale)


class ADMMDiSMechSolver(DiSMechSolver):
    """Implicit discrete elastic rods with contact and friction, solved with ADMM.

    Build rods with :meth:`add_rod` (:func:`~dismech_newton.builder.add_colliding_rod`). Any other
    collision shape in the model (ground plane, static obstacles) is a fixed obstacle. Contacts are
    the caller's: run a :class:`~newton.CollisionPipeline` on ``state_in`` and pass its
    :class:`~newton.Contacts` to :meth:`step`. The solver poses the proxies and sets their
    velocities at the end of every step, so the next ``collide`` (with speculative contacts) sees
    the current rod. Create the pipeline with ``soft_contact_max=0``: the rod nodes are particles,
    and Newton would otherwise generate particle contacts against every shape, which are unused here.
    ``contact_matching="latest"`` warm-starts the contact forces across steps. With ``theta < 1``
    the contacts act at ``q_theta``, not at the end of the step.

    Args:
        model: Model built with :meth:`add_rod`; must be on a CUDA device.
        energy: Triplet energy (see :mod:`dismech_newton.triplet`).
        iterations: Maximum ADMM iterations per step.
        tol: Stop when the local residuals and the change of ``q`` (relative to the mean edge
            length) are below ``tol``; ``0`` always runs ``iterations``. The test runs on the
            device (``wp.capture_while``), so a step can be graph-captured either way.
        check_every: Test convergence every this many iterations (``iterations`` rounds up to a multiple).
        rho_scale, rho_abs_ratio, local_iterations: See :class:`TripletTerm`.
        self_contact: Keep rod-rod contacts (rod-obstacle contacts are always kept).
        friction: Coulomb coefficient ``mu``.
        contact_rho_scale: Contact penalties relative to the largest that provably converges, set per
            contact from its coupling to the others through ``H`` (see :class:`ContactTerm`).
        pose_proxies: See :class:`~dismech_newton.DiSMechSolver`; collision detection needs them.
        linear_solver: ``"dense"`` (one dense inverse per block of ``H``), ``"cudss"``, or
            ``"auto"``: dense while the blocks are small.
        theta: See :class:`~dismech_newton.DiSMechSolver`. With ``theta = 1/2`` nothing damps the
            stiff modes, so the residual ``tol`` leaves must be small: ``1e-5`` holds an undamped
            cantilever, ``1e-4`` lets it blow up within seconds.
    """

    add_rod = staticmethod(add_colliding_rod)

    def __init__(
        self,
        model: Model,
        *,
        energy=linear_energy,
        iterations: int = 200,
        tol: float = 1.0e-4,
        check_every: int = 10,
        rho_scale: float = 0.3,
        rho_abs_ratio: float = 0.1,
        local_iterations: int = 2,
        self_contact: bool = True,
        friction: float = 0.3,
        contact_rho_scale: float = 1.8,
        pose_proxies: bool = True,
        linear_solver: str = "auto",
        theta: float = 1.0,
    ):
        if linear_solver not in ("auto", "dense", "cudss"):
            raise ValueError(f"linear_solver must be 'auto', 'dense' or 'cudss', got {linear_solver!r}")
        # Read by _build_system, which the base constructor calls.
        self.linear_solver = linear_solver
        self.iterations = iterations
        self.tol = tol
        self.check_every = max(1, check_every)
        self._term_options = dict(rho_scale=rho_scale, rho_abs_ratio=rho_abs_ratio, local_iterations=local_iterations)
        self._contact_options = dict(friction=friction, self_contact=self_contact, rho_scale=contact_rho_scale)
        super().__init__(model, energy=energy, pose_proxies=pose_proxies, theta=theta)

    def _build_system(self) -> None:
        dev = self.device
        self.elastic = TripletTerm(self.triplets, **self._term_options)
        self.contact = ContactTerm(self.model, self.fixed, **self._contact_options)
        self._length_scale = float(np.mean(self.der.edge_length.numpy()))
        self._b = wp.zeros(self.num_dofs, dtype=float, device=dev)
        self._rhs = [wp.zeros(self.num_dofs, dtype=float, device=dev) for _ in range(2)]  # read / being built
        self._q_prev = wp.zeros(self.num_dofs, dtype=float, device=dev)
        self._stats = wp.zeros(2, dtype=float, device=dev)  # [local residuals, change of q]
        self._linear = None
        self._factored_alpha = None

    def contact_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """``(pairs, force)`` of the active contacts of the last step (see :meth:`ContactTerm.forces`)."""
        return self.contact.forces()

    def _factorize(self, alpha: float) -> None:
        """``H = M alpha + S^T P S`` on the free DOFs, identity rows on the fixed ones."""
        n = self.num_dofs
        fixed = self.fixed.numpy() != 0
        mass = self.mass.numpy().astype(np.float64)
        r, c, v = self.elastic.penalty()
        rows, cols = np.concatenate([np.arange(n), r]), np.concatenate([np.arange(n), c])
        vals = np.concatenate([np.where(fixed, 1.0, mass * alpha), v])
        keep = (~fixed[rows] & ~fixed[cols]) | (np.arange(len(rows)) < n)
        H = SymmetricCSR.from_scipy(sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n)), self.device)
        dense = self.linear_solver == "dense" or (self.linear_solver == "auto" and BlockInverseSolver.fits(H))
        self._linear = BlockInverseSolver(H) if dense else CudssSolver(H, refactorize=False)
        self._factored_alpha = alpha

    def _solve(self, state_in: State, state_out: State, contacts: Contacts | None, dt: float) -> None:
        """ADMM iterations on ``state_out.dismech.q``, with a convergence test every ``check_every``."""
        q = state_out.dismech.q
        if self._factored_alpha != self.alpha:
            self._factorize(self.alpha)
        wp.launch(
            _inertia_rhs_kernel,
            dim=self.num_dofs,
            inputs=[self.q_pred, state_in.dismech.q, self.mass, self.alpha, self.fixed, self.model.gravity,
                    self._particle_f(state_in), self.num_node_dofs],
            outputs=[self._b],
            device=self.device,
        )
        self.elastic.begin_step(state_in)
        self.contact.begin_step(state_in, contacts, self._linear.solve)

        # The first right-hand side comes from the warm-started z and u.
        wp.copy(self._rhs[0], self._b)
        self._local(state_in, q, dt, self._rhs[0], update=0)
        self._iterate(self._admm_block, self.iterations, self.check_every, self.tol, self._stats,
                      state_in=state_in, q=q, dt=dt)
        self.contact.end_step()

    def _admm_block(self, state_in: State, q: wp.array, dt: float) -> None:
        """``check_every`` ADMM iterations; with ``tol``, the residuals of the last one in ``self._stats``."""
        check = self.tol > 0.0
        n = self.check_every
        for i in range(n):
            if check and i == n - 1:
                wp.copy(self._q_prev, q)
                self._stats.zero_()
            rhs, nxt = self._rhs[i % 2], self._rhs[(i + 1) % 2]
            self._linear.solve(rhs, q, reset=(nxt, self._b))  # the next right-hand side starts from b
            self._local(state_in, q, dt, nxt, update=1)
        if n % 2:  # every block starts from self._rhs[0]
            wp.copy(self._rhs[0], self._rhs[1])
        if check:
            wp.launch(_step_change_kernel, dim=self.num_node_dofs, inputs=[q, self._q_prev, self._length_scale,
                      self._stats], device=self.device)

    def _local(self, state_in: State, q: wp.array, dt: float, rhs: wp.array, update: int) -> None:
        self.elastic.local(state_in, q, dt, rhs, self._stats, update)
        self.contact.local(q, rhs, self._stats, update)
