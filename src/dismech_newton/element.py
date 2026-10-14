"""The generic element block of strain-space ADMM: an element type supplies a ``wp.struct`` view, ``strain`` and
``jacobian`` of its local DOFs ``x`` (a ``vecD``), and ``dofs``; :func:`element_kernels` builds its kernels and
:class:`ElementBlock` runs them.

Per element ``W = 1/2 |eps - rest|^2_K + 1/2 |eps - eps_prev|^2_{C/dt}`` with ``K``, ``C`` diagonal (``params`` per
element are ``[K, C]``), penalty ``P = beta (K + C/dt)``, and the closed-form local step
``z = (K rest + (C/dt) eps_prev + P d) / (K + C/dt + P)`` with ``d = eps(q) + u``, ``u = d - z``.

An element type whose strains are quadratic in its first three nodes' positions (a shell's first form) supplies
``geometric(v, i, sigma) -> mat33``: the stress ``sigma = P (eps - z + u)`` the global step sees, as the matrix ``G``
with ``sum_r sigma_r d^2 eps_r = G (x) I3`` on those nodes (its tensile part: the stiffness Gauss-Newton drops).
"""

from functools import cache
from types import SimpleNamespace

import warp as wp
from newton import State

from .linear import SymmetricCSR, triu_slot
from .state import block_sum, flatten_state, padded


def rest_strains(elem, model, vec) -> wp.array:
    """``elem``'s strains (``vec`` each) in ``model``'s initial state."""
    out = wp.zeros(elem.count, dtype=vec, device=model.device)
    if elem.count:
        rest = model.state()
        flatten_state(rest)
        wp.launch(elem.kernels.measure, dim=elem.count, inputs=[rest.dismech.q, elem.bind(rest.dismech)],
                  outputs=[out], device=model.device)
    return out


class ElementBlock:
    """One element type's strain-ADMM block: penalty ``P``, local variables ``z``, duals ``u``, the step's previous
    strains (the damping's reference) and candidate strains. The type ``elem`` supplies ``kernels``
    (:func:`element_kernels`), ``count``, ``bind(dismech) -> view`` of a state, ``params`` ``[K, C]``, ``rest``
    strains and ``stored``, the state attribute holding its strains."""

    def __init__(self, elem, *, slots, fixed, num_candidates, beta):
        self.k, self.count, self.bind, self.params = elem.kernels, elem.count, elem.bind, elem.params
        self.rest, self.stored, self.slots, self.fixed = elem.rest, elem.stored, slots, fixed
        self.beta, self.device, vec = beta, fixed.device, elem.kernels.vec
        self.P, self.z, self.u, self.eps_prev = (wp.zeros(self.count, dtype=vec, device=self.device) for _ in range(4))
        self.eps_cand = wp.zeros((num_candidates, self.count), dtype=vec, device=self.device)

    def _launch(self, kernel, inputs, outputs, dim=None):
        wp.launch(kernel, dim=dim or self.count, inputs=inputs, outputs=outputs, device=self.device)

    def reset(self, state: State) -> None:
        """``z = eps(q)``, ``u = 0`` at ``state``."""
        self._launch(self.k.measure, [state.dismech.q, self.bind(state.dismech)], [self.z])
        self.u.zero_()

    def begin_step(self, state_in: State, dt: float, exact: bool) -> None:
        """Bind ``state_in``, take its stored strains as the previous ones, set the penalties."""
        s = state_in.dismech
        self.view, self.dt, self.exact = self.bind(s), dt, int(exact)
        self._launch(self.k.previous, [getattr(s, self.stored), s.q, self.view], [self.eps_prev])
        self._launch(self.k.penalty, [self.params, self.rest, self.eps_prev, 1.0 if exact else self.beta, dt,
                                      self.exact], [self.P, self.z, self.u])

    def store(self, s) -> None:
        """The strains of ``s`` (a state's ``dismech``, its frames current) into its stored strains."""
        self._launch(self.k.measure, [s.q, self.bind(s)], [getattr(s, self.stored)])

    def assemble(self, q: wp.array, A: SymmetricCSR) -> None:
        self._launch(self.k.assemble, [q, self.view, self.P, self.z, self.u, self.fixed], [self.slots, A.vals])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self._launch(self.k.grad, [q, self.view, self.P, self.z, self.u, self.fixed], [b])

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(self.k.energy, [q_cand, self.view, self.P, self.z, self.u], [self.eps_cand, energy],
                     dim=(q_cand.shape[0], padded(self.count)))

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self._launch(self.k.commit, [q, self.view, pick, self.eps_cand, self.params, self.rest, self.eps_prev, self.P,
                                     self.dt, self.exact, self.fixed], [self.z, self.u, b])


@cache
def element_kernels(View, strain, jacobian, dofs, N: int, D: int, geometric=None):
    """The block kernels of an element (module docstring); ``params`` per element are ``[K, C]``."""
    vec = wp.types.vector(N, float)
    params_vec = wp.types.vector(2 * N, float)
    dofs_vec = wp.types.vector(D, wp.int32)
    xvec = wp.types.vector(D, float)
    # Loops over the DOFs unrolled: the adjoint kernels differentiate them. -O2: nvrtc -O3 builds a shell's strain
    # adjoint that faults (CUDA 700) on sm_120 (Warp 1.17).
    unrolled = {"max_unroll": 32, "optimization_level": 2}

    @wp.func
    def local(q: wp.array[float], v: View, i: int) -> xvec:
        d = dofs(v, i)
        x = xvec()
        for k in range(D):
            x[k] = q[d[k]]
        return x

    @wp.func
    def prox(p: params_vec, rest: vec, eps_prev: vec, P: vec, d: vec, dt: float) -> vec:
        """``argmin_z W(z) + 1/2 |z - d|^2_P``; a component without stiffness or penalty is ``d``'s."""
        z = vec()
        for j in range(N):
            w = p[N + j] / dt
            den = p[j] + w + P[j]
            z[j] = wp.where(den > 0.0, (p[j] * rest[j] + w * eps_prev[j] + P[j] * d[j]) / wp.max(den, 1.0e-30), d[j])
        return z

    @wp.func
    def grad_into(
        q: wp.array[float], v: View, i: int, P: vec, z: vec, u: vec, fixed: wp.array[wp.int32], b: wp.array[wp.float64]
    ):
        """``b += J^T P (eps(q) - z + u)`` on the free DOFs."""
        eps, J = jacobian(local(q, v, i), v, i)
        g = wp.transpose(J) * wp.cw_mul(P, eps - z + u)
        d = dofs(v, i)
        for k in range(D):
            if fixed[d[k]] == 0:
                wp.atomic_add(b, d[k], wp.float64(g[k]))

    @wp.kernel(module="unique", module_options=unrolled)
    def measure(q: wp.array[float], v: View, out: wp.array[vec]):
        i = wp.tid()
        out[i] = strain(local(q, v, i), v, i)

    @wp.kernel(module="unique", module_options=unrolled)
    def previous(stored: wp.array[vec], q: wp.array[float], v: View, out: wp.array[vec]):
        """The stored strains, measured at ``q`` where they are NaN (a state no step has written)."""
        i = wp.tid()
        s = stored[i]
        if wp.isnan(s[0]):
            s = strain(local(q, v, i), v, i)
        out[i] = s

    @wp.kernel(module="unique")
    def penalty(
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        beta: float,
        dt: float,
        exact: int,
        P: wp.array[vec],
        z: wp.array[vec],
        u: wp.array[vec],
    ):
        """``P = beta (K + C/dt)``; ``exact``: ``z`` the energy's minimiser, ``u = 0``."""
        i = wp.tid()
        p = params[i]
        out = vec()
        for j in range(N):
            out[j] = beta * (p[j] + p[N + j] / dt)
        P[i] = out
        if exact != 0:
            z[i] = prox(p, rest[i], eps_prev[i], vec(), vec(), dt)
            u[i] = vec()

    @wp.kernel(module="unique")
    def assemble(
        q: wp.array[float],
        v: View,
        P: wp.array[vec],
        z: wp.array[vec],
        u: wp.array[vec],
        fixed: wp.array[wp.int32],
        slots: wp.array2d[wp.int32],
        vals: wp.array[wp.float64],
    ):
        """``J^T P J`` (Gauss-Newton; plus ``geometric``'s ``G (x) I3``) into the upper CSR; ``slots``:
        :meth:`SymmetricCSR.slots` of the DOFs."""
        i = wp.tid()
        eps, J = jacobian(local(q, v, i), v, i)
        d = dofs(v, i)
        p = P[i]
        G = wp.mat33()
        if wp.static(geometric is not None):
            G = geometric(v, i, wp.cw_mul(p, eps - z[i] + u[i]))
        free = dofs_vec()
        for a in range(D):
            free[a] = int(fixed[d[a]] == 0)
        for a in range(D):
            for c in range(a, D):
                val = float(0.0)
                for r in range(N):
                    val = val + J[r, a] * p[r] * J[r, c]
                if wp.static(geometric is not None):
                    if c < 9 and a % 3 == c % 3:
                        val = val + G[a // 3, c // 3]
                if val != 0.0 and free[a] != 0 and free[c] != 0:
                    wp.atomic_add(vals, slots[triu_slot(a, c, D), i], wp.float64(val))

    @wp.kernel(module="unique")
    def grad(
        q: wp.array[float],
        v: View,
        P: wp.array[vec],
        z: wp.array[vec],
        u: wp.array[vec],
        fixed: wp.array[wp.int32],
        b: wp.array[wp.float64],
    ):
        i = wp.tid()
        grad_into(q, v, i, P[i], z[i], u[i], fixed, b)

    @wp.kernel(module="unique")
    def energy(
        q_cand: wp.array2d[float],
        v: View,
        P: wp.array[vec],
        z: wp.array[vec],
        u: wp.array[vec],
        eps_cand: wp.array2d[vec],
        energy: wp.array[wp.float64],
    ):
        k, i = wp.tid()
        e = wp.float64(0.0)
        if i < z.shape[0]:
            eps = strain(local(q_cand[k], v, i), v, i)
            eps_cand[k, i] = eps
            r = eps - z[i] + u[i]
            e = wp.float64(0.5 * wp.dot(wp.cw_mul(P[i], r), r))
        block_sum(energy, k, e)

    @wp.kernel(module="unique")
    def commit(
        q: wp.array[float],
        v: View,
        pick: wp.array[wp.int32],
        eps_cand: wp.array2d[vec],
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        P: wp.array[vec],
        dt: float,
        exact: int,
        fixed: wp.array[wp.int32],
        z: wp.array[vec],
        u: wp.array[vec],
        b: wp.array[wp.float64],
    ):
        """The closed-form prox and dual step from the picked candidate's strains (not with ``exact``), then the
        gradient at the new ``q``."""
        i = wp.tid()
        zi = z[i]
        ui = u[i]
        if exact == 0:
            d = eps_cand[pick[0], i] + ui
            zi = prox(params[i], rest[i], eps_prev[i], P[i], d, dt)
            ui = d - zi
            z[i] = zi
            u[i] = ui
        grad_into(q, v, i, P[i], zi, ui, fixed, b)

    # -- the adjoint's: the step's residual per element, differentiated by Warp ----------------------------------

    @wp.kernel(module="unique", module_options=unrolled)
    def gather(q: wp.array[float], v: View, xs: wp.array[xvec]):
        i = wp.tid()
        xs[i] = local(q, v, i)

    @wp.kernel(module="unique", module_options=unrolled)
    def forces(
        xs: wp.array[xvec],
        v: View,
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        dt: float,
        g: wp.array[xvec],
    ):
        """``g = J^T (K (eps - rest) + C/dt (eps - eps_prev))`` in the element's DOFs ``xs``."""
        i = wp.tid()
        eps, J = jacobian(xs[i], v, i)
        p = params[i]
        sigma = vec()
        for j in range(N):
            sigma[j] = p[j] * (eps[j] - rest[i][j]) + p[N + j] / dt * (eps[j] - eps_prev[i][j])
        g[i] = wp.transpose(J) * sigma

    @wp.kernel(module="unique", module_options=unrolled)
    def scatter(g: wp.array[xvec], v: View, fixed: wp.array[wp.int32], r: wp.array[float]):
        i = wp.tid()
        d = dofs(v, i)
        gi = g[i]
        for k in range(D):
            if fixed[d[k]] == 0:
                wp.atomic_add(r, d[k], gi[k])

    @wp.kernel(module="unique")
    def hessian_row(
        row: int,
        h: wp.array[xvec],
        v: View,
        fixed: wp.array[wp.int32],
        slots: wp.array2d[wp.int32],
        vals: wp.array[wp.float64],
    ):
        """Row ``row`` of each element's Hessian (``h = d g[row] / dx``) into the upper CSR (``slots`` as
        :meth:`assemble`'s)."""
        i = wp.tid()
        d = dofs(v, i)
        if fixed[d[row]] != 0:
            return
        hi = h[i]
        for c in range(row, D):
            if fixed[d[c]] == 0 and hi[c] != 0.0:
                wp.atomic_add(vals, slots[triu_slot(row, c, D), i], wp.float64(hi[c]))

    return SimpleNamespace(vec=vec, xvec=xvec, D=D, measure=measure, previous=previous, penalty=penalty,
                           assemble=assemble, grad=grad, energy=energy, commit=commit, gather=gather, forces=forces,
                           scatter=scatter, hessian_row=hessian_row)
