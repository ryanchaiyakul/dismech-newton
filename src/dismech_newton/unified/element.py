"""The generic element block of strain-space ADMM: an element type supplies a ``wp.struct`` view, ``strain``,
``jacobian`` and ``dofs``; :func:`element_kernels` builds its kernels and :class:`ElementBlock` runs them.

Per element ``W = 1/2 |eps - rest|^2_K + 1/2 |eps - eps_prev|^2_{C/dt}`` with ``K``, ``C`` diagonal (``params`` per
element are ``[K, C]``), penalty ``P = beta (K + C/dt)``, and the closed-form local step
``z = (K rest + (C/dt) eps_prev + P d) / (K + C/dt + P)`` with ``d = eps(q) + u``, ``u = d - z``.

An element type whose strains are quadratic in its first three nodes' positions (a shell's first form) supplies
``geometric(v, i, sigma) -> mat33``: the stress ``sigma = P (eps - z + u)`` the global step sees, as the matrix ``G``
with ``sum_r sigma_r d^2 eps_r = G (x) I3`` on those nodes (its tensile part: the stiffness Gauss-Newton drops).
"""

from functools import cache

import warp as wp

from .state import SymmetricCSR


class ElementBlock:
    """One element type's strain-ADMM block: penalty ``P``, local variables ``z``, duals ``u``, candidate strains."""

    def __init__(self, kernels, count: int, slots: wp.array, fixed: wp.array, num_candidates: int, beta: float, device):
        self.k, self.count, self.slots, self.fixed = kernels, count, slots, fixed
        self.beta, self.device = beta, device
        self.P, self.z, self.u = (wp.zeros(count, dtype=kernels.vec, device=device) for _ in range(3))
        self.eps_cand = wp.zeros((num_candidates, count), dtype=kernels.vec, device=device)

    def _launch(self, kernel, dim, inputs, outputs):
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=self.device)

    def reset(self, view, q: wp.array) -> None:
        """``z = eps(q)``, ``u = 0``."""
        self._launch(self.k.reset, self.count, [q, view], [self.z, self.u])

    def begin_step(self, view, params: wp.array, rest: wp.array, eps_prev: wp.array, dt: float, exact: bool) -> None:
        self.view, self.params, self.rest, self.eps_prev, self.dt, self.exact = view, params, rest, eps_prev, dt, exact
        self._launch(
            self.k.penalty,
            self.count,
            [params, rest, eps_prev, 1.0 if exact else self.beta, dt, int(exact)],
            [self.P, self.z, self.u],
        )

    def assemble(self, q: wp.array, A: SymmetricCSR) -> None:
        self._launch(self.k.assemble, self.count, [q, self.view, self.P, self.z, self.u, self.fixed],
                     [self.slots, A.vals])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self._launch(self.k.grad, self.count, [q, self.view, self.P, self.z, self.u, self.fixed], [b])

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(
            self.k.energy,
            (q_cand.shape[0], self.count),
            [q_cand, self.view, self.P, self.z, self.u],
            [self.eps_cand, energy],
        )

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self._launch(
            self.k.commit,
            self.count,
            [
                q,
                self.view,
                pick,
                self.eps_cand,
                self.params,
                self.rest,
                self.eps_prev,
                self.P,
                self.dt,
                int(self.exact),
                self.fixed,
            ],
            [self.z, self.u, b],
        )


@cache
def element_kernels(View, strain, jacobian, dofs, N: int, D: int, geometric=None):
    """The block kernels of an element (module docstring); ``params`` per element are ``[K, C]``."""
    vec = wp.types.vector(N, float)
    params_vec = wp.types.vector(2 * N, float)
    dofs_vec = wp.types.vector(D, wp.int32)

    @wp.func
    def grad_into(
        q: wp.array[float], v: View, i: int, P: vec, z: vec, u: vec, fixed: wp.array[wp.int32], b: wp.array[wp.float64]
    ):
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
    def measure(q: wp.array[float], v: View, out: wp.array[vec]):
        i = wp.tid()
        out[i] = strain(q, v, i)

    @wp.kernel(module="unique")
    def previous(stored: wp.array[vec], q: wp.array[float], v: View, out: wp.array[vec]):
        """The stored strains, measured at ``q`` where they are NaN (a state no step has written)."""
        i = wp.tid()
        s = stored[i]
        if wp.isnan(s[0]):
            s = strain(q, v, i)
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
        eps, J = jacobian(q, v, i)
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
                    wp.atomic_add(vals, slots[a * D - (a * (a - 1)) // 2 + (c - a), i], wp.float64(val))

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
        eps = strain(q_cand[k], v, i)
        eps_cand[k, i] = eps
        r = eps - z[i] + u[i]
        wp.atomic_add(energy, k, wp.float64(0.5 * wp.dot(wp.cw_mul(P[i], r), r)))

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
            p = params[i]
            Pi = P[i]
            d = eps_cand[pick[0], i] + ui
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

    return type(
        "ElementKernels",
        (),
        dict(vec=vec, reset=reset, measure=measure, previous=previous, penalty=penalty, assemble=assemble, grad=grad, energy=energy, commit=commit),
    )
