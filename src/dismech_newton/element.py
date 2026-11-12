"""The generic element block of projected strain-space Newton: an element type supplies a ``wp.struct`` view, ``strain``
and ``jacobian`` of its local DOFs ``x`` (a ``vecD``), and ``dofs``; :func:`element_kernels` builds its kernels and
:class:`ElementBlock` runs them.

Per element ``W = 1/2 |eps - rest|^2_K + 1/2 |eps - eps_prev|^2_{C/dt}`` with ``K``, ``C`` diagonal (``params`` per
element are ``[K, C]``), stress ``sigma = K (eps - rest) + (C/dt) (eps - eps_prev)``, and SPD strain curvature
``H = K + C/dt`` (or a custom ``material(eps, p, rest, eps_prev, inv_dt) -> (energy, sigma, H)``).

An element type whose strains are quadratic in its first three nodes' positions (a shell's first form) supplies
``geometric(v, i, sigma) -> mat33``: the stress ``sigma`` the global step sees, as the matrix ``G``
with ``sum_r sigma_r d^2 eps_r = G (x) I3`` on those nodes (its tensile part: the stiffness Gauss-Newton drops).
"""

from functools import cache
from types import SimpleNamespace

import warp as wp
from newton import State

from .linear import SymmetricCSR, triu_slot
from .state import Worlds, flatten_state, padded, world_sum


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
    """One element type's strain-space Newton block: the step's previous strains (the damping's reference) and
    effective parameters. The type ``elem`` supplies ``kernels`` (:func:`element_kernels`), ``count``,
    ``bind(dismech) -> view`` of a state, ``params`` ``[K, C]``, ``rest`` strains and ``stored``, the state attribute
    holding its strains."""

    def __init__(self, elem, *, slots, fixed, num_candidates: int = 0):
        self.k, self.count, self.bind, self.params = elem.kernels, elem.count, elem.bind, elem.params
        self.rest, self.stored, self.slots, self.fixed = elem.rest, elem.stored, slots, fixed
        self.effective = getattr(elem, "effective", None)  # (state, dt) -> the step's (params, rest), else theirs
        self.params_step, self.rest_step = self.params, self.rest
        self.device, vec = fixed.device, elem.kernels.vec
        self.eps_prev = wp.zeros(self.count, dtype=vec, device=self.device)
        self.inv_dt = 0.0
        self.worlds = None  # Worlds, the solver's
        self.anchor = None  # per DOF, the stiffness assembly drops coupling it to a fixed DOF (the solver's)

    def _launch(self, kernel, inputs, outputs, dim=None):
        wp.launch(kernel, dim=dim or self.count, inputs=inputs, outputs=outputs, device=self.device)

    def reset(self, state: State, mask: wp.array | None = None) -> None:
        """No-op (strain-space Newton carries no auxiliary ADMM state across steps)."""

    def begin_step(self, state_in: State, dt: float, exact: bool = False) -> None:
        """Bind ``state_in``, take its stored strains as the previous ones, update step parameters if needed."""
        s = state_in.dismech
        self.view, self.inv_dt = self.bind(s), 1.0 / dt
        self._launch(self.k.previous, [getattr(s, self.stored), s.q, self.view], [self.eps_prev])
        if self.effective is not None:
            self.params_step, self.rest_step = self.effective(s, dt)

    def store(self, s) -> None:
        """The strains of ``s`` (a state's ``dismech``, its frames current) into its stored strains."""
        self._launch(self.k.measure, [s.q, self.bind(s)], [getattr(s, self.stored)])

    def assemble(self, q: wp.array, A: SymmetricCSR) -> None:
        self._launch(self.k.assemble, [q, self.view, self.params_step, self.rest_step, self.eps_prev, self.inv_dt,
                                       self.fixed], [self.slots, A.vals, self.anchor])

    def grad(self, q: wp.array, b: wp.array) -> None:
        self._launch(self.k.grad, [q, self.view, self.params_step, self.rest_step, self.eps_prev, self.inv_dt,
                                   self.fixed], [b])

    def energy(self, q_cand: wp.array, energy: wp.array) -> None:
        self._launch(self.k.energy, [q_cand, self.view, self.params_step, self.rest_step, self.eps_prev, self.inv_dt,
                                     self.worlds], [energy], dim=(q_cand.shape[0], padded(self.count)))

    def commit(self, q: wp.array, pick: wp.array, b: wp.array) -> None:
        self._launch(self.k.commit, [q, self.view, self.worlds, self.params_step, self.rest_step, self.eps_prev,
                                     self.inv_dt, self.fixed], [b])


@cache
def element_kernels(View, strain, jacobian, dofs, N: int, D: int, geometric=None, material=None):
    """The block kernels of an element (module docstring); ``params`` per element are ``[K, C]``."""
    vec = wp.types.vector(N, float)
    params_vec = wp.types.vector(2 * N, float)
    dofs_vec = wp.types.vector(D, wp.int32)
    xvec = wp.types.vector(D, float)
    # Loops over the DOFs unrolled: the adjoint kernels differentiate them. -O2: nvrtc -O3 builds a shell's strain
    # adjoint that faults (CUDA 700) on sm_120 (Warp 1.17).
    unrolled = {"max_unroll": 32, "optimization_level": 2}

    if material is None:
        @wp.func
        def material(eps: vec, p: params_vec, rest: vec, eps_prev: vec, inv_dt: float):
            """Diagonal quadratic strain energy + Kelvin-Voigt damping: ``(energy, sigma, H_diag)``."""
            e = float(0.0)
            sigma = vec()
            H = vec()
            for j in range(N):
                k = p[j]
                c = p[N + j] * inv_dt
                w = k + c
                s = k * (eps[j] - rest[j]) + c * (eps[j] - eps_prev[j])
                if w > 0.0:
                    e = e + 0.5 * s * s / w
                sigma[j] = s
                H[j] = w
            return e, sigma, H

    @wp.func
    def local(q: wp.array[float], v: View, i: int) -> xvec:
        d = dofs(v, i)
        x = xvec()
        for k in range(D):
            x[k] = q[d[k]]
        return x

    @wp.func
    def grad_into(
        q: wp.array[float], v: View, i: int, p: params_vec, rest: vec, eps_prev: vec, inv_dt: float,
        fixed: wp.array[wp.int32], b: wp.array[wp.float64]
    ):
        """``b += J^T sigma`` on the free DOFs."""
        eps, J = jacobian(local(q, v, i), v, i)
        w_e, sigma, H = material(eps, p, rest, eps_prev, inv_dt)
        g = wp.transpose(J) * sigma
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
    def assemble(
        q: wp.array[float],
        v: View,
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        inv_dt: float,
        fixed: wp.array[wp.int32],
        slots: wp.array2d[wp.int32],
        vals: wp.array[wp.float64],
        anchor: wp.array[float],
    ):
        """``J^T H J`` (Gauss-Newton; plus ``geometric``'s ``G (x) I3``) into the upper CSR; ``slots``:
        :meth:`SymmetricCSR.slots` of the DOFs. ``anchor``: a free DOF's coupling to a fixed one, dropped from ``A``:
        the support's force there is resolved only to it times the free coordinate's float32 ulp."""
        i = wp.tid()
        eps, J = jacobian(local(q, v, i), v, i)
        d = dofs(v, i)
        w_e, sigma, H = material(eps, params[i], rest[i], eps_prev[i], inv_dt)
        G = wp.mat33()
        if wp.static(geometric is not None):
            G = geometric(v, i, sigma)
        free = dofs_vec()
        for a in range(D):
            free[a] = int(fixed[d[a]] == 0)
        for a in range(D):
            for c in range(a, D):
                val = float(0.0)
                for r in range(N):
                    val = val + J[r, a] * H[r] * J[r, c]
                if wp.static(geometric is not None):
                    if c < 9 and a % 3 == c % 3:
                        val = val + G[a // 3, c // 3]
                if val != 0.0 and free[a] != 0 and free[c] != 0:
                    wp.atomic_add(vals, slots[triu_slot(a, c, D), i], wp.float64(val))
                if free[a] != free[c]:
                    wp.atomic_add(anchor, d[wp.where(free[a] != 0, a, c)], wp.abs(val))

    @wp.kernel(module="unique")
    def grad(
        q: wp.array[float],
        v: View,
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        inv_dt: float,
        fixed: wp.array[wp.int32],
        b: wp.array[wp.float64],
    ):
        i = wp.tid()
        grad_into(q, v, i, params[i], rest[i], eps_prev[i], inv_dt, fixed, b)

    @wp.kernel(module="unique")
    def energy(
        q_cand: wp.array2d[float],
        v: View,
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        inv_dt: float,
        wd: Worlds,
        energy: wp.array[wp.float64],
    ):
        k, i = wp.tid()
        e = wp.float64(0.0)
        w = int(-1)
        if i < rest.shape[0]:
            w = wd.of[dofs(v, i)[0]]
            eps = strain(local(q_cand[k], v, i), v, i)
            w_e, sigma, H = material(eps, params[i], rest[i], eps_prev[i], inv_dt)
            e = wp.float64(w_e)
        world_sum(energy, w, k, wd.stride, e)

    @wp.kernel(module="unique")
    def commit(
        q: wp.array[float],
        v: View,
        wd: Worlds,
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        inv_dt: float,
        fixed: wp.array[wp.int32],
        b: wp.array[wp.float64],
    ):
        """The element gradient at the new ``q``; nothing in a world not iterating."""
        i = wp.tid()
        w = wd.of[dofs(v, i)[0]]
        if wd.active[w] == 0:
            return
        grad_into(q, v, i, params[i], rest[i], eps_prev[i], inv_dt, fixed, b)

    # -- the adjoint's: the step's residual per element, differentiated by Warp ----------------------------------

    @wp.kernel(module="unique", module_options=unrolled)
    def gather(q: wp.array[float], v: View, xs: wp.array[xvec]):
        i = wp.tid()
        xs[i] = local(q, v, i)

    @wp.func
    def force(x: xvec, v: View, i: int, p: params_vec, rest: vec, eps_prev: vec, dt: float) -> xvec:
        eps, J = jacobian(x, v, i)
        sigma = vec()
        for j in range(N):
            sigma[j] = p[j] * (eps[j] - rest[j]) + p[N + j] / dt * (eps[j] - eps_prev[j])
        return wp.transpose(J) * sigma

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
        g[i] = force(xs[i], v, i, params[i], rest[i], eps_prev[i], dt)

    @wp.kernel(module="unique", module_options=unrolled)
    def forces_rows(
        xs: wp.array2d[xvec],
        v: View,
        params: wp.array[params_vec],
        rest: wp.array[vec],
        eps_prev: wp.array[vec],
        dt: float,
        g: wp.array2d[xvec],
    ):
        """:attr:`forces` once per local DOF (``xs[i, row]``: element ``i``'s DOFs), so one reverse pass seeded
        with ``g[i, row]``'s ``row``-th component gives every row of the Hessians."""
        i, row = wp.tid()
        g[i, row] = force(xs[i, row], v, i, params[i], rest[i], eps_prev[i], dt)

    @wp.kernel(module="unique", module_options=unrolled)
    def scatter(g: wp.array[xvec], v: View, fixed: wp.array[wp.int32], r: wp.array[float]):
        i = wp.tid()
        d = dofs(v, i)
        gi = g[i]
        for k in range(D):
            if fixed[d[k]] == 0:
                wp.atomic_add(r, d[k], gi[k])

    @wp.kernel(module="unique")
    def hessian_rows(
        h: wp.array2d[xvec],
        v: View,
        fixed: wp.array[wp.int32],
        slots: wp.array2d[wp.int32],
        vals: wp.array[wp.float64],
    ):
        """The elements' Hessians (``h[i, row] = d g[row] / dx``, :attr:`forces_rows`) into the upper CSR
        (``slots`` as :meth:`assemble`'s)."""
        i, row = wp.tid()
        d = dofs(v, i)
        if fixed[d[row]] != 0:
            return
        hi = h[i, row]
        for c in range(row, D):
            if fixed[d[c]] == 0 and hi[c] != 0.0:
                wp.atomic_add(vals, slots[triu_slot(row, c, D), i], wp.float64(hi[c]))

    return SimpleNamespace(vec=vec, xvec=xvec, D=D, measure=measure, previous=previous, assemble=assemble, grad=grad,
                           energy=energy, commit=commit, gather=gather, forces=forces, scatter=scatter,
                           forces_rows=forces_rows, hessian_rows=hessian_rows)
