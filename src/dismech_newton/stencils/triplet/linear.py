"""Concrete triplet stencils with linear energies.

:class:`LinearTriplet` stores one stiffness per strain (5 parameters).
:class:`LinearDampedTriplet` adds strain-rate viscosity and stores stiffness and damping side
by side (10 parameters), so a rod without damping carries no damping data and pays for no
start-of-step strain pass.

An energy is an inlined ``wp.func`` of the strain vector ``[eps_e, eps_f, kappa1, kappa2, tau]``
returning ``sigma = dE/d eps`` and ``C = d^2E/d eps^2``; the stencil kernels apply the chain
rule, so a different (e.g. learned) energy only has to provide this pair.
"""

import numpy as np
import warp as wp

from .stencil import TripletStencil
from .strains.helper import mat55f, param_vec, vec5f

vec10f = param_vec(10)


@wp.func
def linear_energy(eps: vec5f, eps_prev: vec5f, rest: vec5f, k: vec5f, dt: float):
    """``E = 1/2 sum k_i (eps_i - rest_i)^2``."""
    sigma = wp.cw_mul(k, eps - rest)
    C = mat55f()
    for i in range(5):
        C[i, i] = k[i]
    return sigma, C


@wp.func
def linear_damped_energy(eps: vec5f, eps_prev: vec5f, rest: vec5f, p: vec10f, dt: float):
    """:func:`linear_energy` plus strain-rate damping ``c_i d eps_i / dt``; ``p = [k, c]``."""
    rate = (eps - eps_prev) / dt
    sigma = vec5f()
    C = mat55f()
    for i in range(5):
        k = p[i]
        c = p[5 + i]
        sigma[i] = k * (eps[i] - rest[i]) + c * rate[i]
        C[i, i] = k + c / dt
    return sigma, C


class LinearTriplet(TripletStencil):
    """``E = 1/2 sum k_i (eps_i - rest_i)^2``; parameters ``k``."""

    energy = linear_energy

    @classmethod
    def pack(cls, k: np.ndarray, c: np.ndarray) -> np.ndarray:
        return k


class LinearDampedTriplet(TripletStencil):
    """:class:`LinearTriplet` plus strain-rate damping ``c_i d eps_i / dt``; parameters ``[k, c]``."""

    energy = linear_damped_energy
    uses_rate = True

    @classmethod
    def pack(cls, k: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.hstack((k, c))
