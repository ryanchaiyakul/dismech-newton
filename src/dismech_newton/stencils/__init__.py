"""Stencil designs: each owns its attributes, strains, assembly kernel and per-stencil state.

A stencil family (:class:`TripletStencil`) is abstract; concrete subclasses supply its energy
(:class:`LinearTriplet`, :class:`LinearDampedTriplet`, or your own) and are passed to
:class:`~dismech_newton.solver.DiSMechSolver`.
"""

from .base import NAMESPACE, STENCILS, Stencil
from .triplet import LinearDampedTriplet, LinearTriplet, TripletStencil

__all__ = ["NAMESPACE", "STENCILS", "LinearDampedTriplet", "LinearTriplet", "Stencil", "TripletStencil"]
