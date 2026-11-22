"""Discrete elastic rods and midedge shells for Newton: one implicit solver, projected strain-space Gauss-Newton elements with barrier
contact, friction and an adjoint (:class:`SolverDiSMech`)."""

from .builder import add_cloth_mesh, add_rod, fix_segment, register_custom_attributes
from .dofs import flatten_state, suspended_tape
from .elements import Dahl
from .solver import SolverDiSMech

__all__ = [
    "Dahl",
    "SolverDiSMech",
    "add_cloth_mesh",
    "add_rod",
    "fix_segment",
    "flatten_state",
    "register_custom_attributes",
    "suspended_tape",
]
