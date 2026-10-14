"""Discrete elastic rods and midedge shells for Newton: one implicit solver, strain-space ADMM with contact,
friction and an adjoint (:class:`DiSMechSolver`)."""

from .builder import add_cloth_mesh, add_rod, fix_segment, register_custom_attributes
from .solver import DiSMechSolver
from .state import flatten_state, suspended_tape

__all__ = [
    "DiSMechSolver",
    "add_cloth_mesh",
    "add_rod",
    "fix_segment",
    "flatten_state",
    "register_custom_attributes",
    "suspended_tape",
]
