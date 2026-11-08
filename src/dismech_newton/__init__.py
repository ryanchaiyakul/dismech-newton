"""Discrete elastic rods and midedge shells for Newton: one implicit solver, strain-space ADMM elements with barrier
contact, friction and an adjoint (:class:`SolverDiSMech`)."""

from .builder import add_cloth_mesh, add_rod, fix_segment, register_custom_attributes
from .solver import SolverDiSMech
from .state import flatten_state, suspended_tape

__all__ = [
    "SolverDiSMech",
    "add_cloth_mesh",
    "add_rod",
    "fix_segment",
    "flatten_state",
    "register_custom_attributes",
    "suspended_tape",
]

from .schur_exp import install as _schur_install  # EXPERIMENT (DM_SCHUR)

_schur_install()
