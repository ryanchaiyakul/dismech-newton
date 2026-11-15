"""Implicit step solver, sparse linear solvers, PCG, convergence, and IFT adjoint."""

import importlib

from ..constants import TOLERANCE, ULP

__all__ = ["SolverDiSMech", "StepAdjoint", "TOLERANCE", "ULP"]


def __getattr__(name: str):
    if name == "SolverDiSMech":
        return importlib.import_module(".solver", __name__).SolverDiSMech
    if name == "StepAdjoint":
        return importlib.import_module(".adjoint", __name__).StepAdjoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
