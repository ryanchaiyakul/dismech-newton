import warp as wp

from ..system import SymmetricCSR
from .base import LinearSolverBase
from .cudss import CudssSolver
from .superlu import SuperLUSolver


def get_linear_solver(A: SymmetricCSR, device: wp.DeviceLike) -> LinearSolverBase:
    device = wp.get_device(device)
    if device.is_cpu:
        return SuperLUSolver(A, device)
    return CudssSolver(A, device)


__all__ = ["get_linear_solver"]
