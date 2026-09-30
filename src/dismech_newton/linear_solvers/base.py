"""Interface shared by all linear solvers for the Newton system."""

from abc import ABC, abstractmethod

import warp as wp

from ..system import SymmetricCSR


class LinearSolverBase(ABC):
    """Solve ``A x = b`` for a symmetric CSR matrix with a fixed sparsity pattern."""

    def __init__(self, A: SymmetricCSR, device: wp.DeviceLike) -> None:
        self.device = wp.get_device(device)
        self.A = A

    @abstractmethod
    def solve(self, b: wp.array, x: wp.array) -> None:
        """Write the solution of ``A x = b`` into ``x``"""
