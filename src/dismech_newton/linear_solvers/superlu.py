import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warp as wp

from .base import LinearSolverBase


class SuperLUSolver(LinearSolverBase):
    def solve(self, b: wp.array, x: wp.array) -> None:
        A = self.A
        upper = sp.csr_matrix(
            (A.vals.numpy(), A.indices.numpy(), A.indptr.numpy()), shape=(A.n, A.n)
        )
        full = (upper + upper.T - sp.diags(upper.diagonal())).tocsc()
        x.numpy()[:] = spla.splu(full).solve(b.numpy().astype(np.float64))
