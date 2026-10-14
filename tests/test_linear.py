"""The direct solves against dense ones: symmetric (upper CSR) and general; SciPy on the CPU, cuDSS (float64 and
float32) on CUDA."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
from fd import assert_close

from dismech_newton.linear import CudssSolver, GeneralCSR, ScipySolver, SymmetricCSR, cudss


def _system(rng, device, n: int = 40):
    """A random SPD matrix on the pattern of overlapping 4-DOF stencils, as an upper CSR, and the full matrix."""
    stencils = np.array([np.arange(k, k + 4) for k in range(0, n - 3, 2)])
    A = SymmetricCSR(n, (stencils,), device)
    full = n * np.eye(n)
    for d in stencils:
        block = rng.normal(size=(4, 4))
        full[np.ix_(d, d)] += block @ block.T
    indptr, indices = A.pattern
    rows = np.repeat(np.arange(n), np.diff(indptr))
    A.vals.assign(full[rows, indices])
    return A, full


def _solvers(A, device):
    """Each solver and its tolerance (the float32 solution's at best)."""
    if not device.is_cuda or cudss is None:  # (SciPy on a CUDA matrix warns: only without nvmath)
        return {"scipy": (ScipySolver(A), 1.0e-6)}
    return {"cudss": (CudssSolver(A), 1.0e-6), "cudss float32": (CudssSolver(A, single=True), 1.0e-5)}


@pytest.mark.parametrize("general", [False, True], ids=["symmetric", "general"])
def test_solvers_match_dense(device, rng, general):
    A, full = _system(rng, device)
    if general:  # make it nonsymmetric
        full = full + np.triu(rng.normal(size=full.shape), 1) * (full != 0)
        A = GeneralCSR(sp.csr_matrix(full), device)
    b = rng.normal(size=full.shape[0])
    expected = np.linalg.solve(full, b)
    for name, (solver, rtol) in _solvers(A, device).items():
        x = wp.zeros(len(b), dtype=float, device=device)
        solver.solve(wp.array(b, dtype=wp.float64, device=device), x)
        assert_close(x.numpy(), expected, rtol, name)


def test_invalidate_refactorises(device, rng):
    """New values take effect only after ``invalidate``."""
    A, full = _system(rng, device)
    b = rng.normal(size=full.shape[0])
    for name, (solver, _) in _solvers(A, device).items():
        x = wp.zeros(len(b), dtype=float, device=device)
        solver.solve(wp.array(b, dtype=wp.float64, device=device), x)
        A.vals.assign(2.0 * A.vals.numpy())
        solver.invalidate()
        solver.solve(wp.array(b, dtype=wp.float64, device=device), x)
        A.vals.assign(0.5 * A.vals.numpy())
        np.testing.assert_allclose(x.numpy(), np.linalg.solve(2.0 * full, b), rtol=1.0e-4, err_msg=name)
