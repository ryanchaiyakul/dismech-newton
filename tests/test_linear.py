"""The linear solvers against SciPy, plain and in increment form."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp

from dismech_newton.linear import BlockInverseSolver, CudssSolver, ScipySolver, TridiagonalSolver, cudss, cusparse
from dismech_newton.sparse import GeneralCSR, SymmetricCSR

pytestmark = pytest.mark.filterwarnings("ignore:Solving a CUDA system with SciPy:RuntimeWarning")


def _array(x) -> wp.array:
    return wp.array(np.asarray(x, dtype=np.float32), dtype=float)


def _chains(rng, lengths=(1, 2, 3, 40, 117), singles=3, scale=2.0e4):
    """A shuffled block-diagonal of SPD tridiagonal chains (an M-matrix, like ADMM's H) and single DOFs."""
    blocks = []
    for n in lengths:
        off = -rng.uniform(0.5, 1.0, n - 1) * scale
        diag = rng.uniform(1.0e-3, 1.0, n) * scale + np.abs(np.concatenate([off, [0.0]])) + np.abs(
            np.concatenate([[0.0], off]))
        blocks.append(sp.diags([off, diag, off], [-1, 0, 1]))
    blocks += [sp.identity(1) * rng.uniform(0.5, 2.0) for _ in range(singles)]
    H = sp.block_diag(blocks, format="csr")
    p = rng.permutation(H.shape[0])
    return H[p][:, p].tocsr()


def _solvers(A, device):
    """Every solver available on ``device``, by name: ``make(increment)``."""
    out = {"block": lambda inc: BlockInverseSolver(A, increment=inc),
           "scipy": lambda inc: ScipySolver(A, increment=inc)}
    if device.is_cuda and cudss is not None:
        out["cudss"] = lambda inc: CudssSolver(A, increment=inc)
    if device.is_cuda and cusparse is not None:
        out["tridiagonal"] = lambda inc: TridiagonalSolver(A, increment=inc)
    return out


@pytest.mark.parametrize("increment", [False, True])
def test_solvers_match_scipy(device, rng, increment):
    H = _chains(rng)
    A = SymmetricCSR.from_scipy(H, device)
    n = H.shape[0]
    b = rng.normal(size=n) * 1.0e4  # large, like ADMM's M alpha q_pred
    x_ref = sla.spsolve(H.tocsc(), b)
    for name, make in _solvers(A, device).items():
        solver = make(increment)
        x = _array(x_ref + rng.normal(size=n) * 1.0e-2 if increment else np.zeros(n))  # a previous iterate
        dst, src = wp.zeros(n, dtype=float), _array(np.arange(n))
        solver.solve(_array(b), x, reset=(dst, src))
        err = np.abs(x.numpy() - x_ref).max() / np.abs(x_ref).max()
        assert err < 1.0e-5, f"{name}: relative error {err:.1e}"
        np.testing.assert_array_equal(dst.numpy(), src.numpy(), err_msg=f"{name}: reset")


def test_increment_rounds_with_the_correction(device, rng):
    """Near the solution, the increment form's error scales with the correction, not with ``b``."""
    H = _chains(rng, lengths=(400,), singles=0)
    A = SymmetricCSR.from_scipy(H, device)
    n = H.shape[0]
    x_ref = rng.normal(size=n)
    b = _array(H @ x_ref)
    x_near = x_ref + 1.0e-4 * rng.normal(size=n)
    for name, make in _solvers(A, device).items():
        errs = []
        for solver in (make(False), make(True)):
            x = _array(x_near)
            solver.solve(b, x)
            errs.append(np.abs(x.numpy() - x_ref).max())
        assert errs[1] <= errs[0] * 1.5 + 1.0e-6, f"{name}: increment {errs[1]:.1e} vs plain {errs[0]:.1e}"


@pytest.mark.usefixtures("needs_cudss")
def test_cudss_increment_general(device, rng):
    n = 30
    M = sp.random(n, n, density=0.2, random_state=1) + 10.0 * sp.identity(n)
    b = rng.normal(size=n)
    x = _array(rng.normal(size=n))
    CudssSolver(GeneralCSR(M, device), increment=True).solve(_array(b), x)
    np.testing.assert_allclose(x.numpy(), sla.spsolve(M.tocsc(), b), rtol=1e-4, atol=1e-5)


@pytest.mark.usefixtures("needs_cusparse")
def test_tridiagonal_fits(device, rng):
    assert TridiagonalSolver.fits(SymmetricCSR.from_scipy(_chains(rng), device))
    star = sp.identity(4, format="lil") * 4.0
    for j in (1, 2, 3):
        star[0, j] = star[j, 0] = -1.0
    cycle = sp.diags([-np.ones(3), 4 * np.ones(4), -np.ones(3)], [-1, 0, 1], format="lil")
    cycle[0, 3] = cycle[3, 0] = -1.0
    for H in (star, cycle):
        A = SymmetricCSR.from_scipy(H.tocsr(), device)
        assert not TridiagonalSolver.fits(A)
        with pytest.raises(ValueError):
            TridiagonalSolver(A)
