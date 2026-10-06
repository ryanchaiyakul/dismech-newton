"""Strain derivatives against finite differences, for current frames and frames lagged behind the tangents."""

import numpy as np
import pytest
import warp as wp
from fd import assert_close, hessian, jacobian
from triplets import NODE_DOFS, THETA_DOFS, Z_EDGE_DOFS, Z_THETA_DOFS, to_z, unpack_sym8

from dismech_newton.strains import packed_index

STRAINS = ["eps_e", "eps_f", "kappa1", "kappa2", "tau"]


def _derivatives(strains, cfg):
    """``(J, J_fd, H, H_fd)``, dimensionless by the DOF scale."""
    out = strains(cfg.q, cfg)
    f = lambda Q: strains(Q, cfg).strain  # noqa: E731
    D = cfg.dof_scale
    J_fd = jacobian(f, cfg.q, cfg.steps(1.0e-2))
    H_fd = hessian(f, cfg.q, cfg.steps(3.0e-2))
    DD = np.outer(D, D)
    return out.J[0] * D, J_fd * D, out.H[0] * DD, H_fd * DD


LAGS = pytest.mark.parametrize("lag", [0.0, 0.1, 0.3], ids=["current", "lagged", "far"])


@LAGS
def test_jacobian_matches_finite_differences(strains, triplet, lag):
    J, J_fd, _, _ = _derivatives(strains, triplet.lagged(lag))
    for i, name in enumerate(STRAINS):
        assert_close(J[i], J_fd[i], 1.0e-3, name)


@LAGS
def test_hessians_match_finite_differences(strains, triplet, lag):
    _, _, H, H_fd = _derivatives(strains, triplet.lagged(lag))
    blocks = {"xx": (NODE_DOFS, NODE_DOFS), "x-theta": (NODE_DOFS, THETA_DOFS), "theta-theta": (THETA_DOFS, THETA_DOFS)}
    for i, name in enumerate(STRAINS):
        scale = np.abs(H_fd[i]).max()
        for block, (r, c) in blocks.items():
            assert_close(H[i][np.ix_(r, c)], H_fd[i][np.ix_(r, c)], 1.0e-2, f"{name} {block}", scale=scale)


@LAGS
def test_hessians_symmetric(strains, triplet, lag):
    cfg = triplet.lagged(lag)
    H = strains(cfg.q, cfg).H[0]
    np.testing.assert_allclose(H, np.swapaxes(H, 1, 2), rtol=0.0, atol=1.0e-6 * np.abs(H).max())


@pytest.mark.parametrize("lag", [0.0, 0.1], ids=["current", "lagged"])
def test_strain_gradient_matches_jacobian(strains, triplet, rng, lag):
    cfg = triplet.lagged(lag)
    sigma = rng.normal(size=5)
    out = strains(cfg.q, cfg, sigma=sigma)
    assert_close(out.grad[0], out.J[0].T @ sigma, 1.0e-5, "J^T sigma")


# -- derivatives in ADMM's local variable z = [e, theta_e, f, theta_f - theta_e] ----------


def _z_scale(cfg) -> np.ndarray:
    s = np.full(8, cfg.edge_length)
    s[Z_THETA_DOFS] = 1.0
    return s


def _local_derivatives(local_strains, cfg):
    """``(out, J_fd, H_fd)`` at ``z(cfg.q)``, all dimensionless by the z scale."""
    z = to_z(cfg.q)[0]
    D = _z_scale(cfg)
    DD = np.outer(D, D)
    out = local_strains(z, cfg)
    f = lambda Z: local_strains(Z, cfg).strain  # noqa: E731
    J_fd = jacobian(f, z, 1.0e-2 * D) * D
    H_fd = hessian(f, z, 3.0e-2 * D) * DD
    out.J, out.J_ref = out.J[0] * D, out.J_ref[0] * D
    out.H, out.H_ref = out.H[0] * DD, out.H_ref[0] * DD
    return out, J_fd, H_fd


@LAGS
def test_local_jacobian_matches_finite_differences(local_strains, triplet, lag):
    out, J_fd, _ = _local_derivatives(local_strains, triplet.lagged(lag))
    for i, name in enumerate(STRAINS):
        assert_close(out.J[i], J_fd[i], 1.0e-3, name)


@LAGS
def test_local_hessians_match_finite_differences(local_strains, triplet, lag):
    out, _, H_fd = _local_derivatives(local_strains, triplet.lagged(lag))
    E, TH = Z_EDGE_DOFS, Z_THETA_DOFS
    blocks = {"xx": (E, E), "x-theta": (E, TH), "theta-theta": (TH, TH)}
    for i, name in enumerate(STRAINS):
        scale = np.abs(H_fd[i]).max()
        for block, (r, c) in blocks.items():
            assert_close(out.H[i][np.ix_(r, c)], H_fd[i][np.ix_(r, c)], 1.0e-2, f"{name} {block}", scale=scale)


@LAGS
def test_local_derivatives_match_reduced(local_strains, triplet, lag):
    """Against ``strain_derivatives`` reduced by ``T = dq/dz`` (``J T``, ``T^T H T``), to float32 roundoff."""
    out, _, _ = _local_derivatives(local_strains, triplet.lagged(lag))
    J_scale = np.abs(out.J_ref).max()
    for i, name in enumerate(STRAINS):
        assert_close(out.J[i], out.J_ref[i], 1.0e-5, f"J {name}", scale=J_scale)
        assert_close(out.H[i], out.H_ref[i], 1.0e-5, f"H {name}", scale=np.abs(out.H_ref[i]).max() + J_scale)


@LAGS
def test_local_hessian_is_sigma_weighted(local_strains, triplet, rng, lag):
    cfg = triplet.lagged(lag)
    sigma = rng.normal(size=5)
    out = local_strains(to_z(cfg.q)[0], cfg, sigma=sigma)
    expected = np.einsum("i,ijk->jk", sigma, out.H[0])
    assert_close(out.H_sigma[0], expected, 1.0e-5, "sum sigma_i H_i", scale=np.abs(out.H[0]).max())


@wp.kernel
def _packed_index_kernel(out: wp.array2d[wp.int32]):
    i, j = wp.tid()
    out[i, j] = -1
    if i <= j:
        out[i, j] = packed_index(i, j)


def test_packed_index_matches_unpack(device):
    """``packed_index`` enumerates the row-major upper triangle, the order ``unpack_sym8`` reads."""
    out = wp.zeros((8, 8), dtype=wp.int32, device=device)
    wp.launch(_packed_index_kernel, dim=(8, 8), outputs=[out], device=device)
    i, j = np.triu_indices(8)
    np.testing.assert_array_equal(out.numpy()[i, j], np.arange(36))
    np.testing.assert_array_equal(unpack_sym8(np.arange(36.0))[i, j], np.arange(36))

