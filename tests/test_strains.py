"""Strain derivatives against finite differences: exact for current frames, first order in their lag."""

import numpy as np
import pytest
from fd import assert_close, hessian, jacobian
from triplets import NODE_DOFS, THETA_DOFS

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


def test_jacobian_matches_finite_differences(strains, triplet):
    J, J_fd, _, _ = _derivatives(strains, triplet)
    for i, name in enumerate(STRAINS):
        assert_close(J[i], J_fd[i], 1.0e-3, name)


def test_hessians_match_finite_differences(strains, triplet):
    _, _, H, H_fd = _derivatives(strains, triplet)
    blocks = {"xx": (NODE_DOFS, NODE_DOFS), "x-theta": (NODE_DOFS, THETA_DOFS), "theta-theta": (THETA_DOFS, THETA_DOFS)}
    for i, name in enumerate(STRAINS):
        scale = np.abs(H_fd[i]).max()
        for block, (r, c) in blocks.items():
            assert_close(H[i][np.ix_(r, c)], H_fd[i][np.ix_(r, c)], 1.0e-2, f"{name} {block}", scale=scale)


def test_hessians_symmetric(strains, triplet):
    H = strains(triplet.q, triplet).H[0]
    np.testing.assert_allclose(H, np.swapaxes(H, 1, 2), rtol=0.0, atol=1.0e-6 * np.abs(H).max())


@pytest.mark.parametrize("lag", [0.0, 0.1], ids=["current", "lagged"])
def test_strain_gradient_matches_jacobian(strains, triplet, rng, lag):
    cfg = triplet.lagged(lag)
    sigma = rng.normal(size=5)
    out = strains(cfg.q, cfg, sigma=sigma)
    assert_close(out.grad[0], out.J[0].T @ sigma, 1.0e-5, "J^T sigma")


def test_lagged_frames_converge(strains, triplet):
    """Errors are linear in the frames' lag."""
    lags = [0.08, 0.04, 0.02]
    errors = []
    for lag in lags:
        cfg = triplet.lagged(lag)
        J, J_fd, H, H_fd = _derivatives(strains, cfg)
        errors.append([np.abs(J - J_fd).max() / np.abs(J_fd).max(), np.abs(H - H_fd).max() / np.abs(H_fd).max()])
    errors = np.array(errors)
    for k, name in enumerate(["J", "H"]):
        ratios = errors[1:, k] / errors[:-1, k]
        assert np.all((ratios > 0.35) & (ratios < 0.65)), f"{name}: errors {errors[:, k]} not linear in the lag"
