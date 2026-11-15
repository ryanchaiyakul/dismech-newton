"""Richardson-extrapolated central finite differences (``jacobian``: batched, ``f`` maps ``(N, n)`` to
``(N, m)``) and the tests' tolerance checks."""

import numpy as np


def _richardson(central, h):
    return (4.0 * central(0.5 * h) - central(h)) / 3.0


def derivative(f, h: float) -> float:
    """``f'(0)``."""
    return _richardson(lambda e: (f(e) - f(-e)) / (2.0 * e), h)


def jacobian(f, x: np.ndarray, h, error: bool = False):
    """Shape ``(m, n)``; with ``error``, also its error estimate ``|D(h) - D(h/2)|``, the two central differences
    apart: their truncation error and round-off, which bounds the extrapolation's."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size

    def central(h):
        D = np.diag(np.broadcast_to(h, n))
        y = f(np.concatenate([x + D, x - D]))
        return ((y[:n] - y[n:]) / (2.0 * np.diag(D))[:, None]).T

    h = np.asarray(h, dtype=np.float64)
    coarse, fine = central(h), central(0.5 * h)
    J = (4.0 * fine - coarse) / 3.0
    return (J, np.abs(coarse - fine)) if error else J


def assert_close(actual: np.ndarray, expected: np.ndarray, rtol: float, name: str = "", scale: float | None = None):
    """``max |actual - expected| <= rtol * scale``, ``scale`` defaulting to ``max |expected|``."""
    if scale is None:
        scale = np.abs(expected).max()
    err = np.abs(actual - expected).max()
    assert err <= rtol * scale, f"{name}: max error {err:.3e} > {rtol:g} * scale {scale:.3e}"


def assert_derivative(actual: float, fd: float, est: float, rtol: float, name: str = "", scale: float | None = None,
                      noise: float = 4.0, resolved: float | None = None):
    """``|actual - fd| <= rtol * scale + noise * est``, ``scale`` defaulting to ``|fd|``: a direction the differences
    resolve poorly (``est``, from :func:`jacobian`; ``noise`` standard deviations) says so instead of needing a looser
    ``rtol``. ``resolved``: also ``est <= resolved |fd|``."""
    err = abs(actual - fd)
    scale = abs(fd) if scale is None else scale
    assert err <= rtol * scale + noise * est, (
        f"{name}: adjoint {actual:.6e}, differences {fd:.6e} (error estimate {est:.1e}): error {err:.2e} > "
        f"{rtol:g} * scale {scale:.2e} + {noise:g} * {est:.1e}")
    if resolved is not None:
        assert est <= resolved * abs(fd), f"{name}: differences {fd:.6e} unresolved: error estimate {est:.1e}"


def make_probe_dirs(rng, g: np.ndarray, mask: np.ndarray | None = None, k: int = 2) -> np.ndarray:
    """Return ``(k, g.size)`` probe directions: normalized (masked) gradient plus ``k - 1`` unit random vectors."""
    g_m = g if mask is None else g * mask
    g_norm = float(np.linalg.norm(g_m))
    assert g_norm > 1e-5, f"gradient norm too small to probe: {g_norm}"
    d = rng.normal(size=(k, g.size))
    if mask is not None:
        d *= mask
    d[0] = g_m / g_norm
    if k > 1:
        d[1:] /= np.linalg.norm(d[1:], axis=1, keepdims=True)
    return d


def check_vjp_probe(
    label: str,
    dirs: np.ndarray,
    g_vec: np.ndarray,
    eval_fn,
    h_fd: float,
    rtol: float = 1.0e-3,
    *,
    tag: str = "",
    scale_max: bool = False,
):
    """Compare ``dirs @ g_vec`` against central Richardson FD ``jacobian(eval_fn, 0, h_fd)``."""
    fd_j, est_j = jacobian(eval_fn, np.zeros(dirs.shape[0]), h_fd, error=True)
    adj = dirs @ np.asarray(g_vec, dtype=np.float64)
    scale = max(float(np.abs(adj).max()), float(np.abs(fd_j[0]).max())) if scale_max else float(abs(adj[0]))
    prefix = f"[{tag}] " if tag else ""
    rel0 = abs(adj[0] - fd_j[0, 0]) / max(abs(fd_j[0, 0]), 1.0e-12)
    max_rel = float(np.max(np.abs(adj - fd_j[0])) / max(scale, 1.0e-12))
    print(
        f"{prefix}{label}: ad0={adj[0]:+.6e} fd0={fd_j[0, 0]:+.6e} est0={est_j[0, 0]:.1e} "
        f"rel0={rel0:.3e} max_rel={max_rel:.3e}"
    )
    for j in range(adj.shape[0]):
        assert_derivative(
            float(adj[j]),
            float(fd_j[0, j]),
            float(est_j[0, j]),
            rtol,
            name=f"{label}_{j}",
            scale=scale,
        )
    if not scale_max:
        assert abs(adj[0] - fd_j[0, 0]) <= rtol * abs(fd_j[0, 0]) + 4.0 * est_j[0, 0]
