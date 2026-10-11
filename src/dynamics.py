"""Gradient descent on regularized scalar factorization, with optional noise.

Loss (Liang & Montufar, arXiv:2509.25351, Section 3.2 with d = 1):
    L(u, v) = 0.5 * (u*v - y)**2 + 0.5 * lam * (u**2 + v**2)

Update (both coordinates use the old state):
    u <- u - eta_t * ( (u*v - y)*v + lam*u + xi_u )
    v <- v - eta_t * ( (u*v - y)*u + lam*v + xi_v )

Noise models
    "step"     : eta_t = eta * (1 + sigma * U(-1, 1)),   xi = 0
    "additive" : eta_t = eta,  xi ~ N(0, sigma**2) per coordinate (a stylized SGD proxy)

Coupling
    "independent" : every trajectory draws its own noise (used for probabilities)
    "common"      : all trajectories in a batch share one noise path
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Params:
    y: float = 0.5
    lam: float = 0.2
    eta: float = 1.0
    steps: int = 200
    escape: float = 1e8      # |u| or |v| above this counts as divergence
    tol: float = 1e-6        # distance to a stationary point counted as "near" it

    @property
    def minimizer(self) -> float:
        return float(np.sqrt(max(0.0, self.y - self.lam)))


def loss(u, v, p: Params = Params()):
    return 0.5 * (u * v - p.y) ** 2 + 0.5 * p.lam * (u * u + v * v)


def gradient(u, v, p: Params = Params()):
    r = u * v - p.y
    return r * v + p.lam * u, r * u + p.lam * v


def grid(bounds, n):
    """Return (U, V) meshgrid of shape (n, n); rows index v, columns index u."""
    u = np.linspace(bounds[0], bounds[1], n)
    v = np.linspace(bounds[2], bounds[3], n)
    return np.meshgrid(u, v)


def simulate(u0, v0, p: Params = Params(), mode="deterministic", sigma=0.0,
             coupling="independent", rng=None):
    """Run noisy GD on arrays of initial conditions.

    Returns (u, v, escaped). `escaped` is True for trajectories that left the
    box |u|, |v| <= p.escape (or became non-finite) at any step. This is a
    finite-horizon notion: not escaping within p.steps is not a proof of convergence.
    """
    if mode not in ("deterministic", "step", "additive"):
        raise ValueError(f"unknown mode {mode!r}")
    if coupling not in ("independent", "common"):
        raise ValueError(f"unknown coupling {coupling!r}")
    if sigma < 0 or (mode == "step" and sigma >= 1):
        raise ValueError("invalid noise amplitude")
    rng = np.random.default_rng(0) if rng is None else rng
    u = np.array(u0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    escaped = np.zeros(u.shape, dtype=bool)
    shape = () if coupling == "common" else u.shape
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(p.steps):
            gu, gv = gradient(u, v, p)
            if mode == "step" and sigma > 0:
                eta = p.eta * (1.0 + sigma * rng.uniform(-1.0, 1.0, size=shape))
                u, v = u - eta * gu, v - eta * gv
            elif mode == "additive" and sigma > 0:
                u = u - p.eta * (gu + sigma * rng.standard_normal(shape))
                v = v - p.eta * (gv + sigma * rng.standard_normal(shape))
            else:
                u, v = u - p.eta * gu, v - p.eta * gv
            out = ~np.isfinite(u) | ~np.isfinite(v) | (np.abs(u) > p.escape) | (np.abs(v) > p.escape)
            escaped |= out
            u[escaped] = 0.0  # frozen; values of escaped trajectories are never used again
            v[escaped] = 0.0
    return u, v, escaped


def classify(u, v, escaped, p: Params = Params()):
    """Finite-horizon labels: 0 escaped, 1 near (+m,+m), 2 near (-m,-m), 3 near saddle, 4 unresolved."""
    m = p.minimizer
    labels = np.full(u.shape, 4, dtype=np.int8)
    labels[np.hypot(u - m, v - m) < p.tol] = 1
    labels[np.hypot(u + m, v + m) < p.tol] = 2
    labels[np.hypot(u, v) < p.tol] = 3
    labels[escaped] = 0
    return labels


def survival_probability(u0, v0, p: Params, mode, sigma, trials=128, seed=0, chunk=32):
    """Estimate P(no escape within p.steps) per initial condition from independent runs."""
    rng = np.random.default_rng(seed)
    survived = np.zeros(np.shape(u0), dtype=np.int64)
    done = 0
    while done < trials:
        k = min(chunk, trials - done)
        ub = np.broadcast_to(u0, (k,) + np.shape(u0))
        vb = np.broadcast_to(v0, (k,) + np.shape(v0))
        _, _, esc = simulate(ub, vb, p, mode, sigma, "independent", rng)
        survived += (~esc).sum(axis=0)
        done += k
    return survived / trials
