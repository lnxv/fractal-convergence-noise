"""Gradient descent on regularized scalar factorization, with optional noise.

Loss (Liang & Montufar, arXiv:2509.25351, Section 3.2 with d = 1):
    L(u, v) = 0.5 * (u*v - y)**2 + 0.5 * lam * (u**2 + v**2)

Update (both coordinates use the old state):
    u <- u - eta_t * ( (u*v - y_t)*v + lam*u + xi_u )
    v <- v - eta_t * ( (u*v - y_t)*u + lam*v + xi_v )

Noise models (sigma is the noise level)
    "step"     : eta_t = eta * (1 + sigma * U(-1, 1)); the minimizers stay fixed points
    "additive" : xi ~ N(0, sigma**2) per coordinate; isotropic, state-independent
    "sgd"      : y_t = y + sigma * r_t with r_t = +-1 equally likely. This is SGD with batch
                 size 1 on the finite sum L = mean_i 0.5*(uv - y_i)**2 + reg over two data
                 points y_i = y -+ sigma, which has the same minimizers as the deterministic
                 problem. The gradient noise -sigma*r_t*(v, u) depends on the state.

Coupling
    "independent" : every trajectory draws its own noise (used for probabilities)
    "common"      : all trajectories in a batch share one noise path
"""
from dataclasses import dataclass
import numpy as np

MODES = ("deterministic", "step", "additive", "sgd")


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


def gradient(u, v, p: Params = Params(), y=None):
    r = u * v - (p.y if y is None else y)
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
    if mode not in MODES:
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
    noisy = sigma > 0 and mode != "deterministic"
    check_every = 8  # divergence is checked periodically; diverging iterates only grow (or become inf/nan) in between
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(p.steps):
            if noisy and mode == "sgd":
                gu, gv = gradient(u, v, p, y=p.y + sigma * (2.0 * rng.integers(0, 2, size=shape) - 1.0))
            else:
                gu, gv = gradient(u, v, p)
            if noisy and mode == "step":
                eta = p.eta * (1.0 + sigma * rng.uniform(-1.0, 1.0, size=shape))
                u, v = u - eta * gu, v - eta * gv
            elif noisy and mode == "additive":
                u = u - p.eta * (gu + sigma * rng.standard_normal(shape))
                v = v - p.eta * (gv + sigma * rng.standard_normal(shape))
            else:
                u, v = u - p.eta * gu, v - p.eta * gv
            if (t + 1) % check_every == 0 or t + 1 == p.steps:
                out = ~np.isfinite(u) | ~np.isfinite(v) | (np.abs(u) > p.escape) | (np.abs(v) > p.escape)
                if out.any():
                    escaped |= out
                    u[escaped] = 0.0  # reset; values of escaped trajectories are never used again
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


def outcome_probabilities(u0, v0, p: Params, mode, sigma, trials=128, seed=0, chunk=32, conv_tol=1e-3):
    """Estimate outcome probabilities per initial condition from independent noisy runs.

    Returns a dict of arrays shaped like u0:
      survive   P(no escape within p.steps)
      plus      P(no escape and final state closest to the minimizer (+m, +m))
      minus     P(no escape and final state closest to the minimizer (-m, -m))
      converged P(final state within conv_tol of a minimizer)
    For persistent noise (additive, sgd) iterates keep fluctuating, so `plus`/`minus`
    record which minimizer the run is attracted to rather than exact convergence.
    """
    rng = np.random.default_rng(seed)
    m = p.minimizer
    acc = {k: np.zeros(np.shape(u0), dtype=np.int64) for k in ("survive", "plus", "minus", "converged")}
    done = 0
    while done < trials:
        k = min(chunk, trials - done)
        ub = np.broadcast_to(u0, (k,) + np.shape(u0))
        vb = np.broadcast_to(v0, (k,) + np.shape(v0))
        u, v, esc = simulate(ub, vb, p, mode, sigma, "independent", rng)
        alive = ~esc
        d_plus, d_minus, d_zero = np.hypot(u - m, v - m), np.hypot(u + m, v + m), np.hypot(u, v)
        acc["survive"] += alive.sum(axis=0)
        acc["plus"] += (alive & (d_plus < d_minus) & (d_plus < d_zero)).sum(axis=0)
        acc["minus"] += (alive & (d_minus <= d_plus) & (d_minus < d_zero)).sum(axis=0)
        acc["converged"] += (alive & (np.minimum(d_plus, d_minus) < conv_tol)).sum(axis=0)
        done += k
    return {key: val / trials for key, val in acc.items()}


def survival_probability(u0, v0, p: Params, mode, sigma, trials=128, seed=0, chunk=32):
    """Estimate P(no escape within p.steps) per initial condition from independent runs."""
    return outcome_probabilities(u0, v0, p, mode, sigma, trials, seed, chunk)["survive"]


def critical_step_size(u0, v0, y):
    """Theorem 1 of Liang & Montufar for lambda = 0 (with 1/0 read as +infinity)."""
    s = u0 ** 2 + v0 ** 2
    disc = np.sqrt(np.maximum(s ** 2 - 16 * y * (u0 * v0 - y), 0.0))
    first = np.inf if y == 0 else 1.0 / abs(y)
    return np.minimum(first, 8.0 / (s + disc))


def projected_map(P, w, p: Params = Params()):
    """GD in the coordinates (P, w) = (u.v, |u|^2 + |v|^2) (Proposition 3 of the paper)."""
    r, a = P - p.y, 1.0 - p.eta * p.lam
    return a * a * P - p.eta * a * r * w + p.eta ** 2 * r * r * P, (a * a + p.eta ** 2 * r * r) * w - 4 * p.eta * a * r * P
