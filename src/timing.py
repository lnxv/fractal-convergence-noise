"""Timing of random step-size perturbations near the deterministic convergence boundary.

Same loss and baseline parameters as before (y = 0.5, lambda = 0.2, eta = 1). Three experiments:

  init     perturb the initialization, (u0, v0) + eps * N(0, I), then run deterministic GD
  early    random step sizes eta_t = eta * (1 + sigma * U(-1, 1)) for the first K updates,
           then deterministic GD
  persist  random step sizes for all updates

Outcome definitions (per run, finite horizon T):
  escape      |u| or |v| exceeds 1e8 (or becomes non-finite) at some step
  plus        did not escape and the final state is within R_CONV of the minimizer (+m, +m)
  minus       did not escape and the final state is within R_CONV of the minimizer (-m, -m)
  unresolved  did not escape and the final state is within R_CONV of neither minimizer

Being within R_CONV of a minimizer is a stronger requirement than being closer to it than to the
other stationary points; the earlier code used the weaker "closest stationary point" rule.

Per initialization, probabilities are estimated from n independent runs. Minimizer selection
q = p_plus / (p_plus + p_minus) is only defined where at least one run reached a minimizer; if no run
did, selection is marked unresolved (NaN) instead of being assigned q = 1/2.
"""
import numpy as np

from dynamics import Params, gradient

R_CONV = 1e-3
LO, HI = 0.1, 0.9


def run(u0, v0, p: Params, mode, level, K=None, rng=None, check_every=8):
    """Simulate one batch. u0, v0 have shape (n_runs, ...). Returns final (u, v) and an escape mask."""
    rng = np.random.default_rng(0) if rng is None else rng
    u = np.array(u0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    if mode == "init" and level > 0:
        u += level * rng.standard_normal(u.shape)
        v += level * rng.standard_normal(v.shape)
    noisy_steps = 0
    if mode == "persist":
        noisy_steps = p.steps
    elif mode == "early":
        noisy_steps = min(K, p.steps)
    escaped = np.zeros(u.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(p.steps):
            gu, gv = gradient(u, v, p)
            if t < noisy_steps and level > 0:
                eta = p.eta * (1.0 + level * rng.uniform(-1.0, 1.0, size=u.shape))
            else:
                eta = p.eta
            u, v = u - eta * gu, v - eta * gv
            if (t + 1) % check_every == 0 or t + 1 == p.steps:
                out = ~np.isfinite(u) | ~np.isfinite(v) | (np.abs(u) > p.escape) | (np.abs(v) > p.escape)
                if out.any():
                    escaped |= out
                    u[escaped] = 0.0
                    v[escaped] = 0.0
    return u, v, escaped


def outcome_probabilities(U, V, p: Params, mode, level, K=None, trials=64, seed=0, chunk=16):
    """Probabilities of escape / plus / minus / unresolved per initialization, from independent runs."""
    rng = np.random.default_rng(seed)
    m = p.minimizer
    acc = {k: np.zeros(U.shape) for k in ("escape", "plus", "minus", "unresolved")}
    done = 0
    while done < trials:
        k = min(chunk, trials - done)
        ub = np.broadcast_to(U, (k,) + U.shape)
        vb = np.broadcast_to(V, (k,) + V.shape)
        u, v, esc = run(ub, vb, p, mode, level, K, rng)
        alive = ~esc
        near_p = alive & (np.hypot(u - m, v - m) < R_CONV)
        near_m = alive & (np.hypot(u + m, v + m) < R_CONV)
        acc["escape"] += esc.sum(axis=0)
        acc["plus"] += near_p.sum(axis=0)
        acc["minus"] += near_m.sum(axis=0)
        acc["unresolved"] += (alive & ~near_p & ~near_m).sum(axis=0)
        done += k
    return {key: val / trials for key, val in acc.items()}


def selection(probs):
    """q = P(+ | reached a minimizer); NaN where no run reached either minimizer."""
    tot = probs["plus"] + probs["minus"]
    return np.where(tot > 0, probs["plus"] / np.where(tot > 0, tot, 1.0), np.nan)


def summarize(probs):
    pe = probs["escape"]
    q = selection(probs)
    usually_survives = pe < 0.5
    resolved = ~np.isnan(q)
    return {
        "uncertain_escape": float(np.mean((pe > LO) & (pe < HI))),
        "uncertain_selection": float(np.mean(usually_survives & resolved & (q > LO) & (q < HI))),
        "selection_unresolved": float(np.mean(usually_survives & ~resolved)),
        "mean_unresolved_prob": float(np.mean(probs["unresolved"])),
        "mean_escape_prob": float(np.mean(pe)),
    }


def power_law_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    n = len(lx)
    se = float(np.sqrt(ss_res / max(n - 2, 1) / np.sum((lx - lx.mean()) ** 2))) if n > 2 else float("nan")
    return {"exponent": float(coef[0]), "se": se, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n": int(n)}
