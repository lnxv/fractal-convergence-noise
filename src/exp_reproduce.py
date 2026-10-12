"""Step 1: recover the deterministic phenomena of Liang & Montufar before adding noise.

  (a) the convergence region of Figure 3 (middle) on [-4, 4]^2;
  (b) the box-counting dimension of the projected boundary T(dD'') (Figure 3, left/right; paper: 1.249),
      computed with the planar map of Proposition 3;
  (c) the critical step size of Theorem 1 (unregularized case);
  (d) the uncertainty exponent of the boundary in the two windows used for the noise experiments
      (Grebogi, McDonald, Ott & Yorke 1983), an independent route to the dimension.
"""
import os

import numpy as np

from config import OUT, P, WINDOWS, UNC_SAMPLES, PROJ_RES, log, save_json
from dynamics import Params, grid, simulate, classify, critical_step_size, projected_map
from metrics import erode, box_counting_dimension, power_law_fit

LABELS = ["escaped", "near_plus_min", "near_minus_min", "near_saddle", "unresolved"]


def baseline(n=401):
    U, V = grid((-4, 4, -4, 4), n)
    out, labels = {}, None
    for steps in (100, 400):
        u, v, esc = simulate(U, V, Params(steps=steps))
        labels = classify(u, v, esc)
        out[f"steps_{steps}"] = {name: int(np.sum(labels == k)) for k, name in enumerate(LABELS)}
    return out, labels


def projected_boundary(p=P, n=PROJ_RES):
    """Boundary of the convergence region in (P, w) = (u.v, |u|^2 + |v|^2), feasible set w >= 2|P|."""
    PP, WW = np.meshgrid(np.linspace(p.y - 2.5, p.y + 3.0, n), np.linspace(0.0, 10.0, n))
    feasible = WW >= 2 * np.abs(PP)
    Pc, wc = PP[feasible], WW[feasible]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(p.steps):
            Pc, wc = projected_map(Pc, wc, p)
    region = np.zeros_like(feasible)
    region[feasible] = np.abs(wc - 2 * (p.y - p.lam)) < 1e-5  # converged to a minimizer, where w = 2(y - lam)
    boundary = (region ^ erode(region)) & (np.abs(WW - 2 * np.abs(PP)) > 1e-2)  # drop the edge w = 2|P|
    pts = np.c_[PP[boundary], WW[boundary]]
    return box_counting_dimension(pts), pts


def critical_step_check(n=20000, y=1.0, steps=4000, seed=3):
    """Theorem 1: GD converges for eta below eta*(u0, v0) and fails to converge above it."""
    rng = np.random.default_rng(seed)
    u0, v0 = rng.uniform(-2.5, 2.5, n), rng.uniform(-2.5, 2.5, n)
    eta_star = critical_step_size(u0, v0, y)
    out = {"y": y, "samples": n, "steps": steps}
    with np.errstate(over="ignore", invalid="ignore"):
        for margin in (0.03, 0.1):
            for side, factor in (("below", 1 - margin), ("above", 1 + margin)):
                eta, u, v = factor * eta_star, u0.copy(), v0.copy()
                for _ in range(steps):
                    r = u * v - y
                    u, v = u - eta * r * v, v - eta * r * u
                converged = np.abs(u * v - y) < 1e-6
                out[f"{side}_{margin}"] = float(np.mean(converged if side == "below" else ~converged))
    return out


def uncertainty_exponents(eps_list=(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2), seed=20261011):
    """Fraction of initial conditions whose outcome changes under a perturbation of size eps, f ~ eps^(2-D).

    Two outcomes are used: escape vs no escape (boundary of the convergence region), and the final
    state (escape, +minimizer, -minimizer), which also includes the boundary between the two basins.
    """
    rng = np.random.default_rng(seed)
    out = []
    for wname, b in WINDOWS.items():
        u0, v0 = rng.uniform(b[0], b[1], UNC_SAMPLES), rng.uniform(b[2], b[3], UNC_SAMPLES)

        def outcome(du, dv):
            u, v, esc = simulate(u0 + du, v0 + dv, P)
            return esc, classify(u, v, esc, Params(tol=1e-3))

        base_esc, base_lab = outcome(0.0, 0.0)
        rows = []
        for eps in eps_list:
            th = rng.uniform(0, 2 * np.pi, UNC_SAMPLES)
            e1, l1 = outcome(eps * np.cos(th), eps * np.sin(th))
            e2, l2 = outcome(-eps * np.cos(th), -eps * np.sin(th))
            rows.append({"eps": eps,
                         "escape_flip": float(np.mean((e1 != base_esc) | (e2 != base_esc))),
                         "outcome_flip": float(np.mean((l1 != base_lab) | (l2 != base_lab)))})
        res = {"window": wname, "samples": UNC_SAMPLES, "rows": rows}
        for key in ("escape_flip", "outcome_flip"):
            f = power_law_fit([r["eps"] for r in rows], [r[key] for r in rows])
            res[key] = {**f, "implied_dimension": 2 - f["exponent"]}
        out.append(res)
        log(f"uncertainty {wname}: escape 2-D = {res['escape_flip']['exponent']:.3f} +/- {res['escape_flip']['se']:.3f}, "
            f"final state 2-D = {res['outcome_flip']['exponent']:.3f} +/- {res['outcome_flip']['se']:.3f}")
    return out


def main():
    base, labels = baseline()
    log(f"baseline: {base}")
    dim, pts = projected_boundary()
    log(f"projected boundary: D = {dim['exponent']:.3f} (R^2 = {dim['r2']:.4f}, {len(pts)} boundary points; paper: 1.249)")
    crit = critical_step_check()
    log(f"critical step size (Theorem 1): {crit}")
    unc = uncertainty_exponents()
    save_json("reproduce.json", {"baseline": base, "projected_boundary_dimension": dim, "critical_step_size": crit,
                                 "uncertainty": unc})
    keep = np.random.default_rng(0).permutation(len(pts))[:150000]
    np.savez_compressed(os.path.join(OUT, "maps_reproduce.npz"), baseline_labels=labels.astype(np.int8),
                        boundary_points=pts[keep].astype(np.float32))


if __name__ == "__main__":
    main()
