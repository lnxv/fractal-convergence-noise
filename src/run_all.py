"""Reproduce every number and figure in the report.

    python run_all.py            # full run (a few minutes on a laptop)
    QUICK=1 python run_all.py    # smaller grids/trials for a fast smoke test

Outputs: results/results.json and results/fig_*.png
"""
import json
import os
import time

import numpy as np

from dynamics import Params, grid, simulate, classify, survival_probability
from metrics import band_fraction, interface_fraction, shift_fraction, power_law_fit

QUICK = bool(int(os.environ.get("QUICK", "0")))
OUT = os.environ.get("OUT", "results")
os.makedirs(OUT, exist_ok=True)

P = Params(steps=200)
WINDOWS = {"W1": (2.0, 2.7, -0.3, 0.3), "W2": (1.7, 2.3, 0.5, 1.1)}
LEVELS = {"step": [0.01, 0.02, 0.05, 0.1, 0.2], "additive": [0.0025, 0.005, 0.01, 0.02, 0.04]}
RES, TRIALS = (61, 32) if QUICK else (121, 128)
UNC_SAMPLES = 5000 if QUICK else 40000
EPS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def baseline():
    """Deterministic GD over the Figure 3 domain."""
    U, V = grid((-4, 4, -4, 4), 401)
    out = {}
    for steps in (100, 400):
        u, v, esc = simulate(U, V, Params(steps=steps))
        lab = classify(u, v, esc)
        out[f"steps_{steps}"] = {name: int(np.sum(lab == k)) for k, name in
                                 enumerate(["escaped", "near_plus_min", "near_minus_min", "near_saddle", "unresolved"])}
    return out, lab


def noise_sweep():
    records, maps = [], {}
    for wname, b in WINDOWS.items():
        U, V = grid(b, RES)
        _, _, det_esc = simulate(U, V, P)
        maps[(wname, "deterministic", 0.0)] = (~det_esc).astype(float)
        records.append({"window": wname, "mode": "deterministic", "sigma": 0.0,
                        "survival": float(np.mean(~det_esc)),
                        "median_interface": interface_fraction(~det_esc)})
        for mode, levels in LEVELS.items():
            for sigma in levels:
                t0 = time.time()
                prob = survival_probability(U, V, P, mode, sigma, TRIALS, seed=424242)
                maps[(wname, mode, sigma)] = prob
                rec = {"window": wname, "mode": mode, "sigma": sigma, "trials": TRIALS, "resolution": RES,
                       "survival": float(prob.mean()), "band_fraction": band_fraction(prob),
                       "shift_fraction": shift_fraction(prob, det_esc),
                       "median_interface": interface_fraction(prob >= 0.5)}
                records.append(rec)
                log(f"{wname} {mode:8s} sigma={sigma:<7} band={rec['band_fraction']:.4f} "
                    f"shift={rec['shift_fraction']:.4f} survival={rec['survival']:.4f} ({time.time() - t0:.1f}s)")
    fits = []
    for wname in WINDOWS:
        for mode in LEVELS:
            rows = [r for r in records if r["window"] == wname and r["mode"] == mode]
            f = power_law_fit([r["sigma"] for r in rows], [r["band_fraction"] for r in rows])
            fits.append({"window": wname, "mode": mode, **f})
            log(f"fit {wname} {mode}: beta = {f['exponent']:.3f} +/- {f['se']:.3f} (R^2 = {f['r2']:.4f})")
    return records, fits, maps


def uncertainty_exponent():
    """Grebogi-McDonald-Ott-Yorke uncertainty exponent for deterministic GD."""
    rng = np.random.default_rng(20261011)
    out = []
    for wname, b in WINDOWS.items():
        u0 = rng.uniform(b[0], b[1], UNC_SAMPLES)
        v0 = rng.uniform(b[2], b[3], UNC_SAMPLES)
        _, _, base = simulate(u0, v0, P)
        rows = []
        for eps in EPS:
            th = rng.uniform(0, 2 * np.pi, UNC_SAMPLES)
            du, dv = eps * np.cos(th), eps * np.sin(th)
            _, _, a = simulate(u0 + du, v0 + dv, P)
            _, _, c = simulate(u0 - du, v0 - dv, P)
            rows.append({"eps": eps, "uncertain_fraction": float(np.mean((a != base) | (c != base)))})
        f = power_law_fit([r["eps"] for r in rows], [r["uncertain_fraction"] for r in rows])
        out.append({"window": wname, "samples": UNC_SAMPLES, "rows": rows, **f, "implied_dimension": 2 - f["exponent"]})
        log(f"uncertainty {wname}: alpha = {f['exponent']:.3f} +/- {f['se']:.3f}  ->  D = {2 - f['exponent']:.3f}")
    return out


def controls():
    """Grid-resolution, horizon, and noise-coupling checks at step noise sigma = 0.1 in W1."""
    b, out = WINDOWS["W1"], {}
    for res in (81, 161):
        U, V = grid(b, res)
        prob = survival_probability(U, V, P, "step", 0.1, 64, seed=424242)
        out[f"resolution_{res}"] = band_fraction(prob)
    U, V = grid(b, 81)
    out["horizon_400"] = band_fraction(survival_probability(U, V, Params(steps=400), "step", 0.1, 64, seed=424242))
    U, V = grid(b, 91)
    rng = np.random.default_rng(1009)
    _, _, common = simulate(U, V, P, "step", 0.1, "common", rng)
    _, _, indep = simulate(U, V, P, "step", 0.1, "independent", rng)
    prob = survival_probability(U, V, P, "step", 0.1, 64, seed=424242)
    out["coupling"] = {"common_interface": interface_fraction(common), "independent_interface": interface_fraction(indep),
                       "majority_interface": interface_fraction(prob >= 0.5)}
    log(f"controls: {json.dumps(out)}")
    return out, (U, V, common, indep, prob)


if __name__ == "__main__":
    t_all = time.time()
    log(f"start (quick={QUICK}, res={RES}, trials={TRIALS})")
    base, base_labels = baseline(); log(f"baseline: {base}")
    records, fits, maps = noise_sweep()
    unc = uncertainty_exponent()
    ctrl, coupling_data = controls()
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump({"params": P.__dict__, "windows": WINDOWS, "resolution": RES, "trials": TRIALS,
                   "baseline": base, "noise_sweep": records, "noise_fits": fits,
                   "uncertainty": unc, "controls": ctrl}, fh, indent=2)
    try:
        import figures
        figures.make_all(OUT, base_labels, maps, records, fits, unc, coupling_data, WINDOWS)
    except ImportError as exc:  # matplotlib missing: numbers are still saved
        log(f"skipping figures ({exc})")
    log(f"done in {time.time() - t_all:.0f}s")
