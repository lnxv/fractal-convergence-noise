"""Step 2: add noise gradually and measure how the boundary changes, in two windows near the boundary.

For each noise model and level, outcome probabilities are estimated from TRIALS independent runs per
initial condition. Measured quantities:
  band_fraction        fraction with 0.1 < P(survive) < 0.9 (convergence boundary becomes uncertain)
  selection_fraction   fraction that usually survives but whose selected minimizer is uncertain
  shift_fraction       where the majority outcome differs from deterministic GD
  median_interface     roughness of the P(survive) = 1/2 contour
Select noise models with MODES=step,additive,sgd.
"""
import os

import numpy as np

from config import OUT, P, WINDOWS, LEVELS, RES, TRIALS, SEED, log, save_json
from dynamics import grid, simulate, classify, outcome_probabilities
from metrics import band_fraction, selection_uncertain_fraction, shift_fraction, interface_fraction, power_law_fit

MODES = [m for m in os.environ.get("MODES", "step,additive,sgd").split(",") if m]


def main():
    records, fits, maps = [], [], {}
    for wname, b in WINDOWS.items():
        U, V = grid(b, RES)
        u, v, det_esc = simulate(U, V, P)
        lab = classify(u, v, det_esc)
        det = {"survive": (~det_esc).astype(float), "plus": (lab == 1).astype(float), "minus": (lab == 2).astype(float)}
        maps[f"{wname}_deterministic_0_survive"] = det["survive"].astype(np.float32)
        maps[f"{wname}_deterministic_0_plus"] = det["plus"].astype(np.float32)
        maps[f"{wname}_deterministic_0_minus"] = det["minus"].astype(np.float32)
        records.append({"window": wname, "mode": "deterministic", "sigma": 0.0, "survival": float(det["survive"].mean()),
                        "median_interface": interface_fraction(~det_esc), "basin_interface": interface_fraction(lab)})
        for mode in MODES:
            rows = []
            for sigma in LEVELS[mode]:
                probs = outcome_probabilities(U, V, P, mode, sigma, TRIALS, seed=SEED)
                rec = {"window": wname, "mode": mode, "sigma": sigma, "trials": TRIALS, "resolution": RES,
                       "survival": float(probs["survive"].mean()),
                       "converged": float(probs["converged"].mean()),
                       "band_fraction": band_fraction(probs["survive"]),
                       "selection_fraction": selection_uncertain_fraction(probs),
                       "shift_fraction": shift_fraction(probs["survive"], det_esc),
                       "median_interface": interface_fraction(probs["survive"] >= 0.5)}
                rows.append(rec)
                for key in ("survive", "plus", "minus"):
                    maps[f"{wname}_{mode}_{sigma}_{key}"] = probs[key].astype(np.float32)
                log(f"{wname} {mode:8s} sigma={sigma:<7} band={rec['band_fraction']:.4f} sel={rec['selection_fraction']:.4f} "
                    f"shift={rec['shift_fraction']:.4f} survival={rec['survival']:.4f} conv={rec['converged']:.4f}")
            records += rows
            for key in ("band_fraction", "selection_fraction"):
                ys = [r[key] for r in rows]
                if min(ys) > 0:
                    f = power_law_fit([r["sigma"] for r in rows], ys)
                    fits.append({"window": wname, "mode": mode, "quantity": key, **f})
                    log(f"fit {wname} {mode} {key}: exponent = {f['exponent']:.3f} +/- {f['se']:.3f} (R^2 = {f['r2']:.4f})")
    tag = "_".join(MODES)
    save_json(f"windows_{tag}.json", {"modes": MODES, "trials": TRIALS, "resolution": RES, "records": records, "fits": fits})
    np.savez_compressed(os.path.join(OUT, f"maps_windows_{tag}.npz"), **maps)


if __name__ == "__main__":
    main()
