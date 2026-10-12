"""Step 3: larger noise over the whole domain [-4, 4]^2, comparable to Figure 3 (middle) of the paper.

Answers "what happens as the noise level is increased further": total probability of not escaping,
how it splits between the two minimizers, how much of the domain has an uncertain outcome, and
(for step noise, where the minimizers stay fixed points) how much actually converges.
"""
import os

import numpy as np

from config import OUT, P, GLOBAL_LEVELS, GLOBAL_RES, GLOBAL_TRIALS, SEED, log, save_json
from dynamics import grid, simulate, classify, outcome_probabilities
from metrics import band_fraction, selection_uncertain_fraction

BOUNDS = (-4.0, 4.0, -4.0, 4.0)


def summarize(mode, sigma, probs):
    return {"mode": mode, "sigma": sigma, "survival": float(probs["survive"].mean()),
            "plus": float(probs["plus"].mean()), "minus": float(probs["minus"].mean()),
            "converged": float(probs["converged"].mean()),
            "band_fraction": band_fraction(probs["survive"]),
            "selection_fraction": selection_uncertain_fraction(probs)}


def main():
    U, V = grid(BOUNDS, GLOBAL_RES)
    u, v, esc = simulate(U, V, P)
    lab = classify(u, v, esc)
    det = {"survive": (~esc).astype(float), "plus": (lab == 1).astype(float), "minus": (lab == 2).astype(float),
           "converged": ((lab == 1) | (lab == 2)).astype(float)}
    records = [summarize("deterministic", 0.0, det)]
    maps = {f"deterministic_0_{k}": det[k].astype(np.float32) for k in ("survive", "plus", "minus")}
    for mode, levels in GLOBAL_LEVELS.items():
        for sigma in levels:
            probs = outcome_probabilities(U, V, P, mode, sigma, GLOBAL_TRIALS, seed=SEED)
            rec = summarize(mode, sigma, probs)
            records.append(rec)
            for k in ("survive", "plus", "minus"):
                maps[f"{mode}_{sigma}_{k}"] = probs[k].astype(np.float32)
            log(f"global {mode:8s} sigma={sigma:<5} survival={rec['survival']:.4f} plus={rec['plus']:.4f} "
                f"minus={rec['minus']:.4f} converged={rec['converged']:.4f} band={rec['band_fraction']:.4f} "
                f"sel={rec['selection_fraction']:.4f}")
    save_json("global.json", {"bounds": BOUNDS, "resolution": GLOBAL_RES, "trials": GLOBAL_TRIALS, "records": records})
    np.savez_compressed(os.path.join(OUT, "maps_global.npz"), **maps)


if __name__ == "__main__":
    main()
