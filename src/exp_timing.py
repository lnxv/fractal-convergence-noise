"""Experiment driver: timing of step-size perturbations in window W1.

run_set(tag, T, trials, res, groups) evaluates every (mode, K, level) in `groups` and writes
timing_<tag>.json (summary numbers) and timing_<tag>_maps.npz (probability maps at selected levels).
"""
import json
import time

import numpy as np

from dynamics import Params, grid, simulate
from timing import outcome_probabilities, summarize, R_CONV

W1 = (2.0, 2.7, -0.3, 0.3)
LEVELS = {
    "init": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    "early": [0.01, 0.02, 0.05, 0.1, 0.2],
    "persist": [0.01, 0.02, 0.05, 0.1, 0.2],
}
GROUPS = [("init", None), ("early", 10), ("early", 50), ("persist", None)]
MAP_LEVELS = {"init": [0.005, 0.02], "early": [0.05, 0.2], "persist": [0.05, 0.2]}
SEED = 20261101


def deterministic(U, V, p):
    m = p.minimizer
    u, v, esc = simulate(U, V, p)
    alive = ~esc
    probs = {"escape": esc.astype(float),
             "plus": (alive & (np.hypot(u - m, v - m) < R_CONV)).astype(float),
             "minus": (alive & (np.hypot(u + m, v + m) < R_CONV)).astype(float)}
    probs["unresolved"] = alive.astype(float) - probs["plus"] - probs["minus"]
    return probs


def run_set(tag, T=200, trials=64, res=101, groups=GROUPS, levels=LEVELS):
    p = Params(steps=T)
    U, V = grid(W1, res)
    t0 = time.time()
    det = deterministic(U, V, p)
    out = {"tag": tag, "T": T, "trials": trials, "res": res, "window": W1, "r_conv": R_CONV,
           "deterministic": summarize(det), "records": []}
    maps = {f"det_{k}": v.astype(np.float32) for k, v in det.items()}
    print(f"[{tag}] deterministic unresolved={out['deterministic']['mean_unresolved_prob']:.4f}", flush=True)
    for mode, K in groups:
        for lvl in levels[mode]:
            probs = outcome_probabilities(U, V, p, mode, lvl, K, trials, seed=SEED)
            rec = {"mode": mode, "K": K, "level": lvl, **summarize(probs)}
            out["records"].append(rec)
            name = mode if K is None else f"{mode}{K}"
            if lvl in MAP_LEVELS.get(mode, []):
                for k, v in probs.items():
                    maps[f"{name}_{lvl}_{k}"] = v.astype(np.float32)
            print(f"[{tag}] {name:8s} level={lvl:<6} unc_esc={rec['uncertain_escape']:.4f} "
                  f"unc_sel={rec['uncertain_selection']:.4f} sel_unres={rec['selection_unresolved']:.4f} "
                  f"unres={rec['mean_unresolved_prob']:.4f} t={time.time() - t0:.0f}s", flush=True)
    with open(f"timing_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(f"timing_{tag}_maps.npz", **maps)
    return out
