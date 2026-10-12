"""Step 4: checks that the measurements reflect the dynamics and not the numerical setup.

  - grid resolution: uncertain fraction at 81 vs 161 points per side;
  - horizon: 200 vs 400 steps;
  - noise coupling: one shared noise path vs independent noise per pixel vs the probability map.
"""
import os

import numpy as np

from config import OUT, P, WINDOWS, SEED, log, save_json
from dynamics import Params, grid, simulate, survival_probability
from metrics import band_fraction, interface_fraction


def main(mode="step", sigma=0.1):
    b, out = WINDOWS["W1"], {"mode": mode, "sigma": sigma}
    for res in (81, 161):
        U, V = grid(b, res)
        out[f"band_resolution_{res}"] = band_fraction(survival_probability(U, V, P, mode, sigma, 64, seed=SEED))
    U, V = grid(b, 81)
    out["band_horizon_400"] = band_fraction(survival_probability(U, V, Params(steps=400), mode, sigma, 64, seed=SEED))
    U, V = grid(b, 91)
    rng = np.random.default_rng(1009)
    _, _, common = simulate(U, V, P, mode, sigma, "common", rng)
    _, _, indep = simulate(U, V, P, mode, sigma, "independent", rng)
    prob = survival_probability(U, V, P, mode, sigma, 64, seed=SEED)
    _, _, det = simulate(U, V, P)
    out["interface"] = {"deterministic": interface_fraction(det), "common_noise": interface_fraction(common),
                        "independent_noise": interface_fraction(indep), "probability_majority": interface_fraction(prob >= 0.5)}
    log(f"controls: {out}")
    save_json("controls.json", out)
    np.savez_compressed(os.path.join(OUT, "maps_controls.npz"), U=U.astype(np.float32), V=V.astype(np.float32),
                        common=common, independent=indep, probability=prob.astype(np.float32))


if __name__ == "__main__":
    main()
