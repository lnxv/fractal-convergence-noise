"""Shared experiment settings. Set QUICK=1 for a fast smoke test and OUT=<dir> for the output folder."""
import json
import os
import time

from dynamics import Params

QUICK = bool(int(os.environ.get("QUICK", "0")))
OUT = os.environ.get("OUT", "results")
os.makedirs(OUT, exist_ok=True)

P = Params(steps=200)  # y = 0.5, lambda = 0.2, eta = 1: the Figure 3 setting of the paper

# Two windows straddling the fractal part of the boundary.
WINDOWS = {"W1": (2.0, 2.7, -0.3, 0.3), "W2": (1.7, 2.3, 0.5, 1.1)}
LEVELS = {
    "step": [0.01, 0.02, 0.05, 0.1, 0.2],
    "additive": [0.0025, 0.005, 0.01, 0.02, 0.04],
    "sgd": [0.0025, 0.005, 0.01, 0.02, 0.04],
}
# Larger noise over the whole domain [-4, 4]^2 ("what happens as the noise level is increased further").
GLOBAL_LEVELS = {
    "step": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    "sgd": [0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
    "additive": [0.01, 0.02, 0.05, 0.1, 0.2],
}
RES, TRIALS = (61, 32) if QUICK else (101, 96)
GLOBAL_RES, GLOBAL_TRIALS = (81, 8) if QUICK else (161, 32)
UNC_SAMPLES = 5000 if QUICK else 40000
PROJ_RES = 600 if QUICK else 2000
SEED = 424242


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(name, obj):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=2)
