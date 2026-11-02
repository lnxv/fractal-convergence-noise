"""When is the outcome of deterministic GD decided, for initializations in W1?

For each initialization we record the first step at which the iterate is either far out
(|u| or |v| > 10, after which it always escaped in this window) or inside a small ball
of radius 0.05 around a minimizer (inside which every step size in [0.8, 1.2] contracts,
since the Hessian eigenvalues at the minimizers are 0.4 and 0.6).
"""
import json
import numpy as np
from dynamics import Params, grid, gradient, simulate

p = Params(steps=200)
m = p.minimizer
U, V = grid((2.0, 2.7, -0.3, 0.3), 401)
u, v = U.copy(), V.copy()
decided = np.full(U.shape, -1)
far_first = np.zeros(U.shape, bool)
with np.errstate(over="ignore", invalid="ignore"):
    for t in range(p.steps):
        gu, gv = gradient(u, v, p)
        u, v = u - gu, v - gv
        far = ~np.isfinite(u) | (np.abs(u) > 10) | (np.abs(v) > 10)
        near = (np.hypot(u - m, v - m) < 0.05) | (np.hypot(u + m, v + m) < 0.05)
        new = (decided < 0) & (far | near)
        decided[new] = t + 1
        far_first[new & far] = True
        u = np.where(far, 0.0, u); v = np.where(far, 0.0, v)
_, _, esc = simulate(U, V, p)
consistent = float(np.mean(far_first == esc))
d = decided[decided > 0]
q = {str(k): int(np.percentile(d, k)) for k in (50, 90, 99, 99.9)}
out = {"undecided_after_200": float(np.mean(decided < 0)), "quantiles_of_decision_step": q,
       "frac_decided_by_10": float(np.mean((decided > 0) & (decided <= 10))),
       "frac_decided_by_50": float(np.mean((decided > 0) & (decided <= 50))),
       "far_matches_escape": consistent}
print(json.dumps(out))
json.dump(out, open("decision_time.json", "w"), indent=1)
