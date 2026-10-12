"""Measurements on probability maps, boundary geometry, and power-law fits."""
import numpy as np


def band_fraction(prob, lo=0.1, hi=0.9):
    """Fraction of initial conditions whose outcome is uncertain under noise (lo < p < hi)."""
    return float(np.mean((prob > lo) & (prob < hi)))


def selection_uncertain_fraction(probs, lo=0.1, hi=0.9, min_survive=0.5):
    """Fraction of initial conditions that usually survive but whose selected minimizer is uncertain.

    q = P(+m | survive). Counted if P(survive) >= min_survive and lo < q < hi.
    """
    s, plus, minus = probs["survive"], probs["plus"], probs["minus"]
    tot = plus + minus
    q = np.divide(plus, tot, out=np.full_like(s, 0.5), where=tot > 0)
    return float(np.mean((s >= min_survive) & (q > lo) & (q < hi)))


def interface_fraction(mask):
    """Fraction of horizontally/vertically adjacent cell pairs with different values."""
    mask = np.asarray(mask)
    n_r, n_c = mask.shape
    edges = np.count_nonzero(mask[:, 1:] != mask[:, :-1]) + np.count_nonzero(mask[1:, :] != mask[:-1, :])
    return edges / (n_r * (n_c - 1) + (n_r - 1) * n_c)


def shift_fraction(prob, det_escaped):
    """Fraction of cells where the majority outcome (p >= 1/2 survives) differs from deterministic GD."""
    return float(np.mean((prob >= 0.5) != ~np.asarray(det_escaped, dtype=bool)))


def erode(mask):
    """Binary erosion with the 4-neighbour cross; pixels outside the array count as False."""
    e = mask.copy()
    e[1:, :] &= mask[:-1, :]
    e[:-1, :] &= mask[1:, :]
    e[:, 1:] &= mask[:, :-1]
    e[:, :-1] &= mask[:, 1:]
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = False
    return e


def box_counting_dimension(points, ks=range(2, 9)):
    """Box-counting dimension of a planar point set, fitted over box sizes 2^-k of its bounding square."""
    pts = np.asarray(points, float)
    pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0)).max()
    eps, counts = [], []
    for k in ks:
        m = 2 ** k
        idx = np.clip((pts * m).astype(np.int64), 0, m - 1)
        counts.append(len(np.unique(idx[:, 0] * m + idx[:, 1])))
        eps.append(1.0 / m)
    f = power_law_fit(1.0 / np.array(eps), np.array(counts, float))
    return {**f, "eps": eps, "counts": counts}


def power_law_fit(x, y):
    """Least-squares fit of log y = a + b log x. Returns exponent b, its standard error, and R^2."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([np.ones_like(lx), lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    dof = len(lx) - 2
    s2 = resid @ resid / dof
    se = float(np.sqrt(s2 / np.sum((lx - lx.mean()) ** 2)))
    r2 = float(1 - resid @ resid / np.sum((ly - ly.mean()) ** 2))
    return {"exponent": float(coef[1]), "se": se, "r2": r2, "intercept": float(coef[0])}
