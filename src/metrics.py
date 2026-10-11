"""Measurements on survival-probability maps and power-law fits."""
import numpy as np


def band_fraction(prob, lo=0.1, hi=0.9):
    """Fraction of initial conditions whose outcome is uncertain under noise (lo < p < hi)."""
    return float(np.mean((prob > lo) & (prob < hi)))


def interface_fraction(mask):
    """Fraction of horizontally/vertically adjacent cell pairs with different values."""
    mask = np.asarray(mask, dtype=bool)
    n_r, n_c = mask.shape
    edges = np.count_nonzero(mask[:, 1:] != mask[:, :-1]) + np.count_nonzero(mask[1:, :] != mask[:-1, :])
    return edges / (n_r * (n_c - 1) + (n_r - 1) * n_c)


def shift_fraction(prob, det_escaped):
    """Fraction of cells where the majority outcome (p >= 1/2 survives) differs from deterministic GD."""
    return float(np.mean((prob >= 0.5) != ~np.asarray(det_escaped, dtype=bool)))


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
