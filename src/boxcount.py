"""
Box-counting fractal dimension estimator, matching the method in the
authors' own Fig3.ipynb: extract boundary points (cells adjacent to a
differently-classified neighbor), normalize into the unit square, count
occupied boxes at scales eps=1/2^k for k=2..8, fit log N(eps) vs log(1/eps)
with a line; the slope is the estimated box-counting dimension.
"""
import numpy as np


def extract_boundary_points(U0, V0, cls):
    """cls: integer grid (0=diverge,1=neg,2=pos) aligned with U0,V0 meshgrid."""
    diff_down = cls[1:, :] != cls[:-1, :]
    diff_right = cls[:, 1:] != cls[:, :-1]
    boundary = np.zeros_like(cls, dtype=bool)
    boundary[1:, :] |= diff_down
    boundary[:-1, :] |= diff_down
    boundary[:, 1:] |= diff_right
    boundary[:, :-1] |= diff_right
    pts = np.stack([U0[boundary], V0[boundary]], axis=1)
    return pts


def box_counting_dimension(points, k_range=range(2, 9)):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    scale = (maxs - mins).max()
    pts_norm = (points - mins) / scale

    eps_list, N_list = [], []
    for k in k_range:
        eps = 1.0 / (2 ** k)
        m = 2 ** k
        idx = np.clip((pts_norm / eps).astype(int), 0, m - 1)
        n_boxes = len(np.unique(idx, axis=0))
        eps_list.append(eps)
        N_list.append(n_boxes)

    X = np.log(1.0 / np.array(eps_list))
    Yv = np.log(np.array(N_list))
    coef = np.polyfit(X, Yv, 1)
    slope, intercept = coef
    Y_pred = np.polyval(coef, X)
    ss_res = np.sum((Yv - Y_pred) ** 2)
    ss_tot = np.sum((Yv - Yv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return X, Yv, slope, intercept, r2, eps_list, N_list


if __name__ == "__main__":
    from baseline import run_grid
    U0, V0, cls, _ = run_grid(resolution=1200)
    pts = extract_boundary_points(U0, V0, cls)
    print(f"boundary points found: {len(pts)}")
    X, Yv, slope, intercept, r2, eps_list, N_list = box_counting_dimension(pts)
    print(f"Estimated box-counting dimension: {slope:.4f} (paper reports 1.249)")
    print(f"R^2 = {r2:.4f}")

















































