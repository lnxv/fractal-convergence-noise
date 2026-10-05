"""
Deterministic fractal convergence boundary, reproducing Liang & Montufar
(arXiv:2509.25351) Figure 3 (middle panel), using the authors' exact
parameters and convergence criterion from their released code:
https://anonymous.4open.science/r/chaos-matrix-factorization-07C5 (Fig3.ipynb)

Loss:   L(u,v) = 0.5*(u*v - y)^2 + 0.5*r*(u^2 + v^2)
GD:     u_{t+1} = u_t - h*(u_t*v_t - y)*v_t - r*u_t
        v_{t+1} = v_t - h*(u_t*v_t - y)*u_t - r*v_t
Params: y = 0.5, h (step size) = 1.0, r (regularization) = 0.2
Minimizers: (+sqrt(y-r), +sqrt(y-r)) and (-sqrt(y-r), -sqrt(y-r))

Convergence criterion (matching the authors' code): a point has converged if
(u*v - y)^2 / 2 < lbound, where lbound = 1e-6 + r*y - 0.5*r^2.
"""
import numpy as np
import matplotlib.pyplot as plt

Y = 0.5
H = 1.0
R = 0.2
MIN_VAL = np.sqrt(Y - R)
LBOUND = 1e-6 + R * Y - 0.5 * R ** 2
N_STEP = 150
DIVERGE_THRESHOLD = 1e8


def gd_step(u, v):
    z = u * v - Y
    return u - H * z * v - R * u, v - H * z * u - R * v


def run_grid(u_range=(-4, 4), v_range=(-4, 4), resolution=1200, n_step=N_STEP):
    """Vectorized grid evaluation: run every initialization in parallel via numpy."""
    np.seterr(over='ignore', invalid='ignore')
    u_vals = np.linspace(*u_range, resolution)
    v_vals = np.linspace(*v_range, resolution)
    U0, V0 = np.meshgrid(u_vals, v_vals)
    U, V = U0.copy(), V0.copy()

    for _ in range(n_step):
        U, V = gd_step(U, V)

    diverged = (np.abs(U) > DIVERGE_THRESHOLD) | (np.abs(V) > DIVERGE_THRESHOLD) | np.isnan(U) | np.isnan(V)
    loss_term = (U * V - Y) ** 2 / 2.0
    converged = (~diverged) & (loss_term < LBOUND)

    # classification grid: 0 = diverge/unconverged, 1 = neg minimizer, 2 = pos minimizer
    cls = np.zeros_like(U, dtype=np.int8)
    cls[converged & (U > 0)] = 2
    cls[converged & (U <= 0)] = 1
    return U0, V0, cls, U  # also return final U for sign-based coloring


def plot_baseline(out_path="../results/figures/baseline_deterministic.png", resolution=1200):
    U0, V0, cls, Ufinal = run_grid(resolution=resolution)
    fig, ax = plt.subplots(figsize=(7, 7))
    mask = cls > 0
    ax.scatter(U0[mask], V0[mask], c=Ufinal[mask], cmap='RdBu_r', s=0.3, marker='.')
    ax.scatter(MIN_VAL, MIN_VAL, color='black', marker='^', s=80, zorder=5, label='Minimizer (pos)')
    ax.scatter(-MIN_VAL, -MIN_VAL, color='black', marker='o', s=80, zorder=5, label='Minimizer (neg)')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'Deterministic convergence boundary (h={H}, r={R}, y={Y})')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    counts = {'diverge': int((cls == 0).sum()), 'neg': int((cls == 1).sum()), 'pos': int((cls == 2).sum())}
    print(f"counts: {counts}")
    return counts


if __name__ == "__main__":
    print(f"Minimizers: (+-{MIN_VAL:.4f}, +-{MIN_VAL:.4f}); lbound={LBOUND:.6f}")
    plot_baseline()































































