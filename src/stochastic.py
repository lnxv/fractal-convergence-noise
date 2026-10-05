"""
Stochastic extension of the deterministic baseline (baseline.py): does noise
smooth or roughen the fractal convergence boundary? Two noise models, fully
vectorized with numpy so every grid point and every GD step is a single
array operation (no Python-level loop over pixels).

  (A) grad_noise: additive i.i.d. Gaussian noise on the gradient each step,
      mimicking SGD gradient noise.
  (B) step_noise:  step size h_t ~ Uniform[h*(1-sigma), h*(1+sigma)] each
      step, resampled independently per grid point and per iteration.
"""
import numpy as np
import matplotlib.pyplot as plt
from baseline import Y, H, R, LBOUND, DIVERGE_THRESHOLD

N_STEP = 100  # matches the authors' N_step exactly


def run_grid_stochastic(sigma, mode, u_range, v_range, resolution, n_step=N_STEP, seed=0):
    """Vectorized noisy GD: every (u0, v0) in the grid evolves simultaneously."""
    rng = np.random.default_rng(seed)
    np.seterr(over='ignore', invalid='ignore')

    u_vals = np.linspace(*u_range, resolution)
    v_vals = np.linspace(*v_range, resolution)
    U0, V0 = np.meshgrid(u_vals, v_vals)
    U, V = U0.copy(), V0.copy()

    for _ in range(n_step):
        z = U * V - Y
        g_u, g_v = z * V, z * U
        if mode == 'grad_noise':
            U = U - H * (g_u + sigma * rng.standard_normal(U.shape)) - R * U
            V = V - H * (g_v + sigma * rng.standard_normal(V.shape)) - R * V
        elif mode == 'step_noise':
            h_t = H * (1.0 + sigma * rng.uniform(-1.0, 1.0, size=U.shape))
            U = U - h_t * g_u - R * U
            V = V - h_t * g_v - R * V
        else:
            raise ValueError(mode)

    diverged = (np.abs(U) > DIVERGE_THRESHOLD) | (np.abs(V) > DIVERGE_THRESHOLD) | np.isnan(U) | np.isnan(V)
    loss_term = (U * V - Y) ** 2 / 2.0
    converged = (~diverged) & (loss_term < LBOUND)

    cls = np.zeros_like(U, dtype=np.int8)  # 0 = diverge, 1 = neg, 2 = pos
    cls[converged & (U > 0)] = 2
    cls[converged & (U <= 0)] = 1
    return U0, V0, cls, U


def plot_noise_sweep(sigmas, mode, region, resolution=600, out_dir="../results/figures"):
    fig, axs = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4.2))
    if len(sigmas) == 1:
        axs = [axs]
    for ax, sigma in zip(axs, sigmas):
        U0, V0, cls, Ufinal = run_grid_stochastic(sigma, mode, region[0], region[1], resolution)
        mask = cls > 0
        ax.scatter(U0[mask], V0[mask], c=Ufinal[mask], cmap='RdBu_r', s=0.5, marker='.')
        ax.set_title(f"$\\sigma$={sigma}", fontsize=11)
        ax.set_xlabel('u')
        ax.set_aspect('equal')
        counts = {'diverge': int((cls == 0).sum()), 'neg': int((cls == 1).sum()), 'pos': int((cls == 2).sum())}
        print(f"mode={mode} sigma={sigma}: {counts}")
    axs[0].set_ylabel('v')
    plt.tight_layout()
    out_path = f"{out_dir}/noise_sweep_{mode}.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # zoom into one of the fractal spike regions (found by boundary search,
    # see analysis.py) for high-resolution detail
    region = ((2.0, 2.7), (-0.3, 0.3))
    plot_noise_sweep([0.0, 0.005, 0.01, 0.02, 0.05], 'grad_noise', region)
    plot_noise_sweep([0.0, 0.02, 0.05, 0.1, 0.2], 'step_noise', region)

































































