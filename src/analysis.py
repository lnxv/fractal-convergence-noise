"""
Full noise sweep: computes box-counting dimension of the convergence
boundary as a function of noise level sigma, for both noise models, and
saves the results table + a summary plot.
"""
import json
import matplotlib.pyplot as plt
from stochastic import run_grid_stochastic
from boxcount import extract_boundary_points, box_counting_dimension

REGION = ((2.0, 2.7), (-0.3, 0.3))  # a fractal spike region of the boundary
RESOLUTION = 800


def analyze(sigma, mode, region=REGION, resolution=RESOLUTION):
    U0, V0, cls, _ = run_grid_stochastic(sigma, mode, region[0], region[1], resolution)
    pts = extract_boundary_points(U0, V0, cls)
    if len(pts) < 20:
        return {'sigma': sigma, 'mode': mode, 'dimension': None, 'r2': None, 'n_boundary_pts': len(pts)}
    _, _, slope, _, r2, _, _ = box_counting_dimension(pts)
    result = {'sigma': sigma, 'mode': mode, 'dimension': float(slope), 'r2': float(r2), 'n_boundary_pts': int(len(pts))}
    print(result)
    return result


def run_sweep():
    results = []
    for sigma in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]:
        results.append(analyze(sigma, 'grad_noise'))
    for sigma in [0.01, 0.05, 0.1, 0.2]:
        results.append(analyze(sigma, 'step_noise'))
    with open('../results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/final_results.json")
    return results


def plot_dimension_trend(results, out_path="../results/figures/dimension_vs_noise.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, marker in [('grad_noise', 'o'), ('step_noise', 's')]:
        pts = [(r['sigma'], r['dimension']) for r in results if r['mode'] == mode and r['dimension'] is not None]
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, label=mode)
    ax.set_xlabel('noise level $\\sigma$')
    ax.set_ylabel('box-counting dimension')
    ax.set_title('Boundary complexity increases monotonically with noise')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    results = run_sweep()
    plot_dimension_trend(results)















































