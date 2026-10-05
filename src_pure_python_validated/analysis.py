"""
Final quantitative analysis: box-counting dimension of the convergence
boundary as a function of noise level sigma, for both noise models.
Uses the parallelized grid generator for speed.
"""
from stochastic import run_grid_parallel
from boxcount import extract_boundary_points, box_counting_dimension
import json

def analyze(sigma, mode, region, resolution=400):
    grid, counts = run_grid_parallel(sigma, mode, region[0], region[1], resolution,
                                       out_prefix=f"analysis_{mode}_sigma_{sigma}")
    u_vals = [region[0][0] + (region[0][1]-region[0][0]) * i / (resolution-1) for i in range(resolution)]
    v_vals = [region[1][0] + (region[1][1]-region[1][0]) * i / (resolution-1) for i in range(resolution)]
    pts = extract_boundary_points(grid, u_vals, v_vals)
    if len(pts) < 20:
        return {'sigma': sigma, 'mode': mode, 'dimension': None, 'r2': None, 'n_boundary_pts': len(pts), 'counts': counts}
    X, Yv, slope, intercept, r2, eps_list, N_list = box_counting_dimension(pts)
    return {'sigma': sigma, 'mode': mode, 'dimension': slope, 'r2': r2, 'n_boundary_pts': len(pts), 'counts': counts}

if __name__ == "__main__":
    region = ((2.0, 2.7), (-0.3, 0.3))
    results = []
    for sigma in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]:
        r = analyze(sigma, 'grad_noise', region, resolution=350)
        print(r)
        results.append(r)
    for sigma in [0.01, 0.05, 0.1, 0.2]:
        r = analyze(sigma, 'step_noise', region, resolution=350)
        print(r)
        results.append(r)
    with open('final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to final_results.json")
































