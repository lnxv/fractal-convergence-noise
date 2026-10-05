"""
Reproduction of Liang & Montufar (arXiv:2509.25351) Figure 3 middle panel,
using the EXACT parameters and methodology from the authors' own released
code (https://anonymous.4open.science/r/chaos-matrix-factorization-07C5,
Fig3.ipynb), adapted to pure Python (no numpy, see note below) since this
sandbox cannot load compiled numpy extensions.

Loss: L(u,v) = 0.5*(u*v - y)^2 + 0.5*r*(u^2 + v^2)
GD:   u_{t+1} = u_t - h*(u_t*v_t - y)*v_t - r*u_t
      v_{t+1} = v_t - h*(u_t*v_t - y)*u_t - r*v_t
Params (from authors' code): y = 0.5, h (step size) = 1.0, r (reg) = 0.2
Grid: u,v in [-4, 4], N_step = 100 (we use 150 for a safety margin)
Convergence test (from authors' code): (u*v - y)^2 / 2 < lbound,
  lbound = 1e-6 + r*y - 0.5*r^2
Minimizers: (+sqrt(y-r), +sqrt(y-r)) and (-sqrt(y-r), -sqrt(y-r))
"""
import math

Y = 0.5
H = 1.0      # step size (eta), authors call it h
R = 0.2      # regularization (lambda), authors call it r
MIN_VAL = math.sqrt(Y - R)
LBOUND = 1e-6 + R * Y - 0.5 * R * R
N_STEP = 150
DIVERGE_THRESHOLD = 1e8

def gd_step(u, v):
    z = u * v - Y
    u_new = u - H * z * v - R * u
    v_new = v - H * z * u - R * v
    return u_new, v_new

def run_point(u0, v0, n_step=N_STEP):
    u, v = u0, v0
    for _ in range(n_step):
        u, v = gd_step(u, v)
        if abs(u) > DIVERGE_THRESHOLD or abs(v) > DIVERGE_THRESHOLD or math.isnan(u) or math.isnan(v):
            return None  # diverged / not in mask
    loss_term = (u * v - Y) ** 2 / 2.0
    if loss_term < LBOUND:
        return u  # converged; return final u (sign tells which minimizer)
    return None

def classify(u0, v0):
    """Returns 'pos', 'neg', or 'diverge' matching authors' coolwarm-by-sign coloring."""
    result = run_point(u0, v0)
    if result is None:
        return 'diverge'
    return 'pos' if result > 0 else 'neg'

def write_ppm(filename, grid, width, height):
    color_map = {
        'pos': (178, 24, 43),     # tab:red-ish (ColorBrewer RdBu)
        'neg': (33, 102, 172),    # tab:blue-ish
        'diverge': (255, 255, 255),  # unplotted/background = white, matching authors
    }
    with open(filename, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        buf = bytearray()
        for row in grid:
            for cls in row:
                buf.extend(color_map[cls])
        f.write(bytes(buf))

def run_grid(u_range, v_range, resolution, out_prefix):
    width = height = resolution
    u_vals = [u_range[0] + (u_range[1]-u_range[0]) * i / (width-1) for i in range(width)]
    v_vals = [v_range[0] + (v_range[1]-v_range[0]) * i / (height-1) for i in range(height)]
    grid = []
    counts = {'pos': 0, 'neg': 0, 'diverge': 0}
    for v in reversed(v_vals):
        row = []
        for u in u_vals:
            cls = classify(u, v)
            counts[cls] += 1
            row.append(cls)
        grid.append(row)
    ppm_path = f"{out_prefix}.ppm"
    write_ppm(ppm_path, grid, width, height)
    print(f"H={H} R={R} Y={Y}: counts={counts}")
    return grid, ppm_path, counts

if __name__ == "__main__":
    print(f"Minimizers: (+-{MIN_VAL:.4f}, +-{MIN_VAL:.4f})")
    print(f"lbound = {LBOUND:.6f}")
    run_grid((-4, 4), (-4, 4), resolution=500, out_prefix="v2_fullview")














































































