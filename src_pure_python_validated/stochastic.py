"""
Multiprocessing-parallelized stochastic noise sweep (16 cores available).
Pure Python (sandbox blocks compiled numpy extensions), but parallelized
across rows to make the sweep tractable.
"""
import math
from multiprocessing import Pool
from baseline import Y, H, R, LBOUND, DIVERGE_THRESHOLD

N_STEP = 100  # matches authors' N_step exactly

def lcg_next(state):
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF

def gauss_from_lcg(state):
    state, u1 = lcg_next(state)
    u1 = max(u1, 1e-12)
    state, u2 = lcg_next(state)
    return state, math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

def classify_stochastic(u0, v0, sigma, mode, seed):
    state = seed & 0xFFFFFFFF
    u, v = u0, v0
    for _ in range(N_STEP):
        z = u * v - Y
        g_u = z * v
        g_v = z * u
        if mode == 'grad_noise':
            state, n1 = gauss_from_lcg(state)
            state, n2 = gauss_from_lcg(state)
            u_new = u - H * (g_u + sigma * n1) - R * u
            v_new = v - H * (g_v + sigma * n2) - R * v
        else:  # step_noise
            state, uu = lcg_next(state)
            h_t = H * (1.0 + sigma * (2 * uu - 1.0))
            u_new = u - h_t * g_u - R * u
            v_new = v - h_t * g_v - R * v
        u, v = u_new, v_new
        if abs(u) > DIVERGE_THRESHOLD or abs(v) > DIVERGE_THRESHOLD or math.isnan(u) or math.isnan(v):
            return 'diverge'
    loss_term = (u * v - Y) ** 2 / 2.0
    if loss_term < LBOUND:
        return 'pos' if u > 0 else 'neg'
    return 'diverge'

def classify_row(args):
    v, u_vals, sigma, mode, seed_row_base = args
    row = []
    for i, u in enumerate(u_vals):
        cls = classify_stochastic(u, v, sigma, mode, seed_row_base + i)
        row.append(cls)
    return row

def run_grid_parallel(sigma, mode, u_range, v_range, resolution, out_prefix, seed_base=777, nproc=16):
    width = height = resolution
    u_vals = [u_range[0] + (u_range[1]-u_range[0]) * i / (width-1) for i in range(width)]
    v_vals = [v_range[0] + (v_range[1]-v_range[0]) * i / (height-1) for i in range(height)]
    v_vals_rev = list(reversed(v_vals))

    tasks = [(v, u_vals, sigma, mode, seed_base + row_idx*width) for row_idx, v in enumerate(v_vals_rev)]
    with Pool(nproc) as pool:
        grid = pool.map(classify_row, tasks)

    counts = {'pos': 0, 'neg': 0, 'diverge': 0}
    for row in grid:
        for cls in row:
            counts[cls] += 1

    write_ppm(f"{out_prefix}.ppm", grid, width, height)
    print(f"sigma={sigma} mode={mode}: counts={counts}")
    return grid, counts

def write_ppm(filename, grid, width, height):
    color_map = {'pos': (178, 24, 43), 'neg': (33, 102, 172), 'diverge': (255, 255, 255)}
    with open(filename, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        buf = bytearray()
        for row in grid:
            for cls in row:
                buf.extend(color_map[cls])
        f.write(bytes(buf))

if __name__ == "__main__":
    import time
    region = ((2.0, 2.7), (-0.3, 0.3))
    for sigma in [0.0, 0.002, 0.008, 0.02, 0.05]:
        t0 = time.time()
        run_grid_parallel(sigma, 'grad_noise', region[0], region[1],
                           resolution=400, out_prefix=f"fast_grad_sigma_{sigma}")
        print(f"  took {time.time()-t0:.1f}s")
















































































