"""
Box-counting fractal dimension estimator, matching the authors' method in
Fig3.ipynb: extract boundary points (grid cells adjacent to a differently
classified neighbor), normalize into the unit square, count occupied boxes
at several scales eps=1/2^k, then fit log N(eps) vs log(1/eps) with a line;
slope = estimated box-counting dimension. Pure Python (no numpy/sklearn).
"""
import math

def extract_boundary_points(grid, u_vals, v_vals):
    """grid[row][col] in {'pos','neg','diverge'}; rows correspond to v
    descending (see baseline_v2.run_grid). Returns list of (u,v) boundary pts."""
    height = len(grid)
    width = len(grid[0])
    pts = []
    for r in range(height):
        for c in range(width):
            cls = grid[r][c]
            is_boundary = False
            for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < height and 0 <= cc < width and grid[rr][cc] != cls:
                    is_boundary = True
                    break
            if is_boundary:
                pts.append((u_vals[c], v_vals[height-1-r]))  # undo the v-reversal from run_grid
    return pts

def normalize(points):
    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)
    scale = max(umax - umin, vmax - vmin)
    if scale == 0:
        scale = 1.0
    return [((u - umin)/scale, (v - vmin)/scale) for u, v in points]

def box_counting_dimension(points, k_range=range(2, 9)):
    pts_norm = normalize(points)
    eps_list, N_list = [], []
    for k in k_range:
        eps = 1.0 / (2 ** k)
        m = 2 ** k
        boxes = set()
        for u, v in pts_norm:
            iu = min(int(u / eps), m - 1)
            iv = min(int(v / eps), m - 1)
            boxes.add((iu, iv))
        eps_list.append(eps)
        N_list.append(len(boxes))
    X = [math.log(1.0/e) for e in eps_list]
    Yv = [math.log(n) for n in N_list]
    # least-squares linear fit
    n = len(X)
    mean_x = sum(X)/n
    mean_y = sum(Yv)/n
    num = sum((X[i]-mean_x)*(Yv[i]-mean_y) for i in range(n))
    den = sum((X[i]-mean_x)**2 for i in range(n))
    slope = num/den
    intercept = mean_y - slope*mean_x
    # R^2
    ss_res = sum((Yv[i] - (slope*X[i]+intercept))**2 for i in range(n))
    ss_tot = sum((Yv[i]-mean_y)**2 for i in range(n))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    return X, Yv, slope, intercept, r2, eps_list, N_list

if __name__ == "__main__":
    from baseline import run_grid
    grid, _, _ = run_grid((-4, 4), (-4, 4), resolution=500, out_prefix="v2_fullview_boxcount")
    u_vals = [-4 + 8 * i / 499 for i in range(500)]
    v_vals = [-4 + 8 * i / 499 for i in range(500)]
    pts = extract_boundary_points(grid, u_vals, v_vals)
    print(f"boundary points found: {len(pts)}")
    X, Yv, slope, intercept, r2, eps_list, N_list = box_counting_dimension(pts)
    print(f"Estimated box-counting dimension: {slope:.3f} (paper reports 1.249)")
    print(f"R^2 = {r2:.4f}")
    for e, n in zip(eps_list, N_list):
        print(f"  eps={e:.5f}  N(eps)={n}")











































































