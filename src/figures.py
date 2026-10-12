"""All figures, built from the files written by the experiment scripts (run after them)."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import OUT, WINDOWS

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "savefig.bbox": "tight"})
RED, BLUE, GRAY, GOLD = (np.array(c) for c in ([0.75, 0.22, 0.17], [0.17, 0.44, 0.73], [0.93, 0.93, 0.93], [0.95, 0.77, 0.06]))
WCOL = {"W1": "#2c6fbb", "W2": "#c0392b"}
MSTYLE = {"step": dict(ls="-", marker="o"), "additive": dict(ls="--", marker="o", mfc="white"), "sgd": dict(ls="-.", marker="^")}
GCOL = {"step": "#2c6fbb", "sgd": "#c0392b", "additive": "#27ae60"}
EXT = (-4, 4, -4, 4)


def _json(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


def _pm(x):
    return f"{x:.3f}" if x < 0.01 else f"{x:.2f}"


def rgb(survive, plus, minus):
    """Red = P(+ minimizer), blue = P(- minimizer), gray = P(escape), gold = survives near neither."""
    other = np.clip(survive - plus - minus, 0, 1)
    img = plus[..., None] * RED + minus[..., None] * BLUE + (1 - survive)[..., None] * GRAY + other[..., None] * GOLD
    return np.clip(img, 0, 1)


def load_windows():
    records, fits, maps = [], [], {}
    for f in sorted(glob.glob(os.path.join(OUT, "windows_*.json"))):
        d = json.load(open(f))
        records += [r for r in d["records"] if r["mode"] != "deterministic" or not any(
            q["mode"] == "deterministic" and q["window"] == r["window"] for q in records)]
        fits += d["fits"]
    for f in sorted(glob.glob(os.path.join(OUT, "maps_windows_*.npz"))):
        with np.load(f) as z:
            maps.update({k: z[k] for k in z.files})
    return records, fits, maps


def fig_reproduction(rep):
    z = np.load(os.path.join(OUT, "maps_reproduce.npz"))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    cmap = ListedColormap([GRAY, RED, BLUE, [0.56, 0.27, 0.68], GOLD])
    ax[0].imshow(z["baseline_labels"], origin="lower", extent=EXT, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax[0].set(xlabel="$u_0$", ylabel="$v_0$", title="(a) convergence region, 401$\\times$401 grid")
    pts = z["boundary_points"]
    ax[1].scatter(pts[:, 0], pts[:, 1], s=0.05, c="#c0392b", rasterized=True)
    xx = np.linspace(-2, 3, 200)
    ax[1].plot(xx, 2 * np.abs(xx), color="gray", lw=1)
    ax[1].set(xlim=(-2, 3), ylim=(0, 10), xlabel="$uv$", ylabel="$u^2+v^2$", title="(b) projected boundary $T(\\partial D''_\\eta)$")
    d = rep["projected_boundary_dimension"]
    x = 1 / np.array(d["eps"])
    ax[2].loglog(x, d["counts"], "o", color="#2c6fbb")
    ax[2].loglog(x, np.exp(d["intercept"]) * x ** d["exponent"], "--", color="#555")
    ax[2].set(xlabel="$1/\\varepsilon$", ylabel="occupied boxes $N(\\varepsilon)$",
              title=f"(c) box-counting dimension {d['exponent']:.3f} (paper: 1.249)")
    _save(fig, "fig_reproduction.png")


def fig_global(glob_res):
    z = np.load(os.path.join(OUT, "maps_global.npz"))
    cols = {"step": [0.1, 0.4, 0.8], "sgd": [0.05, 0.2, 0.4], "additive": [0.02, 0.1, 0.2]}
    fig, ax = plt.subplots(3, 4, figsize=(10, 7.6), sharex=True, sharey=True)
    det = rgb(z["deterministic_0_survive"], z["deterministic_0_plus"], z["deterministic_0_minus"])
    for i, mode in enumerate(cols):
        ax[i, 0].imshow(det, origin="lower", extent=EXT, interpolation="nearest")
        ax[i, 0].set_title("deterministic GD")
        ax[i, 0].set_ylabel(f"{mode} noise\n$v_0$")
        for j, s in enumerate(cols[mode]):
            key = f"{mode}_{s}"
            ax[i, j + 1].imshow(rgb(z[key + "_survive"], z[key + "_plus"], z[key + "_minus"]), origin="lower",
                                extent=EXT, interpolation="nearest")
            ax[i, j + 1].set_title(f"{mode}, $\\sigma={s}$")
    for a in ax[-1]:
        a.set_xlabel("$u_0$")
    fig.suptitle("Outcome probabilities over the whole domain (red: + minimizer, blue: - minimizer, gray: escape, "
                 "gold: near neither)", y=0.995, fontsize=9)
    _save(fig, "fig_global.png")

    recs = glob_res["records"]
    det_rec = recs[0]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for mode in ("step", "sgd", "additive"):
        rr = [r for r in recs if r["mode"] == mode]
        s = [r["sigma"] for r in rr]
        ax[0].semilogx(s, [r["survival"] for r in rr], marker="o", color=GCOL[mode], label=mode)
        ax[1].semilogx(s, [min(1.0, r["selection_fraction"] / max(r["survival"], 1e-9)) for r in rr], marker="o",
                       color=GCOL[mode], label=mode)
        ax[2].semilogx(s, [(r["plus"] + r["minus"]) / max(r["survival"], 1e-9) for r in rr], marker="o", color=GCOL[mode], label=mode)
    ax[0].axhline(det_rec["survival"], color="gray", ls=":", label="deterministic")
    ax[0].set(xlabel="noise level $\\sigma$", ylabel="fraction of domain", title="(a) does not escape in 200 steps")
    ax[1].set(xlabel="noise level $\\sigma$", ylabel="fraction of surviving region", title="(b) minimizer choice uncertain")
    ax[2].set(xlabel="noise level $\\sigma$", ylabel="fraction of surviving runs", title="(c) ends near a minimizer")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    _save(fig, "fig_global_curves.png")


def fig_window_maps(maps):
    b = WINDOWS["W1"]
    rows = {"step": [0.02, 0.05, 0.2], "sgd": [0.005, 0.01, 0.04]}
    fig, ax = plt.subplots(2, 4, figsize=(10, 5.3), sharex=True, sharey=True)
    det = rgb(maps["W1_deterministic_0_survive"], maps["W1_deterministic_0_plus"], maps["W1_deterministic_0_minus"])
    for i, mode in enumerate(rows):
        ax[i, 0].imshow(det, origin="lower", extent=b, interpolation="nearest", aspect="auto")
        ax[i, 0].set_title("deterministic GD")
        ax[i, 0].set_ylabel(f"{mode} noise\n$v_0$")
        for j, s in enumerate(rows[mode]):
            k = f"W1_{mode}_{s}"
            ax[i, j + 1].imshow(rgb(maps[k + "_survive"], maps[k + "_plus"], maps[k + "_minus"]), origin="lower",
                                extent=b, interpolation="nearest", aspect="auto")
            ax[i, j + 1].set_title(f"{mode}, $\\sigma={s}$")
    for a in ax[-1]:
        a.set_xlabel("$u_0$")
    _save(fig, "fig_window_maps.png")


def fig_scaling(records, fits, rep):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, key, ukey, title in ((ax[0], "band_fraction", "escape_flip", "(a) whether the run escapes"),
                                (ax[1], "selection_fraction", "outcome_flip", "(b) which minimizer is selected")):
        for f in [f for f in fits if f["quantity"] == key]:
            rr = [r for r in records if r["window"] == f["window"] and r["mode"] == f["mode"]]
            a.loglog([r["sigma"] for r in rr], [r[key] for r in rr], color=WCOL[f["window"]], **MSTYLE[f["mode"]], ms=4,
                     label=f"{f['mode']}, {f['window']}: {f['exponent']:.2f}$\\pm${_pm(f['se'])}")
        for u in rep["uncertainty"]:
            g = u[ukey]
            a.loglog([r["eps"] for r in u["rows"]], [r[ukey] for r in u["rows"]], ":", marker="s", ms=3, alpha=0.7,
                     color=WCOL[u["window"]], label=f"deterministic $\\varepsilon$, {u['window']}: {g['exponent']:.2f}$\\pm${_pm(g['se'])}")
        a.set(xlabel="noise level $\\sigma$ (or perturbation $\\varepsilon$)", ylabel="fraction of uncertain initial conditions",
              title=title)
        a.grid(True, which="both", alpha=0.2)
        a.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    _save(fig, "fig_scaling.png")


def fig_coupling():
    z = np.load(os.path.join(OUT, "maps_controls.npz"))
    ext = (z["U"].min(), z["U"].max(), z["V"].min(), z["V"].max())
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.0), sharey=True)
    for a, img, t in zip(ax, [~z["common"], ~z["independent"], z["probability"]],
                         ["one shared noise path", "independent noise per pixel", "probability, 64 runs per pixel"]):
        a.imshow(img.astype(float), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        a.set(title=t, xlabel="$u_0$")
    ax[0].set_ylabel("$v_0$")
    fig.suptitle("Step noise $\\sigma=0.1$ in W1: speckle from independent draws is not geometry", y=1.03, fontsize=9)
    _save(fig, "fig_coupling.png")


def main():
    rep = _json("reproduce.json")
    records, fits, maps = load_windows()
    fig_reproduction(rep)
    fig_global(_json("global.json"))
    fig_window_maps(maps)
    fig_scaling(records, fits, rep)
    fig_coupling()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
