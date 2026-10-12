"""All figures, built from the files written by the experiment scripts (run after them).

Figures are sized for a 6.5 in text width and saved as PDF (for the paper) and PNG.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import OUT, WINDOWS

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.3,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "lines.markersize": 3.2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
RED, BLUE, GRAY, GOLD = (np.array(c) for c in ([0.75, 0.22, 0.17], [0.17, 0.44, 0.73], [0.93, 0.93, 0.93], [0.95, 0.77, 0.06]))
WCOL = {"W1": "#2c6fbb", "W2": "#c0392b"}
MSTYLE = {"step": dict(ls="-", marker="o"), "additive": dict(ls="--", marker="o", mfc="white"), "sgd": dict(ls="-.", marker="^")}
MNAME = {"step": "step", "additive": "additive", "sgd": "SGD"}
GCOL = {"step": "#2c6fbb", "sgd": "#c0392b", "additive": "#27ae60"}
EXT = (-4, 4, -4, 4)
W = 6.5  # text width in inches


def _json(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def _label(ax, text):
    ax.set_title(text, loc="left", fontsize=8)


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
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.15), gridspec_kw={"width_ratios": [1, 1.05, 1.05], "wspace": 0.42})
    cmap = ListedColormap([GRAY, RED, BLUE, [0.56, 0.27, 0.68], GOLD])
    ax[0].imshow(z["baseline_labels"], origin="lower", extent=EXT, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax[0].set(xlabel="$u_0$", ylabel="$v_0$")
    _label(ax[0], "(a)")
    pts = z["boundary_points"]
    ax[1].scatter(pts[:, 0], pts[:, 1], s=0.08, c="#c0392b", lw=0, rasterized=True)
    xx = np.linspace(-2, 3, 200)
    ax[1].plot(xx, 2 * np.abs(xx), color="0.5", lw=0.7)
    ax[1].set(xlim=(-2, 3), ylim=(0, 10), xlabel="$uv$", ylabel="$u^2+v^2$")
    _label(ax[1], "(b)")
    d = rep["projected_boundary_dimension"]
    x = 1 / np.array(d["eps"])
    ax[2].loglog(x, d["counts"], "o", color="#2c6fbb", ms=3.5)
    ax[2].loglog(x, np.exp(d["intercept"]) * x ** d["exponent"], "--", color="0.3", lw=0.8,
                 label=f"slope ${d['exponent']:.3f}$, $R^2={d['r2']:.4f}$")
    ax[2].set(xlabel="$1/\\varepsilon$", ylabel="$N(\\varepsilon)$")
    ax[2].legend(loc="lower right", frameon=False)
    _label(ax[2], "(c)")
    _save(fig, "fig_reproduction")


def fig_window_maps(maps):
    b = WINDOWS["W1"]
    rows = {"step": [0.02, 0.05, 0.2], "sgd": [0.005, 0.01, 0.04]}
    fig, ax = plt.subplots(2, 4, figsize=(W, 3.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08, "hspace": 0.22})
    det = rgb(maps["W1_deterministic_0_survive"], maps["W1_deterministic_0_plus"], maps["W1_deterministic_0_minus"])
    for i, mode in enumerate(rows):
        ax[i, 0].imshow(det, origin="lower", extent=b, interpolation="nearest", aspect="auto")
        ax[i, 0].set_title("$\\sigma=0$")
        ax[i, 0].set_ylabel(f"{MNAME[mode]}\n$v_0$")
        for j, s in enumerate(rows[mode]):
            k = f"W1_{mode}_{s}"
            ax[i, j + 1].imshow(rgb(maps[k + "_survive"], maps[k + "_plus"], maps[k + "_minus"]), origin="lower",
                                extent=b, interpolation="nearest", aspect="auto")
            ax[i, j + 1].set_title(f"$\\sigma={s}$")
    for a in ax[-1]:
        a.set_xlabel("$u_0$")
    _save(fig, "fig_window_maps")


def fig_scaling(records, fits, rep):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw={"wspace": 0.25})
    for a, key, ukey, lab in ((ax[0], "band_fraction", "escape_flip", "(a) escape"),
                              (ax[1], "selection_fraction", "outcome_flip", "(b) minimizer selection")):
        for f in [f for f in fits if f["quantity"] == key]:
            rr = [r for r in records if r["window"] == f["window"] and r["mode"] == f["mode"]]
            a.loglog([r["sigma"] for r in rr], [r[key] for r in rr], color=WCOL[f["window"]], **MSTYLE[f["mode"]])
        for u in rep["uncertainty"]:
            a.loglog([r["eps"] for r in u["rows"]], [r[ukey] for r in u["rows"]], ":", marker="s", ms=2.5, alpha=0.75,
                     color=WCOL[u["window"]])
        a.set(xlabel="$\\sigma$ (noise) or $\\varepsilon$ (perturbation)", ylabel="uncertain fraction")
        a.grid(True, which="major", alpha=0.25, lw=0.5)
        _label(a, lab)
    h = [Line2D([], [], color="k", **MSTYLE[m]) for m in ("step", "sgd", "additive")]
    h += [Line2D([], [], color="k", ls=":", marker="s", ms=2.5), Line2D([], [], color=WCOL["W1"], lw=2.5), Line2D([], [], color=WCOL["W2"], lw=2.5)]
    fig.legend(h, ["step noise", "SGD", "additive noise", "deterministic GD, perturbation $\\varepsilon$", "window W1", "window W2"],
               loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.4, columnspacing=1.2)
    _save(fig, "fig_scaling")


def fig_global(glob_res):
    z = np.load(os.path.join(OUT, "maps_global.npz"))
    cols = {"step": [0.1, 0.4, 0.8], "sgd": [0.05, 0.2, 0.4], "additive": [0.02, 0.1, 0.2]}
    fig, ax = plt.subplots(3, 4, figsize=(W, 4.9), sharex=True, sharey=True, gridspec_kw={"wspace": 0.06, "hspace": 0.2})
    det = rgb(z["deterministic_0_survive"], z["deterministic_0_plus"], z["deterministic_0_minus"])
    for i, mode in enumerate(cols):
        ax[i, 0].imshow(det, origin="lower", extent=EXT, interpolation="nearest")
        ax[i, 0].set_title("$\\sigma=0$")
        ax[i, 0].set_ylabel(f"{MNAME[mode]}\n$v_0$")
        for j, s in enumerate(cols[mode]):
            key = f"{mode}_{s}"
            ax[i, j + 1].imshow(rgb(z[key + "_survive"], z[key + "_plus"], z[key + "_minus"]), origin="lower",
                                extent=EXT, interpolation="nearest")
            ax[i, j + 1].set_title(f"$\\sigma={s}$")
    for a in ax[-1]:
        a.set_xlabel("$u_0$")
    _save(fig, "fig_global")

    recs = glob_res["records"]
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.0), gridspec_kw={"wspace": 0.42})
    for mode in ("step", "sgd", "additive"):
        rr = [r for r in recs if r["mode"] == mode]
        s = [r["sigma"] for r in rr]
        kw = dict(marker="o", color=GCOL[mode], label=MNAME[mode])
        ax[0].semilogx(s, [r["survival"] for r in rr], **kw)
        ax[1].semilogx(s, [min(1.0, r["selection_fraction"] / max(r["survival"], 1e-9)) for r in rr], **kw)
        ax[2].semilogx(s, [(r["plus"] + r["minus"]) / max(r["survival"], 1e-9) for r in rr], **kw)
    ax[0].axhline(recs[0]["survival"], color="0.5", ls=":", lw=0.8)
    ax[0].set(xlabel="$\\sigma$", ylabel="non-escaping fraction")
    ax[1].set(xlabel="$\\sigma$", ylabel="uncertain selection")
    ax[2].set(xlabel="$\\sigma$", ylabel="ends near a minimizer")
    for a, l in zip(ax, ("(a)", "(b)", "(c)")):
        a.grid(alpha=0.25, lw=0.5)
        _label(a, l)
    ax[0].legend(frameon=False, loc="upper left")
    _save(fig, "fig_global_curves")


def fig_coupling():
    z = np.load(os.path.join(OUT, "maps_controls.npz"))
    ext = (z["U"].min(), z["U"].max(), z["V"].min(), z["V"].max())
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, img, t in zip(ax, [~z["common"], ~z["independent"], z["probability"]], ["(a)", "(b)", "(c)"]):
        a.imshow(img.astype(float), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        a.set_xlabel("$u_0$")
        _label(a, t)
    ax[0].set_ylabel("$v_0$")
    _save(fig, "fig_coupling")


def main():
    rep = _json("reproduce.json")
    records, fits, maps = load_windows()
    fig_reproduction(rep)
    fig_window_maps(maps)
    fig_scaling(records, fits, rep)
    fig_global(_json("global.json"))
    fig_coupling()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
