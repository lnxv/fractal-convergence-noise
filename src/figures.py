"""Figures for the report (matplotlib)."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight"})
BLUES = plt.get_cmap("Blues")


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=200)
    plt.close(fig)


def baseline_figure(out, labels):
    cmap = ListedColormap(["#eeeeee", "#c0392b", "#2c6fbb", "#8e44ad", "#f1c40f"])
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(labels, origin="lower", extent=(-4, 4, -4, 4), cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax.set(xlabel="$u_0$", ylabel="$v_0$", title="Deterministic GD ($\\eta=1$, $\\lambda=0.2$, $y=0.5$)")
    _save(fig, out, "fig_baseline.png")


def probability_maps(out, maps, windows):
    b = windows["W1"]
    panels = [("deterministic", 0.0, "deterministic GD"), ("step", 0.02, "step noise $\\sigma=0.02$"),
              ("step", 0.05, "step noise $\\sigma=0.05$"), ("step", 0.2, "step noise $\\sigma=0.2$")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
    for ax, (mode, sigma, title) in zip(axes, panels):
        im = ax.imshow(maps[("W1", mode, sigma)], origin="lower", extent=b, cmap=BLUES, vmin=0, vmax=1,
                       interpolation="nearest", aspect="auto")
        ax.set(title=title, xlabel="$u_0$")
    axes[0].set_ylabel("$v_0$")
    fig.colorbar(im, ax=axes, shrink=0.85, label="P(no escape in 200 steps)")
    _save(fig, out, "fig_probability_maps.png")


def coupling_figure(out, data):
    U, V, common, indep, prob = data
    ext = (U.min(), U.max(), V.min(), V.max())
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
    for ax, img, title in zip(axes, [~common, ~indep, prob],
                              ["one shared noise path", "independent noise per pixel", "probability, 64 runs/pixel"]):
        ax.imshow(img.astype(float), origin="lower", extent=ext, cmap=BLUES, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set(title=title, xlabel="$u_0$")
    axes[0].set_ylabel("$v_0$")
    fig.suptitle("Step noise $\\sigma=0.1$: speckle from independent draws is not geometry", y=1.02)
    _save(fig, out, "fig_coupling.png")


def _pm(x):
    return f"{x:.3f}" if x < 0.01 else f"{x:.2f}"


def scaling_figure(out, records, fits, unc):
    colors = {"W1": "#2c6fbb", "W2": "#c0392b"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for f in fits:
        rows = [r for r in records if r["window"] == f["window"] and r["mode"] == f["mode"]]
        x = np.array([r["sigma"] for r in rows]); y = np.array([r["band_fraction"] for r in rows])
        ls, mfc = ("-", colors[f["window"]]) if f["mode"] == "step" else ("--", "white")
        ax.loglog(x, y, ls, marker="o", color=colors[f["window"]], mfc=mfc,
                  label=f"{f['mode']} noise, {f['window']}: $\\beta={f['exponent']:.2f}\\pm{_pm(f['se'])}$")
    for u in unc:
        x = np.array([r["eps"] for r in u["rows"]]); y = np.array([r["uncertain_fraction"] for r in u["rows"]])
        ax.loglog(x, y, ":", marker="s", ms=4, color=colors[u["window"]], alpha=0.7,
                  label=f"deterministic, perturbation $\\varepsilon$, {u['window']}: $2-D={u['exponent']:.2f}\\pm{_pm(u['se'])}$")
    ax.set(xlabel="noise level $\\sigma$  (or perturbation $\\varepsilon$)", ylabel="fraction of uncertain initial conditions")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.9)
    _save(fig, out, "fig_scaling.png")


def make_all(out, base_labels, maps, records, fits, unc, coupling_data, windows):
    baseline_figure(out, base_labels)
    probability_maps(out, maps, windows)
    coupling_figure(out, coupling_data)
    scaling_figure(out, records, fits, unc)
