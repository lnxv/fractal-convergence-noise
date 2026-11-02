"""Fits and figures for the timing experiment. Reads timing_<tag>.json / timing_<tag>_maps.npz."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from timing import power_law_fit

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "lines.markersize": 3.2, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
W = 6.5
RED, BLUE, GRAY, GOLD = (np.array(c) for c in ([0.75, 0.22, 0.17], [0.17, 0.44, 0.73], [0.93, 0.93, 0.93], [0.95, 0.77, 0.06]))
NAMES = {("init", None): "initial perturbation", ("early", 10): "noise in first 10 steps",
         ("early", 50): "noise in first 50 steps", ("persist", None): "persistent noise"}
COLORS = {("init", None): "#2c6fbb", ("early", 10): "#c0392b", ("early", 50): "#e67e22", ("persist", None): "#27ae60"}
RUNS = {"base": dict(ls="-", marker="o", label="T = 200, n = 64"),
        "long": dict(ls="--", marker="s", mfc="white", label="T = 600, n = 64"),
        "more": dict(ls=":", marker="^", label="T = 200, n = 192")}
QUANTS = ("uncertain_escape", "uncertain_selection")


def load():
    data = {}
    for f in sorted(glob.glob("timing_*.json")):
        if f.endswith("summary.json"):
            continue
        d = json.load(open(f))
        run = d["tag"].split("_")[0]
        for r in d["records"]:
            key = (r["mode"], r["K"])
            data.setdefault(run, {}).setdefault(key, []).append(r)
        data.setdefault(run, {})["_det"] = d["deterministic"]
        data[run]["_meta"] = {"T": d["T"], "trials": d["trials"], "res": d["res"]}
    return data


def fits(data):
    out = []
    for run, groups in data.items():
        for key, rows in groups.items():
            if key in ("_det", "_meta"):
                continue
            rows = sorted(rows, key=lambda r: r["level"])
            x = [r["level"] for r in rows]
            for q in QUANTS:
                y = [r[q] for r in rows]
                for rng_name, sl in (("all levels", slice(None)), ("without end points", slice(1, -1))):
                    f = power_law_fit(x[sl], y[sl])
                    out.append({"run": run, "mode": key[0], "K": key[1], "quantity": q, "range": rng_name,
                                "levels": [x[sl][0], x[sl][-1]], **f})
    return out


def fig_scaling(data):
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"wspace": 0.25})
    for a, q, title in zip(ax, QUANTS, ("(a) uncertain escape", "(b) uncertain minimizer selection")):
        for run, style in RUNS.items():
            for key, rows in data.get(run, {}).items():
                if key in ("_det", "_meta"):
                    continue
                rows = sorted(rows, key=lambda r: r["level"])
                st = dict(style)
                st["label"] = None
                a.loglog([r["level"] for r in rows], [max(r[q], 1e-5) for r in rows], color=COLORS[key], **st)
        a.set(xlabel=r"perturbation size ($\varepsilon$ for initial, $\sigma$ for step size)", ylabel="fraction of $W_1$")
        a.grid(True, which="major", alpha=0.25, lw=0.5)
        a.set_title(title, loc="left", fontsize=8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLORS[k], lw=2.2) for k in NAMES if any(k in data[r] for r in data)]
    lab = [NAMES[k] for k in NAMES if any(k in data[r] for r in data)]
    h += [Line2D([], [], color="k", **{k: v for k, v in s.items() if k != "label"}) for s in RUNS.values()]
    lab += [s["label"] for s in RUNS.values()]
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig("fig_timing_scaling.png", dpi=300)
    fig.savefig("fig_timing_scaling.pdf")
    plt.close(fig)


def rgb(esc, plus, minus, unres):
    img = plus[..., None] * RED + minus[..., None] * BLUE + esc[..., None] * GRAY + unres[..., None] * GOLD
    return np.clip(img, 0, 1)


def fig_maps():
    z = {}
    for f in glob.glob("timing_base_*_maps.npz"):
        with np.load(f) as d:
            z.update({k: d[k] for k in d.files})
    ext = (2.0, 2.7, -0.3, 0.3)
    rows = [("init", [0.005, 0.02], r"\varepsilon"), ("early10", [0.05, 0.2], r"\sigma"), ("persist", [0.05, 0.2], r"\sigma")]
    names = {"init": "initial\nperturbation", "early10": "first 10\nsteps", "persist": "persistent"}
    fig, ax = plt.subplots(3, 3, figsize=(W * 0.78, 4.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.06, "hspace": 0.25})
    det = rgb(z["det_escape"], z["det_plus"], z["det_minus"], z["det_unresolved"])
    for i, (name, lv, sym) in enumerate(rows):
        ax[i, 0].imshow(det, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
        ax[i, 0].set_title("deterministic GD", fontsize=7)
        ax[i, 0].set_ylabel(names[name] + "\n$v_0$", fontsize=7)
        for j, l in enumerate(lv):
            k = f"{name}_{l}"
            if k + "_escape" in z:
                ax[i, j + 1].imshow(rgb(z[k + "_escape"], z[k + "_plus"], z[k + "_minus"], z[k + "_unresolved"]),
                                    origin="lower", extent=ext, aspect="auto", interpolation="nearest")
            ax[i, j + 1].set_title(f"${sym}={l}$", fontsize=7)
    for a in ax[-1]:
        a.set_xlabel("$u_0$")
    fig.savefig("fig_timing_maps.png", dpi=300)
    plt.close(fig)


def main():
    data = load()
    fl = fits(data)
    json.dump({"fits": fl, "deterministic": {r: data[r]["_det"] for r in data},
               "meta": {r: data[r]["_meta"] for r in data},
               "records": {r: {f"{k[0]}{'' if k[1] is None else k[1]}": v for k, v in g.items() if k not in ("_det", "_meta")}
                           for r, g in data.items()}},
              open("timing_summary.json", "w"), indent=1)
    for f in fl:
        print(f"{f['run']:5s} {f['mode']:8s} K={str(f['K']):4s} {f['quantity']:20s} {f['range']:18s} "
              f"beta={f['exponent']:.3f} +/- {f['se']:.3f} R2={f['r2']:.4f}")
    fig_scaling(data)
    if glob.glob("timing_base_*_maps.npz"):
        fig_maps()
    print("analysis done")


if __name__ == "__main__":
    main()
