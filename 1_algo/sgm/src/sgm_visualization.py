# =============================================================================
# sgm_visualization.py  --  Plots for the SGM experiment
# =============================================================================
# APPROACH
# --------
# Written to  1_algo/sgm/data/ :
#   comparison_<scene>.png        left image, ground truth, block-matching
#                                 baselines, SGM raw / post-processed maps for each
#                                 cost, and |error| maps (clipped at 5 px; grey =
#                                 outside the evaluation mask)
#   sgm_vs_block_summary.png      mean MAE: block-matching baselines vs SGM
#   sgm_per_scene.png             the same, per scene
#   sgm_mae_vs_directions.png     mean MAE of raw SGM using 1, 2, 4, 8 path
#                                 directions (one line per cost)
#   sgm_p2_sweep.png              mean MAE of raw SGM vs the penalty P2
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _dcm():
    cm = plt.get_cmap("plasma").copy(); cm.set_bad("black"); return cm


def _ecm():
    cm = plt.get_cmap("hot").copy(); cm.set_bad("0.6"); return cm


def save_comparison(scene, d, out_path):
    """d: left, gt, valid, mask, max_disp, block{label: (disp, mae)},
    sgm{cost: {'raw','post','mae_raw','mae_post'}}"""
    vmax = d["max_disp"]
    panels = [("Left image", d["left"], "img"),
              ("Ground truth", np.where(d["valid"], d["gt"], np.nan), "disp")]
    for label, (disp, e) in d["block"].items():
        panels.append((f"{label}  (MAE {e:.2f})", disp, "disp"))
    for cost, r in d["sgm"].items():
        panels.append((f"SGM-{cost} raw  (MAE {r['mae_raw']:.2f})", r["raw"], "disp"))
        panels.append((f"SGM-{cost} post-processed  (MAE {r['mae_post']:.2f})", r["post"], "disp"))
    first_block = next(iter(d["block"].items()))
    panels.append((f"|error| {first_block[0]}", first_block[1][0], "err"))
    for cost, r in d["sgm"].items():
        panels.append((f"|error| SGM-{cost} post", r["post"], "err"))

    n, ncol = len(panels), 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for a, (title, arr, kind) in zip(axes.ravel(), panels):
        if kind == "img":
            a.imshow(arr, cmap="gray")
        elif kind == "disp":
            a.imshow(np.ma.masked_invalid(arr), cmap=_dcm(), vmin=0, vmax=vmax)
        else:
            err = np.where(d["mask"], np.abs(arr - d["gt"]), np.nan)
            a.imshow(np.ma.masked_invalid(err), cmap=_ecm(), vmin=0, vmax=5)
        a.set_title(title, fontsize=10)
    fig.suptitle(f"{scene} - Semi-Global Matching (disparity 0 .. {vmax} px)", fontsize=14)
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close(fig)


def _method_order(df):
    return list(dict.fromkeys(df.method))


def plot_summary(df, out_path):
    sub = df[df.experiment.isin(["baseline", "main"])]
    methods = _method_order(sub)
    means = [sub[sub.method == m].mae.mean() for m in methods]
    colors = ["0.55" if m.startswith("Block") else
              ("tab:red" if "post" in m else "tab:orange") for m in methods]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(methods)), means, color=colors)
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Mean MAE over scenes (pixels)")
    ax.set_title("Block matching (grey) vs SGM raw (orange) and post-processed (red)")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_per_scene(df, out_path):
    sub = df[df.experiment.isin(["baseline", "main"])]
    methods = _method_order(sub); scenes = list(dict.fromkeys(sub.scene))
    x = np.arange(len(scenes)); w = 0.85 / len(methods)
    cols = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for i, m in enumerate(methods):
        vals = [sub[(sub.scene == s) & (sub.method == m)].mae.mean() for s in scenes]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, label=m, color=cols[i % 10])
    ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=30, ha="right")
    ax.set_ylabel("MAE (pixels)"); ax.set_title("MAE per scene")
    ax.grid(axis="y", alpha=.3); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _line_plot(df, experiment, xlabel, title, out_path, logx=False):
    sub = df[df.experiment == experiment]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in _method_order(sub):
        s = sub[sub.method == m].groupby("param").mae.mean()
        ax.plot(s.index, s.values, "o-", label=m)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("Mean MAE (pixels)")
    ax.set_title(title); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_directions(df, out_path):
    _line_plot(df, "directions", "Number of aggregation directions",
               "SGM (raw): MAE vs number of path directions", out_path)


def plot_p2_sweep(df, out_path):
    _line_plot(df, "p2_sweep", "Penalty P2 (cost units)",
               "SGM (raw): MAE vs P2", out_path, logx=True)
