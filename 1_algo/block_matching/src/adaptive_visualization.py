# =============================================================================
# adaptive_visualization.py  --  Plots for the adaptive block size experiment
# =============================================================================
# APPROACH
# --------
# Written to  1_algo/block_matching/data/adaptive/ :
#   comparison_<scene>.png      3x4 sheet: left image, ground truth, smallest and
#                               largest fixed block, then for each adaptive
#                               strategy its disparity map, absolute error map,
#                               map of the chosen block size, plus a histogram of
#                               chosen sizes and the texture map.
#   adaptive_vs_fixed_summary.png   mean MAE per metric: every fixed size vs the
#                               two adaptive strategies.
#   adaptive_vs_fixed_per_scene.png same for the display metric, per scene.
#   adaptive_tau_sweep.png      MAE of confidence-based selection vs threshold tau,
#                               with the fixed sizes as horizontal reference lines.
# Error maps are clipped at 5 px; pixels outside the evaluation mask are grey.
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

METRICS = ("SAD", "SSD", "MAD", "NCC")
ADAPTIVE = {"adaptive_confidence": ("Adaptive (confidence)", "tab:red"),
            "adaptive_texture": ("Adaptive (texture)", "tab:blue")}


def _disp_cmap():
    cm = plt.get_cmap("plasma").copy(); cm.set_bad("black"); return cm


def _err_cmap():
    cm = plt.get_cmap("hot").copy(); cm.set_bad("0.6"); return cm


def _size_cmap(n):
    return ListedColormap(plt.get_cmap("viridis")(np.linspace(0, 1, n)))


def save_size_map_png(size_map, sizes, path):
    idx = np.searchsorted(sizes, size_map).astype(float)
    idx[size_map == 0] = np.nan
    plt.imsave(path, np.ma.masked_invalid(idx), cmap=_size_cmap(len(sizes)),
               vmin=-0.5, vmax=len(sizes) - 0.5)


def save_comparison(scene, d, sizes, metric, out_path):
    """d: dict with left, gt, valid, mask, max_disp, fixed{k:disp}, conf/tex results."""
    S = len(sizes); vmax = d["max_disp"]
    dc, ec = _disp_cmap(), _err_cmap()
    fig, ax = plt.subplots(3, 4, figsize=(20, 11))
    for a in ax.ravel():
        a.axis("off")

    def show_disp(a, disp, title):
        a.imshow(np.ma.masked_invalid(disp), cmap=dc, vmin=0, vmax=vmax); a.set_title(title)

    def show_err(a, disp, title):
        err = np.where(d["mask"], np.abs(disp - d["gt"]), np.nan)
        a.imshow(np.ma.masked_invalid(err), cmap=ec, vmin=0, vmax=5); a.set_title(title)

    def show_size(a, size_map, title):
        idx = np.searchsorted(sizes, size_map).astype(float); idx[size_map == 0] = np.nan
        im = a.imshow(np.ma.masked_invalid(idx), cmap=_size_cmap(S), vmin=-0.5, vmax=S - 0.5)
        cb = fig.colorbar(im, ax=a, fraction=0.046, ticks=range(S))
        cb.ax.set_yticklabels([f"{k}x{k}" for k in sizes]); a.set_title(title)

    ax[0, 0].imshow(d["left"], cmap="gray"); ax[0, 0].set_title("Left image")
    ax[0, 1].imshow(np.ma.masked_where(~d["valid"], d["gt"]), cmap=dc, vmin=0, vmax=vmax)
    ax[0, 1].set_title("Ground truth")
    ks, kl = sizes[0], sizes[-1]
    show_disp(ax[0, 2], d["fixed"][ks], f"Fixed {ks}x{ks}  (MAE {d['mae'][f'fixed_{ks}']:.2f})")
    show_disp(ax[0, 3], d["fixed"][kl], f"Fixed {kl}x{kl}  (MAE {d['mae'][f'fixed_{kl}']:.2f})")

    for row, key in ((1, "adaptive_confidence"), (2, "adaptive_texture")):
        label = ADAPTIVE[key][0]
        disp, size_map = d[key]
        show_disp(ax[row, 0], disp, f"{label}  (MAE {d['mae'][key]:.2f})")
        show_err(ax[row, 1], disp, f"{label}: |error| (0-5 px)")
        show_size(ax[row, 2], size_map, f"{label}: chosen block size")

    # histogram of chosen sizes
    h = ax[1, 3]; h.axis("on")
    w = 0.38; x = np.arange(S)
    for off, key in ((-w / 2, "adaptive_confidence"), (w / 2, "adaptive_texture")):
        sm = d[key][1][d["mask"]]
        frac = [np.mean(sm == k) for k in sizes]
        h.bar(x + off, frac, w, label=ADAPTIVE[key][0], color=ADAPTIVE[key][1])
    h.set_xticks(x); h.set_xticklabels([f"{k}x{k}" for k in sizes])
    h.set_ylabel("fraction of evaluated pixels"); h.set_title("Chosen block sizes"); h.legend()

    t = ax[2, 3]
    im = t.imshow(d["texture"], cmap="magma"); t.set_title("Local texture (std of left image)")
    fig.colorbar(im, ax=t, fraction=0.046)
    fig.suptitle(f"{scene} - adaptive block size, metric {metric}", fontsize=14)
    fig.tight_layout(); fig.savefig(out_path, dpi=100); plt.close(fig)


def _methods(df):
    fixed = sorted(df[df.approach == "fixed"].block_size.unique())
    return fixed


def plot_summary(df, out_path):
    fixed = _methods(df)
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 0.85, len(fixed)))
    bars = [(f"Fixed {k}x{k}", ("fixed", k), shades[i]) for i, k in enumerate(fixed)]
    bars += [(lab, (key, 0), col) for key, (lab, col) in ADAPTIVE.items()]
    x = np.arange(len(METRICS)); w = 0.8 / len(bars)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (lab, (appr, k), col) in enumerate(bars):
        sub = df[(df.approach == appr) & (df.block_size == k)]
        vals = [sub[sub.metric == m].mae.mean() for m in METRICS]
        ax.bar(x + (i - (len(bars) - 1) / 2) * w, vals, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(METRICS)
    ax.set_ylabel("Mean MAE over scenes (pixels)")
    ax.set_title("Adaptive vs fixed block size"); ax.grid(axis="y", alpha=.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_per_scene(df, metric, out_path):
    df = df[df.metric == metric]
    scenes = list(dict.fromkeys(df.scene)); fixed = _methods(df)
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 0.85, len(fixed)))
    bars = [(f"Fixed {k}x{k}", ("fixed", k), shades[i]) for i, k in enumerate(fixed)]
    bars += [(lab, (key, 0), col) for key, (lab, col) in ADAPTIVE.items()]
    x = np.arange(len(scenes)); w = 0.85 / len(bars)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for i, (lab, (appr, k), col) in enumerate(bars):
        vals = [df[(df.scene == s) & (df.approach == appr) & (df.block_size == k)].mae.mean()
                for s in scenes]
        ax.bar(x + (i - (len(bars) - 1) / 2) * w, vals, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=30, ha="right")
    ax.set_ylabel("MAE (pixels)"); ax.set_title(f"Adaptive vs fixed block size per scene ({metric})")
    ax.grid(axis="y", alpha=.3); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_tau_sweep(sweep_df, fixed_means, metric, out_path):
    """sweep_df: columns tau, mae (mean over scenes). fixed_means: {size: mae}."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sweep_df.tau, sweep_df.mae, "o-", color="tab:red", label="Adaptive (confidence)")
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 0.85, len(fixed_means)))
    for c, (k, v) in zip(shades, sorted(fixed_means.items())):
        ax.axhline(v, color=c, ls="--", lw=1, label=f"Fixed {k}x{k}")
    ax.set_xlabel("Confidence threshold tau (peak ratio)"); ax.set_ylabel("Mean MAE (pixels)")
    ax.set_title(f"Effect of tau on confidence-based adaptive matching ({metric})")
    ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
