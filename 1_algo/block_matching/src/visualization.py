# =============================================================================
# visualization.py  --  Plots and disparity images
# =============================================================================
# APPROACH
# --------
# All figures are written to  1_algo/block_matching/data/ :
#   disparity_maps/<scene>/*.png        colour-coded disparity of every method
#   comparison_<scene>.png              left image, ground truth, pixel-based and
#                                       block-based maps of SAD/SSD/MAD/NCC
#   mae_per_scene.png                   grouped bars: MAE of each metric per scene,
#                                       pixel-based (top) vs block-based (bottom)
#   mae_summary_pixel_vs_block.png      MAE averaged over all scenes, per metric
#   mae_vs_block_size.png               mean MAE as a function of block size
#   mae_vs_block_size_per_scene.png     the same curve for every scene
#
# Disparity maps share one colour scale (plasma, 0 .. max_disp) so they can be
# compared by eye.  Pixels without an estimate (NaN) are drawn black.
# =============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("SAD", "SSD", "MAD", "NCC")
COLORS = {"SAD": "tab:blue", "SSD": "tab:orange", "MAD": "tab:green", "NCC": "tab:red"}
STYLES = {"SAD": "-", "SSD": "-", "MAD": "--", "NCC": "-"}
MARKERS = {"SAD": "o", "SSD": "s", "MAD": "x", "NCC": "^"}


def _cmap():
    cm = plt.get_cmap("plasma").copy()
    cm.set_bad("black")
    return cm


def save_disparity_png(disp, max_disp, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.ma.masked_invalid(disp), cmap=_cmap(), vmin=0, vmax=max_disp)


def save_comparison(scene, maps, block_size, out_path):
    """maps: dict with 'left', 'gt', 'valid', 'max_disp', 'pixel'{m}, 'block'{m}."""
    cm, vmax = _cmap(), maps["max_disp"]
    fig, axes = plt.subplots(3, 5, figsize=(20, 9))
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(maps["left"], cmap="gray"); axes[0, 0].set_title("Left image")
    axes[0, 1].imshow(np.ma.masked_where(~maps["valid"], maps["gt"]),
                      cmap=cm, vmin=0, vmax=vmax)
    axes[0, 1].set_title("Ground truth")
    for j, m in enumerate(METRICS):
        axes[1, j].imshow(np.ma.masked_invalid(maps["pixel"][m]), cmap=cm, vmin=0, vmax=vmax)
        axes[1, j].set_title(f"Pixel-based {m}  (MAE {maps['pixel_mae'][m]:.2f})")
        axes[2, j].imshow(np.ma.masked_invalid(maps["block"][m]), cmap=cm, vmin=0, vmax=vmax)
        axes[2, j].set_title(f"Block {block_size}x{block_size} {m}  (MAE {maps['block_mae'][m]:.2f})")
    fig.suptitle(f"{scene} - disparity maps (0 .. {vmax} px)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_mae_per_scene(df, display_block, out_path):
    scenes = list(dict.fromkeys(df["scene"]))
    x = np.arange(len(scenes)); width = 0.2
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, (title, sub) in zip(axes, [
        ("Pixel-based matching", df[df.approach == "pixel"]),
        (f"Block-based matching ({display_block}x{display_block})",
         df[(df.approach == "block") & (df.block_size == display_block)]),
    ]):
        for i, m in enumerate(METRICS):
            vals = [sub[(sub.scene == s) & (sub.metric == m)].mae.mean() for s in scenes]
            ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=COLORS[m])
        ax.set_ylabel("MAE (pixels)"); ax.set_title(title); ax.grid(axis="y", alpha=.3)
        ax.legend(ncol=4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(scenes, rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_summary(df, display_block, out_path):
    pix = df[df.approach == "pixel"].groupby("metric").mae.mean()
    blk = df[(df.approach == "block") & (df.block_size == display_block)] \
        .groupby("metric").mae.mean()
    x = np.arange(len(METRICS)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, [pix[m] for m in METRICS], w, label="Pixel-based")
    b2 = ax.bar(x + w / 2, [blk[m] for m in METRICS], w,
                label=f"Block-based ({display_block}x{display_block})")
    ax.bar_label(b1, fmt="%.2f"); ax.bar_label(b2, fmt="%.2f")
    ax.set_xticks(x); ax.set_xticklabels(METRICS)
    ax.set_ylabel("Mean MAE over scenes (pixels)")
    ax.set_title("Pixel-based vs block-based matching"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _draw_curves(ax, sub, pixel_sub=None):
    for m in METRICS:
        s = sub[sub.metric == m].groupby("block_size").mae.mean()
        ax.plot(s.index, s.values, STYLES[m], marker=MARKERS[m], color=COLORS[m],
                label=m + (" (same ranking as SAD)" if m == "MAD" else ""))
    ax.set_xlabel("Block size (pixels)"); ax.set_ylabel("MAE (pixels)"); ax.grid(alpha=.3)


def plot_mae_vs_block_size(df, out_path):
    blk = df[df.approach == "block"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_curves(ax, blk)
    ax.set_xticks(sorted(blk.block_size.unique()))
    ax.set_title("MAE vs block size (mean over all scenes)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_mae_vs_block_size_per_scene(df, out_path):
    blk = df[df.approach == "block"]
    scenes = list(dict.fromkeys(blk.scene))
    cols = 5; rows = int(np.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), scenes):
        _draw_curves(ax, blk[blk.scene == s]); ax.set_title(s)
        ax.set_xticks(sorted(blk.block_size.unique()))
    for ax in axes.ravel()[len(scenes):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("MAE vs block size per scene"); fig.tight_layout()
    fig.savefig(out_path, dpi=110); plt.close(fig)
