# =============================================================================
# sgm_blocksize_visualization.py  --  Plots for the SGM block-size sweep
# =============================================================================
# APPROACH
# --------
# Written to  1_algo/sgm/data/blocksize/ :
#   mae_vs_blocksize.png         one panel per cost family (SAD, SSD, Census):
#                                mean MAE vs block size for plain block matching
#                                (grey, dashed) and SGM raw / post-processed.
#                                Census is compared with block NCC (the closest
#                                block counterpart; census has no block form).
#   sgm_gain_vs_blocksize.png    MAE(block matching) - MAE(raw SGM) at the same
#                                block size; positive = SGM is better.
#   mae_vs_blocksize_per_scene.png  the same curves for every scene (display cost)
#   blocksize_maps_<scene>.png   disparity maps, one row per block size:
#                                block matching | SGM raw | SGM post-processed
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY_TITLE = {"sad": "SAD cost  (block baseline: SAD)",
                "ssd": "SSD cost  (block baseline: SSD)",
                "census": "Census cost  (block baseline: NCC)"}
STYLE = {"Block": dict(color="0.4", ls="--", marker="s"),
         "SGM raw": dict(color="tab:orange", ls="-", marker="o"),
         "SGM post": dict(color="tab:red", ls="-", marker="^")}


def _curves(ax, sub, title):
    for method, st in STYLE.items():
        s = sub[sub.method == method].groupby("window").mae.mean()
        if len(s):
            ax.plot(s.index, s.values, label=method, **st)
    ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
    ax.set_xlabel("Block size (pixels)"); ax.set_ylabel("MAE (pixels)")


def plot_mae_vs_blocksize(df, out_path):
    fams = [f for f in ("sad", "ssd", "census") if f in set(df.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(6 * len(fams), 5), squeeze=False)
    for ax, f in zip(axes[0], fams):
        _curves(ax, df[df.family == f], FAMILY_TITLE[f])
        ax.set_xticks(sorted(df[df.family == f].window.unique()))
    axes[0, 0].legend()
    fig.suptitle("MAE vs block size, mean over scenes: block matching vs SGM")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_gain(df, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for f, col in (("sad", "tab:blue"), ("ssd", "tab:green"), ("census", "tab:purple")):
        sub = df[df.family == f]
        if sub.empty:
            continue
        m = sub.groupby(["method", "window"]).mae.mean().unstack("method")
        if "Block" not in m or "SGM raw" not in m:
            continue
        m = m.dropna(subset=["Block", "SGM raw"])
        ax.plot(m.index, m["Block"] - m["SGM raw"], "o-", color=col, label=f"{f}: raw SGM")
        if "SGM post" in m:
            ax.plot(m.index, m["Block"] - m["SGM post"], "^--", color=col, alpha=.6,
                    label=f"{f}: post-processed SGM")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Block size (pixels)")
    ax.set_ylabel("MAE(block matching) - MAE(SGM)  [pixels]")
    ax.set_title("SGM gain over block matching at the same block size (>0 = SGM better)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_per_scene(df, family, out_path):
    sub = df[df.family == family]
    scenes = list(dict.fromkeys(sub.scene))
    cols = min(5, len(scenes)); rows = int(np.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), scenes):
        _curves(ax, sub[sub.scene == s], s)
    for ax in axes.ravel()[len(scenes):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"MAE vs block size per scene - {FAMILY_TITLE[family]}")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def save_maps(scene, d, windows, family, out_path):
    """d: left, gt, valid, max_disp, block{w: (disp, mae)}, raw{w: (disp, mae)}, post{w: (disp, mae)}"""
    cm = plt.get_cmap("plasma").copy(); cm.set_bad("black")
    vmax = d["max_disp"]
    n = len(windows) + 1
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.7 * n), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    axes[0, 0].imshow(d["left"], cmap="gray"); axes[0, 0].set_title("Left image")
    axes[0, 1].imshow(np.ma.masked_where(~d["valid"], d["gt"]), cmap=cm, vmin=0, vmax=vmax)
    axes[0, 1].set_title("Ground truth")
    for i, w in enumerate(windows, 1):
        for j, (key, label) in enumerate((("block", "Block"), ("raw", "SGM raw"),
                                          ("post", "SGM post"))):
            if w in d[key]:
                disp, e = d[key][w]
                axes[i, j].imshow(np.ma.masked_invalid(disp), cmap=cm, vmin=0, vmax=vmax)
                axes[i, j].set_title(f"{label} {w}x{w}  (MAE {e:.2f})", fontsize=9)
    fig.suptitle(f"{scene} - {FAMILY_TITLE[family]}", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=85); plt.close(fig)
