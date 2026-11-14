# =============================================================================
# run_experiments.py  --  Entry point: runs every algorithm and evaluation
# =============================================================================
# APPROACH
# --------
# For each of the first N scenes (default 10, alphabetical) of
# disparity_algorithms/data/trainingQ:
#   1. load the stereo pair + ground truth       (data_loader.py)
#   2. run PIXEL-based matching with SAD, SSD, MAD, NCC   (pixel_matching.py)
#   3. run BLOCK-based matching with SAD, SSD, MAD, NCC for several block sizes
#      (3, 5, 9, 15, 21 by default)                       (block_matching.py)
#   4. score every disparity map with MAE on a common pixel mask (evaluation.py)
#   5. save disparity images, comparison sheets, MAE plots and a CSV of all
#      numbers into 1_algo/block_matching/data/            (visualization.py)
#
# USAGE (from anywhere):
#   python run_experiments.py
#   python run_experiments.py --num-folders 10 --scale 0.5
#   python run_experiments.py --folders Adirondack Motorcycle --block-sizes 3 7 11
#   python run_experiments.py --data-root /path/to/data --max-disp 64
#
# Note: MAD ranks candidates exactly like SAD (it is SAD divided by a constant),
# so the two produce identical disparity maps / MAE; both are reported because
# they were requested.  For a 1x1 pixel window MAD == SAD as well.
# =============================================================================

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import list_scenes, load_scene
from pixel_matching import pixel_matching
from block_matching import block_matching
from evaluation import common_mask, mae
import visualization as viz

METRICS = ("SAD", "SSD", "MAD", "NCC")
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]            # .../disparity_algorithms
OUT_DIR = HERE.parent / "data"            # .../1_algo/block_matching/data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--num-folders", type=int, default=10)
    p.add_argument("--folders", nargs="*", default=None, help="explicit scene names")
    p.add_argument("--scale", type=float, default=0.5, help="image resize factor")
    p.add_argument("--max-disp", type=int, default=None,
                   help="search range; default derived from calib.txt")
    p.add_argument("--block-sizes", type=int, nargs="+", default=[3, 5, 9, 15, 21])
    p.add_argument("--display-block", type=int, default=9,
                   help="block size used in bar charts / comparison sheets")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = sorted(set(args.block_sizes) | {args.display_block})
    out = args.out_dir
    (out / "disparity_maps").mkdir(parents=True, exist_ok=True)

    scenes = list_scenes(args.data_root, limit=args.num_folders, names=args.folders)
    if not scenes:
        raise SystemExit(f"No usable scenes found under {args.data_root}")
    print(f"Scenes ({len(scenes)}): {', '.join(scenes)}")

    rows = []
    for name in scenes:
        t0 = time.perf_counter()
        sc = load_scene(name, args.data_root, args.scale, args.max_disp)
        left, right, gt = sc["left"], sc["right"], sc["gt"]
        md = sc["max_disp"]
        mask = common_mask(sc["valid"], md, max(sizes))
        scene_dir = out / "disparity_maps" / name
        print(f"\n[{name}] size={left.shape[::-1]} max_disp={md} "
              f"eval pixels={int(mask.sum())}")

        viz.save_disparity_png(np.where(sc["valid"], gt, np.nan), md,
                               scene_dir / "ground_truth.png")

        maps = {"left": left, "gt": gt, "valid": sc["valid"], "max_disp": md,
                "pixel": {}, "block": {}, "pixel_mae": {}, "block_mae": {}}

        for m in METRICS:
            d = pixel_matching(left, right, md, m)
            e = mae(d, gt, mask)
            rows.append(dict(scene=name, approach="pixel", block_size=1, metric=m, mae=e))
            viz.save_disparity_png(d, md, scene_dir / f"pixel_{m}.png")
            maps["pixel"][m], maps["pixel_mae"][m] = d, e
            print(f"  pixel         {m}: MAE = {e:.3f}")

        for k in sizes:
            for m in METRICS:
                d = block_matching(left, right, md, k, m)
                e = mae(d, gt, mask)
                rows.append(dict(scene=name, approach="block", block_size=k, metric=m, mae=e))
                viz.save_disparity_png(d, md, scene_dir / f"block{k}x{k}_{m}.png")
                if k == args.display_block:
                    maps["block"][m], maps["block_mae"][m] = d, e
            print("  block {:>2}x{:<2}    ".format(k, k) +
                  "  ".join(f"{m}={rows[-4 + i]['mae']:.3f}" for i, m in enumerate(METRICS)))

        viz.save_comparison(name, maps, args.display_block, out / f"comparison_{name}.png")
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "mae_results.csv", index=False)

    viz.plot_mae_per_scene(df, args.display_block, out / "mae_per_scene.png")
    viz.plot_summary(df, args.display_block, out / "mae_summary_pixel_vs_block.png")
    viz.plot_mae_vs_block_size(df, out / "mae_vs_block_size.png")
    viz.plot_mae_vs_block_size_per_scene(df, out / "mae_vs_block_size_per_scene.png")

    print("\n=== Mean MAE over scenes (pixels) ===")
    table = df.groupby(["approach", "block_size", "metric"]).mae.mean().unstack("metric")
    print(table[list(METRICS)].round(3).to_string())
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
