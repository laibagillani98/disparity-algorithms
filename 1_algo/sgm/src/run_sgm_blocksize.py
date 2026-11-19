# =============================================================================
# run_sgm_blocksize.py  --  SGM for several block sizes, compared with block matching
# =============================================================================
# APPROACH
# --------
# SGM has no block size of its own, but its LOCAL matching cost uses a window:
#     sad / ssd cost : size of the averaging window   (any odd size, 1 = per pixel)
#     census cost    : size of the census window      (3, 5 or 7)
# This script sweeps that window ("block size") and compares, at the same size,
#     plain block matching   (SAD, SSD, NCC; from ../block_matching/src)
#     SGM raw                (aggregation + winner takes all)
#     SGM post-processed     (+ left-right check, filling, median)
# Census has no block-matching equivalent, so it is compared with block NCC, the
# closest radiometric-invariant block metric.
#
# Everything else (scenes, scaling, search range, evaluation mask, SGM penalties)
# is identical to run_sgm.py / run_experiments.py, so numbers are comparable.
# Per scene and block size, SGM aggregation runs once per cost (8 directions).
#
# Typical findings to check in the plots:
#   * block matching improves a lot with size up to a point, then edges blur;
#   * SGM gets its smoothness from aggregation, so it is far less sensitive to
#     the cost window and often prefers small windows;
#   * the gap (gain) between the two shows how much SGM aggregation adds.
#
# USAGE
#   python run_sgm_blocksize.py
#   python run_sgm_blocksize.py --costs sad census --sad-windows 1 3 5 9 15
#   python run_sgm_blocksize.py --folders Adirondack Piano --display-cost census
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1] / "block_matching" / "src"))   # shared code

from data_loader import list_scenes, load_scene        # noqa: E402
from evaluation import common_mask, mae                # noqa: E402
from block_matching import block_matching              # noqa: E402
import visualization as bviz                           # noqa: E402

import sgm as sgm_mod                                   # noqa: E402
from sgm_costs import compute_cost_volume, COSTS        # noqa: E402
import sgm_blocksize_visualization as bsviz             # noqa: E402

OUT_DIR = HERE.parent / "data" / "blocksize"
BLOCK_METRIC = {"sad": "SAD", "ssd": "SSD", "census": "NCC"}   # baseline per family


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--num-folders", type=int, default=10)
    p.add_argument("--folders", nargs="*", default=None)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--max-disp", type=int, default=None)
    p.add_argument("--costs", nargs="+", default=["sad", "ssd", "census"], choices=COSTS)
    p.add_argument("--sad-windows", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 15],
                   help="odd window sizes for the sad / ssd costs")
    p.add_argument("--census-windows", type=int, nargs="+", default=[3, 5, 7],
                   choices=[3, 5, 7])
    p.add_argument("--p1", type=float, default=0.04)
    p.add_argument("--p2", type=float, default=0.2)
    p.add_argument("--no-adaptive-p2", action="store_true")
    p.add_argument("--subpixel", action="store_true")
    p.add_argument("--median", type=int, default=3)
    p.add_argument("--display-cost", default="sad", choices=COSTS,
                   help="cost used for per-scene curves and disparity-map sheets")
    p.add_argument("--mask-block", type=int, default=21,
                   help="border margin of the evaluation mask; must be >= the largest block size")
    return p.parse_args()


def main():
    args = parse_args()
    adaptive = not args.no_adaptive_p2
    sgm_windows = {"sad": sorted(set(args.sad_windows)), "ssd": sorted(set(args.sad_windows)),
                   "census": sorted(set(args.census_windows))}
    costs = [c for c in args.costs]
    block_windows = sorted({w for c in costs for w in sgm_windows[c]})
    if max(block_windows) > args.mask_block:
        raise SystemExit(f"--mask-block ({args.mask_block}) must be >= largest block size "
                         f"({max(block_windows)}) so all methods are scored on the same pixels")
    if args.display_cost not in costs:
        args.display_cost = costs[0]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    scenes = list_scenes(args.data_root, limit=args.num_folders, names=args.folders)
    if not scenes:
        raise SystemExit(f"No usable scenes found under {args.data_root}")
    print(f"Scenes ({len(scenes)}): {', '.join(scenes)}")
    print(f"Costs {costs}; windows: sad/ssd {sgm_windows['sad']}, census {sgm_windows['census']}")

    rows = []
    for name in scenes:
        t0 = time.perf_counter()
        sc = load_scene(name, args.data_root, args.scale, args.max_disp)
        left, right, gt, md = sc["left"], sc["right"], sc["gt"], sc["max_disp"]
        mask = common_mask(sc["valid"], md, args.mask_block)
        print(f"\n[{name}] size={left.shape[::-1]} max_disp={md} eval pixels={int(mask.sum())}")
        maps = dict(left=left, gt=gt, valid=sc["valid"], max_disp=md,
                    block={}, raw={}, post={})

        def add(family, method, w, e):
            rows.append(dict(scene=name, family=family, method=method, window=w, mae=e))

        # ---- block-matching baselines ----------------------------------------
        for family in costs:
            metric = BLOCK_METRIC[family]
            for w in sgm_windows[family]:
                if metric == "NCC" and w < 3:
                    continue
                d = block_matching(left, right, md, w, metric)
                e = mae(d, gt, mask)
                add(family, "Block", w, e)
                if family == args.display_cost:
                    maps["block"][w] = (d, e)

        # ---- SGM for every cost and window -----------------------------------
        for family in costs:
            line = []
            for w in sgm_windows[family]:
                kw = dict(win=w) if family != "census" else dict(census_win=w)
                cost = compute_cost_volume(left, right, md, family, **kw)
                res = sgm_mod.sgm(left, right, cost, args.p1, args.p2, adaptive=adaptive,
                                  subpixel=args.subpixel, median=args.median)
                e_raw, e_post = mae(res["raw"], gt, mask), mae(res["post"], gt, mask)
                add(family, "SGM raw", w, e_raw)
                add(family, "SGM post", w, e_post)
                line.append(f"{w}:{e_raw:.2f}/{e_post:.2f}")
                if family == args.display_cost:
                    maps["raw"][w] = (res["raw"], e_raw)
                    maps["post"][w] = (res["post"], e_post)
                del cost, res
            print(f"  SGM-{family} raw/post MAE by window: " + "  ".join(line))

        win_sheet = sgm_windows[args.display_cost]
        bsviz.save_maps(name, maps, win_sheet, args.display_cost,
                        out / f"blocksize_maps_{name}.png")
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "sgm_blocksize_results.csv", index=False)
    bsviz.plot_mae_vs_blocksize(df, out / "mae_vs_blocksize.png")
    bsviz.plot_gain(df, out / "sgm_gain_vs_blocksize.png")
    bsviz.plot_per_scene(df, args.display_cost, out / "mae_vs_blocksize_per_scene.png")

    print("\n=== Mean MAE over scenes (pixels) ===")
    for family in costs:
        t = (df[df.family == family].groupby(["window", "method"]).mae.mean()
             .unstack("method")[["Block", "SGM raw", "SGM post"]])
        print(f"\n{family} (block baseline: {BLOCK_METRIC[family]})\n{t.round(3).to_string()}")
        for m in ("Block", "SGM raw", "SGM post"):
            s = t[m].dropna()
            print(f"  best {m:<8}: block size {int(s.idxmin())}  (MAE {s.min():.3f})")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
