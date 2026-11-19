# =============================================================================
# run_sgm.py  --  Runs and evaluates Semi-Global Matching
# =============================================================================
# APPROACH
# --------
# Uses the same scenes, scaling, search range and MAE evaluation mask as the
# block-matching experiments (code is imported from ../block_matching/src), so the
# numbers are directly comparable.  For each of the first N scenes:
#   1. Block-matching baselines: SAD and NCC with --baseline-block (default 9).
#   2. For every matching cost in --costs (default sad census):
#        - build the cost volume, aggregate along 8 directions, take the winner
#          ("raw"), then apply left-right check + filling + median ("post");
#        - DIRECTION ABLATION: raw MAE after 1, 2, 4 and 8 directions
#          (computed in the same pass, no extra cost);
#        - P2 SWEEP: raw MAE for each value of --p2-sweep.
#   3. Save disparity maps, comparison sheets, plots and one CSV into
#      1_algo/sgm/data/.
#
# USAGE
#   python run_sgm.py
#   python run_sgm.py --num-folders 10 --costs sad census --p1 0.04 --p2 0.2
#   python run_sgm.py --subpixel --folders Adirondack Piano
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]                       # .../disparity_algorithms
sys.path.insert(0, str(HERE.parents[1] / "block_matching" / "src"))   # shared code

from data_loader import list_scenes, load_scene       # noqa: E402
from evaluation import common_mask, mae               # noqa: E402
from block_matching import block_matching             # noqa: E402
import visualization as bviz                          # noqa: E402  (disparity PNG saver)

import sgm as sgm_mod                                  # noqa: E402
from sgm_costs import compute_cost_volume, COSTS       # noqa: E402
import sgm_visualization as sviz                       # noqa: E402

OUT_DIR = HERE.parent / "data"
DIR_COUNTS = (1, 2, 4, 8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--num-folders", type=int, default=10)
    p.add_argument("--folders", nargs="*", default=None)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--max-disp", type=int, default=None)
    p.add_argument("--costs", nargs="+", default=["sad", "census"], choices=COSTS)
    p.add_argument("--cost-win", type=int, default=3, help="window of sad/ssd cost")
    p.add_argument("--census-win", type=int, default=5, choices=[3, 5, 7])
    p.add_argument("--p1", type=float, default=0.04, help="penalty for 1-px change")
    p.add_argument("--p2", type=float, default=0.2, help="penalty for larger jumps")
    p.add_argument("--no-adaptive-p2", action="store_true")
    p.add_argument("--subpixel", action="store_true", help="parabola sub-pixel refinement")
    p.add_argument("--median", type=int, default=3, help="median filter size (0 = off)")
    p.add_argument("--p2-sweep", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.8])
    p.add_argument("--baseline-block", type=int, default=9)
    p.add_argument("--mask-block", type=int, default=21,
                   help="border margin of the evaluation mask (use the largest block size "
                        "of the block-matching run for identical pixels)")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out_dir
    (out / "disparity_maps").mkdir(parents=True, exist_ok=True)
    adaptive = not args.no_adaptive_p2

    scenes = list_scenes(args.data_root, limit=args.num_folders, names=args.folders)
    if not scenes:
        raise SystemExit(f"No usable scenes found under {args.data_root}")
    print(f"Scenes ({len(scenes)}): {', '.join(scenes)}")
    print(f"Costs {args.costs}, P1={args.p1}, P2={args.p2}, adaptive P2={adaptive}, "
          f"subpixel={args.subpixel}")

    rows = []
    def add(scene, experiment, method, param, e):
        rows.append(dict(scene=scene, experiment=experiment, method=method,
                         param=param, mae=e))

    k = args.baseline_block
    for name in scenes:
        t0 = time.perf_counter()
        sc = load_scene(name, args.data_root, args.scale, args.max_disp)
        left, right, gt, md = sc["left"], sc["right"], sc["gt"], sc["max_disp"]
        mask = common_mask(sc["valid"], md, args.mask_block)
        sdir = out / "disparity_maps" / name
        print(f"\n[{name}] size={left.shape[::-1]} max_disp={md} eval pixels={int(mask.sum())}")
        bviz.save_disparity_png(np.where(sc["valid"], gt, np.nan), md, sdir / "ground_truth.png")

        sheet = dict(left=left, gt=gt, valid=sc["valid"], mask=mask, max_disp=md,
                     block={}, sgm={})
        for metric in ("SAD", "NCC"):
            d = block_matching(left, right, md, k, metric)
            e = mae(d, gt, mask)
            label = f"Block {metric} {k}x{k}"
            add(name, "baseline", label, np.nan, e)
            sheet["block"][label] = (d, e)
            print(f"  {label}: MAE = {e:.3f}")

        for cost_kind in args.costs:
            cost = compute_cost_volume(left, right, md, cost_kind,
                                       args.cost_win, args.census_win)
            S = None
            for n_dirs, S in sgm_mod.aggregate_iter(cost, left, args.p1, args.p2,
                                                    sgm_mod.DIRECTIONS, adaptive):
                if n_dirs in DIR_COUNTS:
                    add(name, "directions", f"SGM-{cost_kind}", n_dirs,
                        mae(sgm_mod.winner_takes_all(S).astype(np.float32), gt, mask))
            res = sgm_mod.sgm(left, right, cost, args.p1, args.p2, adaptive=adaptive,
                              subpixel=args.subpixel, median=args.median, S=S)
            e_raw, e_post = mae(res["raw"], gt, mask), mae(res["post"], gt, mask)
            add(name, "main", f"SGM-{cost_kind} raw", np.nan, e_raw)
            add(name, "main", f"SGM-{cost_kind} post", np.nan, e_post)
            add(name, "lrc", f"SGM-{cost_kind}", np.nan,
                float(1.0 - res["consistent"][mask].mean()))     # share failing the check
            bviz.save_disparity_png(res["raw"], md, sdir / f"sgm_{cost_kind}_raw.png")
            bviz.save_disparity_png(res["post"], md, sdir / f"sgm_{cost_kind}_post.png")
            sheet["sgm"][cost_kind] = dict(raw=res["raw"], post=res["post"],
                                           mae_raw=e_raw, mae_post=e_post)
            print(f"  SGM-{cost_kind}: raw MAE = {e_raw:.3f}  post MAE = {e_post:.3f}  "
                  f"(inconsistent pixels {100 * (1 - res['consistent'][mask].mean()):.1f}%)")
            del S, res

            for p2 in args.p2_sweep:
                S2 = sgm_mod.aggregate(cost, left, args.p1, p2, sgm_mod.DIRECTIONS, adaptive)
                add(name, "p2_sweep", f"SGM-{cost_kind}", p2,
                    mae(sgm_mod.winner_takes_all(S2).astype(np.float32), gt, mask))
                del S2

        sviz.save_comparison(name, sheet, out / f"comparison_{name}.png")
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "sgm_results.csv", index=False)
    sviz.plot_summary(df, out / "sgm_vs_block_summary.png")
    sviz.plot_per_scene(df, out / "sgm_per_scene.png")
    sviz.plot_directions(df, out / "sgm_mae_vs_directions.png")
    sviz.plot_p2_sweep(df, out / "sgm_p2_sweep.png")

    print("\n=== Mean MAE over scenes (pixels) ===")
    main = df[df.experiment.isin(["baseline", "main"])].groupby("method", sort=False).mae.mean()
    print(main.round(3).to_string())
    for exp, title in (("directions", "directions"), ("p2_sweep", "P2")):
        t = df[df.experiment == exp].groupby(["param", "method"]).mae.mean().unstack("method")
        print(f"\nRaw SGM MAE vs {title}:\n{t.round(3).to_string()}")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
