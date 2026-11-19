# =============================================================================
# run_adaptive.py  --  Runs and evaluates adaptive block size matching
# =============================================================================
# APPROACH
# --------
# For each of the first N scenes (same loading, scaling and evaluation mask as
# run_experiments.py, so numbers are directly comparable):
#   1. For every block size and each metric (SAD, SSD, NCC) compute the winner
#      disparity and its confidence (adaptive_block_matching.compute_candidates).
#      The FIXED-size results come from the same computation.
#   2. Build two ADAPTIVE disparity maps per metric:
#         confidence : smallest block whose peak ratio >= --tau
#         texture    : block size chosen from local image texture
#   3. Score everything with MAE on the common mask (evaluation.py).
#   4. For the display metric (--metric) also sweep tau over --taus.
#   5. Save maps, comparison sheets, plots and a CSV to
#      1_algo/block_matching/data/adaptive/.
# MAD gives the same result as SAD (see adaptive_block_matching.py), so its rows
# are copied from SAD.
#
# USAGE
#   python run_adaptive.py
#   python run_adaptive.py --num-folders 10 --metric NCC --tau 1.3
#   python run_adaptive.py --block-sizes 3 7 15 --folders Adirondack Piano
# =============================================================================

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import list_scenes, load_scene
from evaluation import common_mask, mae
from adaptive_block_matching import (compute_candidates, select_by_confidence,
                                     select_by_texture)
import visualization as viz
import adaptive_visualization as aviz

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
OUT_DIR = HERE.parent / "data" / "adaptive"
COMPUTED = ("SAD", "SSD", "NCC")          # MAD is copied from SAD


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--num-folders", type=int, default=10)
    p.add_argument("--folders", nargs="*", default=None)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--max-disp", type=int, default=None)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[3, 5, 9, 15, 21])
    p.add_argument("--tau", type=float, default=1.2, help="confidence threshold")
    p.add_argument("--taus", type=float, nargs="+", default=[1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0])
    p.add_argument("--texture-win", type=int, default=9)
    p.add_argument("--metric", default="SAD", choices=["SAD", "SSD", "NCC"],
                   help="metric shown in comparison sheets and tau sweep")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = sorted(set(args.block_sizes))
    out = args.out_dir
    (out / "disparity_maps").mkdir(parents=True, exist_ok=True)

    scenes = list_scenes(args.data_root, limit=args.num_folders, names=args.folders)
    if not scenes:
        raise SystemExit(f"No usable scenes found under {args.data_root}")
    print(f"Scenes ({len(scenes)}): {', '.join(scenes)}")
    print(f"Block sizes {sizes}, tau={args.tau}, display metric {args.metric}")

    rows, sweep_rows = [], []
    for name in scenes:
        t0 = time.perf_counter()
        sc = load_scene(name, args.data_root, args.scale, args.max_disp)
        left, right, gt, md = sc["left"], sc["right"], sc["gt"], sc["max_disp"]
        mask = common_mask(sc["valid"], md, max(sizes))
        sdir = out / "disparity_maps" / name
        print(f"\n[{name}] size={left.shape[::-1]} max_disp={md} eval pixels={int(mask.sum())}")

        def add(approach, k, metric, e):
            rows.append(dict(scene=name, approach=approach, block_size=k, metric=metric, mae=e))
            if metric == "SAD":                       # MAD == SAD
                rows.append(dict(scene=name, approach=approach, block_size=k,
                                 metric="MAD", mae=e))

        sheet = None
        for metric in COMPUTED:
            cand = compute_candidates(left, right, md, sizes, metric)
            res = {}
            for k in sizes:
                res[f"fixed_{k}"] = mae(cand[k][0], gt, mask)
                add("fixed", k, metric, res[f"fixed_{k}"])

            d_conf, s_conf = select_by_confidence(cand, args.tau)
            d_tex, s_tex, tex = select_by_texture(cand, left, args.texture_win)
            res["adaptive_confidence"] = mae(d_conf, gt, mask)
            res["adaptive_texture"] = mae(d_tex, gt, mask)
            add("adaptive_confidence", 0, metric, res["adaptive_confidence"])
            add("adaptive_texture", 0, metric, res["adaptive_texture"])

            for tag, dm, sm in (("confidence", d_conf, s_conf), ("texture", d_tex, s_tex)):
                aviz_dir = sdir
                viz.save_disparity_png(dm, md, aviz_dir / f"adaptive_{tag}_{metric}.png")
                aviz.save_size_map_png(sm, sizes, aviz_dir / f"adaptive_{tag}_{metric}_sizes.png")

            fixed_txt = "  ".join(f"{k}:{res[f'fixed_{k}']:.3f}" for k in sizes)
            print(f"  {metric}: fixed[{fixed_txt}]  "
                  f"conf={res['adaptive_confidence']:.3f}  tex={res['adaptive_texture']:.3f}")

            if metric == args.metric:
                for tau in args.taus:
                    dt, _ = select_by_confidence(cand, tau)
                    sweep_rows.append(dict(scene=name, tau=tau, mae=mae(dt, gt, mask)))
                sheet = dict(left=left, gt=gt, valid=sc["valid"], mask=mask, max_disp=md,
                             fixed={k: cand[k][0] for k in sizes}, mae=res, texture=tex,
                             adaptive_confidence=(d_conf, s_conf),
                             adaptive_texture=(d_tex, s_tex))
        aviz.save_comparison(name, sheet, sizes, args.metric, out / f"comparison_{name}.png")
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "adaptive_results.csv", index=False)
    sweep = pd.DataFrame(sweep_rows).groupby("tau").mae.mean().reset_index()
    sweep.to_csv(out / "adaptive_tau_sweep.csv", index=False)

    aviz.plot_summary(df, out / "adaptive_vs_fixed_summary.png")
    aviz.plot_per_scene(df, args.metric, out / "adaptive_vs_fixed_per_scene.png")
    fixed_means = {k: df[(df.approach == "fixed") & (df.block_size == k) &
                         (df.metric == args.metric)].mae.mean() for k in sizes}
    aviz.plot_tau_sweep(sweep, fixed_means, args.metric, out / "adaptive_tau_sweep.png")

    print("\n=== Mean MAE over scenes (pixels) ===")
    df["method"] = np.where(df.approach == "fixed",
                            "fixed " + df.block_size.astype(str), df.approach)
    table = df.groupby(["method", "metric"]).mae.mean().unstack("metric")
    print(table[["SAD", "SSD", "MAD", "NCC"]].round(3).to_string())
    print(f"\nTau sweep ({args.metric}):\n{sweep.round(3).to_string(index=False)}")
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
