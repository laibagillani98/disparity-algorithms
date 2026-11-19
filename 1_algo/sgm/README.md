# Semi-Global Matching (SGM)

SGM (Hirschmüller) improves on block matching by adding a **smoothness constraint**
between neighbouring pixels. Instead of deciding each pixel on its own, it solves a
1-D optimisation exactly along several paths through the image and sums the results.

It reuses the data loader, evaluation mask and block-matching baselines from
`../block_matching/src`, so MAE numbers are directly comparable with that experiment.

## Layout

```
1_algo/sgm/
├── src/
│   ├── sgm_costs.py           # matching costs: sad, ssd, census (all scaled to 0..1)
│   ├── sgm.py                 # path aggregation, WTA, sub-pixel, left-right check, filling
│   ├── sgm_visualization.py   # plots
│   ├── run_sgm.py             # entry point (single configuration)
│   ├── run_sgm_blocksize.py   # sweep of block sizes, compared with block matching
│   └── sgm_blocksize_visualization.py
└── data/                      <- generated plots, disparity images, CSV
```

## Run

```bash
python 1_algo/sgm/src/run_sgm.py
python 1_algo/sgm/src/run_sgm.py --costs sad census --p1 0.04 --p2 0.2 --subpixel
python 1_algo/sgm/src/run_sgm.py --folders Adirondack Piano --scale 0.5
```

| Flag | Default | Meaning |
|---|---|---|
| `--costs` | sad census | Matching costs to run (`sad`, `ssd`, `census`) |
| `--cost-win` | 3 | Window of the sad/ssd cost |
| `--census-win` | 5 | Census window (3, 5 or 7) |
| `--p1` | 0.04 | Penalty for a 1-pixel disparity change (slanted surfaces) |
| `--p2` | 0.2 | Penalty for larger jumps (depth edges) |
| `--no-adaptive-p2` | off | Disable lowering P2 at intensity edges |
| `--subpixel` | off | Parabola sub-pixel refinement |
| `--median` | 3 | Median filter size after filling (0 = off) |
| `--p2-sweep` | 0.05 … 0.8 | P2 values for the sweep plot |
| `--baseline-block` | 9 | Block size of the block-matching baselines |
| `--mask-block` | 21 | Border margin of the evaluation mask (21 = same pixels as the block-matching run) |
| others | | `--data-root`, `--out-dir`, `--num-folders`, `--folders`, `--scale`, `--max-disp` |

## Block-size sweep

SGM has no block size of its own, but its local matching cost uses a window: the
averaging window of the `sad`/`ssd` cost (any odd size, 1 = per pixel) and the census
window (3, 5 or 7). `run_sgm_blocksize.py` sweeps it and compares, **at the same block
size**, plain block matching with raw and post-processed SGM. Census has no block form,
so it is compared with block NCC.

```bash
python 1_algo/sgm/src/run_sgm_blocksize.py
python 1_algo/sgm/src/run_sgm_blocksize.py --costs sad census --sad-windows 1 3 5 9 15
python 1_algo/sgm/src/run_sgm_blocksize.py --folders Adirondack Piano --display-cost census
```

| Flag | Default | Meaning |
|---|---|---|
| `--costs` | sad ssd census | Cost families to sweep |
| `--sad-windows` | 1 3 5 7 9 11 15 | Window sizes for `sad` and `ssd` |
| `--census-windows` | 3 5 7 | Census window sizes (7 is the maximum) |
| `--display-cost` | sad | Cost used for per-scene curves and disparity-map sheets |
| `--mask-block` | 21 | Evaluation border margin; must be at least the largest block size |
| others | | `--p1`, `--p2`, `--no-adaptive-p2`, `--subpixel`, `--median`, `--data-root`, `--out-dir`, `--num-folders`, `--folders`, `--scale`, `--max-disp` |

Outputs go to `sgm/data/blocksize/`:

| Output | Content |
|---|---|
| `mae_vs_blocksize.png` | One panel per cost: mean MAE vs block size for block matching, SGM raw, SGM post-processed |
| `sgm_gain_vs_blocksize.png` | MAE(block matching) − MAE(raw SGM) per block size; above 0 means SGM is better |
| `mae_vs_blocksize_per_scene.png` | The same curves for each scene (display cost) |
| `blocksize_maps_<scene>.png` | One row per block size: block matching, SGM raw, SGM post-processed |
| `sgm_blocksize_results.csv` | Columns: scene, family, method (`Block`, `SGM raw`, `SGM post`), window, mae |

The console output also lists the best block size per method.

Caveat: `P1` and `P2` are fixed in cost units, but the three costs have different
typical magnitudes (SSD values are much smaller than SAD values). The penalties therefore
act relatively stronger on SSD, which makes SGM-SSD look insensitive to block size. For a
fair comparison between costs, tune `--p1`/`--p2` per cost.

## How it works

1. **Cost volume** `C(y, x, d)`: local cost of matching left `(y, x)` with right `(y, x−d)`.
   `census` compares the order of intensities in a window (robust to brightness
   changes); `sad`/`ssd` average the intensity difference over a small window.
2. **Path aggregation.** For each direction `r`:
   `L_r(p,d) = C(p,d) + min(L_r(p−r,d), L_r(p−r,d±1)+P1, min_k L_r(p−r,k)+P2) − min_k L_r(p−r,k)`.
   Eight directions are used (2 horizontal, 2 vertical, 4 diagonal). P2 is lowered where
   the image has an intensity edge, because depth discontinuities usually coincide with them.
3. **Winner takes all** on `S = Σ_r L_r`, optionally with sub-pixel refinement.
4. **Post-processing:**
   - *left-right consistency check*, reading the right-view disparity from the same `S`;
     failing pixels are mostly occlusions or mismatches;
   - *filling* each failed pixel with the smaller (more distant) disparity of its nearest
     valid neighbours in the row, since occluded areas belong to the background;
   - *median filter* to remove speckles.

Penalties are in units of the 0..1 cost volume, so `P1`/`P2` mean the same thing for every cost.

## Outputs (`sgm/data/`)

| Output | Content |
|---|---|
| `comparison_<scene>.png` | Left image, ground truth, block baselines, SGM raw and post-processed maps per cost, error maps |
| `sgm_vs_block_summary.png` | Mean MAE: block SAD/NCC vs SGM raw and post-processed |
| `sgm_per_scene.png` | The same per scene |
| `sgm_mae_vs_directions.png` | Raw SGM MAE with 1, 2, 4 and 8 directions |
| `sgm_p2_sweep.png` | Raw SGM MAE vs P2 |
| `sgm_results.csv` | Every number: scene, experiment (`baseline`, `main`, `directions`, `p2_sweep`, `lrc`), method, param, mae |
| `disparity_maps/<scene>/` | `sgm_<cost>_raw.png`, `sgm_<cost>_post.png`, `ground_truth.png` |

The `lrc` rows give the share of evaluated pixels that fail the left-right check.

## Notes

- **Memory:** the cost volume and the running sum are each `height × width × max_disp`
  float32, and the left-right check needs one more copy. At `--scale 0.5` this is tens of
  MB. At `--scale 1.0` expect roughly 0.5 GB per scene.
- **Penalties depend on the data:** P1 and P2 are not tuned for Middlebury. Check
  `sgm_p2_sweep.png` and adjust `--p1`, `--p2` accordingly.
- **Raw vs post-processed MAE:** the post-processed map is filled everywhere, so its MAE
  includes the filled occluded pixels. Compare it with the `lrc` share to see how much
  filling is involved.
