# 1_algo – Pixel-based and Block-based Stereo Matching

Classical stereo disparity estimation on the Middlebury dataset. Given a rectified
left/right image pair, each algorithm estimates, for every left pixel, how far
(in pixels) it has shifted in the right image. Results are scored against the
ground-truth disparity with Mean Absolute Error (MAE).

Everything lives in `block_matching/`.

## Folder layout

```
disparity_algorithms/
├── data/                              <- input dataset (you provide it)
│   ├── trainingQ/<scene>/             im0.png (or img0.png), im1.png (or img1.png), calib.txt
│   └── trainingDisparity/<scene>/     disp0GT.pfm
├── requirements.txt
└── 1_algo/
    ├── README.md
    └── block_matching/
        ├── src/                       <- code
        └── data/                      <- generated plots, disparity images, CSV
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy, opencv-python, matplotlib, pandas
```

## Run

```bash
python 1_algo/block_matching/src/run_experiments.py
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--data-root` | `disparity_algorithms/data` | Folder containing `trainingQ` and `trainingDisparity` |
| `--out-dir` | `block_matching/data` | Where results are saved |
| `--num-folders` | 10 | Number of scenes (first N alphabetically) |
| `--folders A B ...` | – | Explicit scene names instead of the first N |
| `--scale` | 0.5 | Image resize factor (for speed) |
| `--max-disp` | from `calib.txt` | Disparity search range |
| `--block-sizes` | 3 5 9 15 21 | Odd block sizes to evaluate |
| `--display-block` | 9 | Block size used in bar charts and comparison sheets |

Examples:

```bash
python 1_algo/block_matching/src/run_experiments.py --folders Adirondack Motorcycle
python 1_algo/block_matching/src/run_experiments.py --block-sizes 3 5 7 9 11 15 --max-disp 64
```

## Algorithms

For a left pixel `(y, x)` and candidate disparity `d`, the match is the right pixel
`(y, x - d)`. For every `d` in `0 .. max_disp-1` a matching cost is computed and the
disparity with the best cost is kept per pixel (winner takes all).

| Metric | Cost | Better when |
|---|---|---|
| SAD | sum of \|L − R\| | lower |
| SSD | sum of (L − R)² | lower |
| MAD | SAD / (k·k) | lower |
| NCC | zero-mean normalised cross-correlation of the two blocks | higher |

**Pixel-based** (`pixel_matching.py`) compares single pixels (a 1×1 window). It is
fast but noisy, because one intensity value is ambiguous.

**Block-based** (`block_matching.py`) compares k×k blocks. Costs are summed over
every window at once with `cv2.boxFilter`, so runtime does not depend on k. Larger
blocks give smoother maps but blur depth edges.

Notes:

- **MAD ≡ SAD:** dividing by a constant never changes which disparity wins, so MAD
  gives the same map and MAE as SAD. Both are reported because both were requested.
  For a 1×1 window MAD equals SAD as well.
- **Pixel-based NCC:** a single pixel has no variance, so true NCC is undefined. The
  code uses a bounded pixel-level analogue, `2·zL·zR / (zL² + zR²)`, on globally
  z-scored images. Block-based NCC is the standard zero-mean version.
- **Borders:** a pixel gets a block estimate only if its whole window lies inside both
  images. Pixels without one are NaN and drawn black.

## Ground truth and evaluation

- The PFM is read, multiplied by the |scale| in its header, resized to the image size
  with nearest-neighbour interpolation, and its values scaled by `--scale`.
- Valid ground-truth pixels are finite and > 0.
- The search range comes from `vmax` (else `ndisp`) in `calib.txt`, scaled like the
  images, falling back to 64. A warning is printed if the ground truth exceeds it.
- `MAE = mean(|predicted − ground truth|)` in pixels, computed on the **same pixel
  mask for every method in a scene**: valid ground truth, away from borders for the
  largest block and disparity, and finite prediction.

## Source files

| File | Purpose |
|---|---|
| `data_loader.py` | Reads images, PFM ground truth, calibration; resizes and scales |
| `pixel_matching.py` | Pixel-based SAD / SSD / MAD / NCC |
| `block_matching.py` | Block-based SAD / SSD / MAD / NCC for any odd block size |
| `evaluation.py` | Common evaluation mask and MAE |
| `visualization.py` | Disparity images and all plots |
| `run_experiments.py` | Runs everything and writes the outputs |

## Outputs (`block_matching/data/`)

| Output | Content |
|---|---|
| `disparity_maps/<scene>/` | Colour-coded PNGs: ground truth, 4 pixel-based maps, every block size × metric |
| `comparison_<scene>.png` | Left image, ground truth, pixel-based and block-based maps with MAE in titles |
| `mae_per_scene.png` | Grouped bars of the 4 metrics per scene, pixel-based vs block-based |
| `mae_summary_pixel_vs_block.png` | MAE averaged over scenes, pixel-based vs block-based |
| `mae_vs_block_size.png` | Mean MAE vs block size, one line per metric |
| `mae_vs_block_size_per_scene.png` | The same curve for each scene |
| `mae_results.csv` | All numbers: scene, approach, block_size, metric, mae |

All disparity maps share one colour scale (plasma, 0 to `max_disp`) so they can be
compared by eye. A pixel-based result has `block_size = 1` in the CSV.
