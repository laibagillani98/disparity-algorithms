# =============================================================================
# data_loader.py  --  Loading Middlebury stereo scenes and their ground truth
# =============================================================================
# APPROACH
# --------
# Every scene lives in two places (paths relative to  disparity_algorithms/data):
#
#     trainingQ/<scene>/im0.png  (or img0.png)   left  image
#     trainingQ/<scene>/im1.png  (or img1.png)   right image
#     trainingQ/<scene>/calib.txt                camera calibration (vmin/vmax/ndisp)
#     trainingDisparity/<scene>/disp0GT.pfm      ground-truth disparity (left view)
#
# Steps performed for each scene (they mirror the notebook):
#   1. Read both images as grayscale and resize by `scale` (default 0.5) so the
#      experiments run quickly.  cv2.INTER_AREA is used for down-sampling.
#   2. Read the PFM ground truth and multiply by the |PFM scale| factor stored in
#      its header (the notebook verified this gives disparities that agree with
#      the calibration range).
#   3. Resize the ground truth to the image size (nearest neighbour, so that
#      invalid pixels are not blended with valid ones) and multiply the disparity
#      values by `scale`, because a horizontal shift of d pixels becomes d*scale
#      pixels when the image width is scaled.
#   4. Build a boolean validity mask: ground-truth pixels that are finite and > 0
#      (Middlebury stores occluded / unknown pixels as inf).
#   5. Choose the disparity search range from calib.txt (vmax, else ndisp), scaled
#      like the images; fall back to 64 if no calibration is available.
#      It can be overridden from the command line (--max-disp).
# =============================================================================

from pathlib import Path
import math
import re

import cv2
import numpy as np

LEFT_NAMES = ("im0.png", "img0.png")
RIGHT_NAMES = ("im1.png", "img1.png")


def _find_file(folder: Path, names):
    for n in names:
        p = folder / n
        if p.exists():
            return p
    return None


def list_scenes(data_root: Path, limit=None, names=None):
    """Return sorted scene names that have images and a ground-truth file."""
    q_dir = Path(data_root) / "trainingQ"
    gt_dir = Path(data_root) / "trainingDisparity"
    if not q_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {q_dir}")

    if names:
        scenes = list(names)
    else:
        scenes = sorted(p.name for p in q_dir.iterdir() if p.is_dir())

    ok = []
    for s in scenes:
        has_imgs = _find_file(q_dir / s, LEFT_NAMES) and _find_file(q_dir / s, RIGHT_NAMES)
        has_gt = (gt_dir / s / "disp0GT.pfm").exists()
        if has_imgs and has_gt:
            ok.append(s)
        else:
            print(f"[skip] {s}: missing images or disp0GT.pfm")
    return ok[:limit] if limit else ok


def _read_pfm_manual(path: Path):
    """Minimal PFM reader (fallback if OpenCV was built without PFM support)."""
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        channels = 3 if header == "PF" else 1
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, dtype=endian + "f4")
    img = data.reshape((h, w, channels) if channels == 3 else (h, w))
    return np.flipud(img).astype(np.float32)   # PFM is stored bottom-to-top


def read_pfm(path: Path):
    """Return (raw array, |scale| from header)."""
    with open(path, "rb") as f:
        f.readline()
        f.readline()
        pfm_scale = abs(float(f.readline().decode().strip()))
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raw = _read_pfm_manual(path)
    return raw.astype(np.float32), pfm_scale


def _read_calib(calib_path: Path):
    """Parse 'key=value' lines of a Middlebury calib.txt into a dict of floats."""
    values = {}
    if not calib_path.exists():
        return values
    for line in calib_path.read_text().splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*([-+0-9.eE]+)\s*$", line)
        if m:
            values[m.group(1)] = float(m.group(2))
    return values


def load_scene(name, data_root, scale=0.5, max_disp=None):
    """Load one scene and return a dict with everything the algorithms need."""
    data_root = Path(data_root)
    folder = data_root / "trainingQ" / name

    left = cv2.imread(str(_find_file(folder, LEFT_NAMES)), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(_find_file(folder, RIGHT_NAMES)), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError(f"Could not read stereo images for scene {name}")

    gt_raw, pfm_scale = read_pfm(data_root / "trainingDisparity" / name / "disp0GT.pfm")
    gt = gt_raw * pfm_scale
    valid = np.isfinite(gt) & (gt > 0)

    orig_h, orig_w = left.shape
    if scale != 1.0:
        new_size = (int(round(orig_w * scale)), int(round(orig_h * scale)))
        left = cv2.resize(left, new_size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, new_size, interpolation=cv2.INTER_AREA)
        gt_clean = np.where(valid, gt, 0).astype(np.float32)
        gt = cv2.resize(gt_clean, new_size, interpolation=cv2.INTER_NEAREST) * scale
        valid = cv2.resize(valid.astype(np.uint8), new_size,
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    gt = np.where(valid, gt, 0).astype(np.float32)

    # ---- disparity search range -------------------------------------------
    calib = _read_calib(folder / "calib.txt")
    if max_disp is None:
        ref = calib.get("vmax", calib.get("ndisp"))
        max_disp = int(math.ceil(ref * scale)) + 1 if ref else 64
    if gt[valid].size and gt[valid].max() > max_disp:
        print(f"[warn] {name}: ground truth reaches {gt[valid].max():.1f}px but "
              f"max_disp={max_disp}. Check PFM scaling or pass --max-disp.")

    return {"name": name, "left": left, "right": right, "gt": gt,
            "valid": valid, "max_disp": int(max_disp)}
