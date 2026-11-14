# =============================================================================
# block_matching.py  --  Block (window) based stereo matching
#                        SAD, SSD, MAD and NCC for any odd block size
# =============================================================================
# APPROACH
# --------
# Instead of comparing single pixels, compare a k x k BLOCK around each left
# pixel with the k x k block around the candidate pixel in the right image
# (shifted left by d).  Using neighbours makes the match far less ambiguous and
# less noisy, at the price of blurring depth edges for large blocks.
#
# For each candidate disparity d = 0 .. max_disp-1:
#   1. Build a per-pixel difference image between left[:, d:] and right[:, :w-d].
#   2. Sum it over every k x k window.  This is done for ALL pixels at once with
#      cv2.boxFilter (a running-sum filter), so the cost is independent of k.
#   3. Keep, per pixel, the disparity with the best aggregated cost
#      (winner takes all).
#
# Metrics (lower cost = better, except NCC where we negate the similarity):
#   SAD : sum over block of |L - R|
#   SSD : sum over block of (L - R)^2          (punishes large errors more)
#   MAD : SAD / (k*k)                          mean absolute difference.  Dividing
#         by a constant does not change which d wins, so MAD returns the same
#         disparity map as SAD; it only changes the cost scale.
#   NCC : zero-mean normalised cross-correlation of the two blocks
#             sum((L-mL)(R-mR)) / sqrt(sum (L-mL)^2 * sum (R-mR)^2)
#         in [-1, 1]; 1 = identical up to gain and offset.  It is invariant to
#         brightness / contrast changes between the two cameras.  The sums are
#         obtained from box filters of L, R, L^2, R^2 and L*R.  Cost = -NCC.
#         Flat (textureless) blocks have ~0 variance -> NCC ~ 0 -> ambiguous.
#
# Border handling: a pixel only gets an estimate if its whole block lies inside
# both images for that d (x >= d + k//2, x < w - k//2, k//2 <= y < h - k//2).
# Pixels that never get a valid estimate stay NaN; evaluation ignores them.
#
# Disparity convention: left(y, x) <-> right(y, x - d), d >= 0.
# =============================================================================

import cv2
import numpy as np

METRICS = ("SAD", "SSD", "MAD", "NCC")


def _box_sum(img, k):
    return cv2.boxFilter(img, ddepth=-1, ksize=(k, k), normalize=False,
                         borderType=cv2.BORDER_REFLECT101)


def block_matching(left, right, max_disp=64, block_size=9, metric="SAD"):
    """Return a float32 disparity map with NaN where no estimate is possible."""
    metric = metric.upper()
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")

    l = left.astype(np.float64)
    r = right.astype(np.float64)
    h, w = l.shape
    k = block_size
    rad = k // 2
    n = float(k * k)

    if metric == "NCC":                      # statistics that do not depend on d
        sl, sl2 = _box_sum(l, k), _box_sum(l * l, k)
        sr, sr2 = _box_sum(r, k), _box_sum(r * r, k)
        var_l = sl2 - sl * sl / n

    best = np.full((h, w), np.inf)
    disp = np.full((h, w), np.nan, dtype=np.float32)

    for d in range(min(max_disp, w)):
        a = l[:, d:]
        b = r[:, : w - d]

        if metric in ("SAD", "MAD"):
            c = np.zeros((h, w)); c[:, d:] = np.abs(a - b)
            agg = _box_sum(c, k)
            if metric == "MAD":
                agg /= n
        elif metric == "SSD":
            c = np.zeros((h, w)); c[:, d:] = (a - b) ** 2
            agg = _box_sum(c, k)
        else:  # NCC
            prod = np.zeros((h, w)); prod[:, d:] = a * b
            sr_d = np.zeros((h, w)); sr_d[:, d:] = sr[:, : w - d]
            sr2_d = np.zeros((h, w)); sr2_d[:, d:] = sr2[:, : w - d]
            cov = _box_sum(prod, k) - sl * sr_d / n
            var_r = sr2_d - sr_d * sr_d / n
            ncc = cov / np.sqrt(np.maximum(var_l, 0) * np.maximum(var_r, 0) + 1e-6)
            agg = -ncc

        # invalidate positions whose full window is not inside both images
        agg[:, : d + rad] = np.inf
        agg[:, w - rad:] = np.inf
        agg[:rad, :] = np.inf
        agg[h - rad:, :] = np.inf

        better = agg < best
        best[better] = agg[better]
        disp[better] = d
    return disp
