# =============================================================================
# pixel_matching.py  --  Pixel-based stereo matching (SAD, SSD, MAD, NCC)
# =============================================================================
# APPROACH
# --------
# Pixel-based matching looks at ONE pixel of the left image and compares it
# with ONE pixel of the right image.  For a left pixel (y, x) and a candidate
# disparity d, the matching right pixel is (y, x - d).  For every candidate
# d = 0 .. max_disp-1 we compute a matching cost and keep the disparity with the
# best cost per pixel ("winner takes all").
#
# With a 1x1 window the four metrics become:
#   SAD : |L - R|                      (sum of absolute differences, 1 sample)
#   SSD : (L - R)^2                    (sum of squared differences, 1 sample)
#   MAD : |L - R| / 1                  (mean absolute difference, 1 sample)
#         -> numerically identical to SAD for a 1x1 window, so SAD and MAD give
#            the same disparity map here.  They differ only in name; the block
#            version divides by the window area but still ranks candidates
#            identically.
#   NCC : normalised cross-correlation needs more than one sample (the variance
#         of one pixel is zero), so a true window NCC is impossible.  As a
#         pixel-level analogue we z-score each IMAGE globally (subtract its mean,
#         divide by its std) and use the bounded similarity
#                 s = 2*zL*zR / (zL^2 + zR^2)        in [-1, 1]
#         which equals 1 only when zL == zR.  Cost = -s (higher similarity wins).
#
# Because a single pixel carries very little information, many wrong candidates
# have a cost similar to the right one -> the result is noisy.  This is the
# baseline that block matching improves on.
#
# Disparity convention: left(y, x) <-> right(y, x - d), d >= 0.
# Pixels with x < d have no right-image partner and are skipped for that d.
# =============================================================================

import numpy as np

METRICS = ("SAD", "SSD", "MAD", "NCC")


def _pixel_cost(l, r, d, metric, eps=1e-6):
    """Cost (lower = better) between l[:, d:] and r[:, :w-d]."""
    a = l[:, d:]
    b = r[:, : l.shape[1] - d]
    if metric in ("SAD", "MAD"):
        return np.abs(a - b)
    if metric == "SSD":
        return (a - b) ** 2
    if metric == "NCC":
        return -(2.0 * a * b) / (a * a + b * b + eps)
    raise ValueError(f"Unknown metric: {metric}")


def pixel_matching(left, right, max_disp=64, metric="SAD"):
    """Return a float32 disparity map (same shape as the images)."""
    metric = metric.upper()
    l = left.astype(np.float64)
    r = right.astype(np.float64)

    if metric == "NCC":                      # global z-score of each image
        l = (l - l.mean()) / (l.std() + 1e-6)
        r = (r - r.mean()) / (r.std() + 1e-6)

    h, w = l.shape
    best = np.full((h, w), np.inf)
    disp = np.zeros((h, w), dtype=np.float32)

    for d in range(min(max_disp, w)):
        cost = _pixel_cost(l, r, d, metric)
        better = cost < best[:, d:]
        best[:, d:][better] = cost[better]
        disp[:, d:][better] = d
    return disp
