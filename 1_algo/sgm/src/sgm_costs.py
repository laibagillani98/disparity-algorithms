# =============================================================================
# sgm_costs.py  --  Pixel matching costs that feed Semi-Global Matching
# =============================================================================
# APPROACH
# --------
# SGM starts from a cost volume C[y, x, d]: how badly left pixel (y, x) matches
# right pixel (y, x - d).  Unlike block matching, the costs here are only LOCAL
# (a tiny window); the smoothness between neighbouring pixels is added later by
# the path aggregation in sgm.py.
#
# Three costs are available, all scaled to [0, 1] so the SGM penalties P1 / P2
# mean the same thing for every cost:
#   sad    : |L - R| / 255, averaged over a small window (default 3x3)
#   ssd    : (L - R)^2 / 255^2, averaged over the same window
#   census : Census transform (each pixel -> bit string telling which neighbours
#            in a win x win window are darker than the centre); the cost is the
#            Hamming distance of the two bit strings divided by the number of
#            bits.  It depends only on the ORDER of intensities, so it is robust
#            to brightness / contrast differences between the two cameras.
#
# Disparity convention: left(y, x) <-> right(y, x - d), d >= 0.
# Entries with x < d have no partner in the right image and get the maximum
# cost 1.0.
# =============================================================================

import cv2
import numpy as np

COSTS = ("sad", "ssd", "census")


def _popcount64(x):
    """Number of set bits of every element of a uint64 array."""
    u = np.uint64
    x = x - ((x >> u(1)) & u(0x5555555555555555))
    x = (x & u(0x3333333333333333)) + ((x >> u(2)) & u(0x3333333333333333))
    x = (x + (x >> u(4))) & u(0x0F0F0F0F0F0F0F0F)
    return (x * u(0x0101010101010101)) >> u(56)


def census_transform(img, win=5):
    """uint64 census code per pixel; win must be odd and <= 7 (<= 48 bits)."""
    if win % 2 == 0 or win > 7 or win < 3:
        raise ValueError("census window must be 3, 5 or 7")
    r = win // 2
    h, w = img.shape
    pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REPLICATE)
    centre = pad[r:r + h, r:r + w]
    code = np.zeros((h, w), dtype=np.uint64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            nb = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            code = (code << np.uint64(1)) | (nb < centre).astype(np.uint64)
    return code, win * win - 1


def compute_cost_volume(left, right, max_disp, kind="census", win=3, census_win=5):
    """Return float32 cost volume (h, w, D), D = min(max_disp, w), values in [0, 1]."""
    kind = kind.lower()
    if kind not in COSTS:
        raise ValueError(f"Unknown cost: {kind}")
    h, w = left.shape
    D = min(max_disp, w)
    vol = np.ones((h, w, D), dtype=np.float32)

    if kind == "census":
        cl, nbits = census_transform(left, census_win)
        cr, _ = census_transform(right, census_win)
        for d in range(D):
            ham = _popcount64(cl[:, d:] ^ cr[:, : w - d])
            vol[:, d:, d] = ham.astype(np.float32) / nbits
        return vol

    l = left.astype(np.float32) / 255.0
    r = right.astype(np.float32) / 255.0
    for d in range(D):
        diff = np.abs(l[:, d:] - r[:, : w - d])
        if kind == "ssd":
            diff = diff * diff
        if win > 1:
            diff = cv2.blur(diff, (win, win), borderType=cv2.BORDER_REFLECT101)
        vol[:, d:, d] = diff
    return vol
