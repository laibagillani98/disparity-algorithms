# =============================================================================
# adaptive_block_matching.py  --  Per-pixel adaptive block size stereo matching
# =============================================================================
# PROBLEM
# -------
# A fixed block size is a compromise:
#   * small blocks  -> sharp depth edges, but ambiguous / noisy in flat regions
#   * large blocks  -> smooth and reliable in flat regions, but blur depth edges
#                      and swallow thin objects
# No single size is best everywhere in an image.
#
# APPROACH
# --------
# 1. For every candidate block size (e.g. 3, 5, 9, 15, 21) run ordinary block
#    matching (block_matching.cost_volume) and keep, per pixel,
#        - the winning disparity of that size, and
#        - a CONFIDENCE for it: the peak ratio (PKRN)
#              PKRN = (second-best cost + eps) / (best cost + eps)
#          where the second-best is taken over disparities that are NOT the
#          winner or its two neighbours (so a smooth cost valley is not mistaken
#          for ambiguity).  PKRN ~ 1 -> several disparities fit equally well
#          (ambiguous); PKRN >> 1 -> one clearly distinct match.
# 2. Choose a size for each pixel with one of two strategies:
#
#    "confidence"  Go from the SMALLEST to the LARGEST size and take the first
#                  size whose PKRN >= tau.  Pixels where no size reaches tau use
#                  the most confident size (ties -> larger size).
#                  Idea: use fine windows wherever they are already reliable,
#                  enlarge the window only where the match is ambiguous.
#
#    "texture"     Measure local texture = standard deviation of the LEFT image
#                  in a small window.  Rank pixels by texture and split them into
#                  as many equal-population classes as there are block sizes:
#                  most textured class -> smallest block, flattest class ->
#                  largest block.  Needs no matching costs and no threshold.
#
# 3. The final disparity of a pixel is the one found with its chosen block size.
#
# Neither strategy uses the ground truth: only the stereo pair is needed.  The
# ground truth is used afterwards for evaluation only.
#
# Metrics: SAD, SSD, MAD, NCC (as in block_matching.py).  MAD is SAD divided by a
# constant, so its PKRN and choices are identical to SAD; the runner therefore
# reports MAD = SAD.
#
# Disparity convention: left(y, x) <-> right(y, x - d), d >= 0.
# =============================================================================

import numpy as np

from block_matching import cost_volume, _box_sum


def disparity_and_confidence(vol, eps=1e-6):
    """Winner disparity and peak-ratio confidence from a cost volume (h, w, D).

    Returns (disp float32 with NaN where invalid, pkrn float64, 0 where invalid).
    """
    h, w, D = vol.shape
    d1 = np.argmin(vol, axis=2)
    c1 = np.take_along_axis(vol, d1[..., None], axis=2)[..., 0]
    invalid = ~np.isfinite(c1)

    masked = vol.copy()                      # remove winner and its neighbours
    for off in (-1, 0, 1):
        idx = np.clip(d1 + off, 0, D - 1)
        np.put_along_axis(masked, idx[..., None], np.inf, axis=2)
    c2 = masked.min(axis=2)

    pkrn = np.zeros_like(c1)                 # 0 = no estimate possible
    ok = ~invalid
    pkrn[ok] = (c2[ok] + eps) / (c1[ok] + eps)   # c2 may be +inf -> inf (confident)
    disp = d1.astype(np.float32)
    disp[invalid] = np.nan
    return disp, pkrn


def compute_candidates(left, right, max_disp, sizes, metric):
    """Run block matching for every size; returns {size: (disp, pkrn)}.

    Only one cost volume is held in memory at a time.
    """
    out = {}
    for k in sorted(sizes):
        vol = cost_volume(left, right, max_disp, k, metric)
        out[k] = disparity_and_confidence(vol)
        del vol
    return out


def select_by_confidence(candidates, tau=1.2):
    """Smallest block size whose peak ratio >= tau (else most confident size).

    Returns (disp, size_map); size_map holds the chosen block size (0 = none).
    """
    sizes = sorted(candidates)
    disps = np.stack([candidates[k][0] for k in sizes])      # (S, h, w)
    confs = np.stack([candidates[k][1] for k in sizes])
    S = len(sizes)

    ok = (confs >= tau) & np.isfinite(disps)
    first_ok = np.argmax(ok, axis=0)                          # first True along S
    has_ok = ok.any(axis=0)

    # fallback: most confident size; reversed so ties go to the LARGER block
    best_rev = np.argmax(confs[::-1], axis=0)
    fallback = S - 1 - best_rev

    idx = np.where(has_ok, first_ok, fallback)
    return _gather(disps, idx, sizes)


def texture_map(img, win=9):
    """Local standard deviation of intensity in a win x win window."""
    f = img.astype(np.float64)
    n = float(win * win)
    mean = _box_sum(f, win) / n
    var = _box_sum(f * f, win) / n - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def select_by_texture(candidates, left, win=9):
    """Textured pixels -> small blocks, flat pixels -> large blocks.

    Pixels are ranked by local texture and split into len(sizes) equal groups.
    Returns (disp, size_map, texture).
    """
    sizes = sorted(candidates)
    disps = np.stack([candidates[k][0] for k in sizes])
    confs = np.stack([candidates[k][1] for k in sizes])
    S = len(sizes)

    tex = texture_map(left, win)
    # ties (e.g. perfectly flat images) are broken by position via rank-based
    # classes, so every class is populated
    order = np.argsort(tex.ravel(), kind="stable")
    rank = np.empty_like(order); rank[order] = np.arange(order.size)
    level = np.minimum((rank * S) // order.size, S - 1).reshape(tex.shape)  # 0 = flattest
    idx = S - 1 - level                                                     # flat -> large
    disp, size_map = _gather(disps, idx, sizes)
    return disp, size_map, tex


def _gather(disps, idx, sizes):
    disp = np.take_along_axis(disps, idx[None], axis=0)[0]
    size_map = np.asarray(sizes)[idx]
    # pixels that no size could estimate stay NaN / size 0
    none = ~np.isfinite(disp)
    size_map = np.where(none, 0, size_map)
    return disp.astype(np.float32), size_map.astype(np.int32)
