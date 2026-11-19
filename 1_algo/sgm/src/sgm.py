# =============================================================================
# sgm.py  --  Semi-Global Matching (SGM, Hirschmueller 2005/2008)
# =============================================================================
# IDEA
# ----
# Block matching decides every pixel independently, so it is noisy in flat or
# ambiguous areas.  A global method would minimise an energy over the whole image
#       E(D) = sum_p C(p, D_p)
#            + sum_{q in N(p)} P1 * [ |D_p - D_q| == 1 ]
#            + sum_{q in N(p)} P2 * [ |D_p - D_q|  > 1 ]
# (data cost + small penalty P1 for 1-pixel disparity changes, which allows
# slanted surfaces, + larger penalty P2 for jumps) but that is NP-hard in 2-D.
# SGM approximates it by solving the problem exactly along many 1-D paths
# through the image and summing the results.
#
# STEPS
# -----
# 1. Cost volume C(p, d)                                       (sgm_costs.py)
# 2. PATH AGGREGATION.  For a path direction r, recursively
#        L_r(p, d) = C(p, d) + min( L_r(p-r, d),
#                                   L_r(p-r, d-1) + P1,
#                                   L_r(p-r, d+1) + P1,
#                                   min_k L_r(p-r, k) + P2 )
#                    - min_k L_r(p-r, k)
#    The last term only keeps values from growing; it does not change argmins.
#    8 directions are used (horizontal, vertical, 4 diagonals).  Each direction
#    is processed as a loop along its main axis that is vectorised over the
#    other axis, so the Python loop has only w or h iterations.
#    P2 is ADAPTIVE: P2_eff = max(P1, P2 * min(1, T / |I(p) - I(p-r)|)), i.e.
#    it is lowered at intensity edges, where depth discontinuities are likely.
# 3. S(p, d) = sum_r L_r(p, d);  disparity = argmin_d S(p, d)  (winner takes all).
# 4. Optional sub-pixel refinement: parabola through S at d-1, d, d+1.
# 5. POST-PROCESSING (standard SGM):
#      * left-right consistency check: the disparity of the right image is read
#        from the same S (right pixel x' with disparity d <-> left pixel x'+d);
#        a pixel is kept only if |dL(x) - dR(x - dL(x))| <= 1.  Failures are
#        mostly occlusions and mismatches.
#      * invalid pixels are filled per row with the SMALLER (= more distant)
#        disparity of the nearest valid pixel on the left / right, because
#        occluded regions belong to the background.
#      * a small median filter removes remaining speckles.
#
# Penalties are in units of the [0, 1] cost volume (see sgm_costs.py).
# Disparity convention: left(y, x) <-> right(y, x - d), d >= 0.
# =============================================================================

import cv2
import numpy as np

# order chosen so that the first 1, 2 and 4 entries are meaningful subsets:
# 1 = left->right, 2 = horizontal, 4 = horizontal + vertical, 8 = all
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
GRAD_T = 8.0 / 255.0        # intensity step above which P2 starts to shrink


def _align(a, dy):
    """aligned[y] = a[y - dy]; rows without a source are zero."""
    if dy == 0:
        return a
    out = np.zeros_like(a)
    if dy > 0:
        out[dy:] = a[:-dy]
    else:
        out[:dy] = a[-dy:]
    return out


def _step(prev, c, p1, p2v):
    """One recursion step for n paths at once. prev, c: (n, D); p2v: (n,)."""
    m = prev.min(axis=1, keepdims=True)
    best = prev.copy()
    np.minimum(best[:, 1:], prev[:, :-1] + p1, out=best[:, 1:])     # from d-1
    np.minimum(best[:, :-1], prev[:, 1:] + p1, out=best[:, :-1])    # from d+1
    np.minimum(best, m + p2v[:, None], out=best)                    # any jump
    return c + best - m


def _p2_vector(img_a, img_b, p1, p2, adaptive):
    if not adaptive:
        return np.full(img_a.shape, max(p1, p2), dtype=np.float32)
    di = np.abs(img_a - img_b)
    return np.maximum(p1, p2 * np.minimum(1.0, GRAD_T / np.maximum(di, 1e-6))).astype(np.float32)


def aggregate_direction(cost, img, dy, dx, p1, p2, adaptive=True):
    """Aggregated cost L_r for one direction r = (dy, dx); predecessor is (y-dy, x-dx)."""
    h, w, D = cost.shape
    L = np.empty_like(cost)

    if dx != 0:                                         # loop over columns
        xs = range(w) if dx > 0 else range(w - 1, -1, -1)
        for i, x in enumerate(xs):
            c = cost[:, x, :]
            if i == 0:
                L[:, x, :] = c
                continue
            px = x - dx
            prev = _align(L[:, px, :], dy)
            p2v = _p2_vector(img[:, x], _align(img[:, px], dy), p1, p2, adaptive)
            new = _step(prev, c, p1, p2v)
            if dy > 0:                                  # rows with no predecessor
                new[:dy] = c[:dy]
            elif dy < 0:
                new[dy:] = c[dy:]
            L[:, x, :] = new
    else:                                               # vertical: loop over rows
        ys = range(h) if dy > 0 else range(h - 1, -1, -1)
        for i, y in enumerate(ys):
            c = cost[y]
            if i == 0:
                L[y] = c
                continue
            py = y - dy
            p2v = _p2_vector(img[y], img[py], p1, p2, adaptive)
            L[y] = _step(L[py], c, p1, p2v)
    return L


def aggregate_iter(cost, img, p1=0.04, p2=0.2, directions=DIRECTIONS, adaptive=True):
    """Yield (n_done, S) after each direction, S = running sum of the L_r.

    S is updated in place, so use it (e.g. argmin) before advancing.
    """
    imgf = img.astype(np.float32) / 255.0
    S = np.zeros_like(cost)
    for i, (dy, dx) in enumerate(directions, 1):
        S += aggregate_direction(cost, imgf, dy, dx, p1, p2, adaptive)
        yield i, S


def aggregate(cost, img, p1=0.04, p2=0.2, directions=DIRECTIONS, adaptive=True):
    S = None
    for _, S in aggregate_iter(cost, img, p1, p2, directions, adaptive):
        pass
    return S


def winner_takes_all(S):
    return np.argmin(S, axis=2)


def subpixel_refine(S, d):
    """Parabola fit through S[d-1], S[d], S[d+1]; returns float32 disparity."""
    D = S.shape[2]
    dc = np.clip(d, 1, D - 2)
    take = lambda k: np.take_along_axis(S, (dc + k)[..., None], axis=2)[..., 0]
    s0, sm, sp = take(0), take(-1), take(1)
    denom = sm - 2 * s0 + sp
    shift = np.where(denom > 1e-9, (sm - sp) / (2 * denom + 1e-12), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    out = d.astype(np.float32)
    interior = (d >= 1) & (d <= D - 2)
    out[interior] += shift[interior].astype(np.float32)
    return out


def left_right_check(S, d_left, max_diff=1):
    """Boolean mask of left pixels whose disparity agrees with the right view."""
    h, w, D = S.shape
    s_right = np.full_like(S, np.inf)
    for d in range(D):                       # right pixel x' with d <-> left x'+d
        s_right[:, : w - d, d] = S[:, d:, d]
    d_right = np.argmin(s_right, axis=2)
    del s_right
    x = np.arange(w)[None, :]
    xr = x - d_left.astype(int)
    inside = xr >= 0
    d_right_at = np.take_along_axis(d_right, np.clip(xr, 0, w - 1), axis=1)
    return inside & (np.abs(d_left.astype(int) - d_right_at) <= max_diff)


def fill_invalid(disp, ok):
    """Per row, fill invalid pixels with the smaller of the nearest valid neighbours."""
    h, w = disp.shape
    cols = np.arange(w)[None, :]
    li = np.maximum.accumulate(np.where(ok, cols, -1), axis=1)
    ri = np.minimum.accumulate(np.where(ok, cols, w)[:, ::-1], axis=1)[:, ::-1]
    lv = np.take_along_axis(disp, np.clip(li, 0, w - 1), axis=1)
    rv = np.take_along_axis(disp, np.clip(ri, 0, w - 1), axis=1)
    has_l, has_r = li >= 0, ri < w
    fill = np.where(has_l & has_r, np.minimum(lv, rv),
                    np.where(has_l, lv, np.where(has_r, rv, 0.0)))
    return np.where(ok, disp, fill).astype(np.float32)


def sgm(left, right, cost, p1=0.04, p2=0.2, directions=DIRECTIONS, adaptive=True,
        subpixel=False, postprocess=True, median=3, S=None):
    """Run SGM from a cost volume.

    Returns dict with 'raw' (winner-takes-all), 'post' (consistency check, filling,
    median; equals raw if postprocess=False), 'consistent' (bool mask) and 'S'.
    """
    if S is None:
        S = aggregate(cost, left, p1, p2, directions, adaptive)
    d_int = winner_takes_all(S)
    d_flt = subpixel_refine(S, d_int) if subpixel else d_int.astype(np.float32)
    out = {"raw": d_flt, "S": S}
    if not postprocess:
        out["post"], out["consistent"] = d_flt, np.ones(d_flt.shape, bool)
        return out
    ok = left_right_check(S, d_int)
    post = fill_invalid(d_flt, ok)
    if median and median > 1:
        post = cv2.medianBlur(post, median)
    out["post"], out["consistent"] = post, ok
    return out
