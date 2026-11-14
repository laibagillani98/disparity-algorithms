# =============================================================================
# evaluation.py  --  Mean Absolute Error (MAE) against the ground truth
# =============================================================================
# APPROACH
# --------
#   MAE = mean( | predicted_disparity - ground_truth_disparity | )   [pixels]
#
# To compare algorithms fairly, every algorithm of a scene is scored on the SAME
# set of pixels:
#   * ground truth is valid (finite and > 0),
#   * the pixel is far enough from the borders for the LARGEST block size and the
#     LARGEST disparity (so no method is penalised for lacking a full window),
#   * the prediction is finite.
# =============================================================================

import numpy as np


def common_mask(valid, max_disp, max_block):
    rad = max_block // 2
    mask = valid.copy()
    h, w = mask.shape
    mask[:, : max_disp - 1 + rad] = False
    mask[:, w - rad:] = False
    mask[:rad, :] = False
    mask[h - rad:, :] = False
    return mask


def mae(pred, gt, mask):
    m = mask & np.isfinite(pred)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m])))
