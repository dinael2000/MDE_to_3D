# < -------------------------------------------------------------------
# Deprecated functions
# ------------------------------------------------------------------->

import torch
import cv2

import numpy as np

def fit_circle(mask):
    """
    Takes a binary mask of pixels on the rim of a seal/coin and 
    fits a circle to them using least-squares
    
    :param mask: Valid mask

    :return (cx, cy): Estimated center
    :return R: Estimated radius
    """

    # Extracts rim pixel coordinates
    ys, xs = np.where(mask)

    # Fallback in case of not enough points
    if len(xs) < 20:
        H, W = mask.shape
        # Returns image center, half the smaller dimension as radius
        return (W / 2, H / 2), min(H, W) / 2

    # Constructs linear least-squares linear system 
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = -(xs**2 + ys**2)

    # Solves least-squares system (A·c≈b)
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Converts algebraic parameters to geometric ones,
    # to recover center and radius
    cx = -c[0] / 2
    cy = -c[1] / 2
    R  = np.sqrt((c[0]**2 + c[1]**2) / 4 - c[2])

    return (cx, cy), R

def rmse_linear(pred, gt, valid_mask=None):
    """
    Computes per-sample RMSE within the linear space
    over spatial dimensions, between predicted 
    (affine invariant) and ground truth (metric)
    depth maps, within a masked region.

    :return: Average over batch
    """
    # Calculates raw difference between depth maps
    diff = pred - gt

    # Applies mask to difference
    # if applicable
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]

    # Squares the error
    diff2 = torch.pow(diff, 2)

    # Computes Mean Squared Error (MSE) over 
    # spatial dimensions
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()

def load_depth_map_old(path, clip_start=1e-6, clip_end=10.0):
    """
    A function that loads pre-existing Ground Truth (GT) depth maps.
    Works for depth maps created within the Blender ecosystem.
    Converts depth map to single-channel 2D array.

    :param path: path to GT depth map
    :param clip_start: Minimum distance to visible range
    from orthogonal perspective camera
    :param clip_end: Maximum distance to visible range
    from orthogonal perspective camera
    """

    gt_depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if gt_depth is None:
        raise ValueError(f"Could not load depth map: {path}")

    if gt_depth.ndim == 3:
        gt_depth = gt_depth[..., 0]

    if gt_depth.dtype == np.uint16:
        gt_depth = clip_start + (gt_depth.astype(np.float32) / 65535.0) * (clip_end - clip_start)

    elif gt_depth.dtype in (np.float16, np.float32, np.float64):
        gt_depth = gt_depth.astype(np.float32)

    else:
        max_val = np.iinfo(gt_depth.dtype).max
        print(f"[WARN] Depth map {path} is {gt_depth.dtype}, expected uint16. Rescaling as fallback.")
        gt_depth = clip_start + (gt_depth.astype(np.float32) / max_val) * (clip_end - clip_start)

    gt_depth[np.isnan(gt_depth)] = 0
    gt_depth[np.isinf(gt_depth)] = 0

    return gt_depth