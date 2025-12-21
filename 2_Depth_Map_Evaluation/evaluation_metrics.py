# < -------------------------------------------------------------------------
# A module with functions to calculate standard metrics 
# for the evaluation of monocular depth estimation depth maps 
# (affive invariant) to ground truth depth maps (metric).
# 
# Citations: See citations.txt
# ------------------------------------------------------------------------- >

import cv2
import torch

import numpy as np

from alignment import *
from detail_metrics import *

########################
## Evaluation Metrics ##
########################

def abs_rel(pred, gt, mask):
    """
    Computes Absolute Relative Error (AbsRel) between
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region.
    Essentially, measures the absolute difference between
    the aformentioned, scale by the true depth value.
    """
    diff = np.abs(pred[mask] - gt[mask]) / gt[mask]
    return diff.mean()

def sq_rel(pred, gt, mask):
    """ Computes Squared Relative Error (SqRel) between
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region.
    """
    diff = ((pred[mask] - gt[mask])**2) / gt[mask]
    return diff.mean()

def rmse(pred, gt, mask):
    """
    Computes Relative Mean Squared Error (RMSE) between
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region.
    """
    return np.sqrt(((pred[mask] - gt[mask])**2).mean())

def gradient_rmse(pred, gt, mask):
    """
    Computes RMSE of depth gradients between
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region. 
    Measures how well local slopes match.
    """
    gy_p, gx_p = np.gradient(pred)
    gy_g, gx_g = np.gradient(gt)
    diff = (gx_p - gx_g)**2 + (gy_p - gy_g)**2
    return np.sqrt(diff[mask].mean())

def rmse_log(pred, gt, mask):
    """
    Computes log-space RMSE between predicted 
    (affine invariant) and ground truth (metric)
    depth maps, within a masked region.
    """
    diff = np.log(pred[mask]) - np.log(gt[mask])
    return np.sqrt((diff**2).mean())

def silog(pred, gt, mask):
    """
    Computes Scale-Invariant Logarithmic Error (SILog) between 
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region. Removes global scale bias.
    """
    d = np.log(pred[mask]) - np.log(gt[mask])
    return np.sqrt((d**2).mean() - d.mean()**2)

def silog_100(pred, gt, mask):
    """
    Computes Scale-Invariant Logarithmic Error (SILog) multiplied by 100
    between predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region. Removes global scale bias.
    """
    d = np.log(pred[mask]) - np.log(gt[mask])
    return 100 * np.sqrt((d**2).mean() - d.mean()**2)

def delta_accuracy(pred, gt, mask, thresh):
    """
    Computes δ-threshold accuracy between 
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region. Measures how close
    pred is to gt regardless of scale
    
    :param thresh: Given theshold
    """
    ratio = np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return (ratio < thresh).mean()

def laplacian_error(pred, gt, mask):
    """
    Computes absolute error of Laplacians between 
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region. 
    For more info, see: detail_metric.py module.
    """
    lap_p = cv2.Laplacian(pred, cv2.CV_32F)
    lap_g = cv2.Laplacian(gt, cv2.CV_32F)
    return np.mean(np.abs(lap_p[mask] - lap_g[mask]))


def highfreq_ratio(pred, gt, mask):
    """
    Computes relative amount of high-frequency detail between 
    predicted (affine invariant) and ground truth (metric)
    depth maps, within a masked region. 

    c. 1.0 -> perfect detail contrast
    < 1.0 -> too smooth
    1.0 -> too noisy
    For more info, see: detail_metric.py module.
    """

    # High-pass filtering using Laplacian
    hp_p = cv2.Laplacian(pred, cv2.CV_32F)
    hp_g = cv2.Laplacian(gt, cv2.CV_32F)

    # Computes high-frequency energy inside mask
    num = np.sum(hp_p[mask]**2)
    den = np.sum(hp_g[mask]**2) + 1e-8
    
    return num / den

def normal_error(pred, gt, mask):
    """
    Computes per-pixel angular error between predicted 
    and ground truth depth maps using the dot product.
    :return: Angular error in radians
    """
    # Computes normals
    n_p = compute_normals(pred)
    n_g = compute_normals(gt)

    # Computes angular error
    dot = np.sum(n_p * n_g, axis=2)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)

    return np.mean(ang[mask])

