# < -------------------------------------------------------------------
# Module with functions to produce masks for given depth maps
# ------------------------------------------------------------------->

import cv2

import numpy as np

def build_valid_mask(gt, pred=None):
    """
    Builds boolean mask with valid, usable depth values 
    from given depth map.
    
    :param gt: Ground truth depth map
    :param pred: Predicted depth map

    :return mask: Combined mask of gt and pred depth map
    """

    # Calculates valid pixels of gt depth map
    mask = (gt > 1e-6) & np.isfinite(gt) # Filters zero, negative or finite pixels
    
    # Calculates valid pixels of pred depth map
    # (if provided)
    if pred is not None:
        mask &= (pred > 1e-6) & np.isfinite(pred)

    return mask

def build_relief_mask(gt):
    """
    Builds relief mask from high-curvature details in a depth map
    by thresholding its absolute Laplacian to the top 10% curvature values.
    
    :param gt: Ground truth depth map

    :return mask_relief: Boolean mask of sharpest curvature regions
    """

    # Computes Laplacian of gt depth map
    lap = cv2.Laplacian(gt, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)

    # Eliminates weakest 90% pixels, as corresponding 
    # to low-curvature values.
    hi = np.percentile(lap_abs, 90)

    # Fallback in case of nearly flat depth maps 
    # (i.e. with little curvature)
    if hi < 1e-8:
        hi = np.percentile(lap_abs, 70)

    # Builds relief mask
    mask_relief = lap_abs >= hi
    
    return mask_relief

def build_region_masks(gt):
    """
    Percentile-based region masks for Option C
    """
    valid = gt > 0
    g = gt[valid]

    p10, p20, p80, p90 = np.percentile(g, [10,20,80,90])

    mask_center   = (gt >= p20) & (gt <= p80)
    mask_mid      = ((gt >= p10) & (gt < p20)) | ((gt > p80) & (gt <= p90))
    mask_boundary = (gt > 0) & ((gt < p10) | (gt > p90))

    return mask_center, mask_mid, mask_boundary