# < -------------------------------------------------------------------------
# A module with functions to debug alignment of depth maps
# ------------------------------------------------------------------------- >

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

#################
## Diagnostics ##
################# 

    
def alignment_diagnostics(gt, pred, mask, out_dir, image_name="sample"):
    """
    Generates diagnostic figures to verify validity of alignment 
    between a ground truth and a predicted depth map.
    Saves 4 plots:

    1. Scatter plot, to confirm correlation between depth maps

    2. Depth direction histogram, to confirm
    whether the depth map has the correct
    direction or is inverted (near is far, far is near)

    3. Percentile Distribution Comparison, to confirm
    whether shape and spread match between depth maps

    4. Visual overlay of depth maps
    
    :param gt: Ground truth depth map
    :param pred: Predicted depth map
    :param mask: Given mask
    :param out_dir: Path to save diagnostics
    :param image_name: Object name
    """
    # Initializes output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Converts depth maps to floats
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    # Applies mask to depth maps
    gt_valid = gt[mask]
    pred_valid = pred[mask]

    # Visualizes scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(gt_valid, pred_valid, s=1, alpha=0.3)
    plt.xlabel("GT depth")
    plt.ylabel("Pred raw depth")
    plt.title("GT vs Pred Raw Scatter\n" + image_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{image_name}_scatter_raw.png", dpi=150)
    plt.close()

    # Calculates and visualizes direction of depth map
    corr = np.corrcoef(gt_valid, pred_valid)[0, 1]

    plt.figure(figsize=(6, 4))
    plt.hist(pred_valid, bins=80, alpha=0.6, label="Pred Depth")
    plt.hist(gt_valid, bins=80, alpha=0.6, label="GT Depth")
    plt.legend()
    plt.title(f"Depth Direction Test\nCorrelation = {corr:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{image_name}_direction_test.png", dpi=150)
    plt.close()

    # Calculates percentile distribution
    p = np.linspace(0, 100, 50)
    gt_p = np.percentile(gt_valid, p)
    pred_p = np.percentile(pred_valid, p)

    plt.figure(figsize=(6, 4))
    plt.plot(p, gt_p, label="GT")
    plt.plot(p, pred_p, label="Pred Raw")
    plt.xlabel("Percentile")
    plt.ylabel("Depth Value")
    plt.title("Percentile Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{image_name}_percentiles.png", dpi=150)
    plt.close()

    # Creates visual overlay
    # after computing affine fit
    m = mask
    A = np.vstack([pred[m], np.ones_like(pred[m])]).T
    b = gt[m]
    scale, shift = np.linalg.lstsq(A, b, rcond=None)[0]
    pred_aligned = pred * scale + shift

    # Normalizes for visualization
    def norm(x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    # Creates plots
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(norm(gt), cmap="inferno")
    plt.title("GT Depth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(norm(pred_aligned), cmap="inferno")
    plt.title("Aligned Prediction")
    plt.axis("off")

    plt.tight_layout()

    # Saves plots
    plt.savefig(out_dir / f"{image_name}_aligned_overlay.png", dpi=150)
    plt.close()

    # Returns numerical diagnostics
    return {
        "correlation_raw": float(corr),
        "gt_min": float(gt_valid.min()),
        "gt_max": float(gt_valid.max()),
        "pred_min": float(pred_valid.min()),
        "pred_max": float(pred_valid.max()),
        "scale_est": float(scale),
        "shift_est": float(shift)
    }
