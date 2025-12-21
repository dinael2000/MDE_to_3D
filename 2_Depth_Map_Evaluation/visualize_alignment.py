import cv2

import numpy as np

def normalize_for_vis(depth):
    d = depth.copy().astype(np.float32)
    mask = np.isfinite(d)
    if mask.sum() == 0:
        return np.zeros_like(d, np.uint8)
    mn, mx = d[mask].min(), d[mask].max()
    if mx == mn:
        return np.zeros_like(d, np.uint8)
    d = (d - mn) / (mx - mn)
    return (d * 255).clip(0,255).astype(np.uint8)


def colorize_error_map(err):
    err = err.astype(np.float32)
    err = err / (np.percentile(err,95)+1e-6)
    err = np.clip(err,0,1)
    return cv2.applyColorMap((err*255).astype(np.uint8), cv2.COLORMAP_JET)


def visualize_alignment_steps(pred_raw, pred_geo, pred_fixed, pred_aligned, gt,
                              save_path, title="Alignment Debug"):

    H_g, W_g = gt.shape

    def to_vis(img, is_depth=True):
        if is_depth:
            img = normalize_for_vis(img)
        else:
            img = img.astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # resize everything to GT size for clean stacking
        return cv2.resize(img, (W_g, H_g), interpolation=cv2.INTER_NEAREST)

    # depth views
    raw_vis     = to_vis(pred_raw)
    geo_vis     = to_vis(pred_geo)
    fixed_vis   = to_vis(pred_fixed)
    aligned_vis = to_vis(pred_aligned)
    gt_vis      = to_vis(gt)

    # error maps (need numeric resize to GT first)
    raw_err     = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_raw.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)
    geo_err     = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_geo.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)
    fixed_err   = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_fixed.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)
    aligned_err = to_vis(colorize_error_map(
                         np.abs(cv2.resize(pred_aligned.astype(np.float32),(W_g,H_g)) - gt)),
                         is_depth=False)

    zero_err    = to_vis(colorize_error_map(np.zeros((H_g,W_g), np.float32)),
                         is_depth=False)

    def label(img, text):
        h, w = img.shape[:2]
        canvas = np.zeros((h+40, w, 3), np.uint8)
        canvas[40:] = img
        cv2.putText(canvas, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return canvas

    row1 = np.hstack([
        label(raw_vis,     "1. Raw Pred"),
        label(geo_vis,     "2. Resized (Geo)"),
        label(fixed_vis,   "3. Flipped"),
        label(aligned_vis, "4. Scale+Shift"),
        label(gt_vis,      "5. GT")
    ])

    row2 = np.hstack([
        label(raw_err,     "Raw Err"),
        label(geo_err,     "Geo Err"),
        label(fixed_err,   "Flip Err"),
        label(aligned_err, "Final Err"),
        label(zero_err,    "")
    ])

    final = np.vstack([row1, row2])

    # Title bar
    pad_top = np.zeros((70, final.shape[1], 3), np.uint8)
    cv2.putText(pad_top, title, (20,45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 3)

    out = np.vstack([pad_top, final])
    cv2.imwrite(str(save_path), out)
    print("[debug] Saved alignment visualization →", save_path)