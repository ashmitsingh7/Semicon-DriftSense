"""Shared utilities for localization modules."""

import numpy as np
import cv2


def prep_search(reference_img, search_img, nominal_downsample=10.0,
                sigma_prefilter=0.6):
    """Normalize, mild denoise, and compute nominal template size.
    Returns (ref_dn, srch_dn, nominal_side, nominal_templ)."""
    ref = reference_img.astype(np.float32)
    srch = search_img.astype(np.float32)
    if ref.max() > 1.5:
        ref = ref / 255.0
    if srch.max() > 1.5:
        srch = srch / 255.0

    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=sigma_prefilter)
    srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=sigma_prefilter)
    srch_dn = np.clip(srch_dn, 0, 1).astype(np.float32)

    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
    nominal_side = min(nominal_side, srch_dn.shape[0] - 1, srch_dn.shape[1] - 1)
    nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side),
                                interpolation=cv2.INTER_AREA)
    return ref_dn, srch_dn, nominal_side, nominal_templ


def build_nominal_template(ref_dn, srch_dn, nominal_downsample=10.0):
    """Build nominal-scale template for Stage 1 coarse search."""
    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
    nominal_side = min(nominal_side, srch_dn.shape[0] - 1, srch_dn.shape[1] - 1)
    return cv2.resize(ref_dn, (nominal_side, nominal_side),
                       interpolation=cv2.INTER_AREA), nominal_side


def subpixel_peak(surface, iy, ix):
    """Parabolic sub-pixel refinement (Tian & Huhns 1986)."""
    h, w = surface.shape
    if 0 < iy < h - 1 and 0 < ix < w - 1:
        denom_y = surface[iy + 1, ix] - 2 * surface[iy, ix] + surface[iy - 1, ix] + 1e-9
        denom_x = surface[iy, ix + 1] - 2 * surface[iy, ix] + surface[iy, ix - 1] + 1e-9
        dy = 0.5 * (surface[iy + 1, ix] - surface[iy - 1, ix]) / denom_y
        dx = 0.5 * (surface[iy, ix + 1] - surface[iy, ix - 1]) / denom_x
        dy = np.clip(dy, -1, 1)
        dx = np.clip(dx, -1, 1)
        return float(iy - dy), float(ix - dx)
    return float(iy), float(ix)


def peak_sharpness(surface, iy, ix, exclude_radius):
    """Ratio of peak to best peak outside exclusion window."""
    masked = surface.copy()
    y0, y1 = max(0, iy - exclude_radius), min(surface.shape[0], iy + exclude_radius + 1)
    x0, x1 = max(0, ix - exclude_radius), min(surface.shape[1], ix + exclude_radius + 1)
    masked[y0:y1, x0:x1] = -1.0
    second = masked.max() if masked.size else -1.0
    best = surface[iy, ix]
    if second <= 0:
        return float("inf")
    return float(best / max(second, 1e-6))


def rotate_template(ref_small, angle):
    """Rotate template around center with replicate border."""
    if angle == 0:
        return ref_small
    h, w = ref_small.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(ref_small, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def local_grid_search(local_crop, ref_dn, nominal_downsample,
                      scale_search, rot_search_deg, x_off, y_off):
    """Run scale×rotation grid search on a local crop.
    Returns best dict with score, cx, cy, angle, scale_mult, target_side."""
    best = None
    for scale_mult in scale_search:
        target_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample * scale_mult)))
        if target_side >= local_crop.shape[1] or target_side >= local_crop.shape[0]:
            continue
        ref_small = cv2.resize(ref_dn, (target_side, target_side),
                                interpolation=cv2.INTER_AREA)
        for angle in rot_search_deg:
            templ = rotate_template(ref_small, angle)
            result = cv2.matchTemplate(local_crop, templ.astype(np.float32),
                                        cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if best is None or max_val > best["score"]:
                iy, ix = max_loc[1], max_loc[0]
                sy, sx = subpixel_peak(result, iy, ix)
                best = {
                    "score": float(max_val),
                    "target_side": target_side,
                    "angle": angle,
                    "scale_mult": scale_mult,
                    "cx": x_off + sx + target_side / 2.0,
                    "cy": y_off + sy + target_side / 2.0,
                }
    return best