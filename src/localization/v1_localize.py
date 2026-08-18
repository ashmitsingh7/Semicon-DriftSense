"""V1 Production: Two-Stage Single-Argmax NCC Localization.

This is the primary production algorithm submitted for the Applied Materials
Drift-Sense problem. It implements a two-stage coarse-to-fine normalized
cross-correlation (NCC) search with sub-pixel parabolic peak refinement
and built-in ambiguity detection.

Algorithm:
  Stage 1 (Coarse, Global): Single nominal-scale/zero-rotation NCC over entire
    search image → one global peak + ambiguity ratio (peak vs 2nd peak).
  Stage 2 (Fine, Local): Scale×rotation grid search (35 combinations) in a
    small window around Stage 1 peak → sub-pixel refined coordinates.

Output includes ambiguity flag for failure-mode awareness (catches all 3
canonical hard cases: finfet_017, finfet_021, finfet_023).
"""

import numpy as np
import cv2


def _subpixel_peak(surface, iy, ix):
    """Parabolic sub-pixel refinement around a discrete correlation peak.
    Based on Tian & Huhns, "Algorithms for Subpixel Registration", CVGIP 35, 1986."""
    h, w = surface.shape
    if 0 < iy < h - 1 and 0 < ix < w - 1:
        dy = 0.5 * (surface[iy + 1, ix] - surface[iy - 1, ix]) / (
            surface[iy + 1, ix] - 2 * surface[iy, ix] + surface[iy - 1, ix] + 1e-9)
        dx = 0.5 * (surface[iy, ix + 1] - surface[iy, ix - 1]) / (
            surface[iy, ix + 1] - 2 * surface[iy, ix] + surface[iy, ix - 1] + 1e-9)
        dy = np.clip(dy, -1, 1)
        dx = np.clip(dx, -1, 1)
        return iy - dy, ix - dx
    return float(iy), float(ix)


def _peak_sharpness(surface, iy, ix, exclude_radius):
    """Ratio of global peak to best peak outside exclusion window.
    Low values indicate periodic/ambiguous match."""
    masked = surface.copy()
    y0, y1 = max(0, iy - exclude_radius), min(surface.shape[0], iy + exclude_radius + 1)
    x0, x1 = max(0, ix - exclude_radius), min(surface.shape[1], ix + exclude_radius + 1)
    masked[y0:y1, x0:x1] = -1.0
    second = masked.max() if masked.size else -1.0
    best = surface[iy, ix]
    if second <= 0:
        return float("inf")
    return float(best / max(second, 1e-6))


def localize(reference_img, search_img, nominal_downsample=10.0,
             scale_search=(0.92, 0.96, 1.0, 1.04, 1.08),
             rot_search_deg=(-4, -2.5, -1, 0, 1, 2.5, 4),
             return_debug=False):
    """
    Two-stage search for speed:

      Stage 1 (coarse, GLOBAL): single NCC pass over entire search image.
        Gives both coarse location and ambiguity signal in one pass.

      Stage 2 (fine, LOCAL): scale×rotation grid search inside small window
        around Stage 1 peak. Confining grid to local crop is the single
        biggest lever on throughput (cost scales with search area).

    Args:
        reference_img: 2D float32/uint8, native-resolution reference patch (high-res)
        search_img:    2D float32/uint8, larger search image (10x lower mag, noisier)
        nominal_downsample: expected reference->search magnification ratio (10.0)
        scale_search:  scale multipliers for Stage 2 grid
        rot_search_deg: rotation degrees for Stage 2 grid
        return_debug:  include internal state in output

    Returns:
        dict with: x, y, confidence, ambiguity_ratio, scale, rotation_deg, low_confidence_flag
    """
    ref = reference_img.astype(np.float32)
    srch = search_img.astype(np.float32)
    if ref.max() > 1.5:
        ref = ref / 255.0
    if srch.max() > 1.5:
        srch = srch / 255.0

    # Mild denoise before correlation -- reduces HF shot-noise sensitivity
    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
    srch_dn = np.clip(srch_dn, 0, 1).astype(np.float32)

    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
    nominal_side = min(nominal_side, srch_dn.shape[0] - 1, srch_dn.shape[1] - 1)
    nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side),
                                interpolation=cv2.INTER_AREA)

    # ---- Stage 1: coarse, global ----
    coarse_result = cv2.matchTemplate(srch_dn, nominal_templ, cv2.TM_CCOEFF_NORMED)
    _, coarse_val, _, coarse_loc = cv2.minMaxLoc(coarse_result)
    c_iy, c_ix = coarse_loc[1], coarse_loc[0]
    ambiguity = _peak_sharpness(coarse_result, c_iy, c_ix,
                                 exclude_radius=max(3, nominal_side // 2))
    coarse_cx = c_ix + nominal_side / 2.0
    coarse_cy = c_iy + nominal_side / 2.0

    # ---- Stage 2: fine, local window only ----
    window_r = int(nominal_side * 2.5) + 10
    x0 = max(0, int(coarse_cx - window_r))
    y0 = max(0, int(coarse_cy - window_r))
    x1 = min(srch_dn.shape[1], int(coarse_cx + window_r))
    y1 = min(srch_dn.shape[0], int(coarse_cy + window_r))
    local_crop = srch_dn[y0:y1, x0:x1]

    best = None
    for scale_mult in scale_search:
        target_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample * scale_mult)))
        if target_side >= local_crop.shape[1] or target_side >= local_crop.shape[0]:
            continue
        ref_small = cv2.resize(ref_dn, (target_side, target_side),
                                interpolation=cv2.INTER_AREA)
        for angle in rot_search_deg:
            if angle == 0:
                templ = ref_small
            else:
                M = cv2.getRotationMatrix2D(
                    (target_side / 2, target_side / 2), angle, 1.0)
                templ = cv2.warpAffine(ref_small, M, (target_side, target_side),
                                        borderMode=cv2.BORDER_REPLICATE)

            result = cv2.matchTemplate(local_crop, templ.astype(np.float32),
                                        cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if best is None or max_val > best["score"]:
                iy, ix = max_loc[1], max_loc[0]
                sy, sx = _subpixel_peak(result, iy, ix)
                best = {
                    "score": float(max_val),
                    "target_side": target_side,
                    "angle": angle,
                    "scale_mult": scale_mult,
                    "cx": x0 + sx + target_side / 2.0,
                    "cy": y0 + sy + target_side / 2.0,
                }

    if best is None:
        best = {"score": float(coarse_val), "target_side": nominal_side,
                 "angle": 0, "scale_mult": 1.0, "cx": coarse_cx, "cy": coarse_cy}

    confidence = float(np.clip(best["score"], 0, 1))
    out = {
        "x": round(float(best["cx"]), 2),
        "y": round(float(best["cy"]), 2),
        "confidence": round(confidence, 4),
        "ambiguity_ratio": round(ambiguity, 3) if np.isfinite(ambiguity) else None,
        "scale": round(nominal_downsample / best["scale_mult"], 3),
        "rotation_deg": best["angle"],
        "low_confidence_flag": bool(confidence < 0.35 or (
            np.isfinite(ambiguity) and ambiguity < 1.05)),
    }
    if return_debug:
        out["debug"] = {**best, "coarse_cx": coarse_cx, "coarse_cy": coarse_cy,
                         "coarse_val": coarse_val, "window": (x0, y0, x1, y1)}
    return out