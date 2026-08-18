"""V2 Candidate Pool Localization.

Extends V1 by generating Top-K candidates at Stage 1 (coarse global) with
Non-Maximum Suppression (NMS), then running Stage 2 fine grid search on each
candidate. Deduplicates on refined sub-pixel coordinates.

This provides the candidate pool that gated rerankers operate on.
"""

import numpy as np
import cv2
from .v1_localize import _subpixel_peak, _peak_sharpness


def _nms_peaks(surface, k, min_dist):
    """Non-maximum suppression to extract Top-K distinct peaks.
    Returns list of (iy, ix, score) sorted by score descending."""
    h, w = surface.shape
    used = np.zeros_like(surface, dtype=bool)
    peaks = []
    flat_idx = surface.ravel().argsort()[::-1]
    for idx in flat_idx:
        if len(peaks) >= k:
            break
        iy, ix = divmod(idx, w)
        if used[iy, ix]:
            continue
        peaks.append((iy, ix, surface[iy, ix]))
        y0, y1 = max(0, iy - min_dist), min(h, iy + min_dist + 1)
        x0, x1 = max(0, ix - min_dist), min(w, ix + min_dist + 1)
        used[y0:y1, x0:x1] = True
    return peaks


def localize_v2(reference_img, search_img, nominal_downsample=10.0,
                K=5, nms_radius=3,
                scale_search=(0.92, 0.96, 1.0, 1.04, 1.08),
                rot_search_deg=(-4, -2.5, -1, 0, 1, 2.5, 4),
                return_debug=False):
    """
    Two-stage with candidate pool at Stage 1.

    Stage 1 (coarse, GLOBAL): NCC over entire search image → Top-K peaks via NMS
      Each peak gets ambiguity ratio (peak vs 2nd best outside excl window).

    Stage 2 (fine, LOCAL per candidate): Scale×rotation grid search in
      window around each Stage 1 peak. Results deduped by refined (x,y).

    Args:
        reference_img: 2D float32/uint8, native-resolution reference patch
        search_img:    2D float32/uint8, larger search image
        nominal_downsample: expected reference->search magnification ratio
        K: number of candidates to keep from Stage 1
        nms_radius: pixel radius for NMS suppression
        scale_search: scale multipliers for Stage 2 grid
        rot_search_deg: rotation degrees for Stage 2 grid
        return_debug: include internal state

    Returns:
        dict with: x, y, confidence, ambiguity_ratio, candidates list, low_confidence_flag
    """
    ref = reference_img.astype(np.float32)
    srch = search_img.astype(np.float32)
    if ref.max() > 1.5:
        ref = ref / 255.0
    if srch.max() > 1.5:
        srch = srch / 255.0

    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
    srch_dn = np.clip(srch_dn, 0, 1).astype(np.float32)

    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
    nominal_side = min(nominal_side, srch_dn.shape[0] - 1, srch_dn.shape[1] - 1)
    nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side),
                                interpolation=cv2.INTER_AREA)

    # ---- Stage 1: coarse, global with Top-K ----
    coarse_result = cv2.matchTemplate(srch_dn, nominal_templ, cv2.TM_CCOEFF_NORMED)
    peaks = _nms_peaks(coarse_result, K, nms_radius)

    candidates = []
    for iy, ix, score in peaks:
        ambiguity = _peak_sharpness(coarse_result, iy, ix,
                                     exclude_radius=max(3, nominal_side // 2))
        coarse_cx = ix + nominal_side / 2.0
        coarse_cy = iy + nominal_side / 2.0

        # ---- Stage 2: fine, local window per candidate ----
        window_r = int(nominal_side * 2.5) + 10
        x0 = max(0, int(coarse_cx - window_r))
        y0 = max(0, int(coarse_cy - window_r))
        x1 = min(srch_dn.shape[1], int(coarse_cx + window_r))
        y1 = min(srch_dn.shape[0], int(coarse_cy + window_r))
        local_crop = srch_dn[y0:y1, x0:x1]

        best_local = None
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
                if best_local is None or max_val > best_local["score"]:
                    ly, lx = max_loc[1], max_loc[0]
                    sy, sx = _subpixel_peak(result, ly, lx)
                    best_local = {
                        "score": float(max_val),
                        "target_side": target_side,
                        "angle": angle,
                        "scale_mult": scale_mult,
                        "cx": x0 + sx + target_side / 2.0,
                        "cy": y0 + sy + target_side / 2.0,
                    }

        if best_local is None:
            best_local = {"score": float(score), "target_side": nominal_side,
                          "angle": 0, "scale_mult": 1.0,
                          "cx": coarse_cx, "cy": coarse_cy}

        candidates.append({
            "x": round(float(best_local["cx"]), 2),
            "y": round(float(best_local["cy"]), 2),
            "confidence": round(float(np.clip(best_local["score"], 0, 1)), 4),
            "ambiguity_ratio": round(ambiguity, 3) if np.isfinite(ambiguity) else None,
            "scale": round(nominal_downsample / best_local["scale_mult"], 3),
            "rotation_deg": best_local["angle"],
            "stage1_score": float(score),
            "stage1_x": round(coarse_cx, 2),
            "stage1_y": round(coarse_cy, 2),
        })

    # Deduplicate on refined coordinates
    seen = []
    uniq = []
    for c in sorted(candidates, key=lambda x: -x["confidence"]):
        dup = False
        for s in seen:
            if np.hypot(c["x"] - s["x"], c["y"] - s["y"]) < 2.0:
                dup = True
                break
        if not dup:
            seen.append(c)
            uniq.append(c)

    primary = uniq[0] if uniq else {
        "x": 0.0, "y": 0.0, "confidence": 0.0,
        "ambiguity_ratio": None, "scale": nominal_downsample,
        "rotation_deg": 0, "low_confidence_flag": True
    }

    out = {
        "x": primary["x"],
        "y": primary["y"],
        "confidence": primary["confidence"],
        "ambiguity_ratio": primary["ambiguity_ratio"],
        "scale": primary["scale"],
        "rotation_deg": primary["rotation_deg"],
        "low_confidence_flag": primary["confidence"] < 0.35 or (
            primary["ambiguity_ratio"] is not None and primary["ambiguity_ratio"] < 1.05),
        "candidates": uniq,
    }
    if return_debug:
        out["debug"] = {"coarse_result_shape": coarse_result.shape, "peaks": peaks,
                         "window_r": window_r}
    return out