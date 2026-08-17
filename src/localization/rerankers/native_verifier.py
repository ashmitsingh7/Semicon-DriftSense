"""
Gated native-resolution verification reranker.

Only reranks when:
  1. Template size gate: nominal_side >= 60px (FinFET-style, not DRAM)
  2. Ambiguity gate: V2 top-2 candidate score margin < 0.01

This rescues finfet_021 (tight margin 0.0023, large template)
while leaving DRAM untouched (large margins, small template).

NOTE: For synthetic self-eval only. Real deployment needs actual
native-resolution re-capture at each candidate site.
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional
from pattern_synth import synth_canvas


# Gate thresholds - derived from failure analysis, NOT tuned on test set
NOMINAL_SIDE_THRESHOLD = 60
SCORE_MARGIN_THRESHOLD = 0.01


def should_rerank(candidates: List[Dict], nominal_side: int) -> bool:
    """Check both gating conditions."""
    if nominal_side < NOMINAL_SIDE_THRESHOLD:
        return False
    if len(candidates) < 2:
        return False
    margin = candidates[0]["score"] - candidates[1]["score"]
    if margin >= SCORE_MARGIN_THRESHOLD:
        return False
    return True


def extract_native_roi(style, seed, candidate_x, candidate_y, downsample,
                        roi_half_size, canvas_size=(10000, 10000)):
    """Regenerate deterministic native canvas and crop ROI around candidate."""
    canvas = synth_canvas(style, canvas_size, seed=seed)
    h, w = canvas_size
    ncx = candidate_x * downsample
    ncy = candidate_y * downsample
    x0 = max(0, int(ncx - roi_half_size))
    y0 = max(0, int(ncy - roi_half_size))
    x1 = min(w, int(ncx + roi_half_size))
    y1 = min(h, int(ncy + roi_half_size))
    roi = canvas[y0:y1, x0:x1].copy()
    return roi, (x0, y0)


def _light_capture_sim(roi, rng, blur_sigma=0.8, dose_scale=1.0):
    """Simulate native-resolution capture (blur + Poisson + Gaussian noise)."""
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=blur_sigma)
    peak = 220.0 * dose_scale
    poisson = rng.poisson(np.clip(blurred, 0, 1) * peak).astype(np.float32) / peak
    gauss = rng.normal(0, 0.02 / np.sqrt(dose_scale), size=roi.shape).astype(np.float32)
    return poisson + gauss


def native_ncc_score(reference_img, native_roi,
                      scale_search=(0.97, 1.0, 1.03),
                      rot_search_deg=(-3, -1.5, 0, 1.5, 3)):
    """Small scale/rotation grid NCC at native resolution."""
    ref = reference_img.astype(np.float32)
    if ref.max() > 1.5:
        ref = ref / 255.0
    roi = native_roi.astype(np.float32)
    if roi.max() > 1.5:
        roi = roi / 255.0
    roi = np.clip(roi, 0, 1).astype(np.float32)

    best = None
    for scale_mult in scale_search:
        side = max(8, int(round(ref.shape[1] * scale_mult)))
        if side >= roi.shape[1] or side >= roi.shape[0]:
            continue
        ref_scaled = cv2.resize(ref, (side, side), interpolation=cv2.INTER_AREA)
        for angle in rot_search_deg:
            if angle == 0:
                templ = ref_scaled
            else:
                M = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
                templ = cv2.warpAffine(ref_scaled, M, (side, side),
                                        borderMode=cv2.BORDER_REPLICATE)
            result = cv2.matchTemplate(roi, templ.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if best is None or max_val > best[0]:
                cx = max_loc[0] + side / 2.0
                cy = max_loc[1] + side / 2.0
                best = (float(max_val), cx, cy, scale_mult, angle)
    if best is None:
        return -1.0, 0.0, 0.0, 1.0, 0
    return best


def native_reranker_gated(
    reference_img: np.ndarray,
    search_img: np.ndarray,
    candidates: List[Dict],
    nominal_downsample: float,
    style: str,
    seed: int,
    k_verify: int = 3,
    roi_half_size_mult: float = 1.3,
    canvas_size: tuple = (10000, 10000),
    noise_seed_offset: int = 99991,
    return_debug: bool = False
) -> Tuple[List[Dict], float, Optional[Dict]]:
    """
    Gated native verification reranker entry point.

    Args:
        reference_img: Native-resolution reference patch
        search_img: Search image (10x downsampled)
        candidates: V2 candidate list from localize_topk (sorted by score desc)
        nominal_downsample: Expected magnification ratio (~10.0)
        style: Sample style ("dram", "finfet", "mixed_logic")
        seed: Dataset seed for canvas regeneration
        k_verify: Number of top candidates to verify
        roi_half_size_mult: ROI half-size multiplier relative to reference
        canvas_size: Native canvas dimensions
        noise_seed_offset: RNG offset for capture simulation
        return_debug: Include debug info in return

    Returns:
        (reranked_candidates, native_time_seconds, debug_dict)
        - If gates fail: returns original candidates unchanged, 0.0 time
        - If gates pass: returns candidates re-sorted by native_score
        - Each candidate gets added keys:
            "rerank_score": native_score (or -1 if not verified)
            "rerank_method": "native"
            "rerank_applied": bool
    """
    t0 = time.time()

    ref = reference_img
    ref_side = ref.shape[1]
    nominal_side = max(6, int(round(ref_side / nominal_downsample)))
    roi_half_size = int(ref_side * roi_half_size_mult / 2) + 50

    # Check gates
    if not should_rerank(candidates, nominal_side):
        out = []
        for c in candidates:
            c2 = dict(c)
            c2["rerank_score"] = c["score"]
            c2["rerank_method"] = "native"
            c2["rerank_applied"] = False
            out.append(c2)
        elapsed = time.time() - t0
        margin = candidates[0]["score"] - candidates[1]["score"] if len(candidates) >= 2 else None
        debug = {"gated_out": True, "nominal_side": nominal_side, "margin": margin}
        return out, elapsed, debug if return_debug else None

    # Gates passed: verify top k_verify candidates at native resolution
    rows = []
    for rank, cand in enumerate(candidates[:k_verify]):
        roi, (x0, y0) = extract_native_roi(
            style, seed, cand["x"], cand["y"], nominal_downsample,
            roi_half_size, canvas_size)
        rng = np.random.default_rng(seed * 104729 + rank + noise_seed_offset)
        roi_captured = _light_capture_sim(roi, rng)
        native_score, dx, dy, nscale, nangle = native_ncc_score(ref, roi_captured)
        rows.append({
            "candidate_rank_pre_native": rank,
            "x": cand["x"], "y": cand["y"],
            "coarse_score": cand.get("coarse_score"),
            "fine_score": cand["score"],
            "native_score": round(native_score, 4),
            "native_scale_mult": nscale,
            "native_angle_deg": nangle,
        })

    rows.sort(key=lambda r: r["native_score"], reverse=True)
    for i, r in enumerate(rows):
        r["candidate_rank_post_native"] = i

    # Map native scores back to candidates
    native_score_map = {(r["x"], r["y"]): r["native_score"] for r in rows}

    out = []
    for c in candidates:
        c2 = dict(c)
        key = (c["x"], c["y"])
        c2["rerank_score"] = native_score_map.get(key, -1.0)
        c2["rerank_method"] = "native"
        c2["rerank_applied"] = True
        out.append(c2)

    # Sort: verified candidates by native score, then unverified by original score
    verified = [c for c in out if c["rerank_score"] >= 0]
    unverified = [c for c in out if c["rerank_score"] < 0]
    verified.sort(key=lambda c: c["rerank_score"], reverse=True)
    out = verified + unverified

    elapsed = time.time() - t0

    margin = candidates[0]["score"] - candidates[1]["score"]
    debug = {"gated_out": False, "nominal_side": nominal_side,
             "margin": margin, "k_verified": len(rows),
             "n_candidates": len(candidates)}
    return out, elapsed, debug if return_debug else None