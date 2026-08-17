"""
Gated phase-correlation reranker.

Only reranks when:
  1. Template size gate: nominal_side >= 60px (FinFET-style, not DRAM)
  2. Ambiguity gate: V2 top-2 candidate score margin < 0.01

This rescues finfet_021 (tight margin 0.0023, large template)
while leaving DRAM untouched (large margins, small template).
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional


# Gate thresholds - derived from failure analysis, NOT tuned on test set
NOMINAL_SIDE_THRESHOLD = 60
SCORE_MARGIN_THRESHOLD = 0.01


def _prep(img):
    """Normalize image to float32 [0, 1]."""
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def build_nominal_template(reference_img, nominal_downsample):
    """Build template same as localizer.py Stage 1."""
    ref = _prep(reference_img)
    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
    templ = cv2.resize(ref_dn, (nominal_side, nominal_side),
                        interpolation=cv2.INTER_AREA)
    return templ, nominal_side


def prep_search(search_img):
    """Denoise search image same as localizer.py."""
    srch = _prep(search_img)
    srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
    return np.clip(srch_dn, 0, 1).astype(np.float32)


def _extract_patch(cx, cy, size, img):
    """Extract size x size patch centered at (cx, cy)."""
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = int(np.clip(x0, 0, img.shape[1] - size))
    y0 = int(np.clip(y0, 0, img.shape[0] - size))
    return img[y0:y0 + size, x0:x0 + size].copy(), (x0, y0)


def phase_score(templ, patch, window):
    """Compute phase correlation response."""
    (_, _), resp = cv2.phaseCorrelate(templ.astype(np.float32),
                                       patch.astype(np.float32), window)
    return float(resp)


def should_rerank(candidates: List[Dict], nominal_side: int) -> bool:
    """Check both gating conditions."""
    # Gate 1: Template size (skip DRAM ~30px)
    if nominal_side < NOMINAL_SIDE_THRESHOLD:
        return False
    # Gate 2: Ambiguity (need at least 2 candidates with tight margin)
    if len(candidates) < 2:
        return False
    margin = candidates[0]["score"] - candidates[1]["score"]
    if margin >= SCORE_MARGIN_THRESHOLD:
        return False
    return True


def phase_reranker_gated(
    reference_img: np.ndarray,
    search_img: np.ndarray,
    candidates: List[Dict],
    nominal_downsample: float,
    use_hann: bool = True,
    return_debug: bool = False
) -> Tuple[List[Dict], float, Optional[Dict]]:
    """
    Gated phase reranker entry point.

    Args:
        reference_img: Native-resolution reference patch
        search_img: Search image (10x downsampled)
        candidates: V2 candidate list from localize_topk (sorted by score desc)
        nominal_downsample: Expected magnification ratio (~10.0)
        use_hann: Apply Hann window to template and patch
        return_debug: Include debug info in return

    Returns:
        (reranked_candidates, phase_time_seconds, debug_dict)
        - If gates fail: returns original candidates unchanged, 0.0 time
        - If gates pass: returns candidates re-sorted by phase_response
        - Each candidate gets added keys:
            "rerank_score": phase_response
            "rerank_method": "phase"
            "rerank_applied": bool
    """
    t0 = time.time()

    templ, nominal_side = build_nominal_template(reference_img, nominal_downsample)
    srch_dn = prep_search(search_img)
    window = cv2.createHanningWindow((nominal_side, nominal_side), cv2.CV_32F) \
        if use_hann else None

    # Check gates
    if not should_rerank(candidates, nominal_side):
        # Gates failed: return original candidates with rerank_applied=False
        out = []
        for c in candidates:
            c2 = dict(c)
            c2["rerank_score"] = c["score"]
            c2["rerank_method"] = "phase"
            c2["rerank_applied"] = False
            out.append(c2)
        elapsed = time.time() - t0
        margin = candidates[0]["score"] - candidates[1]["score"] if len(candidates) >= 2 else None
        debug = {"gated_out": True, "nominal_side": nominal_side, "margin": margin}
        return out, elapsed, debug if return_debug else None

    # Gates passed: rerank by phase correlation
    out = []
    for c in candidates:
        patch, (x0, y0) = _extract_patch(c["x"], c["y"], nominal_side, srch_dn)
        resp = phase_score(templ, patch, window)
        c2 = dict(c)
        c2["rerank_score"] = round(resp, 4)
        c2["rerank_method"] = "phase"
        c2["rerank_applied"] = True
        out.append(c2)

    out.sort(key=lambda c: c["rerank_score"], reverse=True)
    elapsed = time.time() - t0

    margin = candidates[0]["score"] - candidates[1]["score"]
    debug = {"gated_out": False, "nominal_side": nominal_side,
             "margin": margin, "n_candidates": len(candidates)}
    return out, elapsed, debug if return_debug else None