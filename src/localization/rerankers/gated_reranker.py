"""
gated_reranker.py
------------------
Production-ready gated reranker for finfet_021/finfet_017 rescue.
Only activates when:
  1. Template size is sufficient (FinFET-style: nominal_side >= 60px)
  2. V2 top-2 candidate score margin is tight (< 0.01)

This avoids the DRAM regression seen in V3 (native) and V4 (phase) full reranking,
while still rescuing the ranking-ambiguity cases that V1/V2 mis-order.

Two reranking strategies:
- PhaseCorrelate (fast, ~14ms, works at search resolution)
- Native verification (slow, self-eval only, needs seed/canvas access)
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Gating conditions
# ──────────────────────────────────────────────────────────────────────

def should_rerank(nominal_side: int,
                  candidates: List[Dict],
                  score_margin_threshold: float = 0.01,
                  min_template_side: int = 60) -> bool:
    """
    Decide whether to invoke reranking.

    Args:
        nominal_side: Template size in search-image pixels (~30 DRAM, ~100 FinFET)
        candidates: V2 candidate list (already sorted by Stage-2 score desc)
        score_margin_threshold: Max gap between top-1 and top-2 to trigger rerank
        min_template_side: Only rerank if template has enough spatial support

    Returns:
        True if reranker should run
    """
    if nominal_side < min_template_side:
        return False
    if len(candidates) < 2:
        return False
    margin = candidates[0]["score"] - candidates[1]["score"]
    return margin < score_margin_threshold


# ──────────────────────────────────────────────────────────────────────
# Phase-correlation reranker (fast, production-ready)
# ──────────────────────────────────────────────────────────────────────

def _prep(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def build_nominal_template(reference_img: np.ndarray, nominal_downsample: float) -> Tuple[np.ndarray, int]:
    """Same construction as localizer.py Stage 1's nominal_templ."""
    ref = _prep(reference_img)
    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
    templ = cv2.resize(ref_dn, (nominal_side, nominal_side), interpolation=cv2.INTER_AREA)
    return templ, nominal_side


def prep_search(search_img: np.ndarray) -> np.ndarray:
    srch = _prep(search_img)
    srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
    return np.clip(srch_dn, 0, 1).astype(np.float32)


def _extract_patch(cx: float, cy: float, size: int, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = int(np.clip(x0, 0, img.shape[1] - size))
    y0 = int(np.clip(y0, 0, img.shape[0] - size))
    return img[y0:y0 + size, x0:x0 + size].copy(), (x0, y0)


def phase_rerank_candidates(reference_img: np.ndarray,
                            search_img: np.ndarray,
                            candidates: List[Dict],
                            nominal_downsample: float) -> Tuple[List[Dict], float]:
    """
    Re-rank V2 candidates by local phase-correlation response.
    Fast (~14ms for 10 candidates). Works at search resolution.
    """
    t0 = time.time()

    templ, nominal_side = build_nominal_template(reference_img, nominal_downsample)
    srch_dn = prep_search(search_img)
    window = cv2.createHanningWindow((nominal_side, nominal_side), cv2.CV_32F)

    out = []
    for c in candidates:
        patch, _ = _extract_patch(c["x"], c["y"], nominal_side, srch_dn)
        (_, _), resp = cv2.phaseCorrelate(templ.astype(np.float32),
                                           patch.astype(np.float32), window)
        c2 = dict(c)
        c2["phase_response"] = round(float(resp), 4)
        out.append(c2)

    out.sort(key=lambda c: c["phase_response"], reverse=True)
    elapsed = time.time() - t0
    return out, elapsed


# ──────────────────────────────────────────────────────────────────────
# Native verification reranker (self-eval only, slow, needs canvas seed)
# ──────────────────────────────────────────────────────────────────────

def _light_capture_sim(roi: np.ndarray, rng: np.random.Generator,
                       blur_sigma: float = 0.8, dose_scale: float = 1.0) -> np.ndarray:
    """Simulate native capture with mild blur + Poisson-Gaussian noise."""
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=blur_sigma)
    peak = 220.0 * dose_scale
    poisson = rng.poisson(np.clip(blurred, 0, 1) * peak).astype(np.float32) / peak
    gauss = rng.normal(0, 0.02 / np.sqrt(dose_scale), size=roi.shape).astype(np.float32)
    return poisson + gauss


def native_ncc_score(reference_img: np.ndarray, native_roi: np.ndarray,
                     scale_search=(0.97, 1.0, 1.03),
                     rot_search_deg=(-3, -1.5, 0, 1.5, 3)) -> Tuple[float, float, float, float, int]:
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


def native_rerank_candidates(reference_img: np.ndarray,
                             candidates: List[Dict],
                             style: str,
                             seed: int,
                             downsample: float = 10.0,
                             k_verify: int = 3,
                             canvas_size: Tuple[int, int] = (10000, 10000)) -> List[Dict]:
    """
    Native verification reranker (SELF-EVAL ONLY).
    Requires: sample style + seed to regenerate deterministic canvas.
    """
    from pattern_synth import synth_canvas

    ref_side = reference_img.shape[1]
    roi_half_size = int(ref_side * 1.3 / 2) + 50

    rows = []
    for rank, cand in enumerate(candidates[:k_verify]):
        # Native coordinates
        ncx = cand["x"] * downsample
        ncy = cand["y"] * downsample

        canvas = synth_canvas(style, canvas_size, seed=seed)
        h, w = canvas_size
        x0 = max(0, int(ncx - roi_half_size))
        y0 = max(0, int(ncy - roi_half_size))
        x1 = min(w, int(ncx + roi_half_size))
        y1 = min(h, int(ncy + roi_half_size))
        roi = canvas[y0:y1, x0:x1].copy()

        rng = np.random.default_rng(seed * 104729 + rank + 99991)
        roi_captured = _light_capture_sim(roi, rng)

        native_score, dx, dy, nscale, nangle = native_ncc_score(reference_img, roi_captured)

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
    return rows


# ──────────────────────────────────────────────────────────────────────
# Unified gated reranker entry point
# ──────────────────────────────────────────────────────────────────────

def gated_rerank(reference_img: np.ndarray,
                 search_img: np.ndarray,
                 candidates: List[Dict],
                 nominal_downsample: float,
                 style: Optional[str] = None,
                 seed: Optional[int] = None,
                 use_native: bool = False,
                 score_margin_threshold: float = 0.01,
                 min_template_side: int = 60) -> Tuple[Optional[List[Dict]], Dict]:
    """
    Main entry point: conditionally rerank V2 candidates.

    Args:
        reference_img, search_img: Input images
        candidates: V2 output from localize_topk (sorted by Stage-2 score desc)
        nominal_downsample: Expected scale ratio
        style, seed: Required only if use_native=True (self-eval)
        use_native: If True, use native verification (self-eval only);
                    If False, use phase correlation (production)
        score_margin_threshold: Trigger rerank if top-2 gap < this
        min_template_side: Only rerank if nominal_side >= this

    Returns:
        (reranked_candidates_or_None, debug_info)
        - If gating fails: returns (None, {"reranked": False, "reason": ...})
        - If gating passes: returns (new_candidate_list, debug)
    """
    if not candidates:
        return None, {"reranked": False, "reason": "no candidates"}

    ref = _prep(reference_img)
    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))

    debug = {
        "nominal_side": nominal_side,
        "top1_score": candidates[0]["score"],
        "top2_score": candidates[1]["score"] if len(candidates) > 1 else None,
        "score_margin": candidates[0]["score"] - candidates[1]["score"] if len(candidates) > 1 else None,
        "reranked": False,
    }

    if not should_rerank(nominal_side, candidates, score_margin_threshold, min_template_side):
        debug["reason"] = "gate_failed"
        if nominal_side < min_template_side:
            debug["reason_detail"] = f"template_too_small ({nominal_side} < {min_template_side})"
        elif len(candidates) < 2:
            debug["reason_detail"] = "insufficient_candidates"
        else:
            debug["reason_detail"] = f"margin_too_large ({debug['score_margin']:.4f} >= {score_margin_threshold})"
        return None, debug

    # Gate passed
    if use_native:
        if style is None or seed is None:
            debug["reason"] = "native_requires_seed"
            return None, debug
        reranked = native_rerank_candidates(reference_img, candidates, style, seed, nominal_downsample)
        debug["method"] = "native"
    else:
        reranked, phase_time = phase_rerank_candidates(reference_img, search_img, candidates, nominal_downsample)
        debug["method"] = "phase"
        debug["phase_time_ms"] = round(phase_time * 1000, 1)

    debug["reranked"] = True
    debug["pre_top1"] = {"x": candidates[0]["x"], "y": candidates[0]["y"], "score": candidates[0]["score"]}
    debug["post_top1"] = {"x": reranked[0]["x"], "y": reranked[0]["y"]}

    return reranked, debug


# ──────────────────────────────────────────────────────────────────────
# Convenience: end-to-end production pipeline with gated reranking
# ──────────────────────────────────────────────────────────────────────

def localize_with_gated_rerank(reference_img: np.ndarray,
                                search_img: np.ndarray,
                                nominal_downsample: float = 10.0,
                                K: int = 40,
                                style: Optional[str] = None,
                                seed: Optional[int] = None,
                                use_native: bool = False) -> Tuple[Dict, Dict]:
    """
    Full pipeline: V1 -> V2 candidate pool -> optional gated rerank -> final prediction.
    Returns (final_prediction, debug_info).
    """
    from localizer import localize, localize_topk

    # V1 baseline (fast, for ambiguity flag + fallback)
    v1_pred = localize(reference_img, search_img, nominal_downsample)

    # V2 candidate pool
    v2_candidates, v2_debug = localize_topk(reference_img, search_img,
                                             nominal_downsample=nominal_downsample,
                                             K=K, return_debug=True)

    # Gated rerank
    reranked, rerank_debug = gated_rerank(reference_img, search_img, v2_candidates,
                                           nominal_downsample, style, seed, use_native)

    # Final decision
    if reranked:
        final = reranked[0]
        final["method"] = f"V2+{rerank_debug['method']}_rerank"
    else:
        final = v2_candidates[0]
        final["method"] = "V2"

    # Merge debug
    debug = {
        "v1": v1_pred,
        "v2_top1": v2_candidates[0],
        "v2_pool_size": len(v2_candidates),
        "rerank": rerank_debug,
        "final": final,
    }
    return final, debug