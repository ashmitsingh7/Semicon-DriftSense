"""
Enhanced Localizer v6 - Incorporates best ideas from competitive analysis:

1. Adaptive Coarse Search (800x800 for large templates, skip for small)
2. NLMeans Denoising pre-processing
3. Structural Reranker: Variance similarity (0.4) + Gradient correlation (0.4) + Landmark overlay (0.2)
4. Wider rotation search: ±8° coarse (13 steps) -> ±1.5° fine (5 steps)
5. Gated reranker - only activates when needed
6. Peak-to-2nd-peak ambiguity detection
7. Sub-pixel quadratic refinement
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    x: float
    y: float
    score: float
    scale: float
    rotation: float
    coarse_score: float = 0.0
    fine_score: float = 0.0
    phase_response: float = 0.0
    structural_score: float = 0.0
    combined_score: float = 0.0

    def to_dict(self):
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "score": round(self.score, 4),
            "confidence": round(self.score, 4),
            "scale": round(self.scale, 4),
            "rotation_deg": round(self.rotation, 2),
            "phase_response": round(self.phase_response, 4),
            "structural_score": round(self.structural_score, 4),
            "combined_score": round(self.combined_score, 4),
        }

# ──────────────────────────────────────────────────────────────────────
# Image Pre-processing
# ──────────────────────────────────────────────────────────────────────

def denoise_image(img: np.ndarray, method: str = "nlmeans") -> np.ndarray:
    """Apply denoising to float32 image in [0, 1]."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if method == "nlmeans":
        denoised_u8 = cv2.fastNlMeansDenoising(u8, None, h=5, templateWindowSize=7, searchWindowSize=21)
    elif method == "bilateral":
        denoised_u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=25, sigmaSpace=25)
    else:
        denoised_u8 = cv2.GaussianBlur(u8, (3, 3), 0.5)
    return denoised_u8.astype(np.float32) / 255.0


def get_feature_map(img: np.ndarray) -> np.ndarray:
    """Combine grayscale intensity and Sobel gradient magnitude."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    combined = 0.6 * img.astype(np.float32) + 0.4 * mag
    return combined / (combined.max() + 1e-6)


def ncc_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """NCC via cv2.matchTemplate (TM_CCOEFF_NORMED)."""
    return cv2.matchTemplate(
        search.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )


# ──────────────────────────────────────────────────────────────────────
# Structural Reranker
# ──────────────────────────────────────────────────────────────────────

def compute_structural_score(cand_crop: np.ndarray, ref_crop: np.ndarray) -> float:
    """Compute structural similarity between candidate crop and reference."""
    if cand_crop.size == 0 or ref_crop.size == 0:
        return 0.0

    ch, cw = cand_crop.shape[:2]
    if ref_crop.shape[:2] != (ch, cw):
        ref_scaled = cv2.resize(ref_crop, (cw, ch), interpolation=cv2.INTER_AREA)
    else:
        ref_scaled = ref_crop

    # 1. Variance Similarity
    var_cand = float(np.var(cand_crop))
    var_ref = float(np.var(ref_scaled))
    var_sim = 1.0 / (1.0 + abs(var_cand - var_ref) * 10.0)

    # 2. Gradient Magnitude Correlation
    gx_c = cv2.Sobel(cand_crop, cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(cand_crop, cv2.CV_32F, 0, 1, ksize=3)
    mag_c = cv2.magnitude(gx_c, gy_c)

    gx_r = cv2.Sobel(ref_scaled, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(ref_scaled, cv2.CV_32F, 0, 1, ksize=3)
    mag_r = cv2.magnitude(gx_r, gy_r)

    denom = (np.std(mag_c) * np.std(mag_r) + 1e-6)
    grad_corr = float(np.mean((mag_c - np.mean(mag_c)) * (mag_r - np.mean(mag_r))) / denom)
    grad_corr = max(0.0, grad_corr)

    # 3. High-Contrast Landmark Overlay
    high_c = (cand_crop > 0.75).astype(np.float32)
    high_r = (ref_scaled > 0.75).astype(np.float32)
    overlay = float(np.mean(high_c * high_r))

    return 0.4 * var_sim + 0.4 * grad_corr + 0.2 * overlay


def structural_rerank_candidates(
    candidates: List[Candidate],
    search_img: np.ndarray,
    ref_img: np.ndarray,
    template_size: int = 100,
) -> List[Candidate]:
    """Apply structural reranking to candidates."""
    if not candidates or ref_img is None:
        return candidates

    sh, sw = search_img.shape[:2]
    ref_scaled = cv2.resize(ref_img, (template_size, template_size), interpolation=cv2.INTER_AREA)

    for cand in candidates:
        cx, cy = int(round(cand.x)), int(round(cand.y))
        tw, th = template_size, template_size
        x0 = max(0, cx - tw // 2)
        x1 = min(sw, cx + tw // 2)
        y0 = max(0, cy - th // 2)
        y1 = min(sh, cy + th // 2)

        cand_crop = search_img[y0:y1, x0:x1]
        if cand_crop.shape[:2] != (th, tw):
            cand_crop = cv2.resize(cand_crop, (tw, th), interpolation=cv2.INTER_AREA)

        struct_score = compute_structural_score(cand_crop, ref_scaled)
        cand.structural_score = struct_score
        cand.combined_score = cand.score + 0.15 * struct_score

    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return candidates


# ──────────────────────────────────────────────────────────────────────
# NMS Peak Extraction
# ──────────────────────────────────────────────────────────────────────

def extract_nms_peaks(
    result: np.ndarray,
    threshold: float,
    min_dist: int = 15,
    max_peaks: int = 10,
    tw: int = 100,
    th: int = 100,
) -> List[Candidate]:
    """Extract local maxima from 2D NCC result map using spatial NMS."""
    peaks = []
    res_copy = result.copy()
    for _ in range(max_peaks):
        val = float(res_copy.max())
        if val < threshold:
            break
        py, px = np.unravel_index(res_copy.argmax(), res_copy.shape)
        peaks.append(Candidate(
            score=val,
            x=float(px + tw / 2.0),
            y=float(py + th / 2.0),
            scale=0.0,
            rotation=0.0,
        ))
        y0 = max(0, py - min_dist)
        y1 = min(res_copy.shape[0], py + min_dist + 1)
        x0 = max(0, px - min_dist)
        x1 = min(res_copy.shape[1], px + min_dist + 1)
        res_copy[y0:y1, x0:x1] = -1.0
    return peaks


# ──────────────────────────────────────────────────────────────────────
# Sub-pixel Quadratic Refinement
# ──────────────────────────────────────────────────────────────────────

def subpixel_refine_1d(arr: np.ndarray, idx: int) -> float:
    """Quadratic interpolation for sub-pixel peak location (1D)."""
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    y0, y1, y2 = arr[idx - 1], arr[idx], arr[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-6:
        return float(idx)
    offset = 0.5 * (y0 - y2) / denom
    return float(idx) + offset


def subpixel_refine_2d(result: np.ndarray, py: int, px: int) -> Tuple[float, float]:
    """Quadratic interpolation for sub-pixel peak location (2D)."""
    x_row = result[py, max(0, px-1):min(result.shape[1], px+2)]
    if len(x_row) == 3:
        sub_px = subpixel_refine_1d(x_row, 1)
    else:
        sub_px = float(px)

    y_col = result[max(0, py-1):min(result.shape[0], py+2), px]
    if len(y_col) == 3:
        sub_py = subpixel_refine_1d(y_col, 1)
    else:
        sub_py = float(py)

    return sub_px, sub_py


# ──────────────────────────────────────────────────────────────────────
# Phase Correlation Reranker (gated)
# ──────────────────────────────────────────────────────────────────────

def build_nominal_template(reference_img: np.ndarray, nominal_downsample: float) -> Tuple[np.ndarray, int]:
    """Build nominal template at search resolution."""
    ref = reference_img.astype(np.float32)
    if ref.max() > 1.5:
        ref = ref / 255.0
    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    nominal_side = max(8, int(round(ref_dn.shape[1] / nominal_downsample)))
    templ = cv2.resize(ref_dn, (nominal_side, nominal_side), interpolation=cv2.INTER_AREA)
    return templ, nominal_side


def phase_rerank_candidates(
    reference_img: np.ndarray,
    search_img: np.ndarray,
    candidates: List[Candidate],
    nominal_downsample: float,
) -> List[Candidate]:
    """Re-rank candidates by local phase-correlation response."""
    templ, nominal_side = build_nominal_template(reference_img, nominal_downsample)
    srch = search_img.astype(np.float32)
    if srch.max() > 1.5:
        srch = srch / 255.0
    srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
    srch_dn = np.clip(srch_dn, 0, 1).astype(np.float32)
    window = cv2.createHanningWindow((nominal_side, nominal_side), cv2.CV_32F)

    for c in candidates:
        x0 = int(round(c.x - nominal_side / 2))
        y0 = int(round(c.y - nominal_side / 2))
        x0 = int(np.clip(x0, 0, srch_dn.shape[1] - nominal_side))
        y0 = int(np.clip(y0, 0, srch_dn.shape[0] - nominal_side))
        patch = srch_dn[y0:y0 + nominal_side, x0:x0 + nominal_side].copy()
        (_, _), resp = cv2.phaseCorrelate(templ.astype(np.float32), patch.astype(np.float32), window)
        c.phase_response = round(float(resp), 4)

    candidates.sort(key=lambda c: c.phase_response, reverse=True)
    return candidates


# ──────────────────────────────────────────────────────────────────────
# Gated Reranker Logic
# ──────────────────────────────────────────────────────────────────────

def should_rerank(nominal_side: int, candidates: List[Candidate], score_margin_threshold: float = 0.01, min_template_side: int = 60) -> bool:
    """Decide whether to invoke reranking (gated)."""
    if nominal_side < min_template_side:
        return False
    if len(candidates) < 2:
        return False
    margin = candidates[0].score - candidates[1].score
    return margin < score_margin_threshold


# ──────────────────────────────────────────────────────────────────────
# Main Enhanced Localizer (v6)
# ──────────────────────────────────────────────────────────────────────

def localize_v6(
    reference_img: np.ndarray,
    search_img: np.ndarray,
    nominal_downsample: float = 10.0,
    use_denoise: bool = False,
    denoise_method: str = "nlmeans",
    use_structural_rerank: bool = True,
    use_phase_rerank: bool = True,
    use_gated_rerank: bool = True,
    use_coarse_search: bool = True,
    return_debug: bool = False,
) -> Dict:
    """
    Enhanced v6 localizer combining competitive best practices.
    """
    t0 = time.time()

    # Pre-processing
    if use_denoise:
        search_proc = denoise_image(search_img, denoise_method)
        ref_proc = denoise_image(reference_img, denoise_method)
    else:
        search_proc = search_img.astype(np.float32)
        if search_proc.max() > 1.5:
            search_proc = search_proc / 255.0
        ref_proc = reference_img.astype(np.float32)
        if ref_proc.max() > 1.5:
            ref_proc = ref_proc / 255.0

    # Feature maps for robust edge alignment
    search_feat = get_feature_map(search_proc)
    ref_feat = get_feature_map(ref_proc)

    rh, rw = ref_proc.shape[:2]
    sh, sw = search_proc.shape[:2]

    # Nominal template at search resolution
    ref_dn = cv2.GaussianBlur(ref_proc, (0, 0), sigmaX=0.6)
    nominal_side = max(8, int(round(rw / nominal_downsample)))
    nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side), interpolation=cv2.INTER_AREA)
    nominal_templ_feat = get_feature_map(nominal_templ)

    # ─── Stage 1: Adaptive Coarse Search (only for larger templates) ───
    coarse_results = []

    # Only use coarse search if template is large enough (FinFET-style ~100px)
    # For small templates (DRAM ~30px), coarse downsampling destroys pattern structure
    min_coarse_template = 50

    if use_coarse_search and nominal_side >= min_coarse_template:
        search_coarse = cv2.resize(search_feat, (800, 800), interpolation=cv2.INTER_AREA)
        scale_coarse = 800.0 / 1000.0
        nominal_side_coarse = max(12, int(nominal_side * scale_coarse))
        nominal_templ_coarse = cv2.resize(nominal_templ_feat, (nominal_side_coarse, nominal_side_coarse), interpolation=cv2.INTER_AREA)

        scales = np.linspace(0.088, 0.112, 4)
        coarse_rotations = np.linspace(-8.0, 8.0, 13)

        for s in scales:
            tw_c = max(12, int(rw * s * scale_coarse))
            th_c = max(12, int(rh * s * scale_coarse))
            if th_c >= 800 or tw_c >= 800:
                continue
            ref_ds = cv2.resize(ref_proc, (tw_c, th_c), interpolation=cv2.INTER_AREA)
            templ_base = get_feature_map(ref_ds)

            for rot in coarse_rotations:
                if abs(rot) > 0.05:
                    M = cv2.getRotationMatrix2D((tw_c / 2.0, th_c / 2.0), rot, 1.0)
                    templ = cv2.warpAffine(templ_base, M, (tw_c, th_c), borderMode=cv2.BORDER_REFLECT)
                else:
                    templ = templ_base

                res = ncc_map(search_coarse, templ)
                val = float(res.max())
                py, px = np.unravel_index(res.argmax(), res.shape)
                coarse_results.append({
                    'score': val,
                    'scale': s,
                    'rotation': rot,
                    'x': float(px / scale_coarse + tw_c / 2.0 / scale_coarse),
                    'y': float(py / scale_coarse + th_c / 2.0 / scale_coarse),
                })

        coarse_results.sort(key=lambda r: -r['score'])
        top_coarse = coarse_results[:2] if len(coarse_results) >= 2 else coarse_results
    else:
        # No coarse search - still do full multi-scale search at fine stage
        top_coarse = [{'scale': s, 'rotation': 0.0} for s in np.linspace(0.088, 0.112, 4)]

    # ─── Stage 2: Fine Angular Refinement at Full 1000x1000 ───
    search_full = cv2.GaussianBlur(search_feat, (3, 3), 0.5)
    all_candidates = []
    global_best = Candidate(x=sw/2, y=sh/2, score=-1, scale=0.10, rotation=0.0)

    for coarse in top_coarse:
        s = coarse['scale']
        best_rot = coarse['rotation']

        tw = max(10, int(rw * s))
        th = max(10, int(rh * s))
        if tw >= sw or th >= sh:
            continue
        ref_ds = cv2.resize(ref_proc, (tw, th), interpolation=cv2.INTER_AREA)
        templ_base = get_feature_map(ref_ds)
        templ_base = cv2.GaussianBlur(templ_base, (3, 3), 0.5)

        # Only do rotation search for templates >= 50px (FinFET); small DRAM templates don't benefit from rotation
        min_rot_template = 50
        if nominal_side >= min_rot_template:
            fine_rotations = np.linspace(best_rot - 1.5, best_rot + 1.5, 5)
        else:
            fine_rotations = [best_rot]  # Single angle (usually 0) for small templates

        for rot in fine_rotations:
            if abs(rot) > 0.05:
                M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), rot, 1.0)
                templ = cv2.warpAffine(templ_base, M, (tw, th), borderMode=cv2.BORDER_REFLECT)
            else:
                templ = templ_base

            res = ncc_map(search_full, templ)
            val = float(res.max())

            if val > global_best.score:
                py, px = np.unravel_index(res.argmax(), res.shape)
                sub_px, sub_py = subpixel_refine_2d(res, py, px)
                global_best = Candidate(
                    score=val,
                    x=float(sub_px + tw / 2.0),
                    y=float(sub_py + th / 2.0),
                    scale=float(s),
                    rotation=float(rot),
                )

            t_thresh = max(0.25, val * 0.85)
            peaks = extract_nms_peaks(res, threshold=t_thresh, min_dist=15, max_peaks=10, tw=tw, th=th)
            for p in peaks:
                p.scale = float(s)
                p.rotation = float(rot)
                py, px = int(round(p.y - th/2)), int(round(p.x - tw/2))
                py = np.clip(py, 0, res.shape[0]-1)
                px = np.clip(px, 0, res.shape[1]-1)
                sub_px, sub_py = subpixel_refine_2d(res, py, px)
                p.x = float(sub_px + tw / 2.0)
                p.y = float(sub_py + th / 2.0)
            all_candidates.extend(peaks)

    # Deduplicate peaks
    all_candidates.sort(key=lambda p: -p.score)
    deduped = []
    for p in all_candidates:
        too_close = any(
            ((p.x - d.x) ** 2 + (p.y - d.y) ** 2) ** 0.5 < 15
            for d in deduped
        )
        if not too_close:
            deduped.append(p)

    if not deduped:
        deduped = [global_best]

    # ─── Stage 3: Structural Reranking ───
    rerank_debug = {"reranked": False, "method": "none"}

    if use_structural_rerank and len(deduped) > 1:
        top_k = deduped[:min(5, len(deduped))]
        original_top = deduped[0]
        structural_rerank_candidates(top_k, search_proc, ref_proc, template_size=100)

        if top_k[0] is not original_top:
            rerank_debug["reranked"] = True
            rerank_debug["method"] = "structural"
            deduped = top_k + deduped[len(top_k):]

    # ─── Stage 4: Gated Phase Reranking ───
    if use_gated_rerank and use_phase_rerank:
        if should_rerank(nominal_side, deduped):
            phase_rerank_candidates(reference_img, search_img, deduped, nominal_downsample)
            rerank_debug["reranked"] = True
            prev_method = rerank_debug.get("method", "none")
            rerank_debug["method"] = prev_method + "_phase" if prev_method != "none" else "phase"
            rerank_debug["nominal_side"] = nominal_side
            rerank_debug["top1_score"] = deduped[0].score
            rerank_debug["top1_phase"] = deduped[0].phase_response

    # ─── Final Decision ───
    final = deduped[0]

    # Ambiguity detection (peak-to-2nd-peak ratio)
    if len(deduped) > 1:
        ambiguity_ratio = final.score / max(deduped[1].score, 1e-6)
    else:
        ambiguity_ratio = 10.0

    confidence = final.score * min(1.0, ambiguity_ratio / 2.0)
    low_confidence_flag = confidence < 0.4 or ambiguity_ratio < 1.1

    elapsed = time.time() - t0

    result = {
        "x": round(final.x, 2),
        "y": round(final.y, 2),
        "score": round(final.score, 4),
        "confidence": round(confidence, 4),
        "scale": round(final.scale, 4),
        "rotation_deg": round(final.rotation, 2),
        "ambiguity_ratio": round(ambiguity_ratio, 2),
        "low_confidence_flag": low_confidence_flag,
        "method": "V6",
        "runtime_ms": round(elapsed * 1000, 1),
    }

    if return_debug:
        result["debug"] = {
            "coarse_results": coarse_results[:5] if coarse_results else [{"note": "skipped - template too small"}],
            "num_candidates": len(deduped),
            "rerank": rerank_debug,
            "top_candidates": [c.to_dict() for c in deduped[:5]],
        }

    return result


# ──────────────────────────────────────────────────────────────────────
# Backwards Compatible Interface
# ──────────────────────────────────────────────────────────────────────

def localize(reference_img: np.ndarray, search_img: np.ndarray, nominal_downsample: float = 10.0) -> Dict:
    """Backwards-compatible entry point using V6."""
    return localize_v6(reference_img, search_img, nominal_downsample)