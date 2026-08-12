"""
localizer.py
------------
Navigation-error recovery algorithm: given a Reference Image (small,
native-resolution patch) and a Search Image (larger, ~10x lower
magnification, noisier), find the pixel location in the Search Image
where the reference pattern appears, and return its center (x, y).

Approach: multi-scale, small-rotation normalized cross-correlation (NCC)
with sub-pixel parabolic peak refinement.

  - NCC (via cv2.matchTemplate, TM_CCOEFF_NORMED) is the standard,
    well-understood similarity measure for this kind of template
    localization task and is efficient to compute at scale
    [Lewis, 1995; Briechle & Hanebeck, 2001]:
      J. P. Lewis, "Fast Normalized Cross-Correlation," Vision Interface,
      1995, pp. 120-123.
      K. Briechle and U. D. Hanebeck, "Template matching using fast
      normalized cross correlation," Proc. SPIE 4387, Aerospace/Defense
      Sensing, Simulation, and Controls, 2001.
  - Because the true magnification ratio and small rotation are not known
    exactly (motion-stage drift also introduces slight rotation/scale
    error, not just translation), the template is matched at a small grid
    of candidate scales and rotations around the nominal 1/10 relationship
    and the best-scoring pose is kept.
  - Because the search pattern is highly periodic, the raw NCC surface has
    many near-tied local maxima at multiples of the pattern pitch. We
    exploit the non-periodic content injected into the scene (local
    brightness drift, sparse defects -- see pattern_synth.py) by not just
    taking the single global maximum but validating peak "sharpness"
    (peak-to-second-peak ratio) so failure cases (ambiguous periodic
    regions) can be flagged rather than silently mis-localized, directly
    answering the "test failure mode awareness" requirement in the
    problem statement.
  - Sub-pixel refinement fits a 2D parabola to the 3x3 neighborhood around
    the discrete NCC peak, a standard sub-pixel correlation-peak
    refinement approach [Tian & Huhns, "Algorithms for Subpixel
    Registration", CVGIP 35, 1986].
"""

import numpy as np
import cv2


def _subpixel_peak(surface, iy, ix):
    """Parabolic sub-pixel refinement around a discrete correlation peak."""
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
    """Ratio of the global peak to the best peak outside a local exclusion
    window -- low values indicate a periodic / ambiguous match."""
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

      Stage 1 (coarse, GLOBAL): a single nominal-scale/zero-rotation NCC
      pass over the *entire* search image. This is what we need anyway to
      measure how ambiguous the site is (peak vs. next-best peak *anywhere*
      in the image) -- doing it once, globally, is cheap and gives both a
      coarse location and the ambiguity signal in one pass.

      Stage 2 (fine, LOCAL): crop a small window around the stage-1 peak
      and only run the full scale x rotation grid inside that window. The
      grid search dominates cost, and matchTemplate cost scales with
      search-image area, so confining it to a small local crop (instead of
      the full 1000x1000 image) is what actually removes the bottleneck --
      this is the single biggest lever on the throughput metric the
      problem statement scores separately from accuracy.

    reference_img: 2D float32/uint8 array, native-resolution reference patch
    search_img:    2D float32/uint8 array, the (larger) search image
    nominal_downsample: expected reference->search magnification ratio

    Returns dict: {"x", "y", "confidence", "scale", "rotation_deg"}
    (and "debug" if return_debug=True)
    """
    ref = reference_img.astype(np.float32)
    srch = search_img.astype(np.float32)
    if ref.max() > 1.5:
        ref = ref / 255.0
    if srch.max() > 1.5:
        srch = srch / 255.0

    # mild denoise before correlation -- reduces high-frequency shot-noise
    # sensitivity without blurring away the structural signal
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
        # local window degenerate (reference bigger than crop) -- fall
        # back to the coarse global estimate
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


# ---------------------------------------------------------------------------
# V2: top-K multi-hypothesis candidate architecture
# ---------------------------------------------------------------------------
#
#   global NCC -> top-K spatially-separated local maxima (NMS on the
#   coarse surface) -> independent Stage-2 refinement per candidate ->
#   dedup on the REFINED coordinates (not the coarse ones) -> ranked list.
#
# Rationale for dedup-after-refinement rather than dedup-on-coarse-NMS
# alone: the Stage-2 local window (~2.5x the coarse template side) is much
# larger than a small coarse NMS radius, so several coarse peaks that look
# spatially distinct at the low-res coarse-surface scale can still land in
# (near-)identical Stage-2 windows and converge to the same refined peak.
# Left undeduped, that makes K=40 behave like K=~1 in practice. Dedup must
# therefore run again on the Stage-2 output, keyed on the refined (cx, cy).
#
# This does not change or call the V1 `localize()` code path above -- V1
# remains the control baseline.

def _topk_coarse_candidates(coarse_surface, k, nms_radius):
    """Greedy NMS peak-picking on the coarse NCC surface: repeatedly take
    the global max, record it, then suppress a radius around it so the
    next iteration is forced to find a spatially distinct peak. This is
    what prevents "peak, peak+1px, peak+2px" from being counted as
    separate hypotheses."""
    work = coarse_surface.copy()
    h, w = work.shape
    peaks = []
    for _ in range(k):
        _, val, _, loc = cv2.minMaxLoc(work)
        if not np.isfinite(val) or val <= -1e8:
            break
        ix, iy = loc
        peaks.append((iy, ix, float(val)))
        y0, y1 = max(0, iy - nms_radius), min(h, iy + nms_radius + 1)
        x0, x1 = max(0, ix - nms_radius), min(w, ix + nms_radius + 1)
        work[y0:y1, x0:x1] = -1e9
    return peaks


def _refine_candidate(ref_dn, srch_dn, coarse_cx, coarse_cy, nominal_downsample,
                       nominal_side, scale_search, rot_search_deg, window_r):
    """Run the same Stage-2 scale x rotation grid search used by V1's
    localize(), but around an arbitrary (coarse_cx, coarse_cy) candidate
    instead of the single global argmax."""
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
        best = {"score": -1.0, "target_side": nominal_side, "angle": 0,
                 "scale_mult": 1.0, "cx": coarse_cx, "cy": coarse_cy}
    return best


def localize_topk(reference_img, search_img, nominal_downsample=10.0,
                   K=40, nms_radius=None, dedup_radius=None,
                   scale_search=(0.92, 0.96, 1.0, 1.04, 1.08),
                   rot_search_deg=(-4, -2.5, -1, 0, 1, 2.5, 4),
                   return_debug=False):
    """
    V2 top-K candidate architecture.

    Returns a list of candidate dicts, ranked by Stage-2 score descending,
    each: {"x", "y", "score", "scale", "rotation_deg",
           "coarse_cx", "coarse_cy"}.
    List length is <= K and, after dedup, typically much smaller than K
    (empirically ~8-12 for the FinFET style at K=40 -- see design notes).

    tie_margin / center-priority re-ranking is deliberately NOT applied
    here -- see docs/design_notes.md: a sweep across tie_margin found it
    regresses accuracy against our synthetic GT (12/15 -> 5/15 at
    tie_margin=0.05), because it can legitimately prefer a different,
    equally-valid periodic occurrence over the one GT happens to record.
    Candidates are returned in pure-score order; callers that want
    center-priority re-ranking for spec-compliance reporting should do so
    explicitly as a separate, labeled pass.
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

    coarse_result = cv2.matchTemplate(srch_dn, nominal_templ, cv2.TM_CCOEFF_NORMED)

    if nms_radius is None:
        nms_radius = max(3, nominal_side // 2)
    coarse_peaks = _topk_coarse_candidates(coarse_result, K, nms_radius)

    window_r = int(nominal_side * 2.5) + 10
    if dedup_radius is None:
        # NOTE: this must NOT be window_r (~260px for FinFET). Using the
        # full Stage-2 window as the dedup radius was tried first and is
        # wrong: on finfet_021 it merges the GT-adjacent candidate cluster
        # (score 0.8020, 0px error) into a *different*, slightly
        # higher-scoring decoy cluster only ~182px away (score 0.8043),
        # silently discarding the correct site. Distinct periodic sites in
        # this dataset can sit within one Stage-2 window's radius of each
        # other, so dedup must use a tighter radius tied to the reference
        # footprint itself (candidates this close ARE the same physical
        # site; anything farther is a genuinely different site).
        dedup_radius = nominal_side

    refined = []
    for (iy, ix, cval) in coarse_peaks:
        coarse_cx = ix + nominal_side / 2.0
        coarse_cy = iy + nominal_side / 2.0
        r = _refine_candidate(ref_dn, srch_dn, coarse_cx, coarse_cy,
                               nominal_downsample, nominal_side,
                               scale_search, rot_search_deg, window_r)
        r["coarse_cx"] = coarse_cx
        r["coarse_cy"] = coarse_cy
        r["coarse_score"] = cval
        refined.append(r)

    # dedup on REFINED coordinates, keeping the highest-scoring survivor
    refined.sort(key=lambda r: r["score"], reverse=True)
    kept = []
    for r in refined:
        if all(np.hypot(r["cx"] - k["cx"], r["cy"] - k["cy"]) > dedup_radius
               for k in kept):
            kept.append(r)

    out = []
    for r in kept:
        out.append({
            "x": round(float(r["cx"]), 2),
            "y": round(float(r["cy"]), 2),
            "score": round(float(r["score"]), 4),
            "scale": round(nominal_downsample / r["scale_mult"], 3),
            "rotation_deg": r["angle"],
            "coarse_cx": round(float(r["coarse_cx"]), 2),
            "coarse_cy": round(float(r["coarse_cy"]), 2),
            "coarse_score": round(float(r["coarse_score"]), 4),
        })

    if return_debug:
        return out, {"n_coarse_peaks": len(coarse_peaks), "n_after_dedup": len(kept),
                      "nms_radius": nms_radius, "dedup_radius": dedup_radius,
                      "window_r": window_r, "nominal_side": nominal_side}
    return out
