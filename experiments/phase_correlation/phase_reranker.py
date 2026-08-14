"""
phase_reranker.py
------------------
V4 (EXPERIMENTAL, not yet validated at scale): local windowed
Fourier-phase-correlation reranking of the EXISTING V2 candidate pool.

This module does not touch V1 (`localizer.localize`), V2
(`localizer.localize_topk`), or V3 (`native_verifier`). It is a pure
post-hoc reranker: it takes whatever candidate list V2 already produced
and asks a single question --

    does reordering V2's candidates by local phase-correlation response
    (instead of V2's own low-res intensity-NCC score) put a more accurate
    candidate at rank 1?

-----------------------------------------------------------------------
Origin / what is reproduced vs. newly implemented
-----------------------------------------------------------------------
A prior agent (not this one) reported, in an ad-hoc audit script outside
the committed repo, that local `cv2.phaseCorrelate` responses cleanly
separated the true `finfet_023` site (response ~0.83) from that sample's
top-20 intensity-NCC false candidates (response ~0.03-0.31), and that
picking the highest-phase-response V2 candidate for `finfet_021` landed
~0.4px from GT. That prior script is NOT part of this repository and its
exact code could not be located here -- only its reported numbers and a
CSV/three PNGs derived from it. Nothing below should be read as having
independently reproduced those exact figures; it is a fresh
implementation of the same stated method (local Hann-windowed
`cv2.phaseCorrelate` between the search-resolution reference template and
a same-size window around each V2 candidate), run here for the first time
against the full 30+10 pair benchmark rather than 3 hand-picked cases.
Where this module's own numbers on finfet_017/021/023 land is reported in
`docs/phase_reranker_experiment.md` and compared explicitly against the
prior agent's reported numbers rather than assumed to match.

-----------------------------------------------------------------------
Method
-----------------------------------------------------------------------
For each V2 candidate (already Stage-2 refined, in search-image
coordinates):

  1. Build the same nominal-scale template `localizer.py` Stage 1 uses
     (reference resized to the nominal 1/10 scale, mildly denoised) --
     NOT a per-candidate scale/rotation-matched template. This keeps the
     reranker a single fixed probe applied identically to every
     candidate, so it cannot be accused of implicitly re-doing V2's own
     scale/rotation search.
  2. Extract a same-size window from the (denoised) search image centered
     on the candidate's refined (x, y).
  3. Apply a Hann window to both (mitigates edge/DC leakage -- see the
     validation checks in `docs/phase_reranker_experiment.md`).
  4. `cv2.phaseCorrelate(template, window)` -> response in [~-0.1, ~1.0].
  5. Re-sort the SAME candidate list by this response, descending.

No global search, no candidate generation, no native-resolution access,
no parameter tuned against the 40-pair set (window choice was fixed
before the full benchmark was run -- see validation script).
"""

import numpy as np
import cv2


def _prep(img):
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def build_nominal_template(reference_img, nominal_downsample):
    """Same construction as localizer.py Stage 1's `nominal_templ`."""
    ref = _prep(reference_img)
    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
    templ = cv2.resize(ref_dn, (nominal_side, nominal_side),
                        interpolation=cv2.INTER_AREA)
    return templ, nominal_side


def prep_search(search_img):
    srch = _prep(search_img)
    srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
    return np.clip(srch_dn, 0, 1).astype(np.float32)


def _extract_patch(cx, cy, size, img):
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = int(np.clip(x0, 0, img.shape[1] - size))
    y0 = int(np.clip(y0, 0, img.shape[0] - size))
    return img[y0:y0 + size, x0:x0 + size].copy(), (x0, y0)


def phase_score(templ, patch, window):
    (_, _), resp = cv2.phaseCorrelate(templ.astype(np.float32),
                                       patch.astype(np.float32), window)
    return float(resp)


def rerank(reference_img, search_img, candidates, nominal_downsample,
           use_hann=True):
    """
    candidates: list of dicts as returned by localizer.localize_topk
                (each has at least "x", "y", "score").

    Returns a NEW list (does not mutate input), each candidate dict with
    an added "phase_response" key, sorted by phase_response descending.
    Also returns timing (seconds) for the reranking pass alone.
    """
    import time
    t0 = time.time()

    templ, nominal_side = build_nominal_template(reference_img, nominal_downsample)
    srch_dn = prep_search(search_img)
    window = cv2.createHanningWindow((nominal_side, nominal_side), cv2.CV_32F) \
        if use_hann else None

    out = []
    for c in candidates:
        patch, (x0, y0) = _extract_patch(c["x"], c["y"], nominal_side, srch_dn)
        resp = phase_score(templ, patch, window)
        c2 = dict(c)
        c2["phase_response"] = round(resp, 4)
        out.append(c2)

    out.sort(key=lambda c: c["phase_response"], reverse=True)
    elapsed = time.time() - t0
    return out, elapsed


def localize_v4(reference_img, search_img, nominal_downsample=10.0,
                 K=40, use_hann=True, topk_kwargs=None, return_debug=False):
    """
    Convenience end-to-end entry point: V2 candidate generation
    (unchanged) + V4 phase reranking. Kept separate from `localizer.py`
    on purpose -- importing `localize_topk` here rather than modifying it.
    """
    from localizer import localize_topk
    import time

    topk_kwargs = topk_kwargs or {}
    t0 = time.time()
    v2_candidates, v2_debug = localize_topk(
        reference_img, search_img, nominal_downsample=nominal_downsample,
        K=K, return_debug=True, **topk_kwargs)
    v2_time = time.time() - t0

    if not v2_candidates:
        return None, None, {"v2_time": v2_time, "phase_time": 0.0}

    reranked, phase_time = rerank(reference_img, search_img, v2_candidates,
                                   nominal_downsample, use_hann=use_hann)

    v2_pred = v2_candidates[0]     # V2's own top-score pick, unchanged
    v4_pred = reranked[0]          # V4's phase-reranked pick

    debug = {"v2_time": v2_time, "phase_time": phase_time,
             "n_candidates": len(v2_candidates), **v2_debug}
    if return_debug:
        return v2_pred, v4_pred, {"v2_candidates": v2_candidates,
                                   "v4_candidates": reranked, **debug}
    return v2_pred, v4_pred, debug
