"""
native_verifier.py
-------------------
V3: native-resolution re-verification of the surviving V2 top-K candidates.

Question this module answers (and ONLY this question, per design intent --
see docs/design_notes.md "native verification"):

    Does native-resolution evidence reorder the surviving candidates such
    that the true location becomes distinguishable?

Architecture (deliberately minimal -- no CNN, no FFT, no learned
embedding, no weighted composite score):

    native reference  +  native candidate ROI  ->  native NCC

For each of the top few V2 candidates (by Stage-2 low-res score), extract
a native-resolution image patch around that candidate's location and NCC
it directly against the (already-native-resolution) Reference Image, with
a small scale/rotation grid to absorb residual pose error. Whichever
candidate scores highest under native NCC is the re-ranked answer.

-----------------------------------------------------------------------
IMPORTANT SELF-EVAL LIMITATION -- read before trusting these numbers
-----------------------------------------------------------------------
On real deployment data, "get a native-resolution image at candidate
(x, y)" means physically re-visiting that stage coordinate and capturing
a new high-mag image there -- exactly the re-navigation the wafer tool
already does. We don't have that capability in this sandbox.

For the *synthetic self-eval dataset only*, we exploit the fact that
build_dataset.py records the exact `seed` used to procedurally generate
each sample's native canvas (pattern_synth.synth_canvas is a
deterministic function of (style, seed)). So here we regenerate the same
10,000x10,000 native canvas from its stored seed and crop the ROI out of
it -- this is a faithful stand-in for "go look at native resolution here"
*only because our canvas generator is reproducible*. It will NOT work on
real SEM data, where no such regeneration is possible; a real deployment
would need an actual native-resolution re-capture step at each candidate
site. This limitation is stated here explicitly per the "never hide a
regression / never overstate a result" submission-quality principle --
do not read the numbers below as validated against anything other than
this synthetic generator.

Coordinate-transform caveat (see handoff §12): do NOT assume a naive
candidate_native = candidate_search * downsample mapping is exact -- the
Search Image also has its own small independent affine warp applied
during dataset generation (build_dataset.py's `apply_geometric
_degradation` on the full downsampled canvas) whose matrix is NOT saved
to ground_truth.json. We therefore do not try to invert it exactly;
instead we (a) extract a generously padded ROI to absorb the resulting
position uncertainty, and (b) run the same kind of small scale/rotation
search grid used in Stage 2, at native resolution, inside that ROI.
"""

import sys
import numpy as np
import cv2

from pattern_synth import synth_canvas


def extract_native_roi(style, seed, candidate_x, candidate_y, downsample,
                        roi_half_size, canvas_size=(10000, 10000)):
    """Regenerate the deterministic native canvas for this sample and crop
    a generously padded ROI around the candidate's native-scale location.
    See module docstring for why this regeneration is a self-eval-only
    stand-in for a real native re-capture."""
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
    """Simulate what a fresh native-resolution capture would actually look
    like (mild blur + sensor noise), rather than NCC-ing the reference
    against a noiseless synthetic ground-truth canvas -- that would be an
    unrealistically easy comparison and would overstate how well native
    verification works on real (noisy) native captures."""
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=blur_sigma)
    peak = 220.0 * dose_scale
    poisson = rng.poisson(np.clip(blurred, 0, 1) * peak).astype(np.float32) / peak
    gauss = rng.normal(0, 0.02 / np.sqrt(dose_scale), size=roi.shape).astype(np.float32)
    return poisson + gauss


def native_ncc_score(reference_img, native_roi,
                      scale_search=(0.97, 1.0, 1.03),
                      rot_search_deg=(-3, -1.5, 0, 1.5, 3)):
    """Small scale/rotation grid NCC between the (native-resolution)
    reference and a native ROI. Returns (best_score, best_dx, best_dy,
    best_scale, best_angle) where (dx, dy) is the offset of the matched
    location's center within the ROI."""
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


def verify_candidates(reference_img, candidates, style, seed,
                       downsample=10.0, k_verify=3,
                       roi_half_size_mult=1.3, canvas_size=(10000, 10000),
                       noise_seed_offset=99991):
    """Native-verify the top `k_verify` V2 candidates (already sorted by
    Stage-2 score, highest first). Returns a list of dicts, one per
    verified candidate, each with coarse/fine/native scores, sorted by
    native score descending (the re-ranked order)."""
    ref = reference_img
    ref_side = ref.shape[1]
    roi_half_size = int(ref_side * roi_half_size_mult / 2) + 50

    rows = []
    for rank, cand in enumerate(candidates[:k_verify]):
        roi, (x0, y0) = extract_native_roi(
            style, seed, cand["x"], cand["y"], downsample,
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
    return rows


def print_diagnostic_table(sample_id, rows, gt_xy):
    gx, gy = gt_xy
    print(f"\n=== {sample_id}  (GT = {gx:.2f}, {gy:.2f}) ===")
    header = f"{'pre-rank':>8} {'post-rank':>9} {'coarse':>8} {'fine':>8} {'native':>8} {'gt_err_px':>10}"
    print(header)
    for r in rows:
        err = float(np.hypot(r["x"] - gx, r["y"] - gy))
        print(f"{r['candidate_rank_pre_native']:>8} {r['candidate_rank_post_native']:>9} "
              f"{r['coarse_score']:>8.4f} {r['fine_score']:>8.4f} {r['native_score']:>8.4f} "
              f"{err:>10.1f}")


if __name__ == "__main__":
    import json
    import os
    import cv2 as _cv2
    from localizer import localize_topk

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/self_eval"
    samples = sys.argv[2:] if len(sys.argv) > 2 else ["finfet_017", "finfet_021", "finfet_023"]

    gt = json.load(open(os.path.join(data_dir, "ground_truth.json")))

    for sid in samples:
        meta = gt[sid]
        ref = _cv2.imread(os.path.join(data_dir, "reference", f"{sid}.png"), _cv2.IMREAD_GRAYSCALE)
        srch = _cv2.imread(os.path.join(data_dir, "search", f"{sid}.png"), _cv2.IMREAD_GRAYSCALE)
        nd = meta["ref_native_size"] / meta["gt_inset_size_px"]

        candidates = localize_topk(ref, srch, nominal_downsample=nd, K=40)
        gx, gy = meta["gt_center_xy"]
        best_pool_err = min(np.hypot(c["x"] - gx, c["y"] - gy) for c in candidates)

        if best_pool_err > 5.0:
            print(f"\n=== {sid} ===")
            print(f"GT is NOT in the V2 candidate pool (best pool error = {best_pool_err:.1f}px). "
                  f"Native verification re-ranks only what's already in the pool -- "
                  f"it cannot rescue a candidate that Stage 1/Stage 2 never produced. "
                  f"This is expected for this sample; see docs/design_notes.md.")
            continue

        rows = verify_candidates(ref, candidates, meta["style"], meta["seed"],
                                  downsample=nd, k_verify=5)
        print_diagnostic_table(sid, rows, (gx, gy))

        top_native = rows[0]
        top_native_err = float(np.hypot(top_native["x"] - gx, top_native["y"] - gy))
        pre_native_best = min(candidates[:5], key=lambda c: c["score"])
        pre_native_top_err = float(np.hypot(candidates[0]["x"] - gx, candidates[0]["y"] - gy))
        print(f"pre-native top-score candidate error : {pre_native_top_err:.1f}px")
        print(f"post-native top-ranked candidate error: {top_native_err:.1f}px")
        print("VERDICT:", "RESCUED" if top_native_err <= 5.0 and pre_native_top_err > 5.0
              else ("ALREADY OK" if pre_native_top_err <= 5.0 else "NOT RESCUED"))
