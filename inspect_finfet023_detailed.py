#!/usr/bin/env python3
"""
Inspect Stage-2 refinement and dedup for finfet_023.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from src.localizer import localize_topk, _refine_candidate, _topk_coarse_candidates

# Load ground truth
with open('/home/singh/semicon/Semicon-DriftSense/data/self_eval/ground_truth.json') as f:
    gt = json.load(f)

finfet_023_gt = gt['finfet_023']
gt_x, gt_y = finfet_023_gt['gt_center_xy']
print(f"GT: ({gt_x:.2f}, {gt_y:.2f})")

# Load images
ref_img = cv2.imread('/home/singh/semicon/Semicon-DriftSense/data/self_eval/reference/finfet_023.png', cv2.IMREAD_GRAYSCALE)
srch_img = cv2.imread('/home/singh/semicon/Semicon-DriftSense/data/self_eval/search/finfet_023.png', cv2.IMREAD_GRAYSCALE)

# Normalize
ref = ref_img.astype(np.float32) / 255.0
srch = srch_img.astype(np.float32) / 255.0

# Denoise
ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
srch_dn = np.clip(srch_dn, 0, 1).astype(np.float32)

# Nominal params
nominal_downsample = 10.0
nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
nominal_side = min(nominal_side, srch_dn.shape[0] - 1, srch_dn.shape[1] - 1)
print(f"Nominal side: {nominal_side}")

# Run localize_topk with debug to see full pipeline
candidates, debug = localize_topk(ref_img, srch_img, nominal_downsample=10.0,
                                   K=40, return_debug=True)

print(f"\nDebug info:")
for k, v in debug.items():
    print(f"  {k}: {v}")

print(f"\nFinal candidates ({len(candidates)}):")
print(f"{'Rank':>4} {'x':>8} {'y':>8} {'score':>8} {'scale':>8} {'rot':>6} {'dist_to_gt':>12}")
print("-" * 70)

for rank, c in enumerate(candidates):
    dist = np.hypot(c['x'] - gt_x, c['y'] - gt_y)
    print(f"{rank+1:4d} {c['x']:8.2f} {c['y']:8.2f} {c['score']:8.4f} {c['scale']:8.3f} {c['rotation_deg']:6d} {dist:12.2f}")

# Now let's trace what happens with the raw NMS peaks
# First let's see what the coarse peaks are (before Stage-2)
nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side), interpolation=cv2.INTER_AREA)
coarse_result = cv2.matchTemplate(srch_dn, nominal_templ, cv2.TM_CCOEFF_NORMED)

nms_radius = max(3, nominal_side // 2)
coarse_peaks = _topk_coarse_candidates(coarse_result, 40, nms_radius)

print(f"\n\nCoarse peaks (K=40, nms_radius={nms_radius}):")
print(f"{'Rank':>4} {'iy':>5} {'ix':>5} {'coarse_score':>12} {'coarse_cx':>10} {'coarse_cy':>10} {'dist_to_gt':>12}")
print("-" * 75)

for rank, (iy, ix, cval) in enumerate(coarse_peaks):
    cx = ix + nominal_side / 2.0
    cy = iy + nominal_side / 2.0
    dist = np.hypot(cx - gt_x, cy - gt_y)
    print(f"{rank+1:4d} {iy:5d} {ix:5d} {cval:12.4f} {cx:10.1f} {cy:10.1f} {dist:12.1f}")

# Now refine each and see what happens
window_r = int(nominal_side * 2.5) + 10
scale_search = (0.92, 0.96, 1.0, 1.04, 1.08)
rot_search_deg = (-4, -2.5, -1, 0, 1, 2.5, 4)

print(f"\n\nStage-2 refinement of each coarse peak:")
print(f"{'Rank':>4} {'coarse_cx':>10} {'coarse_cy':>10} {'refined_cx':>10} {'refined_cy':>10} {'score':>8} {'dist_to_gt':>12}")
print("-" * 85)

refined_results = []
for rank, (iy, ix, cval) in enumerate(coarse_peaks):
    coarse_cx = ix + nominal_side / 2.0
    coarse_cy = iy + nominal_side / 2.0

    r = _refine_candidate(ref_dn, srch_dn, coarse_cx, coarse_cy,
                           nominal_downsample, nominal_side,
                           scale_search, rot_search_deg, window_r)

    dist = np.hypot(r['cx'] - gt_x, r['cy'] - gt_y)
    print(f"{rank+1:4d} {coarse_cx:10.1f} {coarse_cy:10.1f} {r['cx']:10.1f} {r['cy']:10.1f} {r['score']:8.4f} {dist:12.2f}")

    r['coarse_cx'] = coarse_cx
    r['coarse_cy'] = coarse_cy
    r['coarse_score'] = cval
    refined_results.append(r)

# Now dedup on refined coordinates
dedup_radius = nominal_side  # 100 for FinFET
print(f"\n\nDedup radius: {dedup_radius}")

refined_results.sort(key=lambda r: r["score"], reverse=True)
kept = []
for r in refined_results:
    if all(np.hypot(r["cx"] - k["cx"], r["cy"] - k["cy"]) > dedup_radius for k in kept):
        kept.append(r)

print(f"\nAfter dedup (dedup_radius={dedup_radius}): {len(kept)} candidates kept")
print(f"{'Rank':>4} {'x':>8} {'y':>8} {'score':>8} {'dist_to_gt':>12}")
print("-" * 55)
for rank, r in enumerate(kept):
    dist = np.hypot(r['cx'] - gt_x, r['cy'] - gt_y)
    print(f"{rank+1:4d} {r['cx']:8.2f} {r['cy']:8.2f} {r['score']:8.4f} {dist:12.2f}")

# Check if any candidate is close to GT
print(f"\n\nCandidates within 10px of GT:")
for r in kept:
    dist = np.hypot(r['cx'] - gt_x, r['cy'] - gt_y)
    if dist < 10:
        print(f"  Found! x={r['cx']:.2f}, y={r['cy']:.2f}, score={r['score']:.4f}, dist={dist:.2f}")

# Also check if any refined candidate (before dedup) is close to GT
print(f"\nRefined candidates (before dedup) within 10px of GT:")
for r in refined_results:
    dist = np.hypot(r['cx'] - gt_x, r['cy'] - gt_y)
    if dist < 10:
        print(f"  Found! coarse_cx={r['coarse_cx']:.1f}, coarse_cy={r['coarse_cy']:.1f}, refined_cx={r['cx']:.2f}, refined_cy={r['cy']:.2f}, score={r['score']:.4f}, dist={dist:.2f}")