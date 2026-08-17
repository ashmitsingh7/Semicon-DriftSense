#!/usr/bin/env python3
"""
Inspect dedup process more carefully for finfet_023.
"""

import numpy as np
import cv2
import json
from src.localizer import _refine_candidate, _topk_coarse_candidates

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

nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side), interpolation=cv2.INTER_AREA)
coarse_result = cv2.matchTemplate(srch_dn, nominal_templ, cv2.TM_CCOEFF_NORMED)

nms_radius = max(3, nominal_side // 2)
coarse_peaks = _topk_coarse_candidates(coarse_result, 60, nms_radius)  # Get 60 to match ablation note

window_r = int(nominal_side * 2.5) + 10
scale_search = (0.92, 0.96, 1.0, 1.04, 1.08)
rot_search_deg = (-4, -2.5, -1, 0, 1, 2.5, 4)

# Refine all 60
refined_results = []
for rank, (iy, ix, cval) in enumerate(coarse_peaks):
    coarse_cx = ix + nominal_side / 2.0
    coarse_cy = iy + nominal_side / 2.0

    r = _refine_candidate(ref_dn, srch_dn, coarse_cx, coarse_cy,
                           nominal_downsample, nominal_side,
                           scale_search, rot_search_deg, window_r)

    r['coarse_cx'] = coarse_cx
    r['coarse_cy'] = coarse_cy
    r['coarse_score'] = cval
    r['rank'] = rank
    refined_results.append(r)

# Sort by refined score
refined_results.sort(key=lambda r: r["score"], reverse=True)

# Now trace dedup step by step with dedup_radius = nominal_side = 100
dedup_radius = nominal_side
print(f"Dedup radius: {dedup_radius}")
print(f"\nStep-by-step dedup:")

kept = []
for r in refined_results:
    dist_to_gt = np.hypot(r['cx'] - gt_x, r['cy'] - gt_y)

    # Check against kept
    conflicts = []
    for k in kept:
        d = np.hypot(r["cx"] - k["cx"], r["cy"] - k["cy"])
        if d <= dedup_radius:
            conflicts.append((k['rank'], k['cx'], k['cy'], k['score'], d))

    if conflicts:
        print(f"  REJECTED rank {r['rank']:2d}: refined=({r['cx']:7.2f}, {r['cy']:7.2f}) score={r['score']:.4f} dist_gt={dist_to_gt:7.2f}")
        for c_rank, c_cx, c_cy, c_score, d in conflicts:
            print(f"      Conflicts with kept rank {c_rank:2d}: ({c_cx:7.2f}, {c_cy:7.2f}) score={c_score:.4f} dist={d:.2f}")
    else:
        kept.append(r)
        print(f"  KEPT     rank {r['rank']:2d}: refined=({r['cx']:7.2f}, {r['cy']:7.2f}) score={r['score']:.4f} dist_gt={dist_to_gt:7.2f}")

print(f"\n\nFinal kept: {len(kept)} candidates")
for r in kept:
    dist_to_gt = np.hypot(r['cx'] - gt_x, r['cy'] - gt_y)
    print(f"  rank {r['rank']:2d}: ({r['cx']:7.2f}, {r['cy']:7.2f}) score={r['score']:.4f} dist_gt={dist_to_gt:7.2f}")

# Now the key question: are any refined candidates close to GT?
# Let's check ALL refined candidates within, say, 200px of GT
print(f"\n\nAll refined candidates within 200px of GT (before dedup):")
for r in refined_results:
    dist_to_gt = np.hypot(r['cx'] - gt_x, r['cy'] - gt_y)
    if dist_to_gt < 200:
        print(f"  rank {r['rank']:2d}: coarse=({r['coarse_cx']:7.1f}, {r['coarse_cy']:7.1f}) refined=({r['cx']:7.2f}, {r['cy']:7.2f}) score={r['score']:.4f} dist_gt={dist_to_gt:7.2f}")