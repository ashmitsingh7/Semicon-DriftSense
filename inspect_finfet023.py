#!/usr/bin/env python3
"""
Inspect the raw coarse NCC surface for finfet_023 to understand why GT is eliminated.
"""

import numpy as np
import cv2
import json
from pathlib import Path

# Load ground truth
with open('/home/singh/semicon/Semicon-DriftSense/data/self_eval/ground_truth.json') as f:
    gt = json.load(f)

finfet_023_gt = gt['finfet_023']
gt_x, gt_y = finfet_023_gt['gt_center_xy']
print(f"GT: ({gt_x:.2f}, {gt_y:.2f})")

# Load images
ref_img = cv2.imread('/home/singh/semicon/Semicon-DriftSense/data/self_eval/reference/finfet_023.png', cv2.IMREAD_GRAYSCALE)
srch_img = cv2.imread('/home/singh/semicon/Semicon-DriftSense/data/self_eval/search/finfet_023.png', cv2.IMREAD_GRAYSCALE)

print(f"Reference shape: {ref_img.shape}")
print(f"Search shape: {srch_img.shape}")

# Normalize
ref = ref_img.astype(np.float32) / 255.0
srch = srch_img.astype(np.float32) / 255.0

# Denoise (same as localizer.py)
ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
srch_dn = cv2.GaussianBlur(srch, (0, 0), sigmaX=0.6)
srch_dn = np.clip(srch_dn, 0, 1).astype(np.float32)

# Nominal params (same as localizer.py for FinFET)
nominal_downsample = 10.0
nominal_side = max(6, int(round(ref_dn.shape[1] / nominal_downsample)))
nominal_side = min(nominal_side, srch_dn.shape[0] - 1, srch_dn.shape[1] - 1)
print(f"Nominal side: {nominal_side}")

nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side), interpolation=cv2.INTER_AREA)
print(f"Template shape: {nominal_templ.shape}")

# Coarse NCC
coarse_result = cv2.matchTemplate(srch_dn, nominal_templ, cv2.TM_CCOEFF_NORMED)
print(f"Coarse surface shape: {coarse_result.shape}")
print(f"Coarse surface range: [{coarse_result.min():.4f}, {coarse_result.max():.4f}]")

# Find GT location in coarse surface coordinates
# GT in search image is at (gt_x, gt_y) - this is the center
# Coarse surface coordinates: top-left of template
gt_coarse_ix = int(gt_x - nominal_side / 2.0)
gt_coarse_iy = int(gt_y - nominal_side / 2.0)
print(f"GT coarse coords: iy={gt_coarse_iy}, ix={gt_coarse_ix}")

if 0 <= gt_coarse_iy < coarse_result.shape[0] and 0 <= gt_coarse_ix < coarse_result.shape[1]:
    gt_score = coarse_result[gt_coarse_iy, gt_coarse_ix]
    print(f"GT coarse score: {gt_score:.4f}")
else:
    print("GT out of bounds for coarse surface")

# Get global max
_, global_max, _, global_loc = cv2.minMaxLoc(coarse_result)
print(f"Global max score: {global_max:.4f} at {global_loc}")

# Now do NMS to get top 60 peaks (as mentioned in the note)
def topk_coarse_candidates(coarse_surface, k, nms_radius):
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

nms_radius = max(3, nominal_side // 2)
print(f"NMS radius: {nms_radius}")

peaks = topk_coarse_candidates(coarse_result, 60, nms_radius)
print(f"\nTop 60 NMS peaks:")
print(f"{'Rank':>4} {'iy':>5} {'ix':>5} {'coarse_score':>12} {'cx':>8} {'cy':>8} {'dist_to_gt':>12}")
print("-" * 70)

gt_cx = gt_x
gt_cy = gt_y

for rank, (iy, ix, cval) in enumerate(peaks):
    cx = ix + nominal_side / 2.0
    cy = iy + nominal_side / 2.0
    dist = np.hypot(cx - gt_cx, cy - gt_cy)
    marker = " <<< GT" if (iy == gt_coarse_iy and ix == gt_coarse_ix) else ""
    print(f"{rank+1:4d} {iy:5d} {ix:5d} {cval:12.4f} {cx:8.1f} {cy:8.1f} {dist:12.1f}{marker}")

# Check if GT is in the raw NMS pool
gt_in_pool = any(iy == gt_coarse_iy and ix == gt_coarse_ix for iy, ix, _ in peaks)
print(f"\nGT in raw NMS pool (before dedup): {gt_in_pool}")

# How many peaks have higher score than GT?
if 0 <= gt_coarse_iy < coarse_result.shape[0] and 0 <= gt_coarse_ix < coarse_result.shape[1]:
    gt_score = coarse_result[gt_coarse_iy, gt_coarse_ix]
    higher_count = sum(1 for _, _, cval in peaks if cval > gt_score)
    print(f"Peaks with higher coarse score than GT: {higher_count}")
    print(f"GT coarse score: {gt_score:.4f}")

# Also show spatial distribution
print("\nSpatial distribution of top 60 peaks (cx, cy):")
for rank, (iy, ix, cval) in enumerate(peaks):
    cx = ix + nominal_side / 2.0
    cy = iy + nominal_side / 2.0
    print(f"  {cx:.1f}, {cy:.1f}  (score={cval:.4f})")