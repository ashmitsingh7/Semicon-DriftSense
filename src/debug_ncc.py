#!/usr/bin/env python3
"""
Debug NCC on Vishnu's dataset at full resolution.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.localization.enhanced_localizer import ncc_map, get_feature_map


def main():
    sample_id = 0
    ref_path = f"/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/reference/{sample_id:04d}.png"
    search_path = f"/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/search/{sample_id:04d}.png"

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    # Normalize
    ref = ref_img.astype(np.float32) / 255.0
    search = search_img.astype(np.float32) / 255.0

    # Feature maps
    ref_feat = get_feature_map(ref)
    search_feat = get_feature_map(search)

    print("Ref feat stats:", ref_feat.min(), ref_feat.max(), ref_feat.mean())
    print("Search feat stats:", search_feat.min(), search_feat.max(), search_feat.mean())

    # Nominal template at search resolution (~100px for 1000/10)
    nominal_side = 100
    ref_dn = cv2.GaussianBlur(ref, (0, 0), sigmaX=0.6)
    nominal_templ = cv2.resize(ref_dn, (nominal_side, nominal_side), interpolation=cv2.INTER_AREA)
    nominal_templ_feat = get_feature_map(nominal_templ)

    print(f"Nominal template feat stats: {nominal_templ_feat.min():.4f} {nominal_templ_feat.max():.4f} {nominal_templ_feat.mean():.4f}")

    # Full NCC
    res = ncc_map(search_feat, nominal_templ_feat)
    print(f"NCC shape: {res.shape}")
    print(f"NCC max: {res.max():.4f} at {np.unravel_index(res.argmax(), res.shape)}")
    print(f"NCC min: {res.min():.4f}")

    # Check around true location
    gt_x, gt_y = 439.7, 445.0
    # Convert to feature map coordinates (template center)
    py, px = int(gt_y - nominal_side/2), int(gt_x - nominal_side/2)
    if 0 <= py < res.shape[0] and 0 <= px < res.shape[1]:
        print(f"NCC at GT ({px}, {py}): {res[py, px]:.4f}")

    # Check around center (where our algorithm found)
    py, px = int(45 - nominal_side/2), int(45 - nominal_side/2)
    if 0 <= py < res.shape[0] and 0 <= px < res.shape[1]:
        print(f"NCC at center ({px}, {py}): {res[py, px]:.4f}")

    # Top 10 peaks
    flat = res.flatten()
    top_indices = np.argpartition(flat, -10)[-10:]
    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
    print("\nTop 10 NCC peaks:")
    for idx in top_indices:
        py, px = np.unravel_index(idx, res.shape)
        print(f"  ({px:.1f}, {py:.1f}): {flat[idx]:.4f}")

    # Also test raw (non-feature) NCC
    res_raw = ncc_map(search, nominal_templ)
    print(f"\nRaw NCC max: {res_raw.max():.4f} at {np.unravel_index(res_raw.argmax(), res_raw.shape)}")
    if 0 <= py < res_raw.shape[0] and 0 <= px < res_raw.shape[1]:
        print(f"Raw NCC at GT ({px}, {py}): {res_raw[py, px]:.4f}")

    # Check what Vishnu's method does - they use phase correlation
    window = cv2.createHanningWindow((nominal_side, nominal_side), cv2.CV_32F)
    search_dn = cv2.GaussianBlur(search, (0, 0), sigmaX=0.6)
    patch = search_dn[int(gt_y-nominal_side/2):int(gt_y+nominal_side/2), int(gt_x-nominal_side/2):int(gt_x+nominal_side/2)]
    if patch.shape == (nominal_side, nominal_side):
        (dx, dy), resp = cv2.phaseCorrelate(nominal_templ.astype(np.float32), patch.astype(np.float32), window)
        print(f"\nPhase correlation at GT: shift=({dx:.2f}, {dy:.2f}), resp={resp:.4f}")

    # Check center patch
    patch_center = search_dn[45-nominal_side//2:45+nominal_side//2, 45-nominal_side//2:45+nominal_side//2]
    if patch_center.shape == (nominal_side, nominal_side):
        (dx, dy), resp = cv2.phaseCorrelate(nominal_templ.astype(np.float32), patch_center.astype(np.float32), window)
        print(f"Phase correlation at center: shift=({dx:.2f}, {dy:.2f}), resp={resp:.4f}")


if __name__ == "__main__":
    main()