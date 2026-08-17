#!/usr/bin/env python3
"""Evaluate our V2 method on vishnu's dataset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import json
import numpy as np
import time

from src.localization.v2_candidates import localize_v2

# Load vishnu dataset ground truth
gt_path = "/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/ground_truth.json"
with open(gt_path) as f:
    gt = json.load(f)

ref_dir = "/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/reference"
search_dir = "/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/search"

rows = []
t0 = time.time()
for meta in gt:
    sample_id = f"{meta['sample_id']:04d}"
    ref_path = os.path.join(ref_dir, f"{sample_id}.png")
    search_path = os.path.join(search_dir, f"{sample_id}.png")

    if not os.path.exists(ref_path) or not os.path.exists(search_path):
        print(f"Warning: Missing images for {sample_id}, skipping")
        continue

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    # V2 candidate pool
    result = localize_v2(ref_img, search_img, nominal_downsample=10.0)

    # Get top candidate - it's the first one since sorted by score desc
    candidates = result.get("candidates", [])
    if not candidates:
        print(f"  {sample_id}: NO CANDIDATES!")
        continue

    best = candidates[0]
    pred_x = best["x"]
    pred_y = best["y"]

    gt_x = meta["x_center"]
    gt_y = meta["y_center"]
    err = float(np.hypot(pred_x - gt_x, pred_y - gt_y))

    rows.append({
        "sample_id": sample_id,
        "gt_x": gt_x, "gt_y": gt_y,
        "pred_x": pred_x, "pred_y": pred_y,
        "error_px": round(err, 2),
        "confidence": best.get("confidence", 0),
        "success": err <= 5.0,
    })
    print(f"  {sample_id}: pred=({pred_x:.2f}, {pred_y:.2f}) gt=({gt_x:.2f}, {gt_y:.2f}) err={err:.2f}px  conf={best.get('confidence',0):.3f}")

elapsed = time.time() - t0
n = len(rows)
n_success = sum(r["success"] for r in rows)
errors = np.array([r["error_px"] for r in rows])

print(f"\n=== Our V2 on Vishnu Dataset ===")
print(f"n_pairs           : {n}")
print(f"success_rate(@5px): {n_success/n*100:.1f}%  ({n_success}/{n})")
print(f"mean_error_px     : {errors.mean():.2f}")
print(f"median_error_px   : {np.median(errors):.2f}")
print(f"p90_error_px      : {np.percentile(errors, 90):.2f}")
print(f"max_error_px      : {errors.max():.2f}")
print(f"total_inference_s : {elapsed:.2f}  ({elapsed/n*1000:.1f} ms/pair)")