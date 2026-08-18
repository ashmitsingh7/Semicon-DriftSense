#!/usr/bin/env python3
"""
Evaluate localizer on Vishnu's dataset.
"""

import sys
import os
import json
import numpy as np
import cv2
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.localization import localize


def main():
    # Vishnu's dataset paths
    gt_path = "/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/ground_truth.json"
    ref_dir = "/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/reference"
    search_dir = "/tmp/competitor_analysis/vishnu_drift-sense/test_dataset/search"

    with open(gt_path) as f:
        gt = json.load(f)

    rows = []
    t0 = time.time()
    for entry in gt:
        sample_id = entry['sample_id']
        gt_x = entry['x_center']
        gt_y = entry['y_center']

        ref_path = os.path.join(ref_dir, f"{sample_id:04d}.png")
        search_path = os.path.join(search_dir, f"{sample_id:04d}.png")

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            print(f"Warning: Could not load images for sample {sample_id}")
            continue

        # Vishnu's dataset has 1000x1000 reference and search, with template ~100px in search
        # So downsample factor = 10
        pred = localize(ref_img, search_img, nominal_downsample=10.0)

        err = float(np.hypot(pred["x"] - gt_x, pred["y"] - gt_y))
        rows.append({
            "sample_id": sample_id,
            "gt_x": gt_x, "gt_y": gt_y,
            "pred_x": pred["x"], "pred_y": pred["y"],
            "error_px": round(err, 2),
            "confidence": pred["confidence"],
            "ambiguity_ratio": pred["ambiguity_ratio"],
            "low_confidence_flag": pred["low_confidence_flag"],
            "success": err <= 5.0,
        })

    elapsed = time.time() - t0
    n = len(rows)
    n_success = sum(r["success"] for r in rows)
    errors = np.array([r["error_px"] for r in rows])

    print(f"\n{'sample_id':<10}{'err_px':>8}{'conf':>8}{'ambig':>8}  success")
    for r in rows:
        print(f"{r['sample_id']:<10}{r['error_px']:>8.2f}{r['confidence']:>8.3f}{str(r['ambiguity_ratio']):>8}  "
              f"{'OK' if r['success'] else 'MISS'}")

    print("\n--- summary ---")
    print(f"n_pairs           : {n}")
    print(f"success_rate(@5.0px): {n_success/n*100:.1f}%  ({n_success}/{n})")
    print(f"mean_error_px     : {errors.mean():.2f}")
    print(f"median_error_px   : {np.median(errors):.2f}")
    print(f"p90_error_px      : {np.percentile(errors, 90):.2f}")
    print(f"total_inference_s : {elapsed:.2f}  ({elapsed/n*1000:.1f} ms/pair)")

    # Save results
    out_path = "/tmp/competitor_analysis/vishnu_drift-sense/our_v6_on_vishnu.json"
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()