"""
evaluate.py
-----------
Runs the localizer over every (reference, search) pair in a dataset folder,
compares against ground_truth.json, and reports accuracy.

Run:
    python3 evaluate.py --data ../data/self_eval
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from localizer import localize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/self_eval")
    ap.add_argument("--success_px", type=float, default=5.0,
                     help="error threshold (px, in search-image coords) counted as a success")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    gt_path = os.path.join(args.data, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)

    rows = []
    t0 = time.time()
    for sample_id, meta in sorted(gt.items()):
        ref_path = os.path.join(args.data, "reference", f"{sample_id}.png")
        search_path = os.path.join(args.data, "search", f"{sample_id}.png")
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        pred = localize(ref_img, search_img,
                         nominal_downsample=meta["ref_native_size"] / meta["gt_inset_size_px"])

        gt_x, gt_y = meta["gt_center_xy"]
        err = float(np.hypot(pred["x"] - gt_x, pred["y"] - gt_y))
        rows.append({
            "sample_id": sample_id,
            "style": meta["style"],
            "gt_x": gt_x, "gt_y": gt_y,
            "pred_x": pred["x"], "pred_y": pred["y"],
            "error_px": round(err, 2),
            "confidence": pred["confidence"],
            "ambiguity_ratio": pred["ambiguity_ratio"],
            "low_confidence_flag": pred["low_confidence_flag"],
            "success": err <= args.success_px,
        })

    elapsed = time.time() - t0
    n = len(rows)
    n_success = sum(r["success"] for r in rows)
    errors = np.array([r["error_px"] for r in rows])

    print(f"\n{'sample_id':<18}{'style':<14}{'err_px':>8}{'conf':>8}{'ambig':>8}  success")
    for r in rows:
        print(f"{r['sample_id']:<18}{r['style']:<14}{r['error_px']:>8.2f}"
              f"{r['confidence']:>8.3f}{str(r['ambiguity_ratio']):>8}  "
              f"{'OK' if r['success'] else 'MISS'}")

    print("\n--- summary ---")
    print(f"n_pairs           : {n}")
    print(f"success_rate(@{args.success_px}px): {n_success/n*100:.1f}%  ({n_success}/{n})")
    print(f"mean_error_px     : {errors.mean():.2f}")
    print(f"median_error_px   : {np.median(errors):.2f}")
    print(f"p90_error_px      : {np.percentile(errors, 90):.2f}")
    print(f"total_inference_s : {elapsed:.2f}  ({elapsed/n*1000:.1f} ms/pair)")

    by_style = {}
    for r in rows:
        by_style.setdefault(r["style"], []).append(r["success"])
    for style, vals in by_style.items():
        print(f"success_rate[{style}] : {sum(vals)/len(vals)*100:.1f}%  ({sum(vals)}/{len(vals)})")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
