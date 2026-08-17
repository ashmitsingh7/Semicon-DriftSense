#!/usr/bin/env python3
"""Entry point: Localization / inference per problem statement.

Usage (single pair):
    python3 localize.py --ref data/self_eval/reference/dram_000.png \
                        --search data/self_eval/search/dram_000.png

Usage (batch):
    python3 localize.py --input data/self_eval --output results/self_eval
    python3 localize.py --input data/ood_holdout --output results/ood_holdout

Usage (with gated phase reranker - default):
    python3 localize.py --input data/self_eval --output results/ --method v5_gated

Methods:
    v1          - Two-stage single-argmax NCC (baseline)
    v2          - Top-K candidate pool with NMS + fine refinement
    v5_gated    - V2 + gated phase reranker (PRODUCTION DEFAULT, 95% @5px)
    v5b_gated_native - V2 + gated native verification (SELF-EVAL ONLY, needs seed)
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.io import load_image, save_json
from src.localization.v1_localize import localize as localize_v1
from src.localization.v2_candidates import localize_v2
from src.localization.rerankers.gated_reranker import (
    localize_with_gated_rerank,
    phase_rerank_candidates,
    native_rerank_candidates,
    should_rerank,
)


def run_single(ref_path, search_path, method, nominal_downsample=10.0, style=None, seed=None, use_native=False):
    """Run localization on a single reference/search pair."""
    ref = load_image(ref_path)
    search = load_image(search_path)

    if method == "v1":
        result = localize_v1(ref, search, nominal_downsample)
        result["method"] = "V1"
    elif method == "v2":
        result = localize_v2(ref, search, nominal_downsample)
        result["method"] = "V2"
        # Primary is already the first candidate
        result = result["candidates"][0] if result["candidates"] else {}
    elif method == "v5_gated":
        final, debug = localize_with_gated_rerank(ref, search, nominal_downsample, style=style, seed=seed, use_native=False)
        result = final
    elif method == "v5b_gated_native":
        if style is None or seed is None:
            raise ValueError("v5b_gated_native requires style and seed")
        final, debug = localize_with_gated_rerank(ref, search, nominal_downsample, style=style, seed=seed, use_native=True)
        result = final
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def run_batch(input_dir, output_dir, method, nominal_downsample=10.0, use_native=False):
    """Run localization on all pairs in a dataset directory."""
    ref_dir = os.path.join(input_dir, "reference")
    search_dir = os.path.join(input_dir, "search")
    gt_path = os.path.join(input_dir, "ground_truth.json")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"ground_truth.json not found in {input_dir}")

    with open(gt_path, "r") as f:
        gt = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    predictions = {}
    timings = []
    errors = []

    for sample_id, meta in gt.items():
        ref_path = os.path.join(ref_dir, f"{sample_id}.png")
        search_path = os.path.join(search_dir, f"{sample_id}.png")

        if not os.path.exists(ref_path) or not os.path.exists(search_path):
            print(f"Warning: Missing images for {sample_id}, skipping")
            continue

        style = meta.get("style")
        seed = meta.get("seed")
        gt_x, gt_y = meta["gt_center_xy"]

        t0 = time.time()
        if method in ("v5_gated", "v5b_gated_native"):
            final, debug = localize_with_gated_rerank(
                load_image(ref_path),
                load_image(search_path),
                nominal_downsample=nominal_downsample,
                style=style,
                seed=seed,
                use_native=(method == "v5b_gated_native"),
            )
            result = final
        elif method == "v1":
            result = localize_v1(load_image(ref_path), load_image(search_path), nominal_downsample)
            result["method"] = "V1"
        elif method == "v2":
            v2_result = localize_v2(load_image(ref_path), load_image(search_path), nominal_downsample)
            result = v2_result["candidates"][0] if v2_result["candidates"] else {}
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - t0

        pred_x = result["x"]
        pred_y = result["y"]
        error = np.hypot(pred_x - gt_x, pred_y - gt_y)

        predictions[sample_id] = {
            "pred_x": round(pred_x, 2),
            "pred_y": round(pred_y, 2),
            "gt_x": round(gt_x, 2),
            "gt_y": round(gt_y, 2),
            "error_px": round(float(error), 2),
            "confidence": result.get("confidence", 0),
            "ambiguity_ratio": result.get("ambiguity_ratio"),
            "scale": result.get("scale", nominal_downsample),
            "rotation_deg": result.get("rotation_deg", 0),
            "low_confidence_flag": result.get("low_confidence_flag", False),
            "method": result.get("method", method),
        }
        timings.append(elapsed)
        errors.append(error)

        print(f"  {sample_id}: pred=({pred_x:.2f}, {pred_y:.2f}) gt=({gt_x:.2f}, {gt_y:.2f}) err={error:.2f}px method={result.get('method', method)}")

    # Summary
    errors_arr = np.array(errors)
    summary = {
        "method": method,
        "n_samples": len(errors),
        "mean_error": float(errors_arr.mean()),
        "median_error": float(np.median(errors_arr)),
        "std_error": float(errors_arr.std()),
        "max_error": float(errors_arr.max()),
        "p90_error": float(np.percentile(errors_arr, 90)),
        "p95_error": float(np.percentile(errors_arr, 95)),
        "within_1px": float((errors_arr <= 1).mean()),
        "within_2px": float((errors_arr <= 2).mean()),
        "within_3px": float((errors_arr <= 3).mean()),
        "within_5px": float((errors_arr <= 5).mean()),
        "within_10px": float((errors_arr <= 10).mean()),
        "within_20px": float((errors_arr <= 20).mean()),
        "mean_time_ms": float(np.mean(timings) * 1000),
        "total_time_s": float(sum(timings)),
    }

    # Save outputs
    save_json(predictions, os.path.join(output_dir, "predictions.json"))
    save_json(summary, os.path.join(output_dir, "summary.json"))

    # Also save results.csv for manifest compatibility
    import csv
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "style", "seed", "gt_x", "gt_y",
            "pred_x", "pred_y", "error_px", "confidence",
            "ambiguity_ratio", "scale", "rotation_deg",
            "low_confidence_flag", "method"
        ])
        for sample_id, meta in gt.items():
            if sample_id in predictions:
                p = predictions[sample_id]
                writer.writerow([
                    sample_id, meta.get("style"), meta.get("seed"),
                    p["gt_x"], p["gt_y"], p["pred_x"], p["pred_y"],
                    p["error_px"], p["confidence"], p["ambiguity_ratio"],
                    p["scale"], p["rotation_deg"],
                    p["low_confidence_flag"], p["method"]
                ])

    print(f"\n=== Summary ({method}) ===")
    print(f"  Samples: {summary['n_samples']}")
    print(f"  Mean error: {summary['mean_error']:.2f}px")
    print(f"  Median error: {summary['median_error']:.2f}px")
    print(f"  Within 5px: {summary['within_5px']*100:.1f}%")
    print(f"  Mean time: {summary['mean_time_ms']:.1f}ms")
    print(f"  Output: {output_dir}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="DriftSense Localization")
    ap.add_argument("--ref", help="Reference image path (single mode)")
    ap.add_argument("--search", help="Search image path (single mode)")
    ap.add_argument("--input", help="Input dataset directory (batch mode)")
    ap.add_argument("--output", help="Output directory (batch mode)")
    ap.add_argument("--method", default="v5_gated",
                    choices=["v1", "v2", "v5_gated", "v5b_gated_native"],
                    help="Localization method (default: v5_gated)")
    ap.add_argument("--downsample", type=float, default=10.0,
                    help="Nominal downsample ratio (default: 10.0)")
    ap.add_argument("--style", help="Style (required for gated native)")
    ap.add_argument("--seed", type=int, help="Seed (required for gated native)")
    ap.add_argument("--use-native", action="store_true", help="Use native verification")

    args = ap.parse_args()

    if args.ref and args.search:
        # Single pair mode
        if not os.path.exists(args.ref) or not os.path.exists(args.search):
            print(f"Error: Image not found")
            return 1

        result = run_single(args.ref, args.search, args.method,
                           args.downsample, args.style, args.seed, args.use_native)
        print(json.dumps(result, indent=2))
    elif args.input and args.output:
        # Batch mode
        run_batch(args.input, args.output, args.method, args.downsample, args.use_native)
    else:
        ap.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())