"""
run_inference.py
-----------------
The submission's required inference entry point: takes a directory of
paired reference/search images, runs the localizer, and writes results to
disk. Also reports end-to-end throughput (disk read -> compute -> disk
write), which is scored separately from accuracy.

Expected input layout (same as data/self_eval):
    <input_dir>/reference/<sample_id>.png
    <input_dir>/search/<sample_id>.png

Usage (batch mode):
    python3 run_inference.py --input ../data/self_eval --output ../data/predictions

Usage (single-pair evaluator mode):
    python3 run_inference.py --ref <ref.png> --search <search.png> --nominal_downsample 10.0
    Outputs JSON to stdout: {"x": ..., "y": ..., "confidence": ..., "ambiguity_ratio": ..., ...}
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2

from localizer import localize, localize_topk, localize_v5_phase_gated, localize_v5_native_gated


def run_batch(args):
    """Batch mode: directory input -> predictions.json + timing_report.json"""
    os.makedirs(args.output, exist_ok=True)

    search_paths = sorted(glob.glob(os.path.join(args.input, "search", "*.png")))
    sample_ids = [os.path.splitext(os.path.basename(p))[0] for p in search_paths]

    # Load ground truth once for native method (needs style/seed)
    gt = None
    if args.method == "v5_native":
        gt_path = os.path.join(args.input, "ground_truth.json")
        if os.path.exists(gt_path):
            with open(gt_path) as f:
                gt = json.load(f)

    predictions = {}
    per_sample_ms = []

    t_start = time.time()
    for sample_id in sample_ids:
        t0 = time.time()

        ref_path = os.path.join(args.input, "reference", f"{sample_id}.png")
        search_path = os.path.join(args.input, "search", f"{sample_id}.png")
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)      # disk read
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)  # disk read

        if args.method == "v1":
            pred = localize(ref_img, search_img,
                             nominal_downsample=args.nominal_downsample)
        elif args.method == "v2":
            candidates, _ = localize_topk(ref_img, search_img,
                                           nominal_downsample=args.nominal_downsample, K=40)
            pred = candidates[0] if candidates else {"x": 0, "y": 0, "confidence": 0}
        elif args.method == "v5_phase":
            result = localize_v5_phase_gated(ref_img, search_img,
                                               nominal_downsample=args.nominal_downsample, return_debug=True)
            pred = result[0] if isinstance(result, tuple) else result
        elif args.method == "v5_native":
            if gt is None:
                raise ValueError("v5_native requires ground_truth.json for style/seed")
            meta = gt[sample_id]
            result = localize_v5_native_gated(
                ref_img, search_img, nominal_downsample=args.nominal_downsample,
                style=meta["style"], seed=meta["seed"], return_debug=True)
            pred = result[0] if isinstance(result, tuple) else result
        else:
            raise ValueError(f"Unknown method: {args.method}")

        predictions[sample_id] = pred
        per_sample_ms.append((time.time() - t0) * 1000.0)

    total_s = time.time() - t_start

    pred_path = os.path.join(args.output, "predictions.json")
    with open(pred_path, "w") as f:   # disk write
        json.dump(predictions, f, indent=2)

    n = len(sample_ids)
    timing = {
        "n_samples": n,
        "total_wall_time_s": round(total_s, 3),
        "mean_ms_per_sample": round(sum(per_sample_ms) / max(n, 1), 2),
        "p50_ms_per_sample": round(sorted(per_sample_ms)[n // 2], 2) if n else None,
        "p90_ms_per_sample": round(sorted(per_sample_ms)[int(n * 0.9)], 2) if n else None,
        "throughput_samples_per_s": round(n / total_s, 2) if total_s > 0 else None,
        "method": args.method,
        "note": "CPU timing (this sandbox has no GPU). matchTemplate/warpAffine "
                "calls are drop-in cv2.cuda-accelerated on an actual H100 node; "
                "expected additional speedup is roughly an order of magnitude "
                "for this workload based on typical cv2.cuda template-matching "
                "benchmarks, not independently verified here.",
    }
    timing_path = os.path.join(args.output, "timing_report.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"Wrote {n} predictions to {pred_path}")
    print(f"Timing report: {timing_path}")
    print(json.dumps(timing, indent=2))


def run_single_pair(args):
    """Single-pair evaluator mode: stdin/stdout or file args -> JSON to stdout"""
    ref_img = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)

    if ref_img is None:
        sys.stderr.write(f"Error: could not read reference image: {args.ref}\n")
        sys.exit(1)
    if search_img is None:
        sys.stderr.write(f"Error: could not read search image: {args.search}\n")
        sys.exit(1)

    pred = localize(ref_img, search_img, nominal_downsample=args.nominal_downsample)
    # Output only the prediction dict as JSON to stdout for easy parsing
    json.dump(pred, sys.stdout)
    sys.stdout.write("\n")


def main():
    ap = argparse.ArgumentParser()
    # Batch mode args
    ap.add_argument("--input",
                     help="directory containing reference/ and search/ subfolders (batch mode)")
    ap.add_argument("--output",
                     help="directory to write predictions.json + timing_report.json (batch mode)")
    # Single-pair mode args
    ap.add_argument("--ref",
                     help="path to reference image (single-pair mode)")
    ap.add_argument("--search",
                     help="path to search image (single-pair mode)")
    # Common
    ap.add_argument("--nominal_downsample", type=float, default=10.0)
    ap.add_argument("--method", choices=["v1", "v2", "v5_phase", "v5_native"],
                    default="v1", help="Localization method to use")
    args = ap.parse_args()

    # Determine mode
    single_pair_mode = (args.ref is not None and args.search is not None)
    batch_mode = (args.input is not None and args.output is not None)

    if single_pair_mode and batch_mode:
        sys.stderr.write("Error: specify either (--ref + --search) for single-pair mode OR (--input + --output) for batch mode, not both.\n")
        sys.exit(1)
    elif single_pair_mode:
        if not args.ref or not args.search:
            sys.stderr.write("Error: single-pair mode requires both --ref and --search.\n")
            sys.exit(1)
        run_single_pair(args)
    elif batch_mode:
        if not args.input or not args.output:
            sys.stderr.write("Error: batch mode requires both --input and --output.\n")
            sys.exit(1)
        run_batch(args)
    else:
        # Default to batch mode if neither specified but input/output given (backward compat)
        # Otherwise show help
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
