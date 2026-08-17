"""
ablation_study.py
------------------
Generates small batches of pairs at increasing noise / rotation severity
(beyond the defaults used for the main self-eval dataset) and reports how
localizer success rate degrades. This is meant to show the operating
envelope of the approach, not just a single pass/fail number on one fixed
dataset.

Usage:
    python3 ablation_study.py --out ../docs/figures/ablation.png
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pattern_synth import (
    synth_canvas, apply_edge_brightening, apply_sensor_noise,
    apply_geometric_degradation, transform_point,
)
from build_dataset import _stamp_landmarks
from localizer import localize

# A smaller canvas than the main dataset's 10,000x10,000 -- same relative
# physics (pitch/downsample/noise ratios unchanged), just cheaper to
# generate so a multi-point severity sweep finishes in reasonable time.
ABLATION_CANVAS_SIZE = (3000, 3000)
ABLATION_REF_SIZE = {"dram": 90, "finfet": 300}
DOWNSAMPLE = 10


def make_pair_at_severity(style, seed, rng_master, dose_scale, max_rot_deg):
    canvas = synth_canvas(style, ABLATION_CANVAS_SIZE, seed=seed)
    ref_size = ABLATION_REF_SIZE[style]
    h, w = ABLATION_CANVAS_SIZE
    margin = 50
    cx = int(rng_master.integers(margin, w - ref_size - margin))
    cy = int(rng_master.integers(margin, h - ref_size - margin))
    canvas = _stamp_landmarks(canvas, cx, cy, ref_size, rng_master)
    ref_crop = canvas[cy:cy + ref_size, cx:cx + ref_size].copy()

    ref_rng = np.random.default_rng(seed * 7919 + 1)
    search_rng = np.random.default_rng(seed * 7919 + 2)

    ref_img = apply_geometric_degradation(ref_crop, ref_rng, max_blur_sigma=1.0,
                                           max_rot_deg=max_rot_deg, scale_jitter=0.03)
    ref_img = apply_edge_brightening(ref_img)
    ref_img = apply_sensor_noise(ref_img, ref_rng, dose_scale=dose_scale * 2.3)

    search_full = cv2.resize(canvas, (w // DOWNSAMPLE, h // DOWNSAMPLE),
                              interpolation=cv2.INTER_AREA)
    search_img, M = apply_geometric_degradation(
        search_full, search_rng, max_blur_sigma=1.4,
        max_rot_deg=max_rot_deg * 0.75, scale_jitter=0.02, return_matrix=True)
    search_img = apply_edge_brightening(search_img, gain=0.5)
    search_img = apply_sensor_noise(search_img, search_rng, dose_scale=dose_scale)

    pre_x = (cx + ref_size / 2.0) / DOWNSAMPLE
    pre_y = (cy + ref_size / 2.0) / DOWNSAMPLE
    gt_x, gt_y = transform_point(pre_x, pre_y, M)

    def to_u8(img):
        return np.clip(img, 0, 1) * 255.0

    return to_u8(ref_img).astype(np.uint8), to_u8(search_img).astype(np.uint8), (gt_x, gt_y)


def run_condition(param_name, values, n_per_value, base_dose, base_rot, success_px=5.0):
    results = []
    for v in values:
        dose = v if param_name == "dose_scale" else base_dose
        rot = v if param_name == "max_rot_deg" else base_rot
        rng_master = np.random.default_rng(int(2000 + v * 1000))
        n_success = 0
        for i in range(n_per_value):
            style = "dram" if i % 2 == 0 else "finfet"
            seed = int(3000 + v * 1000 + i * 37)
            ref_img, search_img, (gt_x, gt_y) = make_pair_at_severity(
                style, seed, rng_master, dose_scale=dose, max_rot_deg=rot)
            pred = localize(ref_img, search_img, nominal_downsample=10.0)
            err = float(np.hypot(pred["x"] - gt_x, pred["y"] - gt_y))
            if err <= success_px:
                n_success += 1
        results.append(n_success / n_per_value)
        print(f"  {param_name}={v}: success_rate={results[-1]*100:.0f}%")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../docs/figures/ablation.png")
    ap.add_argument("--n_per_value", type=int, default=8)
    args = ap.parse_args()

    base_dose, base_rot = 0.6, 1.5  # matches the main self-eval dataset

    print("Sweep 1/2: dose_scale (lower = noisier)")
    dose_values = [0.6, 0.15, 0.04, 0.015, 0.006]
    dose_results = run_condition("dose_scale", dose_values, args.n_per_value,
                                  base_dose, base_rot)

    print("Sweep 2/2: max_rot_deg (search-image rotation)")
    rot_values = [1.5, 6, 12, 18, 25]
    rot_results = run_condition("max_rot_deg", rot_values, args.n_per_value,
                                 base_dose, base_rot)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(dose_values, [r * 100 for r in dose_results], "o-", color="tab:blue")
    axes[0].axvline(base_dose, color="gray", linestyle="--", linewidth=1,
                     label="main dataset setting")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("search-image dose_scale (lower = noisier)")
    axes[0].set_ylabel("success rate (%)  [error \u2264 5px]")
    axes[0].set_title("Robustness to sensor noise")
    axes[0].set_ylim(-5, 105)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(rot_values, [r * 100 for r in rot_results], "o-", color="tab:red")
    axes[1].axvline(base_rot, color="gray", linestyle="--", linewidth=1,
                     label="main dataset setting")
    axes[1].set_xlabel("search-image max rotation (deg)")
    axes[1].set_ylabel("success rate (%)  [error \u2264 5px]")
    axes[1].set_title("Robustness to rotation drift")
    axes[1].set_ylim(-5, 105)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
