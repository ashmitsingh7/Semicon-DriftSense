"""
build_dataset.py
-----------------
Generates the Drift-Sense synthetic dataset: paired (Reference Image,
Search Image) samples for DRAM-style and FinFET-style die architectures,
with recorded ground-truth match locations, for self-evaluation.

Run:
    python3 build_dataset.py --out ../data/self_eval --n 30
"""

import argparse
import json
import os

import cv2
import numpy as np

from pattern_synth import (
    synth_canvas,
    apply_edge_brightening,
    apply_sensor_noise,
    apply_geometric_degradation,
    transform_point,
)

CANVAS_SIZE = (10000, 10000)   # native-resolution capture region
SEARCH_SIZE = (1000, 1000)     # downsampled search image (10x lower mag)
DOWNSAMPLE = 10

REF_SIZE = {
    "dram": 300,     # -> ~30x30 px inset in the search image ("shrunk ~10x")
    "finfet": 1000,  # -> ~100x100 px inset in the search image
    "mixed_logic": 500,  # held-out generalization style, not used for tuning
}

# search-side noise is intentionally higher dose reduction (lower dose_scale
# -> noisier) than reference-side, per mandatory requirement "search image
# will have more noise than the reference image on actual test data"
REF_DOSE_SCALE = 1.4
SEARCH_DOSE_SCALE = 0.6


def to_uint8(img):
    return np.clip(img, 0, 1) * 255.0


def _stamp_landmarks(canvas, cx, cy, ref_size, rng):
    """Guarantee the reference footprint contains a handful of strong,
    high-contrast, non-periodic local features (in addition to the sparse
    background defects already present everywhere). Without this, a
    perfectly periodic pattern plus only-sparse random defects can leave
    some crops with no real distinguishing content at all, making the
    localization task accidentally unsolvable rather than "genuinely
    difficult". This models realistic local process variation / one-off
    contamination landmarks used for site recognition on real wafers."""
    # Radius must stay large enough at NATIVE resolution to still be a
    # multi-pixel, matchable blob after the 10x area-average downsample
    # that builds the Search Image (a native radius of ~10px nearly
    # vanishes into a sub-pixel smudge once downsampled).
    n_marks = int(rng.integers(2, 4))
    for _ in range(n_marks):
        mx = cx + int(rng.integers(int(ref_size * 0.15), int(ref_size * 0.85)))
        my = cy + int(rng.integers(int(ref_size * 0.15), int(ref_size * 0.85)))
        r = int(rng.integers(45, 80))
        val = float(rng.choice([0.03, 0.95]))
        cv2.circle(canvas, (mx, my), r, val, thickness=-1)
    return canvas


def make_pair(style, seed, rng_master):
    canvas = synth_canvas(style, CANVAS_SIZE, seed=seed)

    ref_size = REF_SIZE[style]
    h, w = CANVAS_SIZE
    margin = 50  # keep crop away from the canvas border
    cx = int(rng_master.integers(margin, w - ref_size - margin))
    cy = int(rng_master.integers(margin, h - ref_size - margin))

    canvas = _stamp_landmarks(canvas, cx, cy, ref_size, rng_master)
    ref_crop = canvas[cy:cy + ref_size, cx:cx + ref_size].copy()

    # --- independent per-image RNGs: mandatory "do NOT reuse the same
    # noise on both images" ---
    ref_rng = np.random.default_rng(seed * 7919 + 1)
    search_rng = np.random.default_rng(seed * 7919 + 2)

    # Reference Image pipeline (native resolution). Rotation/scale jitter
    # kept modest (a few degrees / few percent) -- this represents the
    # kind of small navigation/stage drift described in the problem
    # statement, not an arbitrary-orientation search.
    ref_img = apply_geometric_degradation(ref_crop, ref_rng,
                                           max_blur_sigma=1.0,
                                           max_rot_deg=2.0,
                                           scale_jitter=0.03)
    ref_img = apply_edge_brightening(ref_img)
    ref_img = apply_sensor_noise(ref_img, ref_rng, dose_scale=REF_DOSE_SCALE)

    # Search Image pipeline: downsample the WHOLE canvas (this is what
    # guarantees the reference pattern is genuinely, verifiably present
    # inside the search image at the recorded location)
    search_full = cv2.resize(canvas, (w // DOWNSAMPLE, h // DOWNSAMPLE),
                              interpolation=cv2.INTER_AREA)
    search_img, M = apply_geometric_degradation(search_full, search_rng,
                                                 max_blur_sigma=1.4,
                                                 max_rot_deg=1.5,
                                                 scale_jitter=0.02,
                                                 return_matrix=True)
    search_img = apply_edge_brightening(search_img, gain=0.5)
    search_img = apply_sensor_noise(search_img, search_rng,
                                     dose_scale=SEARCH_DOSE_SCALE)

    # ground truth must be tracked THROUGH the affine warp applied to the
    # full search image, otherwise the recorded location silently drifts
    # away from where the reference pattern actually ended up
    pre_warp_x = (cx + ref_size / 2.0) / DOWNSAMPLE
    pre_warp_y = (cy + ref_size / 2.0) / DOWNSAMPLE
    gt_x, gt_y = transform_point(pre_warp_x, pre_warp_y, M)
    gt_inset_size = ref_size / DOWNSAMPLE

    meta = {
        "style": style,
        "seed": seed,
        "ref_native_size": ref_size,
        "search_size": SEARCH_SIZE,
        "gt_center_xy": [round(gt_x, 2), round(gt_y, 2)],
        "gt_inset_size_px": round(gt_inset_size, 2),
    }
    return ref_img, search_img, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/self_eval")
    ap.add_argument("--n", type=int, default=30,
                     help="total pairs (split across --styles)")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--styles", nargs="+", default=["dram", "finfet"],
                     help="which style(s) to generate, e.g. --styles mixed_logic")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "reference"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "search"), exist_ok=True)

    rng_master = np.random.default_rng(args.seed0)
    styles = (args.styles * ((args.n // len(args.styles)) + 1))[:args.n]

    ground_truth = {}
    for i, style in enumerate(styles):
        sample_id = f"{style}_{i:03d}"
        seed = args.seed0 + i * 101
        ref_img, search_img, meta = make_pair(style, seed, rng_master)

        ref_path = os.path.join(args.out, "reference", f"{sample_id}.png")
        search_path = os.path.join(args.out, "search", f"{sample_id}.png")
        cv2.imwrite(ref_path, to_uint8(ref_img))
        cv2.imwrite(search_path, to_uint8(search_img))

        ground_truth[sample_id] = meta
        print(f"[{i+1}/{len(styles)}] {sample_id}  gt_center={meta['gt_center_xy']}")

    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nWrote {len(styles)} pairs to {args.out}")
    print(f"Ground truth: {os.path.join(args.out, 'ground_truth.json')}")


if __name__ == "__main__":
    main()
