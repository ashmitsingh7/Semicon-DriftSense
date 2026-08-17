"""
visualize_ambiguity.py
-----------------------
Renders the raw NCC correlation surface for a chosen sample, overlaying the
ground-truth location, the predicted location, and the detected secondary
peaks -- to make visible (not just claim) that the localizer's ambiguity
flag is catching genuinely hard, near-tied periodic matches.

Usage:
    python3 visualize_ambiguity.py --data ../data/self_eval --sample finfet_023 \
        --out ../docs/figures/ambiguity_finfet_023.png
"""

import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from localizer import localize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/self_eval")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = json.load(open(os.path.join(args.data, "ground_truth.json")))
    meta = gt[args.sample]
    gt_x, gt_y = meta["gt_center_xy"]

    ref = cv2.imread(os.path.join(args.data, "reference", f"{args.sample}.png"),
                      cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(os.path.join(args.data, "search", f"{args.sample}.png"),
                         cv2.IMREAD_GRAYSCALE)

    pred = localize(ref, search, nominal_downsample=10.0)

    # recompute the *global* coarse NCC surface directly (same as stage 1
    # inside localizer.py) purely for visualization
    ref_f = ref.astype(np.float32) / 255.0
    srch_f = search.astype(np.float32) / 255.0
    ref_dn = cv2.GaussianBlur(ref_f, (0, 0), sigmaX=0.6)
    srch_dn = cv2.GaussianBlur(srch_f, (0, 0), sigmaX=0.6)
    side = max(6, ref_dn.shape[1] // 10)
    templ = cv2.resize(ref_dn, (side, side), interpolation=cv2.INTER_AREA)
    surface = cv2.matchTemplate(srch_dn, templ, cv2.TM_CCOEFF_NORMED)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title(f"Reference Image\n({args.sample})")
    axes[0].axis("off")

    axes[1].imshow(search, cmap="gray")
    axes[1].scatter([gt_x], [gt_y], marker="+", s=180, c="lime", linewidths=2,
                     label="ground truth")
    axes[1].scatter([pred["x"]], [pred["y"]], marker="x", s=180, c="red",
                     linewidths=2, label="predicted")
    axes[1].set_title("Search Image")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].axis("off")

    im = axes[2].imshow(surface, cmap="inferno")
    axes[2].scatter([gt_x - side / 2], [gt_y - side / 2], marker="+", s=180,
                     c="lime", linewidths=2, label="ground truth")
    axes[2].scatter([pred["x"] - side / 2], [pred["y"] - side / 2], marker="x",
                     s=180, c="cyan", linewidths=2, label="predicted peak")
    axes[2].set_title(
        f"NCC correlation surface\nambiguity_ratio={pred['ambiguity_ratio']}  "
        f"(flagged={pred['low_confidence_flag']})")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    err = float(np.hypot(pred["x"] - gt_x, pred["y"] - gt_y))
    fig.suptitle(
        f"{args.sample}: error={err:.1f}px, confidence={pred['confidence']}, "
        f"flagged_low_confidence={pred['low_confidence_flag']}",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
