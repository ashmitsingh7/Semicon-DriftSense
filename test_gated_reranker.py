#!/usr/bin/env python3
"""
Test the gated reranker on the three FinFET failure cases.
Validates: finfet_021 rescued, finfet_017/023 flagged appropriately, no DRAM regression.
"""

import cv2
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.localizer import localize, localize_topk
from src.localization.rerankers.gated_reranker import gated_rerank, localize_with_gated_rerank


def test_sample(sample_id, data_dir, use_native=False):
    """Test gated reranker on one sample."""
    gt_path = os.path.join(data_dir, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)

    meta = gt[sample_id]
    ref_path = os.path.join(data_dir, "reference", f"{sample_id}.png")
    search_path = os.path.join(data_dir, "search", f"{sample_id}.png")

    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    downsample = meta["ref_native_size"] / meta["gt_inset_size_px"]
    gt_xy = meta["gt_center_xy"]

    # V1 baseline
    v1_pred = localize(ref, srch, downsample)
    v1_err = np.hypot(v1_pred["x"] - gt_xy[0], v1_pred["y"] - gt_xy[1])

    # V2 pool
    v2_cands, _ = localize_topk(ref, srch, downsample, K=40, return_debug=True)
    v2_top1 = v2_cands[0]
    v2_err = np.hypot(v2_top1["x"] - gt_xy[0], v2_top1["y"] - gt_xy[1])

    best_pool = min(np.hypot(c["x"] - gt_xy[0], c["y"] - gt_xy[1]) for c in v2_cands)

    # Gated rerank
    style = meta["style"] if use_native else None
    seed = meta["seed"] if use_native else None

    reranked, debug = gated_rerank(ref, srch, v2_cands, downsample,
                                    style=style, seed=seed, use_native=use_native)

    if reranked:
        final = reranked[0]
        final_err = np.hypot(final["x"] - gt_xy[0], final["y"] - gt_xy[1])
        method = f"V2+{debug['method']}"
    else:
        final = v2_top1
        final_err = v2_err
        method = "V2 (no rerank)"

    return {
        "sample": sample_id,
        "style": meta["style"],
        "gt": gt_xy,
        "v1": {"x": v1_pred["x"], "y": v1_pred["y"], "err": v1_err, "low_conf": v1_pred["low_confidence_flag"]},
        "v2_top1": {"x": v2_top1["x"], "y": v2_top1["y"], "err": v2_err},
        "v2_pool_best_err": best_pool,
        "gt_in_pool": best_pool <= 5.0,
        "method": method,
        "final": {"x": final["x"], "y": final["y"], "err": final_err},
        "debug": debug
    }


def main():
    # Test on self-eval
    data_dir = os.path.join(os.path.dirname(__file__), "data/self_eval")

    # The three failure cases + a few successes for sanity
    test_samples = [
        "finfet_017",  # Stage-2 refinement corruption
        "finfet_021",  # Ranking failure (should be rescued by phase)
        "finfet_023",  # Candidate generation failure
        "finfet_001",  # Typical FinFET success
        "finfet_009",  # Typical FinFET success
        "dram_000",    # DRAM success (should NOT rerank)
        "dram_010",    # DRAM success
    ]

    print("=" * 80)
    print("GATED RERANKER TEST - Phase Correlation (production mode)")
    print("=" * 80)

    results = []
    for sid in test_samples:
        r = test_sample(sid, data_dir, use_native=False)
        results.append(r)

        status = "✅ RESCUED" if r["v1"]["err"] > 5 and r["final"]["err"] <= 5 else \
                 "✅ OK" if r["final"]["err"] <= 5 else \
                 "❌ FAIL"

        print(f"\n{sid} ({r['style']})")
        print(f"  GT: ({r['gt'][0]:.1f}, {r['gt'][1]:.1f})")
        print(f"  V1: err={r['v1']['err']:.1f}px {'(low_conf)' if r['v1']['low_conf'] else ''}")
        print(f"  V2: top1_err={r['v2_top1']['err']:.1f}px, pool_best={r['v2_pool_best_err']:.1f}px, gt_in_pool={r['gt_in_pool']}")
        print(f"  Gate: {'triggered' if r['debug']['reranked'] else 'NOT triggered'} ({r['debug'].get('reason', 'N/A')})")
        if r['debug']['reranked']:
            print(f"  Rerank: {r['debug']['method']}, pre_top1=({r['debug']['pre_top1']['x']:.1f},{r['debug']['pre_top1']['y']:.1f}) -> post_top1=({r['debug']['post_top1']['x']:.1f},{r['debug']['post_top1']['y']:.1f})")
        print(f"  FINAL: err={r['final']['err']:.1f}px {status}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    finfets = [r for r in results if r['style'] == 'finfet']
    drams = [r for r in results if r['style'] == 'dram']

    print(f"\nFinFET (n={len(finfets)}):")
    for r in finfets:
        print(f"  {r['sample']}: V1={r['v1']['err']:.1f} -> Final={r['final']['err']:.1f} {'✅' if r['final']['err'] <= 5 else '❌'}")

    print(f"\nDRAM (n={len(drams)}):")
    for r in drams:
        print(f"  {r['sample']}: V1={r['v1']['err']:.1f} -> Final={r['final']['err']:.1f} {'✅' if r['final']['err'] <= 5 else '❌'} (gate: {'triggered' if r['debug']['reranked'] else 'off'})")

    # Overall accuracy
    success = sum(1 for r in results if r['final']['err'] <= 5)
    print(f"\nOverall: {success}/{len(results)} @5px")


if __name__ == "__main__":
    main()