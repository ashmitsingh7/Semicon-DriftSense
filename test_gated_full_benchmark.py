#!/usr/bin/env python3
"""
Full 40-pair benchmark with gated phase reranker.
Validates: finfet_021 rescued, no DRAM regression, overall >= V1 baseline.
"""

import cv2
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.localizer import localize, localize_topk
from src.localization.rerankers.gated_reranker import gated_rerank


def test_full_benchmark():
    """Run gated reranker on all 40 pairs (30 self_eval + 10 OOD)."""

    # Self-eval
    se_dir = os.path.join(os.path.dirname(__file__), "data/self_eval")
    with open(os.path.join(se_dir, "ground_truth.json")) as f:
        se_gt = json.load(f)

    # OOD holdout
    ood_dir = os.path.join(os.path.dirname(__file__), "data/ood_holdout")
    with open(os.path.join(ood_dir, "ground_truth.json")) as f:
        ood_gt = json.load(f)

    se_results = []
    ood_results = []

    # Self-eval: 15 DRAM + 15 FinFET
    for sid, meta in se_gt.items():
        ref = cv2.imread(os.path.join(se_dir, "reference", f"{sid}.png"), 0)
        srch = cv2.imread(os.path.join(se_dir, "search", f"{sid}.png"), 0)
        downsample = meta["ref_native_size"] / meta["gt_inset_size_px"]
        gt_xy = meta["gt_center_xy"]

        # V1 baseline
        v1_pred = localize(ref, srch, downsample)
        v1_err = np.hypot(v1_pred["x"] - gt_xy[0], v1_pred["y"] - gt_xy[1])

        # V2 pool + gated phase rerank
        v2_cands, _ = localize_topk(ref, srch, downsample, K=40, return_debug=True)
        reranked, debug = gated_rerank(ref, srch, v2_cands, downsample,
                                        style=meta["style"], seed=meta["seed"],
                                        use_native=False, score_margin_threshold=0.01)

        if reranked:
            final = reranked[0]
            final_err = np.hypot(final["x"] - gt_xy[0], final["y"] - gt_xy[1])
            method = f"V2+{debug['method']}"
        else:
            final = v2_cands[0]
            final_err = np.hypot(final["x"] - gt_xy[0], final["y"] - gt_xy[1])
            method = "V2"

        se_results.append({
            "id": sid, "style": meta["style"],
            "v1_err": v1_err, "final_err": final_err,
            "gate_triggered": debug["reranked"],
            "method": method
        })

    # OOD: 10 mixed_logic
    for sid, meta in ood_gt.items():
        ref = cv2.imread(os.path.join(ood_dir, "reference", f"{sid}.png"), 0)
        srch = cv2.imread(os.path.join(ood_dir, "search", f"{sid}.png"), 0)
        downsample = meta["ref_native_size"] / meta["gt_inset_size_px"]
        gt_xy = meta["gt_center_xy"]

        v1_pred = localize(ref, srch, downsample)
        v1_err = np.hypot(v1_pred["x"] - gt_xy[0], v1_pred["y"] - gt_xy[1])

        v2_cands, _ = localize_topk(ref, srch, downsample, K=40, return_debug=True)
        reranked, debug = gated_rerank(ref, srch, v2_cands, downsample,
                                        style=meta["style"], seed=meta["seed"],
                                        use_native=False, score_margin_threshold=0.01)

        if reranked:
            final = reranked[0]
            final_err = np.hypot(final["x"] - gt_xy[0], final["y"] - gt_xy[1])
        else:
            final = v2_cands[0]
            final_err = np.hypot(final["x"] - gt_xy[0], final["y"] - gt_xy[1])

        ood_results.append({
            "id": sid, "style": meta["style"],
            "v1_err": v1_err, "final_err": final_err,
            "gate_triggered": debug["reranked"],
            "method": "V2+phase" if debug["reranked"] else "V2"
        })

    # Print results
    print("=" * 80)
    print("FULL 40-PAIR BENCHMARK: GATED PHASE RERANKER")
    print("=" * 80)

    for r in se_results:
        status = "✅" if r["final_err"] <= 5 else "❌"
        gate_str = f"[GATED:{r['method']}]" if r["gate_triggered"] else "[V2]"
        print(f"{r['id']:12s} | {r['style']:10s} | V1={r['v1_err']:6.1f} -> Final={r['final_err']:6.1f} {status} {gate_str}")

    print("-" * 80)
    for r in ood_results:
        status = "✅" if r["final_err"] <= 5 else "❌"
        gate_str = f"[GATED:{r['method']}]" if r["gate_triggered"] else "[V2]"
        print(f"{r['id']:12s} | {r['style']:10s} | V1={r['v1_err']:6.1f} -> Final={r['final_err']:6.1f} {status} {gate_str}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # V1 baseline
    v1_se = sum(1 for r in se_results if r["v1_err"] <= 5)
    v1_ood = sum(1 for r in ood_results if r["v1_err"] <= 5)

    # Gated final
    final_se = sum(1 for r in se_results if r["final_err"] <= 5)
    final_ood = sum(1 for r in ood_results if r["final_err"] <= 5)

    # By style
    for style in ["dram", "finfet", "mixed_logic"]:
        if style == "mixed_logic":
            subset = ood_results
        else:
            subset = [r for r in se_results if r["style"] == style]

        v1_s = sum(1 for r in subset if r["v1_err"] <= 5)
        fs = sum(1 for r in subset if r["final_err"] <= 5)
        print(f"\n{style.upper()}: V1 {v1_s}/{len(subset)} @5px -> Gated {fs}/{len(subset)} @5px  Δ={fs-v1_s:+d}")

        # Show gated cases
        gated = [r for r in subset if r["gate_triggered"]]
        if gated:
            for r in gated:
                print(f"  {r['id']}: V1={r['v1_err']:.1f} -> {r['final_err']:.1f} ({r['method']})")

    total_v1 = v1_se + v1_ood
    total_final = final_se + final_ood
    print(f"\nOVERALL: V1 {total_v1}/40 @5px -> Gated {total_final}/40 @5px  Δ={total_final-total_v1:+d}")

    # Verify no DRAM regression
    dram_files = [r for r in se_results if r["style"] == "dram"]
    dram_regressed = [r for r in dram_files if r["v1_err"] <= 5 and r["final_err"] > 5]
    if dram_regressed:
        print(f"\n⚠️  DRAM REGRESSION: {len(dram_regressed)} cases!")
        for r in dram_regressed:
            print(f"  {r['id']}: V1={r['v1_err']:.1f} -> Final={r['final_err']:.1f}")
    else:
        print(f"\n✅ NO DRAM REGRESSION (all {len(dram_files)} DRAM cases preserved)")

    # Verify finfet_021 rescued
    f021 = [r for r in se_results if r["id"] == "finfet_021"][0]
    if f021["v1_err"] > 5 and f021["final_err"] <= 5:
        print(f"✅ finfet_021 RESCUED: {f021['v1_err']:.1f} -> {f021['final_err']:.1f}")

    return {
        "v1_acc": total_v1,
        "gated_acc": total_final,
        "dram_regression": len(dram_regressed),
        "finfet_021_rescued": f021["v1_err"] > 5 and f021["final_err"] <= 5
    }


if __name__ == "__main__":
    test_full_benchmark()