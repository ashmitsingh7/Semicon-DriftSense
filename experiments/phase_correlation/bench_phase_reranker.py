"""
bench_phase_reranker.py
------------------------
Full 40-pair (30 self_eval + 10 ood_holdout) V2-vs-V4 benchmark.
Independent script, does not modify localizer.py / native_verifier.py.

Run:
    python3 bench_phase_reranker.py
"""
import json, os, time
import numpy as np
import cv2

from localizer import localize_topk
from phase_reranker import rerank

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = [
    (os.path.join(REPO, "data", "self_eval"), "self_eval"),
    (os.path.join(REPO, "data", "ood_holdout"), "ood_holdout"),
]
SUCCESS_PX = 5.0
POOL_RECALL_PX = 5.0
K = 40


def load_sample(data_dir, sample_id, meta):
    ref = cv2.imread(os.path.join(data_dir, "reference", f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(os.path.join(data_dir, "search", f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
    nominal_downsample = meta["ref_native_size"] / meta["gt_inset_size_px"]
    return ref, srch, nominal_downsample


def run_one(data_dir, sample_id, meta):
    ref, srch, nd = load_sample(data_dir, sample_id, meta)
    gt_x, gt_y = meta["gt_center_xy"]

    t0 = time.time()
    v2_cands, v2_debug = localize_topk(ref, srch, nominal_downsample=nd, K=K, return_debug=True)
    v2_time = time.time() - t0

    if not v2_cands:
        return None  # degenerate, shouldn't happen on this dataset

    v2_pred = v2_cands[0]
    v2_err = float(np.hypot(v2_pred["x"] - gt_x, v2_pred["y"] - gt_y))

    t1 = time.time()
    v4_cands, phase_time = rerank(ref, srch, v2_cands, nd, use_hann=True)
    v4_pred = v4_cands[0]
    v4_err = float(np.hypot(v4_pred["x"] - gt_x, v4_pred["y"] - gt_y))

    # pool recall: best possible error achievable from this candidate pool
    pool_errs = [float(np.hypot(c["x"] - gt_x, c["y"] - gt_y)) for c in v2_cands]
    best_pool_err = min(pool_errs)
    gt_in_pool = best_pool_err <= POOL_RECALL_PX

    # did the top-1 pick change between V2 and V4?
    changed = float(np.hypot(v2_pred["x"] - v4_pred["x"], v2_pred["y"] - v4_pred["y"])) > 2.0

    return {
        "sample_id": sample_id,
        "style": meta["style"],
        "gt_x": gt_x, "gt_y": gt_y,
        "v2_x": v2_pred["x"], "v2_y": v2_pred["y"], "v2_err": round(v2_err, 2),
        "v4_x": v4_pred["x"], "v4_y": v4_pred["y"], "v4_err": round(v4_err, 2),
        "v2_success": v2_err <= SUCCESS_PX,
        "v4_success": v4_err <= SUCCESS_PX,
        "best_pool_err": round(best_pool_err, 2),
        "gt_in_pool_5px": gt_in_pool,
        "n_candidates": len(v2_cands),
        "changed_top1": changed,
        "v2_time_s": round(v2_time, 3),
        "phase_time_s": round(phase_time, 4),
    }


def main():
    rows = []
    for data_dir, label in DATASETS:
        gt_path = os.path.join(data_dir, "ground_truth.json")
        with open(gt_path) as f:
            gt_all = json.load(f)
        for i, (sample_id, meta) in enumerate(sorted(gt_all.items())):
            print(f"[{label}] {i+1}/{len(gt_all)} {sample_id} ...", flush=True)
            r = run_one(data_dir, sample_id, meta)
            if r is None:
                print(f"  SKIPPED (degenerate candidate pool)")
                continue
            r["dataset"] = label
            rows.append(r)
            print(f"  V2 err={r['v2_err']:.2f}px  V4 err={r['v4_err']:.2f}px  "
                  f"changed_top1={r['changed_top1']}  pool_best={r['best_pool_err']:.2f}px  "
                  f"(v2_time={r['v2_time_s']}s phase_time={r['phase_time_s']}s)")

    out_dir = os.path.join(REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "phase_reranker_ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)

    import csv
    with open(os.path.join(out_dir, "phase_reranker_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows to results/phase_reranker_ablation.{{json,csv}}")


if __name__ == "__main__":
    main()
