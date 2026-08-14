import json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localizer import localize_topk
from phase_reranker import rerank, build_nominal_template, prep_search

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "self_eval")
FIGDIR = os.path.join(REPO, "docs", "figures")
os.makedirs(FIGDIR, exist_ok=True)

gt_all = json.load(open(os.path.join(DATA, "ground_truth.json")))

def extract(cx, cy, size, img):
    x0 = int(np.clip(round(cx - size/2), 0, img.shape[1]-size))
    y0 = int(np.clip(round(cy - size/2), 0, img.shape[0]-size))
    return img[y0:y0+size, x0:x0+size]

cases = ["finfet_017", "finfet_021", "finfet_023"]
fig, axes = plt.subplots(len(cases), 4, figsize=(14, 3.4*len(cases)))

for row, sid in enumerate(cases):
    meta = gt_all[sid]
    gt_x, gt_y = meta["gt_center_xy"]
    ref = cv2.imread(os.path.join(DATA, "reference", f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(os.path.join(DATA, "search", f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    nd = meta["ref_native_size"] / meta["gt_inset_size_px"]

    templ, nominal_side = build_nominal_template(ref, nd)
    srch_dn = prep_search(srch)
    cands, _ = localize_topk(ref, srch, nominal_downsample=nd, K=40, return_debug=True)
    reranked, _ = rerank(ref, srch, cands, nd)

    v2_pred, v4_pred = cands[0], reranked[0]
    v2_err = np.hypot(v2_pred["x"]-gt_x, v2_pred["y"]-gt_y)
    v4_err = np.hypot(v4_pred["x"]-gt_x, v4_pred["y"]-gt_y)
    gt_in_pool = min(np.hypot(c["x"]-gt_x, c["y"]-gt_y) for c in cands) <= 5.0

    panels = [
        (templ, "Reference"),
        (extract(gt_x, gt_y, nominal_side, srch_dn), f"GT crop\n{'(in V2 pool)' if gt_in_pool else '(NOT in V2 pool)'}"),
        (extract(v2_pred["x"], v2_pred["y"], nominal_side, srch_dn),
         f"V2 pick\nerr={v2_err:.1f}px score={v2_pred['score']:.3f}"),
        (extract(v4_pred["x"], v4_pred["y"], nominal_side, srch_dn),
         f"V4 pick\nerr={v4_err:.1f}px phase={v4_pred['phase_response']:.3f}"),
    ]
    for col, (img, title) in enumerate(panels):
        ax = axes[row, col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        color = "green" if "in V2 pool)" in title and "NOT" not in title else (
                "darkred" if "NOT in V2 pool" in title else
                ("green" if ("err=" in title and float(title.split("err=")[1].split("px")[0]) < 5) else "red"))
        ax.set_title(title, fontsize=9, color=color)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(sid, fontsize=11, fontweight="bold")

plt.suptitle("V4 (V2 pool + local phase-correlation reranking): the three canonical hard cases", fontsize=13)
plt.tight_layout()
out = os.path.join(FIGDIR, "v4_diagnostic_cases.png")
plt.savefig(out, dpi=140)
print("wrote", out)
