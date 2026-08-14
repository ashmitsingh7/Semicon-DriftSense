# Failure Analysis — Drift-Sense

This document provides a deep-dive analysis of the three canonical failure cases in the V1/V2 baseline: `finfet_017`, `finfet_021`, and `finfet_023`. These cases represent the core challenge of localizing within a highly periodic semiconductor layout under drift.

---

## The Core Problem

The Drift-Sense task is **not** object detection or template matching in a general scene. It is:

> **Localization of the correct site inside a highly periodic semiconductor layout under drift.**

The reference pattern appears *many* times in the search image (once per die, plus periodic repetitions within a die). All occurrences look nearly identical at the search-image resolution (10x downsampled). The difference between the correct site and a false one is often a few high-contrast "landmark" blobs (simulating process variation / contamination) that survive the 10x downsample — but these are not guaranteed to be distinct at every location.

The V1 two-stage NCC approach:
1. **Stage 1 (coarse, global)**: Single NCC pass at nominal scale/rotation → one global argmax
2. **Stage 2 (fine, local)**: Scale×rotation grid search around the Stage 1 peak

The fundamental flaw: **Stage 1 retains only one hypothesis**. If a periodic decoy scores higher than the true site, the true site is discarded before refinement ever runs.

---

## finfet_017 — Stage-2-Induced Failure

### Ground Truth
- GT center: `(820.45, 830.88)` search-image coordinates
- Style: FinFET (nominal template ~100×100 px in search image)

### V1 Behavior
- **Stage 1 coarse peak**: ~4.5 px from GT (correct neighborhood!)
- **Stage 2 refinement**: Converges to a different periodic occurrence ~214 px away
- **Final error**: **213.81 px**
- Confidence: 0.685, ambiguity_ratio: 1.001 (correctly flagged low-confidence)

### V2 Behavior (K=40)
- GT is the **single BEST coarse peak** (rank 0/100, 4.5 px error)
- Stage 2 refinement on the correct coarse peak *corrupts* it
- Best pool error: **213.81 px** (GT not in pool)
- V2 top-1 error: **266.77 px**

### Gradient Experiment (Sobel magnitude)
- GT **enters candidate pool** at ~1.45 px error
- But gradient-only fails globally (37.5% @5px)

### V4 Phase Correlation
- Makes it **worse**: 266.8 px → 896.2 px
- Closest pool member (213.8 px) phase=0.2142 loses to decoy phase=0.2548

### Root Cause Analysis
This is a **Stage-2 window search failure**, not a candidate generation failure.

The Stage 2 window (~260 px radius for FinFET) is large enough to contain multiple periodic occurrences. The scale×rotation grid search (35 combinations) inside this window finds a *different* high-scoring pose — the correlation surface inside the window has multiple near-tied peaks, and the refinement picks the wrong one.

**Critical insight from design_notes.md §8**: *"Stage 2's local scale/rotation grid search, run on a window centered at that correct coarse peak, converges to a different, higher-scoring pose ~214px away *within that same window*. This is a **Stage-2-induced** failure, not a candidate-generation failure — top-K alone cannot fix it, because Stage 2 corrupts the coordinate before dedup ever sees it."*

### What Could Fix It
1. **Native verification on Stage-1 coarse candidates** (before Stage 2 runs): Follow-up experiment showed the correct coarse candidate (4.5px error) scores native NCC 0.6810 vs decoys 0.6704–0.6735. This would rescue it but needs native subpixel refinement + full A/B.
2. **Constrained Stage 2**: Limit the scale/rotation search to not wander to adjacent periods. The current window is ~2.5× template side.
3. **Multi-hypothesis at Stage 2**: Instead of one refined peak per candidate, keep top-M refined peaks per candidate and dedup globally.

### Verdict for Submission
V1 correctly flags this as low-confidence. No experiment in the repo fixes this without regressing elsewhere. **Document honestly as unresolved.**

---

## finfet_021 — Candidate-Generation Failure (Rescuable)

### Ground Truth
- GT center: `(407.75, 609.69)`
- Style: FinFET

### V1 Behavior
- **Stage 1 coarse**: True site is discarded — a different periodic occurrence wins global argmax
- **Final error**: **570.99 px**
- Confidence: 0.800, ambiguity_ratio: 1.001 (flagged low-confidence)

### V2 Behavior (K=40)
- GT **IS in candidate pool** (ranked 3rd by V2 score)
- Best pool error: **0.02 px** (essentially exact)
- V2 top-1 error: **181.79 px** (decoy scored 0.8043 vs GT 0.8020 — essentially tied)
- Candidate recall: K=10/20/40 all 95% — this case is fixed at pool level

### V3 Native Verification
- **RESCUED**: Post-native top-rank error **0.0 px**
- Native NCC: GT 0.7369 vs decoys 0.7254–0.7257 — clear separation
- This directly validates the native-verification mechanism for cases where GT is in pool but misranked.

### V4 Phase Correlation
- **RESCUED**: V4 top-1 error **0.0 px**
- Phase response: GT 0.8236 vs next-best 0.3250 — wide margin
- Diagnostic figure (`experiments/phase_correlation/v4_diagnostic_cases.png` row 2) shows the true candidate uniquely reproduces the reference's landmark blob.

### Root Cause Analysis
This is a **ranking problem**, not a candidate-generation problem. The true site survives into the V2 pool but is outscored by a decoy under low-res intensity NCC by only 0.0023 (0.8043 vs 0.8020). The decoy and GT are different periodic occurrences that happen to have nearly identical low-res intensity statistics.

The distinguishing feature is a **native-resolution landmark blob** inside the reference footprint that survives downsampling but is not perfectly periodic — it appears differently at each site. Native NCC and phase correlation both latch onto this.

### What Could Fix It (Production-Ready)
1. **Gated native verification**: Only rerank when V2 top-2 scores are within a tight margin (e.g., <0.01). This would catch this case without the DRAM regression of full V3.
2. **Gated phase reranking**: Only rerank when `nominal_side` ≥ 60px (FinFET) AND V2 top-2 margin is tight. This avoids the DRAM (30px) regression seen in V4.
3. **Learned re-ranker**: A small CNN embedding trained on this generator with triplet loss could learn to distinguish periodic occurrences using the landmark context.

### Verdict for Submission
This case *is* rescuable with a gated reranker, but no gated version is validated at scale in the repo. **V1 baseline correctly flags it low-confidence.**

---

## finfet_023 — The Unresolved Hard Case

### Ground Truth
- GT center: `(914.49, 538.52)`
- Style: FinFET

### V1 Behavior
- **Stage 1 coarse**: GT is at ~98.5th percentile of NCC surface
- ~12,000 pixels score higher than GT
- **Final error**: **867.74 px**
- Confidence: 0.734, ambiguity_ratio: 1.002 (flagged low-confidence)

### V2 Behavior (K=40)
- GT **ABSENT** from top-100 coarse peaks (confirmed by direct search)
- Best pool error: **111.88 px** (closest pool member is still 112 px away)
- V2 top-1 error: **867.74 px**

### Gradient Experiment
- GT **still absent** from gradient pool
- Best pool error improved: 111.88 px → 46.80 px
- But still no GT in pool

### V4 Phase Correlation
- GT absent from pool → cannot be rescued by reranking
- V4 top-1: 343.4 px (less wrong, but still a clear miss)
- Prior isolated audit (pre-V2-dedup pool of 60 raw NMS peaks) reported GT phase ~0.83 vs decoys 0.03–0.31. But that pool included many clearly-bad candidates. V2's actual deduped pool (~12 candidates) only has the *best* decoys, and the phase margin narrows/disappears.

### Root Cause Analysis
This is a **candidate-generation failure at the Stage 1 level**. The true site's intensity NCC score is too low to enter even a K=100 pool. It's not a ranking problem (V2/V3/V4) — it's a "GT never generated" problem.

The reference footprint at this location lacks sufficiently distinct landmark content at the search-image resolution to stand out from the periodic background. The low-frequency brightness drift and sparse defects happen to make the GT occurrence look like "just another period cell" at 10x downsample.

**Why increasing K doesn't help**: The coarse NCC surface is essentially translation-invariant across the periodic lattice. The GT occurrence scores in the ~98.5th percentile — meaning tens of thousands of false locations score higher. No practical K captures it.

### What Could Fix It (Research Direction)
1. **Different candidate generation signal**: Not intensity NCC. The Phase 3 periodicity signatures (0%) and gradient (42.5% recall) failed. The phase correlation signal on *coarse* candidates (pre-V2-dedup) showed promise but only in a broader pool.
2. **Semiconductor-specific structure**: The physical layout has two spatial frequencies:
   - Fine: Fin pitch (~36–46 px native → ~4 px search)
   - Coarse: Gate pitch (~260–340 px native → ~26–34 px search)
   
   A descriptor that captures *both* and their relative phase alignment might break the translation invariance that kills plain NCC.
3. **Multiple reference views**: If multiple reference images at different rotations/illumination were available, cross-referencing could disambiguate. Not applicable here.
4. **Stage-coordinate prior**: The problem statement says "choose the one closest to Search-image centre" for ambiguous valid matches. But our GT is *not* center-biased — it's a specific crop origin. Center-priority sweep actually **regressed** FinFET from 12/15 → 5/15 at margin=0.05.

### Verdict for Submission
**This case is not solved by any experiment in the repo.** It is the primary open research problem. V1 correctly flags it as low-confidence. Do not claim it is fixed.

---

## Cross-Case Summary

| Aspect | finfet_017 | finfet_021 | finfet_023 |
|--------|------------|------------|------------|
| **GT in V1 Stage-1 peak?** | ~4.5px (yes, neighborhood) | No (decoy wins) | No (98.5th %ile) |
| **GT in V2 pool (K=40)?** | No (Stage 2 corrupts) | **Yes (0.02px, rank 3)** | No (best 111.88px) |
| **Failure type** | Stage-2 refinement | V2 ranking | Stage-1 candidate generation |
| **Gradient helps?** | Pool yes (~1.45px) | No | Pool better (46.8px) |
| **V3 native verify helps?** | N/A (not in pool) | **Yes** (0px) | N/A (not in pool) |
| **V4 phase rerank helps?** | Worse | **Yes** (0px) | Closer but no |
| **Rescuable by reranking pool?** | No | Yes | No |
| **Requires new candidate gen?** | Maybe (constrained Stage 2) | No | **Yes** |

---

## Implications for the Final Algorithm

1. **V1 is the correct production baseline** — it fails honestly and flags all three as low-confidence.

2. **No single "next step" fixes all three**:
   - `finfet_021` needs a reranker (native or phase) on the V2 pool.
   - `finfet_017` needs Stage-2 constraint or pre-Stage-2 native verify.
   - `finfet_023` needs a fundamentally new candidate generation signal.

3. **Stacking complexity without evidence is counterproductive**: V3, V4, gradient, periodicity all show that "more domain-aware" signals can help one case while hurting others.

4. **The scientific story is**: *"We tested multiple increasingly domain-aware approaches (candidate pools, native verification, structural gradients, periodicity signatures, phase correlation) and retained the robust baseline when they failed to generalize. The three canonical failures map to three distinct root causes in the pipeline, demonstrating that the periodic ambiguity problem requires multiple different solutions, not a single silver bullet."*

---

## Recommended Next Steps (Beyond Submission Scope)

1. **Information audit on finfet_023**: Compare GT crop vs top-20 false candidates across intensity, gradient, Fourier phase, local topology, edge density, residuals. Determine if *any* signal separates them.

2. **Gated reranker for finfet_021/017**: Only invoke native/phase verification when V2 top-2 margin is tight AND template size is sufficient (≥60px). Validate on full 40-pair + OOD.

3. **Semiconductor-specific descriptor**: Combine fin-pitch phase + gate-pitch phase + landmark topology into a joint descriptor. Test as a coarse candidate generator (not just reranker).

4. **Learned embedding**: Train a small CNN on the synthetic generator with contrastive/triplet loss (anchor=reference, positive=GT crop, negative=decoy crops). Compare against NCC on the three hard cases.

---

## Reproducibility

```bash
# Visualize finfet_023 ambiguity surface
python3 src/visualize_ambiguity.py --data data/self_eval --sample finfet_023 --out docs/figures/ambiguity_finfet_023.png

# V2 candidate pool inspection for any sample
python3 -c "
import json, cv2, numpy as np
from localizer import localize_topk
gt = json.load(open('data/self_eval/ground_truth.json'))
for sid in ['finfet_017','finfet_021','finfet_023']:
    m = gt[sid]
    ref = cv2.imread(f'data/self_eval/reference/{sid}.png', 0)
    srch = cv2.imread(f'data/self_eval/search/{sid}.png', 0)
    cands = localize_topk(ref, srch, nominal_downsample=m['ref_native_size']/m['gt_inset_size_px'], K=40)
    gt_x, gt_y = m['gt_center_xy']
    for i, c in enumerate(cands[:5]):
        err = np.hypot(c['x']-gt_x, c['y']-gt_y)
        print(f'{sid} rank {i}: err={err:.1f}px score={c[\"score\"]:.4f}')
    print(f'  Best pool err: {min(np.hypot(c[\"x\"]-gt_x, c[\"y\"]-gt_y) for c in cands):.1f}px')
"

# V3 native verification diagnostic
python3 src/native_verifier.py data/self_eval finfet_021

# V4 phase reranker diagnostic
python3 experiments/phase_correlation/bench_phase_reranker.py
```