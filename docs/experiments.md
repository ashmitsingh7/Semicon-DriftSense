# Experiments — Drift-Sense

This document consolidates all experimental work performed during the Drift-Sense project for the SEMICON India Hackathon 2026 (Applied Materials problem statement).

## Overview

| Version | Description | ≤5px Success | Status |
|---------|-------------|--------------|--------|
| **V1** | Single-argmax NCC baseline (two-stage) | **92.5%** (40-pair) | **Production / Final** |
| **V2** | Top-K candidate generation + NMS + dedup | 92.5% (40-pair) | Production candidate pool |
| **V3** | Native-resolution verification of V2 pool | 67.5% | Rejected |
| **Gradient** | Sobel magnitude NCC (A/B/C) | 37.5% / 70.0% | Rejected |
| **Periodicity** | Orientation/FFT/Gabor signatures | ~0% | Rejected |
| **V4 (Phase)** | Local phase-correlation rerank of V2 pool | 87.5% | Rejected (net regression) |

> **Key principle**: The final submitted algorithm is **V1** (the two-stage single-argmax NCC localizer in `src/localizer.py`). All other versions are documented experiments that failed to improve overall accuracy, even when they helped specific hard cases.

---

## V1: Two-Stage Single-Argmax NCC (BASELINE)

**Location**: `src/localizer.py` → `localize()`

**Architecture**:
```
Reference → resize to nominal Search scale
    → global NCC (cv2.matchTemplate, TM_CCOEFF_NORMED)
    → cv2.minMaxLoc() = ONE coarse hypothesis
    → local scale×rotation refinement (5×7 grid)
    → subpixel parabolic refinement
    → ambiguity ratio / low-confidence flag
    → (x, y)
```

**Results (40-pair: 15 DRAM + 15 FinFET + 10 OOD mixed_logic)**:
- ≤5px: **92.5%**
- ≤4px: 90.0%
- ≤2px: 90.0%
- ≤1px: 90.0%
- Median error: ~0.09 px
- DRAM: 15/15 (100%)
- FinFET: 12/15 (80%)
- OOD: 10/10 (100%)
- Mean runtime: ~132 ms/sample

**Key insight**: The single global argmax (Stage 1) discards the true site if a periodic decoy wins — the ambiguity ratio can only *flag* this, not recover.

**Three canonical failures**:
| Sample | GT | V1 Predict | Error | Root Cause |
|--------|-----|------------|-------|------------|
| finfet_017 | (820.45, 830.88) | (608.95, 862.23) | 213.81 px | Stage 2 refinement converges to wrong periodic occurrence |
| finfet_021 | (407.75, 609.69) | (925.99, 369.98) | 570.99 px | True site discarded by Stage 1 global argmax |
| finfet_023 | (914.49, 538.52) | (52.01, 443.13) | 867.74 px | GT at ~98.5th percentile of NCC surface; ~12,000 pixels score higher |

All three correctly self-flagged as low-confidence (ambiguity_ratio ≈ 1.0).

---

## V2: Top-K Multi-Hypothesis Candidate Architecture

**Location**: `src/localizer.py` → `localize_topk()`

**Architecture**:
```
global NCC
    → top-K spatially-separated coarse peaks (greedy NMS)
    → independent Stage-2 refinement per candidate
    → dedup on REFINED coordinates (key fix: dedup_radius = nominal_side, not window_r)
    → ranked candidate list
```

**Bug discovered & fixed**: Initial implementation used Stage-2 window radius (~260px) as dedup radius. On `finfet_021`, this merged the GT-adjacent candidate (0px error, score 0.8020) into a decoy cluster ~182px away (score 0.8043), silently discarding the correct site. Fixed by tying dedup_radius to `nominal_side` (reference footprint size).

**Results (40-pair, K=40)**:
- ≤5px: **92.5%** (same final accuracy as V1)
- Candidate recall (GT in pool @5px): K=10: 95%, K=20: 95%, K=40: 95%
- Runtime: ~3.37s/sample at K=40 (~24x V1)

**Per-case diagnosis**:
- `finfet_021`: GT **is** in pool (ranked 3rd by V2 score, ~0.02px error). Top-K fixes the "candidate never existed" problem. Ranking remains the issue.
- `finfet_017`: GT is the **single best coarse peak** (rank 0/100, 4.5px error). Stage 2 refinement corrupts it (~214px away). Top-K alone cannot fix — failure is Stage-2-induced.
- `finfet_023`: GT **absent** from top-100 coarse peaks. Not fixable by deeper K under plain NCC.

**Conclusion**: V2 fixes the architectural flaw (single argmax discards truth) but does not improve final ranked accuracy because V2 ranks by low-res NCC score alone, which still favors decoys on hard cases.

---

## V3: Native-Resolution Verification

**Location**: `src/native_verifier.py`

**Hypothesis**: Native-resolution NCC on surviving V2 candidates re-ranks pool so true location wins.

**Architecture**: For each top V2 candidate, regenerate deterministic native canvas from seed → crop native ROI around candidate → native NCC with small scale/rotation grid → re-rank by native score.

**Self-eval caveat**: Regenerating canvas from seed only works for synthetic data. Real deployment needs physical re-capture at candidate coordinates.

**Results (40-pair, top-5 V2 candidates native-verified)**:
- ≤5px: **67.5%** (NET REGRESSION from 92.5%)
- DRAM: 26.7% (4/15) — **severely hurt**
- FinFET: 86.7% (13/15) — slight improvement
- OOD: 100% (10/10)
- Changed selected candidate on 14/40: helped 1, hurt 11, neutral 2

**Per-case**:
| Sample | Pre-native top err | Post-native top err | Verdict |
|--------|-------------------|---------------------|---------|
| finfet_021 | 181.8 px | **0.0 px** | **RESCUED** |
| finfet_017 | GT not in pool | — | Not applicable |
| finfet_023 | GT not in pool | — | Not applicable |

**Follow-up experiment** (not in repo): Native verification on *Stage-1 coarse* candidates (before Stage 2) rescues `finfet_017` (coarse GT at 4.5px scores native NCC 0.6810 vs decoys 0.6704–0.6735). Not implemented as pipeline change — needs native subpixel refinement + full A/B.

**Conclusion**: **Reject V3 as final**. It rescues one case but severely degrades DRAM.

---

## Gradient Experiment (Phase 1/2)

**File**: Was in temporary `layout_aware.py` (not committed)

**Strategies**:
- A: Intensity NCC baseline (V2)
- B: Sobel gradient-magnitude NCC
- C: 0.5 × intensity NCC + 0.5 × gradient NCC

**Results (40-pair)**:
| Strategy | ≤5px | Median | DRAM | FinFET | OOD | Recall@40 |
|----------|------|--------|------|--------|-----|-----------|
| A | 92.5% | 0.09px | 100% | 80% | 100% | 95% |
| B | 37.5% | 244.51px | 33.3% | 33.3% | 50% | 42.5% |
| C | 70.0% | 0.18px | 80% | 40% | 100% | 72.5% |

**Diagnostic clues**:
- `finfet_017`: Gradient-only put GT in pool at ~1.45px (intensity GT absent)
- `finfet_023`: Gradient improved best-pool error 111.88px → 46.80px (still no GT)

**Conclusion**: **Reject B/C as final**. Structural signal exists but generic Sobel is too periodic/weak.

---

## Periodicity/Orientation Signatures (Phase 3)

**Files**: Were in temporary `layout_signature.py`, `bench_layout_ablation.py`, `bench_phase3_ablation.py` (not committed)

**Methods**:
- D1: Structure-tensor orientation/coherence
- D2: Windowed FFT frequencies + Gabor-like periodicity energy
- D3: Combined orientation × periodicity

**Results (40-pair)**:
- D1: 0% ≤5px, median ~658px
- D2: 0%, median ~522px
- D3: 0%, median ~599px
- Recall: essentially zero
- Runtime: ~741ms/sample

**Theoretical reason**: FFT/Gabor magnitude and orientation coherence are approximately translation-invariant within periodic lattices. They say "this is the right type of structure" but not "this is the correct occurrence." Removing phase destroys positional information.

**Conclusion**: **Reject**. Do not resurrect without new evidence.

---

## V4: Phase-Correlation Reranking of V2 Pool

**Location**: `experiments/phase_correlation/phase_reranker.py`, `bench_phase_reranker.py`

**Hypothesis**: Local windowed `cv2.phaseCorrelate` (Hann-windowed) on V2 candidates separates true site from decoys where intensity NCC fails.

**Algorithm**: For each V2 candidate → extract nominal-size window at candidate → Hann-window template & window → `cv2.phaseCorrelate` → re-sort by phase response.

**Results (40-pair)**:
| | ≤5px | ≤4px | ≤2px | ≤1px | Mean err | Median err |
|---|---:|---:|---:|---:|---:|---:|
| **V2 (baseline)** | 92.5% | 90.0% | 90.0% | 90.0% | 33.11px | 0.09px |
| **V4 (phase reranked)** | 87.5% | 85.0% | 85.0% | 85.0% | 50.78px | 0.085px |

**By style**:
| Style (n) | V2 @5px | V4 @5px | V2 mean | V4 mean |
|---|---:|---:|---:|---:|
| DRAM (15) | 100.0% | **80.0%** | 0.17px | 52.38px |
| FinFET (15) | 80.0% | **86.7%** | 88.10px | 82.99px |
| OOD (10) | 100.0% | 100.0% | 0.06px | 0.06px |

**Net**: **-5.0 percentage points overall**. FinFET +6.7%, DRAM -20%.

**Per-case behavior**:
| Sample | V2 top-1 err | V4 top-1 err | Changed? | Notes |
|--------|-------------|-------------|----------|-------|
| finfet_021 | 181.8 px | **0.0 px** | Yes | Phase 0.8236 vs next 0.3250 — decisive rescue |
| finfet_017 | 266.8 px | 896.2 px | Yes | Worse — closest pool member (213.8px) phase=0.2142 lost to decoy phase=0.2548 |
| finfet_023 | 867.7 px | 343.4 px | Yes | Closer but still miss — GT absent from pool |
| dram_008 | 0.12 px | 474.9 px | Yes | **Broken** — decoy phase=0.268 > GT phase=0.262 at 30px window |
| dram_012 | 0.10 px | 198.1 px | Yes | **Broken** |
| dram_026 | 0.09 px | 110.5 px | Yes | **Broken** |

**Validation checks**:
- Hann window off fixes `dram_008`/`dram_026` but not `dram_012` — template size is root cause.
- DRAM `nominal_side` = 30px vs FinFET 100px. Phase correlation is noisy at 30px.
- Pearson r between phase response and V2 NCC score: 0.141 (independent signal, but noisy).
- Prior isolated audit (top-20 of 60 raw NMS peaks, pre-dedup) showed cleaner separation on `finfet_023` (~0.83 vs 0.03–0.31). V2's actual deduped pool (~9–12 candidates) is a harder test.

**Runtime**: Phase reranking adds ~0.014s/pair (0.47% of V2).

**Conclusion**: **Reject V4 as default reranker** — net regression, concentrated in DRAM where window is too small. The FinFET-only signal is mildly positive but ungated one-size-fits-all approach is not viable. A gated version (rerank only when `nominal_side` large enough AND V2 top-2 scores are near-tied) is an untested follow-up idea.

---

## Summary Decision Matrix

| Approach | Helps finfet_021? | Helps finfet_017? | Helps finfet_023? | Overall Δ | Verdict |
|----------|------------------|------------------|------------------|-----------|---------|
| V1 (baseline) | No | No | No | 0% | **KEEP** |
| V2 (top-K pool) | Pool yes, rank no | No | No | 0% | Keep as candidate generator |
| V3 (native verify) | **Yes** | No (pool) | No (pool) | -25% | Reject |
| Gradient (B/C) | No | Pool yes (~1.45px) | Pool better | -22.5% | Reject |
| Periodicity | No | No | No | -92.5% | Reject |
| V4 (phase rerank) | **Yes** | Worse | Closer but no | -5% | Reject |

**Final algorithm**: **V1** (`localizer.localize()`) — the robust, fast, self-flagging baseline.

---

## Reproducibility

All experiments are reproducible from the committed source:

```bash
# V1 baseline (production)
python3 src/evaluate.py --data data/self_eval
python3 src/evaluate.py --data data/ood_holdout

# V2 candidate pool analysis
# (see experiments/v2/bench_v2.py if created, or inline test script)

# V3 native verification
python3 src/native_verifier.py data/self_eval finfet_021

# V4 phase reranker
python3 experiments/phase_correlation/bench_phase_reranker.py

# Ablation plots
python3 src/ablation_study.py --out docs/figures/ablation.png

# Ambiguity visualization
python3 src/visualize_ambiguity.py --data data/self_eval --sample finfet_023 --out docs/figures/ambiguity_finfet_023.png
```