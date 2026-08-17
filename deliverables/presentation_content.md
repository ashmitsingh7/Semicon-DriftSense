## Slide 1: Title Slide
# Drift-Sense
AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

**Semicon India Hackathon 2026 — Applied Materials Problem Statement**
Team: [TEAM NAME], Vellore Institute of Technology

## Slide 2: The Problem
## Pinpointing the Right Periodic Cell

| Aspect | Detail |
|---|---|
| **Task** | Match a 100× native reference crop to a 10× search image |
| **Challenge** | Periodic layout → thousands of visually identical candidates |
| **Failure Mode** | NCC peak often lands on wrong periodic repeat |
| **Key Insight** | Problem is not *detection* — it is *disambiguation* |

- Reference: 1000×1000 px at 100× magnification
- Search: 1000×1000 px at 10× magnification (~10× downsample)
- Nominal scale ratio: 10:1 (robustness tested 9:1 to 11:1)

## Slide 3: Why NCC Alone Fails
## The Periodic Ambiguity Problem

- NCC computes spatial cross-correlation → finds global maximum
- On periodic structures: correlation surface has **many near-equal peaks** at multiples of the pitch
- Global argmax often picks a decoy cell, not the true site

**Three Canonical Failure Modes (Self-Eval Benchmark):**

| Sample | Error | Root Cause |
|---|---:|---|
| `finfet_017` | 214 px | Stage-2 refinement converges to wrong periodic cell |
| `finfet_021` | 571 px | True site in top-K pool but misranked by NCC score |
| `finfet_023` | 868 px | True site absent from top-100 coarse peaks (98.5th percentile) |

**All three correctly self-flagged as low-confidence** via ambiguity ratio.

## Slide 4: DriftSense V1 — Two-Stage NCC Baseline
## Production Algorithm

**Stage 1: Coarse Global Search**
- Single nominal-scale, zero-rotation NCC over full 1000×1000 search image
- Computes ambiguity ratio (best peak vs. best elsewhere) → confidence flag

**Stage 2: Fine Local Refinement**
- 5 scales × 7 rotations grid search in local window around Stage-1 peak
- Parabolic sub-pixel peak refinement
- **3× speedup** (21.9s → 7.2s) by confining grid to local window

**Results (40 pairs: 15 DRAM + 15 FinFET + 10 OOD mixed_logic):**
- ≤5 px: **92.5%** | Median error: **0.09 px**
- DRAM: 100% (15/15) | FinFET: 80% (12/15) | OOD: 100% (10/10)
- Runtime: ~132 ms/sample (CPU)

## Slide 5: DriftSense V2 — Top-K Candidate Pool
## Enabling Reranking Experiments

Global NCC → top-K NMS peaks → independent Stage-2 refinement → dedup on **refined** coordinates → ranked list

| K | Candidate Recall @5px |
|---|---:|
| 10 | 95% |
| 20 | 95% |
| 40 | 95% |

- **Fixes** `finfet_021`: GT enters pool at 0.02 px error
- **Does not fix** `finfet_017` (Stage-2 corruption) or `finfet_023` (candidate generation failure)
- Final ranked accuracy = V1 (92.5%), but enables reranking experiments
- Runtime: ~3.37s/sample at K=40

## Slide 6: Experimental Journey — What We Tried & Rejected

| Experiment | Hypothesis | Result | Why Rejected |
|---|---|---|---|
| **V3 Native Verification** | Native-res NCC re-ranks V2 pool | 67.5% @5px | Rescued `finfet_021` but **broke 11 DRAM cases** (100%→26.7%) |
| **Gradient (Sobel)** | Structural edges break periodicity | 37.5% / 70.0% | Helped `finfet_017` pool but failed globally |
| **Periodicity Signatures** (FFT/Gabor) | Layout frequency identifies cell | ~0% | Translation-invariant — tells structure, not occurrence |
| **V4 Phase Correlation** | Local phase separates true site | 87.5% @5px | Rescued `finfet_021` but **broke 3 correct DRAM cases** (noisy 30px window) |

**Scientific Story**: *"We tested multiple increasingly domain-aware approaches and retained the robust baseline when they failed to generalize."*

## Slide 7: DriftSense V5 — Gated Phase Reranker (Production Default)
## Dual Gating Prevents Regressions

```python
if template_side >= 60px AND top2_score_margin < 0.01:
    apply_phase_reranker()  # Only on ambiguous FinFET candidates
else:
    keep_v2_ranking()       # DRAM / confident cases unchanged
```

**Rationale**: Phase correlation needs spatial support (~60px template). Works for FinFET (~100px) but fails on DRAM (~30px). Gate prevents DRAM regression.

**Results (40 pairs):**
- **95% @5px** overall (DRAM 100%, FinFET 93.3%, mixed_logic 100%)
- Runtime: ~200 ms/sample — minimal overhead over V1
- All 3 failure cases self-flagged; `finfet_021` rescued

## Slide 8: Dataset Design — Synthetic Benchmark with Realism
## Shared Canvas + Independent Degradation

- **Shared native canvas** (10k×10k): Reference = crop, Search = full downsample → GT true by construction
- **Breaking perfect periodicity**: Low-frequency brightness field + sparse defects + guaranteed landmark blobs (2-4 per crop, sized to survive 10× downsample)
- **Independent degradation**: Separate RNGs per image (noise, blur, rotation, scale)
- **SEM Physics** (all cited):
  - Edge brightening (SE escape probability at topographic edges)
  - Mixed Poisson-Gaussian noise (shot noise + read noise)
  - Non-uniform brightness (charging/gain drift)

## Slide 9: Key Results Summary
## Robust, Fast, and Honest

| Metric | V1 Baseline | V5 Gated (Default) |
|---|---:|---:|
| **Overall @5px** | 92.5% | **95.0%** |
| DRAM @5px | 100% | 100% |
| FinFET @5px | 80.0% | **93.3%** |
| mixed_logic @5px | 100% | 100% |
| Median Error | 0.09 px | 0.08 px |
| Runtime (CPU) | ~132 ms | ~200 ms |
| Low-Confidence Flagging | 3/3 correct | 3/3 correct |

## Slide 10: Known Limitations & Future Work
## Honest Assessment

1. **V5 is a strong but simple baseline** — learnable embeddings could beat it on hard periodic regions
2. **CPU-only throughput** — CUDA `cv2.cuda.matchTemplate` expected to give further speedup (not verified)
3. **Synthetic validation only** — noise/edge parameters cited but not validated against real SEM images
4. **finfet_023 unsolved** — GT absent from any NCC-based candidate pool; needs new signal (see SSJD proposal)

## Slide 11: SSJD — Next Algorithmic Contribution
## Semiconductor-Specific Joint Descriptor

**Core Idea**: FinFET has **two incommensurate frequencies** (fins ~4px, gates ~30px). Their **joint phase relationship** is physically fixed but varies across lattice cells.

- **Intensity NCC**: Uses Fourier *magnitude* → translation-invariant
- **Phase Correlation**: Uses Fourier *phase* but needs large template
- **SSJD**: Uses **joint phase coupling** between fin & gate lattices + topological moments of landmarks

**Breaks lattice translation invariance fundamentally** — not just locally like phase correlation.

**Expected**: Recovers `finfet_023` (unique global peak in SSJD surface) + generalizes to DRAM/mixed_logic via auto-detection.

## Slide 12: Thank You
## Drift-Sense: Robust Navigation-Error Recovery

**Key Takeaways:**
- Two-stage NCC + gated phase reranker → **95% @5px**, ~200ms, self-aware
- Systematic experimental journey: tried 4 advanced methods, kept the one that generalizes
- Synthetic benchmark with SEM physics realism + citations
- Clear path forward: SSJD for the remaining hard cases

**Repository**: github.com/[REPO] | **Contact**: [EMAIL]

Drift-Sense — Because the algorithm should know when it doesn't know.
