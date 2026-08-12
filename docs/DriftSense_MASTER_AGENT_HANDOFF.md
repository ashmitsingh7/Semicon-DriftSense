# DriftSense — Master Agent Handoff

## Purpose
Attach this Markdown plus `drift-sense-v2v3.zip` to future coding/research agents. This file is the accumulated project state so agents do not repeat completed audits/experiments.

## Project
**SEMICON India Hackathon 2026 — Applied Materials problem: Drift-Sense, AI-Powered Navigation-Error Recovery for Wafer Inspection Tools.**

Task: given a 100x high-resolution Reference image and a wider 10x Search image, return the target centre `(x,y)` in Search coordinates. Official inputs are 1000x1000 grayscale images; nominal scale is 10:1, robustness may include ~9:1–11:1, rotation ~1–2 degrees, origin top-left. If several valid matches exist, the official statement says choose the one closest to Search-image centre.

The supplied participant document requires:
- realistic synthetic DRAM or FinFET pairs;
- recorded seed/architecture/transforms/noise/scale/rotation/GT metadata;
- literature-supported synthetic data;
- explicit scale handling;
- batch/pair inference without manual source changes;
- runtime measurement;
- failure analysis;
- >=30 varied independent pairs;
- Euclidean error;
- pass rates at 5/4/2/1 px and subpixel performance;
- mean/median/worst error;
- runtime with hardware, Python version and timing method;
- robustness across noise, positions, scales, rotations;
- at least one visualized failure.

It states evaluation weights of 50% localization/inference, 30% synthetic augmentation, 10% failure analysis/explainability, plus RGB bonus; remaining 10% was pending in the supplied document.

Submission-format note: the participant-help document describes a Solution PPT/PPTX, while the separate Idea Submission Template says the current idea submission must use the supplied template, max ~6–7 slides, and be saved/uploaded as PDF. Confirm the current portal/phase before finalizing.

## Physical ZIP state
The main development archive is `drift-sense-v2v3.zip`.

It contains:
```text
drift-sense/
├── src/
│   ├── pattern_synth.py
│   ├── ablation_study.py
│   ├── native_verifier.py
│   ├── run_inference.py
│   ├── visualize_ambiguity.py
│   ├── localizer.py
│   ├── evaluate.py
│   ├── build_submission_pdf.py
│   └── build_dataset.py
├── requirements.txt
├── README.md
├── DriftSense_Submission.pdf
├── .gitignore
├── data/
│   ├── self_eval/
│   ├── predictions/
│   └── ood_holdout/
└── docs/design_notes.md
```

**Important:** later experimental files `layout_aware.py`, `layout_signature.py`, `bench_layout_ablation.py`, and `bench_phase3_ablation.py` are NOT in this ZIP snapshot. Their experiments were run in temporary working copies; their results are recorded below.

## Dataset architecture
`pattern_synth.py` + `build_dataset.py` create one large native semiconductor canvas, crop a native-resolution Reference from it, and downsample the same canvas 10x into Search. Therefore GT is true by construction.

A real bug was found and fixed: after Search-image affine/rotation transformation, GT initially was not transformed correspondingly. Current GT tracks the exact transform. Reference/Search use independent RNGs.

Styles:
- DRAM: word/bit-line grid + contact/via structures.
- FinFET: parallel fins crossed by gate bars.
- `mixed_logic`: irregular row/block style used only as held-out OOD.

Perfect periodicity was found to be mathematically ambiguous: translation by one pitch can produce an indistinguishable pattern. The generator therefore adds low-frequency brightness variation, sparse defects, and guaranteed larger landmarks inside Reference footprints so a unique target exists. Search/Reference degradation is independent and includes blur, scale/rotation jitter, SEM edge brightening and mixed Poisson-Gaussian noise.

## V1 baseline
`src/localizer.py`:
```text
Reference
 → resize to nominal Search scale
 → global NCC
 → cv2.minMaxLoc() = ONE coarse hypothesis
 → local scale×rotation refinement
 → subpixel refinement
 → ambiguity ratio / low-confidence flag
 → (x,y)
```

The key flaw is architectural: Stage 1 retains only one global maximum. If a periodic false occurrence wins, the true site is discarded before refinement.

Original 30-pair self-eval:
- <=5px: 90% (27/30)
- DRAM: 100% (15/15)
- FinFET: 80% (12/15)
- median successful error: ~0.10px
- all 3 misses self-flagged low-confidence
- mixed_logic OOD: 100% (10/10), median ~0.06px
- original CPU throughput ~230ms/sample

Later 40-pair benchmark (15 DRAM + 15 FinFET + 10 OOD):
- <=5px 92.5%
- <=4px 90%
- <=2px 90%
- <=1px 90%
- median ~0.09px
- DRAM 15/15
- FinFET 12/15
- OOD 10/10
- mean runtime ~132ms/sample

Do not mix the 30-pair and 40-pair figures.

## Three canonical failures

### finfet_017
GT ≈ `(820.45,830.88)`.
V1 prediction ≈ `(608.95,862.23)`.
Error ≈ 213.81px.
Stage 1 initially got near the correct neighborhood (~4.5px coarse error), but Stage 2 refinement moved to a wrong periodic occurrence.

### finfet_021
GT ≈ `(407.75,609.69)`.
V1 prediction ≈ `(925.99,369.98)`.
Error ≈ 570.99px.
The correct site was discarded by Stage 1. This proves single global argmax is insufficient.

### finfet_023
GT ≈ `(914.49,538.52)`.
V1 prediction ≈ `(52.01,443.13)`.
Error ≈ 867.74px.
GT is only around the 98.5th percentile of the coarse NCC surface; roughly 12,000 pixels score higher. Increasing K alone cannot solve this. This is the main unresolved case.

## V2: top-K/NMS/dedup
V2 preserves multiple coarse hypotheses:
```text
global NCC
 → top-K peaks
 → NMS
 → Stage-2 refinement
 → post-refinement coordinate dedup
 → ranking
```

A bug was found: large Stage-2 windows caused many distinct coarse candidates to refine to the same location. Fix: deduplicate after refinement in final coordinate space. K=40 often collapsed to ~8–12 genuinely distinct refined candidates.

40-pair V2:
- <=5px 92.5%
- <=4px 90%
- <=2px 90%
- <=1px 90%
- median ~0.09px
- DRAM 15/15
- FinFET 12/15
- OOD 10/10
- mean runtime ~3.368s/sample at K=40
- candidate recall: K=10 38/40=95%, K=20 95%, K=40 95%

Thus V2 did not improve final accuracy, but it fixed the hypothesis-discard architecture.

## Center-priority sweep
Official statement says ambiguous valid matches should choose closest-to-centre. We tested rather than assuming it helps our synthetic GT.

FinFET success:
```text
tie_margin 0.000 → 12/15
0.005 → 12/15
0.010 → 11/15
0.020 → 10/15
0.050 → 5/15
```

Conclusion: broad center-priority tie-breaking regresses against generated GT because GT is actual crop origin while another periodic occurrence can be closer to centre. Do not tune this merely to improve self-eval.

## V3 native-resolution verifier
`src/native_verifier.py` exists in ZIP.

Idea:
```text
V2 candidate pool → native-resolution verification → rerank
```

Native landmark contrast measured ~2.95x vs ~1.84x after 10x downsampling.

Initial V3 regenerated the 10,000x10,000 canvas per candidate. Caching deterministic canvas gave byte-identical output and ~9x speedup on a test (57.6s → 6.3s), but remained expensive.

Full 40-pair V3:
- <=5px 67.5%
- <=4px 65%
- <=2px 65%
- <=1px 65%
- median ~0.10px
- DRAM 26.7% (4/15)
- FinFET 86.7% (13/15)
- OOD 100% (10/10)
- mean ~17.63s/sample
- median ~14.25s/sample
- changed selected candidate 14/40: helped 1, hurt 11, neutral 2

Conclusion: **reject V3 as final**. It rescued `finfet_021` but severely hurt already-correct DRAM cases. Do not resurrect as main algorithm.

V3 case details:
- `finfet_021`: V2 top-score error ~181.79px → V3 ~0.02px.
- `finfet_017`: GT absent from V2 pool, so V3 cannot recover it.
- `finfet_023`: GT absent from V2 pool; best V2 candidate ~111.88px; V3 cannot recover it.

## Phase 1/2: raw gradient experiment
Temporary `layout_aware.py`.

A = intensity/V2.
B = Sobel gradient-magnitude NCC.
C = 0.5 intensity NCC + 0.5 gradient NCC.

40-pair results:
| Strategy | <=5px | Median | DRAM | FinFET | OOD | Recall@40 |
| A | 92.5% | 0.09px | 100% | 80% | 100% | 95% |
| B | 37.5% | 244.51px | 33.3% | 33.3% | 50% | 42.5% |
| C | 70.0% | 0.18px | 80% | 40% | 100% | 72.5% |

Reject B/C as final.

But B produced one important clue:
- `finfet_017`: intensity GT absent from pool, gradient GT entered pool and best candidate was ~1.45px.
- `finfet_023`: gradient improved best-pool error ~111.88px → ~46.80px, but still did not find GT.

Therefore structural information exists, but generic Sobel magnitude is too periodic/weak.

## Phase 3: orientation/periodicity signature
Temporary `layout_signature.py`.

D1: structure-tensor orientation/coherence.
D2: windowed FFT frequencies + Gabor-like periodicity.
D3: combined orientation × periodicity.

40-pair result:
- D1 0% <=5px, median ~658px
- D2 0%, median ~522px
- D3 0%, median ~599px
- recall essentially zero
- runtime ~741ms/sample

Reject.

Theoretical reason: FFT/Gabor magnitude and orientation coherence are approximately translation-invariant within periodic lattices. They say "this is the right type of structure" but not "this is the correct occurrence". Removing phase destroys positional information.

## Current research frontier
Already ruled out:
- generic intensity NCC alone
- single global argmax
- top-K alone as a final accuracy improvement
- native reranking as final
- raw gradient alone
- naive gradient fusion
- orientation-only signature
- translation-invariant periodicity magnitude

The remaining hard problem is finding a **semiconductor-specific AND position-sensitive** signal.

### Immediate next task — do NOT jump straight to V4
Perform an information audit on `finfet_023`.

Compare:
1. GT crop
2. top ~20 intensity-NCC false candidates
3. top ~20 gradient-NCC false candidates

Measure:
- intensity mean/std
- local variance
- gradient X/Y
- gradient magnitude
- gradient orientation histogram
- edge density
- Laplacian
- intensity/gradient histograms
- Fourier magnitude
- Fourier phase
- phase consistency
- local topology
- candidate/reference residuals
- other physically justified descriptors

Do not train a classifier yet. This is to discover whether useful information exists.

Required visualization:
```text
Reference | GT | False 1 | False 2 | False 3 ...
```
at identical scale, plus intensity/gradient/Fourier magnitude/Fourier phase comparisons.

Only if evidence supports it should a next algorithm explore something like:
```text
periodicity + phase + local topology
```
Potential physical interpretation:
- fine phase → fin alignment
- coarse phase → gate alignment
- local topology → contacts/crossings/irregularities
- intensity NCC → local unique appearance

Do not hard-code generator pitches or use GT at inference.

## VLSI-specific differentiation
The concern is that "NCC + NMS" could be produced by any CS/CV engineer.

Generic pieces:
- NCC
- NMS
- top-K
- subpixel interpolation
- Sobel
- FFT

The legitimate semiconductor contribution should come from:
- DRAM/FinFET structure generation
- SEM-specific degradation
- physical repeated-die/periodicity reasoning
- semiconductor layout hierarchy
- scale-aware physical interpretation
- position-sensitive structural features
- failure analysis of periodic semiconductor layouts

Desired narrative:
```text
generic CV baseline
 → observe periodic-semiconductor failure
 → analyze physical/layout structure
 → find position-sensitive semiconductor descriptor
 → use it to recover candidates generic NCC misses
 → retain NCC for precise localization
```

## Frozen benchmark table
| Version | <=5px | <=4px | <=2px | <=1px | Median | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 92.5% | 90% | 90% | 90% | 0.09px | ~132ms |
| V2 | 92.5% | 90% | 90% | 90% | 0.09px | ~3.37s |
| V3 | 67.5% | 65% | 65% | 65% | 0.10px | ~17.63s |
| Gradient | 37.5% | 37.5% | 35% | 22.5% | 244.51px | ~3.96s |
| Fused | 70% | 70% | 70% | 70% | 0.18px | ~6.62s |
| Periodicity/orientation | ~0% | ~0% | ~0% | ~0% | hundreds px | ~0.74s |

V2 candidate recall:
```text
K=10 95%
K=20 95%
K=40 95%
```

## Submission state
Existing `DriftSense_Submission.pdf` contains problem understanding, synthetic data, DRAM/FinFET, SEM degradation, localization, results, ambiguity visualization, robustness, limitations.

It has placeholders:
```text
[TEAM NAME]
[NAMES]
[INSERT REPO LINK]
[INSERT VIDEO LINK]
```

Do not call it final until:
- repo link filled
- team details filled
- final algorithm decided
- threshold-wise results verified
- current portal format confirmed

## Rules for every future agent
1. Inspect actual files first.
2. Preserve V1.
3. Preserve V2/V3 results.
4. Do not overwrite frozen results.
5. Do not rerun completed ablations.
6. Do not assume temporary experiment files are in ZIP.
7. Put new experiments in separate clearly named files.
8. Benchmark every new algorithm against V1/V2.
9. Every claimed semiconductor feature needs a physical/layout rationale.
10. Stop negative experiments rather than stacking complexity.
11. Never optimize only for the three known failures.
12. Never use GT during inference.
13. Keep A/B comparisons controlled.
14. Report negative results honestly.

## Immediate instruction
Start by:
1. inspecting the actual `drift-sense-v2v3` source;
2. reading `docs/design_notes.md`;
3. locating `finfet_023`;
4. performing the GT-vs-top-false-candidates information audit;
5. determining whether phase/topology/other structural information contains a real positional signal;
6. only then proposing or implementing the next algorithm.

If no useful signal exists, say so and stop. The project is already substantial; the next improvement must be evidence-driven.
