# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

**Semicon India Hackathon 2026 — Applied Materials Problem Statement**

---

## Problem

A wafer inspection tool must return to the exact same die site thousands of times a day, with measurements comparable across visits and across tools. In practice, motion stages accumulate small errors between visits — thermal expansion, vibration, mechanical slack — so a revisit can land the tool several pixels away from the intended site.

Because every die on a wafer carries the **same repeating circuit layout**, the landed image looks almost identical to the correct one. The core difficulty is not recognizing the pattern — it is pinpointing **which occurrence** of a highly repetitive pattern is the correct one.

**Formal task**: Given a Reference Image (the known-correct site, native resolution) and a Search Image (a ~10× lower-magnification view captured around where the tool actually landed), output the pixel center `(x, y)` in the Search Image where the reference pattern appears.

---

## Why Ordinary Template Matching Struggles

Standard template matching (normalized cross-correlation, NCC) finds the *single highest peak* in the correlation surface. On a periodic semiconductor layout, the correlation surface has **many near-tied peaks** at multiples of the pattern pitch. The global maximum is often a false periodic occurrence, not the true site.

The three canonical failure modes in our benchmark:

| Case | Error | Root Cause |
|------|-------|------------|
| `finfet_017` | 214 px | Correct coarse peak found, but Stage-2 refinement converges to wrong periodic cell |
| `finfet_021` | 571 px | True site discarded by global argmax; survives in top-K pool but misranked |
| `finfet_023` | 868 px | True site absent from even top-100 coarse peaks (~98.5th percentile) |

All three are **correctly self-flagged as low-confidence** by the ambiguity ratio — the algorithm knows when it doesn't know.

---

## DriftSense Architecture

### V1: Two-Stage NCC Baseline (Production Algorithm)

```
src/localizer.py → localize()
```

1. **Stage 1 (Coarse, Global)**: Single nominal-scale NCC over entire Search image
   - Also computes **ambiguity ratio** (peak vs. best score elsewhere in whole image)
   - Flags low-confidence when ambiguity_ratio < 1.05 or confidence < 0.35

2. **Stage 2 (Fine, Local)**: Scale×rotation grid search (5 scales × 7 rotations) in small window around Stage-1 peak
   - Parabolic sub-pixel peak refinement
   - Confining grid search to local window gives **3× speedup** (21.9s → 7.2s on 30 pairs) with no accuracy loss

**Results (40-pair benchmark: 15 DRAM + 15 FinFET + 10 OOD mixed_logic)**:
- ≤5 px: **92.5%**
- ≤4 px: 90.0%
- ≤2 px: 90.0%
- ≤1 px: 90.0%
- Median error: **0.09 px**
- DRAM: 100% (15/15)
- FinFET: 80% (12/15)
- OOD mixed_logic: 100% (10/10)
- Runtime: ~132 ms/sample (CPU)

### V2: Top-K Candidate Generation (Candidate Pool for Experiments)

```
src/localizer.py → localize_topk()
```

- Global NCC → top-K spatially-separated peaks (NMS) → independent Stage-2 refinement each → dedup on refined coordinates (key fix: dedup_radius = nominal_side, not window radius) → ranked list
- **Candidate recall @5px**: K=10: 95%, K=20: 95%, K=40: 95%
- Fixes the "candidate never existed" problem for `finfet_021` (GT enters pool at 0.02 px) but **not** `finfet_017` (Stage-2-induced) or `finfet_023` (candidate generation failure)
- Final ranked accuracy = V1 (92.5%), but enables reranking experiments
- Runtime: ~3.37s/sample at K=40

---

## Experimental Journey (All Rejected as Final Algorithm)

| Experiment | Hypothesis | Result | Why Rejected |
|------------|------------|--------|--------------|
| **V3 Native Verification** (`src/native_verifier.py`) | Native-res NCC re-ranks V2 pool | 67.5% @5px | Rescued `finfet_021` but hurt 11 DRAM cases (100%→26.7%) |
| **Gradient (Sobel)** | Structural edges break periodicity | 37.5% / 70.0% | Helped `finfet_017` pool (1.45px) but failed globally |
| **Periodicity Signatures** (orientation/FFT/Gabor) | Layout frequency identifies correct cell | ~0% | Translation-invariant within lattice — tells structure type, not which occurrence |
| **V4 Phase Correlation** (`experiments/phase_correlation/`) | Local phase separates true site from decoys | 87.5% @5px | Rescued `finfet_021` but **broke 3 already-correct DRAM cases** (30px window too noisy) |

**The strongest scientific story**: *"We tested multiple increasingly domain-aware approaches and retained the robust baseline when they failed to generalize."*

---

## Repository Structure

```
Semicon-DriftSense/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/                          # Production & core library code
│   ├── localizer.py              # V1 (localize) + V2 (localize_topk)
│   ├── native_verifier.py        # V3 experiment (self-eval only)
│   ├── pattern_synth.py          # Synthetic DRAM/FinFET/mixed Logic canvas
│   ├── build_dataset.py          # Generates self_eval + ood_holdout + GT
│   ├── evaluate.py               # Benchmark runner
│   ├── run_inference.py          # Submission entry point + timing
│   ├── visualize_ambiguity.py    # Correlation surface heatmaps
│   ├── ablation_study.py         # Noise/rotation robustness sweeps
│   └── build_submission_pdf.py   # Hackathon PDF builder
│
├── experiments/                  # Documented experiments (not production)
│   ├── v2/
│   ├── v3/
│   ├── gradient/
│   ├── periodicity/
│   └── phase_correlation/        # V4: local phase-correlation reranker
│       ├── phase_reranker.py
│       ├── bench_phase_reranker.py
│       ├── phase_reranker_experiment.md
│       ├── phase_reranker_ablation.csv
│       ├── make_v4_diagnostics.py
│       └── v4_diagnostic_cases.png
│
├── docs/
│   ├── design_notes.md           # Detailed design decisions + citations
│   ├── experiments.md            # Consolidated experiment log
│   ├── failure_analysis.md       # Deep dive on finfet_017/021/023
│   └── figures/
│       ├── ambiguity_finfet_023.png
│       └── ablation.png
│
├── data/                         # Benchmark datasets (committed)
│   ├── self_eval/                # 30 pairs (DRAM/FinFET) + ground_truth.json
│   ├── ood_holdout/              # 10 mixed_logic pairs (never tuned on)
│   └── predictions/              # run_inference.py output + timing
│
��── submission/
    └── DriftSense_Submission.pdf
```

---

## Quickstart

### Prerequisites
```bash
pip install -r requirements.txt
# numpy>=1.24, opencv-python>=4.8, matplotlib>=3.7, reportlab>=4.0
```

### Generate Datasets
```bash
cd src
python3 build_dataset.py --out ../data/self_eval --n 30                    # main dataset
python3 build_dataset.py --out ../data/ood_holdout --n 10 --styles mixed_logic  # OOD holdout
```

### Run V1 Baseline (Production)
```bash
python3 evaluate.py --data ../data/self_eval
python3 evaluate.py --data ../data/ood_holdout
```

### Run Submission Inference + Timing
```bash
python3 run_inference.py --input ../data/self_eval --output ../data/predictions
```

### Visualize Failure Cases
```bash
python3 visualize_ambiguity.py --data ../data/self_eval --sample finfet_023 --out ../docs/figures/ambiguity_finfet_023.png
```

### Robustness Ablation
```bash
python3 ablation_study.py --out ../docs/figures/ablation.png
```

### Run Documented Experiments
```bash
# V3 native verification (self-eval only; regenerates canvas from seed)
python3 native_verifier.py ../data/self_eval finfet_021

# V4 phase-correlation reranker (full 40-pair benchmark)
python3 ../experiments/phase_correlation/bench_phase_reranker.py
```

---

## Key Design Decisions (with Citations)

See `docs/design_notes.md` for full details:

1. **Shared native canvas** — Reference is a crop, Search is full canvas downsampled 10×. GT true by construction, not assertion.
2. **Breaking perfect periodicity** — Low-frequency brightness field + sparse defects + guaranteed landmark blobs inside every reference footprint (sized to survive 10× downsample).
3. **Independent degradation** — Separate RNGs for Reference/Search; never shared noise realization.
4. **SEM physics** — Edge brightening (SE escape probability), mixed Poisson-Gaussian noise (shot noise + read noise), non-uniform brightness (charging/gain drift).
5. **Two-stage search** — Coarse global NCC (also gives ambiguity signal) + fine local grid search. 3× throughput gain.
6. **Ambiguity flag** — Peak-to-second-peak ratio; all 3 hard misses correctly self-flagged.

---

## Known Limitations

1. **V1 is a strong but simple NCC baseline** — it can be beaten on hard periodic sub-regions by approaches using more global context (e.g., learned embedding with contrastive/triplet loss on this generator).
2. **Throughput measured on CPU only** — `cv2.matchTemplate`/`cv2.warpAffine` have CUDA drop-in equivalents (`cv2.cuda.*`) expected to give further speedup, not yet independently verified.
3. **Noise/edge parameters are citation-backed** but validation against real public-domain SEM images was started (candidates identified on Wikimedia Commons) and not completed.
4. **finfet_023 remains unsolved** — GT absent from candidate pool under any intensity-based method tried. Requires fundamentally new candidate generation signal.

---

## License

MIT License — see `LICENSE` file.

---

## Citation

If you use this work, please cite:
```
Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools
Semicon India Hackathon 2026 — Applied Materials Problem Statement
Team: [TEAM NAME], Vellore Institute of Technology
```