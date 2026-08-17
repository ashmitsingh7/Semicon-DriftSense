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
src/localization/v1_localize.py → localize_v1()
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
src/localization/v2_candidates.py → localize_v2()
```

- Global NCC → top-K spatially-separated peaks (NMS) → independent Stage-2 refinement each → dedup on refined coordinates (key fix: dedup_radius = nominal_side, not window radius) → ranked list
- **Candidate recall @5px**: K=10: 95%, K=20: 95%, K=40: 95%
- Fixes the "candidate never existed" problem for `finfet_021` (GT enters pool at 0.02 px) but **not** `finfet_017` (Stage-2-induced) or `finfet_023` (candidate generation failure)
- Final ranked accuracy = V1 (92.5%), but enables reranking experiments
- Runtime: ~3.37s/sample at K=40

### V5: Gated Phase Reranker (Production Default)

```
src/localization/rerankers/gated_reranker.py → localize_with_gated_rerank()
```

- Dual gating: Only applies phase correlation reranking when (a) template side ≥ 60px (FinFET) AND (b) V2 top-2 score margin < 1%
- **Results**: 95% @5px overall (DRAM 100%, FinFET 93.3%, mixed_logic 100%)
- Runtime: ~200ms/sample (CPU) — minimal overhead over V1

---

## Experimental Journey (All Rejected as Final Algorithm)

| Experiment | Hypothesis | Result | Why Rejected |
|------------|------------|--------|--------------|
| **V3 Native Verification** (`src/localization/rerankers/native_verifier.py`) | Native-res NCC re-ranks V2 pool | 67.5% @5px | Rescued `finfet_021` but hurt 11 DRAM cases (100%→26.7%) |
| **Gradient (Sobel)** | Structural edges break periodicity | 37.5% / 70.0% | Helped `finfet_017` pool (1.45px) but failed globally |
| **Periodicity Signatures** (orientation/FFT/Gabor) | Layout frequency identifies correct cell | ~0% | Translation-invariant within lattice — tells structure type, not which occurrence |
| **V4 Phase Correlation** (`experiments/phase_correlation/`) | Local phase separates true site from decoys | 87.5% @5px | Rescued `finfet_021` but **broke 3 already-correct DRAM cases** (30px window too noisy) |

**The strongest scientific story**: *"We tested multiple increasingly domain-aware approaches and retained the robust baseline when they failed to generalize."*

---

## Deliverables Structure

```
deliverables/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── generate_dataset.py                 # Entry point: generate synthetic benchmark data
├── localize.py                         # Entry point: localization inference (single/batch)
├── generate_manifest.py                # Entry point: consolidate GT + predictions → manifest.csv
├── build_pptx.py                      # Build solution_presentation.pptx from markdown
├── solution_presentation.pptx         # Hackathon presentation
├── configs/
│   ├── dataset_config.yaml            # Dataset generation parameters (cited)
│   └── localization_config.yaml       # Localization algorithm parameters
├── src/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── build_dataset.py           # Synthetic dataset generator (DRAM/FinFET/mixed_logic)
│   │   └── pattern_synth.py           # Semiconductor pattern synthesis with citations
│   ├── localization/
│   │   ├── __init__.py
│   │   ├── base.py                    # Shared utilities
│   │   ├── v1_localize.py             # V1: Two-stage NCC baseline
│   │   ├── v2_candidates.py           # V2: Top-K candidate pool
│   │   └── rerankers/
│   │       ├── __init__.py
│   │       ├── gated_reranker.py      # V5: Gated phase correlation (DEFAULT)
│   │       ├── phase_reranker.py      # V4: Phase correlation (experiment)
│   │       └── native_verifier.py     # V3: Native verification (self-eval only)
│   └── utils/
│       ├── __init__.py
│       ├── io.py                      # Image I/O utilities
│       ├── geometry.py                # Geometric transformations
│       └── metrics.py                 # Evaluation metrics
├── data/                              # Generated datasets (gitignored, created at runtime)
│   ├── self_eval/                     # 30 pairs (DRAM/FinFET) + ground_truth.json
│   ├── ood_holdout/                   # 10 mixed_logic pairs (never tuned on)
│   └── predictions/                   # Inference outputs
├── results/                           # Benchmark results (gitignored)
└── references/                        # Citation PDFs / supplementary material
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
python3 generate_dataset.py --out ./data/self_eval --n 30
python3 generate_dataset.py --out ./data/ood_holdout --n 10 --styles mixed_logic --seed0 9000
```

### Run V1 Baseline (Production)
```bash
python3 localize.py --input ./data/self_eval --output ./results/self_eval_v1 --method v1
python3 localize.py --input ./data/ood_holdout --output ./results/ood_holdout_v1 --method v1
```

### Run V5 Gated Reranker (Default, Best Accuracy)
```bash
python3 localize.py --input ./data/self_eval --output ./results/self_eval_v5 --method v5_gated
python3 localize.py --input ./data/ood_holdout --output ./results/ood_holdout_v5 --method v5_gated
```

### Generate Submission Manifest
```bash
python3 generate_manifest.py --gt ./data/self_eval/ground_truth.json --pred ./results/self_eval_v5/predictions.json --out ./results/self_eval_v5/manifest.csv
```

### Visualize Failure Cases
```bash
python3 -m src.utils.visualize_ambiguity --data ./data/self_eval --sample finfet_023 --out ./results/ambiguity_finfet_023.png
```

### Robustness Ablation
```bash
python3 -m src.ablation_study --out ./results/ablation.png
```

### Run Documented Experiments
```bash
# V3 native verification (self-eval only; regenerates canvas from seed)
python3 -m src.localization.rerankers.native_verifier ./data/self_eval finfet_021

# V4 phase-correlation reranker (full 40-pair benchmark)
python3 -m experiments.phase_correlation.bench_phase_reranker
```

---

## Key Design Decisions (with Citations)

See `REFERENCES.md` and `configs/` for full details:

1. **Shared native canvas** — Reference is a crop, Search is full canvas downsampled 10×. GT true by construction, not assertion.
2. **Breaking perfect periodicity** — Low-frequency brightness field + sparse defects + guaranteed landmark blobs inside every reference footprint (sized to survive 10× downsample).
3. **Independent degradation** — Separate RNGs for Reference/Search; never shared noise realization.
4. **SEM physics** — Edge brightening (SE escape probability), mixed Poisson-Gaussian noise (shot noise + read noise), non-uniform brightness (charging/gain drift).
5. **Two-stage search** — Coarse global NCC (also gives ambiguity signal) + fine local grid search. 3× throughput gain.
6. **Ambiguity flag** — Peak-to-second-peak ratio; all 3 hard misses correctly self-flagged.
7. **Gated phase reranking** — Only apply phase correlation when template is large enough (FinFET) AND ambiguity is high (top-2 margin < 1%). Prevents DRAM regression.

---

## Known Limitations

1. **V5 is a strong NCC + phase baseline** — it can be beaten on hard periodic sub-regions by approaches using more global context (e.g., learned embedding with contrastive/triplet loss on this generator).
2. **Throughput measured on CPU only** — `cv2.matchTemplate`/`cv2.warpAffine` have CUDA drop-in equivalents (`cv2.cuda.*`) expected to give further speedup, not yet independently verified.
3. **Noise/edge parameters are citation-backed** but validation against real public-domain SEM images was started (candidates identified on Wikimedia Commons) and not completed.
4. **finfet_023 remains unsolved** — GT absent from candidate pool under any intensity-based method tried. Requires fundamentally new candidate generation signal (see `docs/SSJD_DESIGN.md` for proposed Semiconductor-Specific Joint Descriptor).

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