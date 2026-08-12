# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

Semicon India Hackathon 2026 — Applied Materials problem statement.

## Problem

A wafer inspection tool must return to the exact same die site thousands of
times a day. Motion-stage drift (thermal expansion, vibration, mechanical
slack) means the tool sometimes lands a few pixels off target. Because every
die on a wafer carries the same repeating circuit layout, the landed image
looks almost identical to the correct one — the challenge is finding *the*
correct site inside a sea of visually near-identical periodic structure.

**Task:** given a Reference Image (the known-correct site, native
resolution) and a Search Image (a ~10x-lower-magnification view around
where the tool actually landed), return the pixel center `(x, y)` in the
Search Image where the reference pattern appears.

## Repo layout

```
src/
  pattern_synth.py       synthetic DRAM/FinFET/mixed_logic generation + SEM-like physics
  build_dataset.py       builds paired reference/search images + ground truth
  localizer.py            the navigation-error-recovery matching algorithm (2-stage NCC)
  evaluate.py              self-evaluation against recorded ground truth
  run_inference.py         directory-in/directory-out submission entry point + timing
  visualize_ambiguity.py   renders the NCC correlation surface for a chosen sample
  ablation_study.py        success rate vs. noise/rotation severity sweep
data/
  self_eval/               30 generated pairs (DRAM+FinFET) + ground_truth.json + results.csv
  ood_holdout/              10 mixed_logic pairs, held out from all tuning, for generalization check
  predictions/               run_inference.py output + timing_report.json
docs/
  design_notes.md            longer write-up of design decisions and citations
  figures/                    ambiguity heatmap + ablation plots
```

## Quickstart

```bash
pip install opencv-python numpy matplotlib
cd src
python3 build_dataset.py --out ../data/self_eval --n 30            # main dataset
python3 build_dataset.py --out ../data/ood_holdout --n 10 --styles mixed_logic  # OOD holdout
python3 evaluate.py --data ../data/self_eval --out_csv ../data/self_eval/results.csv
python3 run_inference.py --input ../data/self_eval --output ../data/predictions  # submission entry point + timing
python3 visualize_ambiguity.py --data ../data/self_eval --sample finfet_023 --out ../docs/figures/ambiguity_finfet_023.png
python3 ablation_study.py --out ../docs/figures/ablation.png
```

## Results

| metric | value |
|---|---|
| success rate (error ≤ 5 px), 30-pair self-eval | 90.0% (27/30) |
| median error on hits | 0.10 px |
| DRAM-style success | 100% (15/15) |
| FinFET-style success | 80% (12/15) |
| all 3 misses self-flagged low-confidence | yes (3/3) |
| **held-out `mixed_logic` style (never tuned on)** | **100% (10/10)**, median error 0.06 px |
| end-to-end throughput (CPU, this sandbox) | ~240 ms/pair, ~4.1 samples/s |
| throughput after 2-stage search optimization | 21.9s → 7.2s total on 30 pairs (~3x) |

The 3 misses are all FinFET cases where the periodic pattern is genuinely
ambiguous (correlation surface has many near-tied peaks — see
`docs/figures/ambiguity_finfet_023.png`); the localizer's ambiguity-ratio
signal flags all 3 as low-confidence rather than silently returning a
wrong answer. The `mixed_logic` holdout — a third layout style the
localizer was never adjusted for — scores *higher* than the tuned styles,
because its irregular (non-periodic) structure has less inherent
ambiguity; see `docs/design_notes.md` §2.

See `docs/figures/ablation.png` for how success rate degrades under
increased sensor noise and rotation drift beyond the main dataset's
settings.

## Dataset generator — design summary

Each sample starts from one large native-resolution synthetic "die" canvas
(10,000×10,000 px) built from vectorized periodic-grid formulas:

- **DRAM-style**: horizontal word-lines + vertical bit-lines crossing at
  right angles, with a contact/via dot at every intersection, following
  standard DRAM 1T1C cell-array layout conventions.
- **FinFET-style**: dense parallel vertical fins crossed by horizontal gate
  bars at the intersection region, following standard multi-fin logic
  layout conventions.
- **mixed_logic-style** (held-out, never tuned on): irregular row-based
  standard-cell blocks, used only to report out-of-distribution
  generalization the way the hackathon's own hidden test set will.

The **Reference Image** is a native-resolution crop from this canvas. The
**Search Image** is the *entire* canvas downsampled 10x — so the reference
pattern is genuinely, verifiably present inside the search image at a
location we record exactly as ground truth.

A perfectly periodic pattern is mathematically ambiguous under translation
by any multiple of the pitch, so every canvas also gets: a smooth
low-frequency brightness field (simulating non-uniform SE yield / charging
drift), a scattering of small background defects, and — inside every
reference footprint — a few guaranteed larger high-contrast landmark blobs
(simulating local process variation / contamination), sized to survive the
10x downsample. This is what makes each site locally unique and the task
actually solvable, while still leaving some sites genuinely hard (see
"Mandatory requirement: highly periodic difficult region" below).

Reference and Search images are then degraded **independently** (separate
RNGs — never the same noise realization on both): Gaussian blur, a small
rotation/scale jitter (representing residual stage drift), a real
SEM-style edge-brightening pass, and a mixed Poisson-Gaussian sensor-noise
model. The Search side uses a lower effective electron dose than the
Reference side, so it is measurably noisier — matching the stated test-time
behaviour.

## Algorithm

`localizer.py` uses a **two-stage** multi-scale, small-rotation
**normalized cross-correlation** (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`)
approach:

1. one coarse, global NCC pass (nominal scale, zero rotation) over the
   *entire* search image — this also yields the ambiguity ratio (peak vs.
   best score elsewhere in the whole image);
2. a fine pass restricted to a small window around the coarse peak, where
   the full scale × rotation grid (5 scales × 7 rotations) is searched,
   with parabolic sub-pixel peak refinement.

Confining the expensive grid search to a small local window (instead of
running it over the full image) is what took total runtime on the 30-pair
self-eval set from 21.9s to 7.2s with no accuracy loss — see
`docs/design_notes.md` §6. It also reports the **ambiguity ratio** and
flags low-confidence matches — directly targeting the "failure-mode
awareness" requirement, since a genuinely periodic/ambiguous region
*should* be flagged rather than silently mis-reported (all 3 misses in the
self-eval set are correctly flagged — see
`docs/figures/ambiguity_finfet_023.png`).

This NCC-based approach is a deliberately simple, fast, fully-explainable
baseline (no training required) — a natural next step for a stronger
submission is a learned matcher (e.g. a small CNN embedding trained with a
contrastive/triplet loss on this same synthetic generator) that can use the
periodic context more cleverly than raw pixel correlation.

## Citations

Augmentation, structural, and algorithmic choices are justified in
`docs/design_notes.md` with the following references:

1. S. M. Sze & K. K. Ng, *Physics of Semiconductor Devices*, 3rd ed., Wiley,
   2007 — DRAM cell-array word-line/bit-line crossing geometry.
2. International Roadmap for Devices and Systems (IRDS), "More Moore"
   chapter, 2022 — FinFET fin pitch / gate pitch layout conventions.
3. ETH Zürich, Dept. of Materials, "Secondary Electron Imaging" teaching
   notes — SEM edge effect definition.
4. Nanoscience Instruments, "Secondary Electrons in SEM: Unlocking Surface
   Insights at the Nanoscale" — edge brightening in SE imaging.
5. St. Cloud State University, Center for Microscopy & Imaging, "SEM A to
   Z: Basic Knowledge for Using the SEM" — edge-effect width and cause.
6. "Poisson shot noise parameter estimation from a single scanning
   electron microscopy image" — SEM shot noise model.
7. Mulapudi & Joy (2003); Cizmar et al. (2008), summarized in "Scanning
   Electron Microscope Image SNR Monitoring" — Poisson+Gaussian SEM noise.
8. "M-Denoiser: Unsupervised image denoising for real-world optical and
   electron microscopy data", ScienceDirect — mixed Poisson-Gaussian
   microscopy noise model.
9. USPTO patent disclosure, "Sample surface structure measuring method" —
   SE brightness dependence on local beam-incidence geometry.
10. J. P. Lewis, "Fast Normalized Cross-Correlation," Vision Interface,
    1995, pp. 120-123.
11. K. Briechle & U. D. Hanebeck, "Template matching using fast normalized
    cross correlation," Proc. SPIE 4387, 2001.

## Known limitations / next steps

- The NCC baseline is a strong-but-simple starting point; it can be beaten
  on the hardest periodic sub-regions by approaches that use more global
  context (e.g. matching multiple candidate peaks against absolute stage
  coordinates, or a learned embedding).
- Throughput has been measured and optimized on CPU (3x speedup via a
  coarse-then-local two-stage search) but not yet profiled on an actual
  H100; `cv2.matchTemplate`/`cv2.warpAffine` have direct `cv2.cuda`
  equivalents, which is the obvious next step.
- The dataset generator is fully synthetic; we started but did not finish
  validating noise/edge-effect parameters against a real public-domain SEM
  image (candidates identified on Wikimedia Commons) — flagged as
  incomplete rather than claimed as done.
