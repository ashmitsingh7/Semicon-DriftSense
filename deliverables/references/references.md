# References

This document lists all literature and technical sources cited in the DriftSense submission, organized by category. Each reference includes a brief note on how it informs our approach.

---

## 1. Semiconductor Device Structures & Layout Geometry

**[1] S. M. Sze & K. K. Ng, "Physics of Semiconductor Devices", 3rd ed., Wiley, 2007.**
Chapter on DRAM cell arrays: word-line/bit-line crossing geometry, 1T1C storage-node placement, and duty-cycle considerations for periodic line/space patterns. Informs our DRAM grid synthesis (`_dram_grid_vectorized`) pitch, line-width, and via-contact parameters.

**[2] International Roadmap for Devices and Systems (IRDS), "More Moore" Chapter, IEEE, 2022.**
FinFET fin pitch, gate pitch, and multi-fin layout conventions for logic standard cells. Our FinFET synthesis (`_finfet_grid_vectorized`) uses incommensurate fin pitch (~36–46px native) and gate pitch (~260–340px native) per this reference. Also covers row-based standard-cell layout conventions used for the `mixed_logic` held-out style.

---

## 2. SEM Image Formation Physics

**[3] ETH Zurich, Electron Microscopy Teaching Notes, "Secondary Electron Imaging".**
Defines the edge effect: increased secondary-electron escape probability near topographic edges, producing characteristic bright rims. Directly models `apply_edge_brightening()`.

**[4] Nanoscience Instruments, "Secondary Electrons in SEM: Unlocking Surface Insights at the Nanoscale".**
Edge enhancement mechanism: primary beam interaction at edges yields excess SE signal, edges appear brighter with characteristic width. Validates Sobel-magnitude-based edge brightening with gain parameter tuning.

**[5] St. Cloud State University Center for Microscopy & Imaging, "SEM A to Z: Basic Knowledge for Using the SEM".**
Edge effect: edges of steps/protrusions appear bright with characteristic width (~5–10nm equivalent). Confirms our `ksize=5` Sobel kernel choice for edge detection.

**[9] "Sample surface structure measuring method" (SEM Patent Disclosure).**
Secondary-electron brightness varies with local beam-incidence geometry; absolute brightness is not translation-invariant across a real sample even for geometrically periodic layouts. Informs our low-frequency brightness field injection (symmetry breaking).

---

## 3. SEM Noise Modeling

**[6] "Poisson shot noise parameter estimation from a single scanning electron microscopy image" (multiple authors, 2010s).**
SEM noise dominated by Poisson-distributed electron shot noise from primary/secondary electron emission, plus additive white Gaussian noise (AWGN) from detection electronics.

**[7] Mulapudi & Joy (2003); Cizmar et al. (2008), "Scanning Electron Microscope Image SNR Monitoring".**
Final SEM image noise modeled as Poisson (electron emission) + Gaussian (readout); Gaussian is a good approximation to Poisson at high mean signal. Our `apply_sensor_noise()` implements this mixed model with separate RNG streams per image.

**[8] Zhang et al., "M-Denoiser: Unsupervised image denoising for real-world optical and electron microscopy data", 2019.**
Microscopy noise = signal-dependent shot noise (Poisson) + signal-independent read noise (Gaussian). Confirms our noise implementation and the intentional non-clipping of search images (real sensors exceed [0,1] due to additive read noise).

---

## 4. Template Matching & Computer Vision

**[10] J. P. Lewis, "Fast Normalized Cross-Correlation," Vision Interface, 1995, pp. 120–123.**
Foundational NCC algorithm with integral-image acceleration. Our NCC uses OpenCV's optimized `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`.

**[11] K. Briechle & U. D. Hanebeck, "Template matching using fast normalized cross correlation," Proc. SPIE 4387, 2001.**
Fast NCC variants and similarity measures. Validates NCC as the standard well-understood measure for template localization tasks.

**[12] Tian & Huhns, "Algorithms for Subpixel Registration", CVGIP 35, 1986.**
Parabolic sub-pixel peak refinement: fitting a 2D parabola to the 3×3 neighborhood of the discrete correlation peak. Our `_subpixel_peak()` implements this directly.

**[13] OpenCV `phaseCorrelate` documentation / Kuglin & Hines (1975).**
Phase correlation for sub-pixel image registration using Fourier shift theorem. Our gated phase reranker uses `cv2.phaseCorrelate` with Hann windowing for fast candidate re-scoring.

---

## 5. Problem Statement & Competition Context

**[14] Applied Materials Semicon India Hackathon 2026 – Problem Statement (7-page PDF).**
Mandatory deliverables (§5), final submission checklist (§9), synthetic data specifications (§3), localization requirements (§4), and evaluation metrics (§6).

---

## Citation Index (for PPT slide references)

| Slides | References |
|--------|------------|
| Problem Understanding | [14] |
| Architecture / DRAM FinFET Structures | [1], [2] |
| Synthetic Data Method (canvas, edge brightening) | [3], [4], [5], [9] |
| Noise Model (Poisson-Gaussian) | [6], [7], [8] |
| Localization Method (NCC, sub-pixel) | [10], [11], [12] |
| Phase Correlation Reranker | [13] |
| Experiments / Results | (internal) |
| Failure Analysis | (internal) |
| Limitations / Next Steps | [1], [2], [9] |

---

## Key Algorithmic Novelties (Internal Documentation)

The following are novel contributions developed for this submission (not prior art):

- **Two-Stage Coarse-to-Fine NCC with Built-in Ambiguity Detection** (V1):
  Single global coarse pass gives both location AND peak-to-2nd-peak ratio; local fine grid in small window removes throughput bottleneck.

- **Gated Phase-Correlation Reranker** (V5, production default):
  Dual-gate architecture (template size ≥60px AND top-2 score margin <0.01) activates reranking ONLY when ambiguity is genuine (FinFET) and ranking is uncertain, avoiding DRAM regression seen in full reranking (V3/V4). Achieves 38/40 @5px (95%) vs. 37/40 (92.5%) V1.

- **Semiconductor-Specific Joint Descriptor (SSJD)** — *Algorithmic Novelty*:
  Joint phase coupling of incommensurate fin/gate frequencies breaks translation invariance of periodic FinFET patterns. Computed via 2D FFT of reference patch; queried at candidate locations in search image. Solves the fundamental candidate-generation failure of `finfet_023` (NCC peak at 98.5th percentile, 12,000+ false positives). Designed for submission as intellectual contribution beyond the graded track.

---

*All internal citations refer to code docstrings in `src/dataset/pattern_synth.py` and `docs/design_notes.md`.*