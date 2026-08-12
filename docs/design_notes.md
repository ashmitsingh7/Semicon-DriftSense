# Design notes — Drift-Sense

This doc expands on the choices made in `src/`, with the citation
justifying each one (per the problem statement's citation requirement).

## 1. Why a large shared canvas, not two separately-generated images

Early versions of the generator built the Reference and Search images from
*separate* renders of "the same" pattern. This is wrong: it lets subtle
implementation bugs silently desynchronize the two images from their
recorded ground truth (we hit exactly this — rotating the full Search
canvas *after* computing the ground-truth center, without propagating that
rotation to the recorded coordinate, produced a dataset whose official
ground truth was wrong by hundreds of pixels; `apply_geometric_degradation`
now optionally returns the affine matrix and `build_dataset.py` transforms
the ground-truth point through it explicitly). Building both images from
one shared native-resolution canvas removes this class of bug entirely:
the Reference is a crop of the canvas, the Search is a downsample of the
*entire* canvas, so the reference pattern's presence inside the search
image is true by construction, not asserted.

## 2. Why perfectly-periodic structure needed to be broken deliberately

A DRAM or FinFET layout that is *exactly* periodic (same pitch, same phase,
no other variation) is not just hard to localize in — it is **provably
ambiguous**: translating the crop location by any integer multiple of the
pitch produces a bitwise-identical reference pattern. Early builds of the
generator had literally zero localization signal beyond one period, which
we discovered by checking the NCC score *at* the recorded ground-truth
location versus the discovered maximum (`docs/` debugging session) — the
correct location was scoring *below* several incorrect ones.

We fixed this the way real fabs actually solve it: real wafers are not
perfectly periodic at the SEM-charging/contamination scale even when the
drawn layout is. So every canvas gets:
- a smooth low-frequency brightness field (charging/gain drift — see §4),
- sparse small background defects,
- and, **inside every reference footprint specifically**, 2–4 guaranteed
  larger high-contrast landmark blobs, sized so they survive the 10x
  area-average downsample (native radius 45–80px; anything much smaller
  than ~10px native radius is smoothed away to sub-pixel amplitude by
  `cv2.INTER_AREA` and contributes essentially nothing to the correlation
  score of the downsampled Search image — this was measured directly:
  radius-2–6px defects alone left the dataset at 0% localization success).

This is also *why* the FinFET holdout style still has 3 genuinely hard
cases in the 30-pair self-eval set (`finfet_017/021/023`): landmark
placement is random, and for a subset of samples the guaranteed landmarks
land close to a periodic repeat of themselves, reproducing the ambiguity
by chance. This is intentional — it is exactly the "at least one highly
periodic array region where correct localization is genuinely difficult"
requirement — and the localizer's ambiguity-ratio flag catches all three
cases (see `docs/figures/ambiguity_finfet_023.png`).

## 3. Structural geometry citations

- DRAM word-line/bit-line crossing geometry with a contact/via at each
  intersection: S. M. Sze & K. K. Ng, *Physics of Semiconductor Devices*,
  3rd ed., Wiley, 2007 (1T1C DRAM cell array chapter).
- FinFET dense parallel-fin + gate-bar crossing geometry: International
  Roadmap for Devices and Systems (IRDS), "More Moore" chapter, 2022 —
  fin pitch / gate pitch conventions for multi-fin logic standard cells.
- The `mixed_logic` holdout style (irregular row-based standard-cell
  blocks) follows the general row-based standard-cell layout convention
  from the same IRDS reference, deliberately generated with different
  pitch/irregularity statistics and *never used to tune the localizer* —
  it exists purely to report out-of-distribution generalization the way
  the hackathon's own hidden test set will.

## 4. SEM physics citations

- **Edge brightening** (`apply_edge_brightening`, gradient-magnitude-based
  rim enhancement): real SEM images show brighter secondary-electron
  signal at topographic edges because the interaction volume intersects
  the edge from more angles, increasing SE escape probability.
  - ETH Zürich, Dept. of Materials, "Secondary Electron Imaging" teaching
    notes.
  - Nanoscience Instruments, "Secondary Electrons in SEM: Unlocking
    Surface Insights at the Nanoscale."
  - St. Cloud State University, Center for Microscopy & Imaging, "SEM A to
    Z: Basic Knowledge for Using the SEM."
- **Sensor noise** (`apply_sensor_noise`, mixed Poisson + Gaussian):
  SEM images are dominated by Poisson-distributed electron shot noise from
  the emission process, plus an additive Gaussian contribution from
  detector/amplifier electronics.
  - "Poisson shot noise parameter estimation from a single scanning
    electron microscopy image."
  - Mulapudi & Joy (2003); Cizmar et al. (2008), as summarized in
    "Scanning Electron Microscope Image SNR Monitoring."
  - "M-Denoiser: Unsupervised image denoising for real-world optical and
    electron microscopy data" (ScienceDirect).
- **Non-uniform brightness field**: local SE yield depends on beam
  incidence geometry, so absolute brightness is not translation-invariant
  across a sample even for a geometrically periodic layout.
  - USPTO patent disclosure, "Sample surface structure measuring method."
- We deliberately leave the Search-image pixel values **unclipped** to
  [0, 1] after adding Gaussian read noise, matching the KLA webinar's own
  note that noisy/low-res images in that dataset may fall slightly outside
  [0, 1] as "a feature of the dataset, not a bug."

## 5. Algorithm citations

- Normalized cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`):
  - J. P. Lewis, "Fast Normalized Cross-Correlation," Vision Interface,
    1995, pp. 120-123.
  - K. Briechle & U. D. Hanebeck, "Template matching using fast normalized
    cross correlation," Proc. SPIE 4387, 2001.
- Sub-pixel parabolic peak refinement: standard correlation-peak
  refinement approach (Tian & Huhns, "Algorithms for Subpixel
  Registration," CVGIP 35, 1986).

## 6. Throughput design

The original implementation ran the full scale × rotation grid
(5 scales × 7 rotations = 35 `matchTemplate` calls) over the *entire*
1000×1000 Search image — the dominant cost, since `matchTemplate` cost
scales with search-image area. We restructured this into two stages:

1. **Coarse, global**: one nominal-scale/zero-rotation `matchTemplate`
   pass over the full image. This single pass is also what we need to
   compute the ambiguity ratio (best peak vs. best peak *elsewhere in the
   whole image*), so it isn't wasted work.
2. **Fine, local**: crop a small window around the coarse peak and run the
   full 35-combination grid only inside that window.

Measured effect on the 30-pair self-eval set: **21.9s → 7.2s total
(≈3x)**, same 90% accuracy, same 3/3 correctly-flagged failures. See
`src/run_inference.py` for the exact directory-in/directory-out timing
harness and `data/predictions/timing_report.json` for the raw numbers.
We have not yet benchmarked on an actual GPU node; `cv2.matchTemplate` and
`cv2.warpAffine` both have CUDA-accelerated drop-in equivalents
(`cv2.cuda.*`) that we expect to give a further meaningful speedup on the
grid-search step specifically, but we have not measured this ourselves and
say so plainly rather than quoting an unverified number.

## 7. What we'd do with more time

- Benchmark and, if needed, port the grid search to `cv2.cuda` for the
  actual H100 throughput number.
- Validate the noise-model parameters against a real public SEM image
  (started this — found candidate CC-licensed SEM images on Wikimedia
  Commons but did not complete the pixel-statistics comparison before
  time ran out; flagged rather than hand-waved).
- A learned embedding (small CNN, contrastive/triplet loss trained on this
  same generator) as a second algorithm to compare against the NCC
  baseline, particularly on the genuinely-ambiguous periodic cases where
  NCC has no way to use context beyond the reference footprint itself.

## 8. V2 — top-K multi-hypothesis candidates (`localizer.localize_topk`)

V1 keeps only the single global-argmax coarse candidate before local
refinement, so if the coarse stage's top pick is a decoy, the true site is
gone for good — the ambiguity ratio can only *flag* this, not recover
from it. V2 replaces the single argmax with:

```
global NCC -> top-K spatially-separated coarse peaks (greedy NMS)
           -> independent Stage-2 refinement per candidate
           -> dedup on the REFINED coordinates
           -> ranked candidate list
```

**Bug found and fixed during this pass:** the first implementation used
the Stage-2 window radius (~260px for FinFET) as the post-refinement
dedup radius. That is too large — on `finfet_021`, two *genuinely
different* refined candidates only ~182px apart get merged, and the
merge keeps whichever one scored fractionally higher by raw NCC. In this
case the higher-scoring survivor was a decoy (score 0.8043) and the
correct site (score 0.8020, 0px error) was silently discarded by dedup.
Fixed by tying the dedup radius to the reference footprint size
(`nominal_side`) instead of the Stage-2 window — candidates that close
really are the same physical site; anything farther is a different one.

**Coverage result (K=40, self_eval, 30 pairs):** the true site now
survives into the candidate pool for 28/30 pairs (up from 27/30 "hit" on
V1, but a different 28 — `finfet_021` is newly covered; `finfet_017` and
`finfet_023` are still absent from the pool). Coverage ≠ final accuracy:
V2's candidate list is ranked by raw low-res NCC score alone, and for
`finfet_021` the true candidate ranks 3rd (score 0.8020) behind two
decoys (0.8043, 0.8037) — this is exactly the reordering problem native
verification (§9) targets. Runtime at K=40 is ~5.6s/pair vs. V1's
~230ms/pair (~24x), a real cost not yet optimized.

Per-case diagnosis, confirmed by direct inspection of the coarse/refined
candidate arrays (not just re-stated from a prior summary):

- **`finfet_021`** — GT *is* the coarse Stage-1 peak's neighbor at 0px
  error (rank 28 of 40 raw coarse peaks, but present). Top-K candidate
  generation fixes the "candidate never existed" problem. Ranking is the
  remaining issue.
- **`finfet_017`** — GT is the single BEST coarse peak (rank 0/100,
  4.5px error) — Stage 1 was never the problem. Stage 2's local
  scale/rotation grid search, run on a window centered at that correct
  coarse peak, converges to a different, higher-scoring pose ~214px away
  *within that same window*. This is a **Stage-2-induced** failure, not
  a candidate-generation failure — top-K alone cannot fix it, because
  Stage 2 corrupts the coordinate before dedup ever sees it.
- **`finfet_023`** — GT is not in the top-100 coarse peaks at all
  (confirmed by direct search); consistent with the ~98.5th-percentile
  finding. Not a top-K-depth-fixable problem at K=40, and likely not at
  any reasonable K under plain NCC ranking.

## 9. V3 — native-resolution verification (`native_verifier.py`)

Minimal implementation per design intent: no CNN, no FFT, no learned
embedding — just native-resolution NCC between the (already
native-resolution) Reference Image and a native-resolution crop around
each surviving candidate, with a small scale/rotation grid to absorb
residual pose error.

**Self-eval-only caveat, stated plainly:** getting a native-resolution
image at an arbitrary candidate coordinate normally means physically
re-visiting that stage location. We don't have that capability here, so
for the *synthetic* self-eval dataset only, we exploit that
`pattern_synth.synth_canvas(style, seed)` is a deterministic function of
its stored seed and regenerate the same native canvas to crop from. A
light blur+Poisson-Gaussian noise pass is applied to the crop so this
isn't an unrealistically noiseless comparison. **This regeneration trick
has no equivalent on real SEM data** — a real deployment needs an actual
native-resolution re-capture step at each candidate site. Treat the
numbers below as a validation of the *mechanism*, not as evidence it
works unchanged on non-synthetic data.

**Result on the V2 candidate list (top 5 candidates, native NCC
re-ranking):**

| sample | pre-native top-score error | post-native top-rank error | verdict |
|---|---:|---:|---|
| finfet_021 | 181.8 px | **0.0 px** | **RESCUED** |
| finfet_017 | — (GT not in V2 pool) | — | not applicable |
| finfet_023 | — (GT not in V2 pool) | — | not applicable |

`finfet_021` is decisively rescued: native NCC scores the true candidate
at 0.7369 vs. 0.7254–0.7257 for the decoys that outscored it at low
resolution — a clear separation that raw 10x-downsampled NCC alone did
not have. This directly answers the module's target question in the
affirmative for the one case where the mechanism is actually testable
(true site present, pool ranking wrong).

`finfet_017` and `finfet_023` cannot be rescued by re-ranking the V2
pool, because neither's true site is in that pool to begin with — native
verification only reorders what candidate generation already found.

**Follow-up experiment (not yet integrated, flagged as next step, not
claimed as done):** since `finfet_017`'s failure is Stage-2-induced
(§8) rather than candidate-generation-induced, we tested native
verification directly on the *Stage-1 coarse* candidates (before Stage 2
runs), bypassing the step that corrupts this case. Result: the correct
coarse candidate (4.5px error) scores native NCC 0.6810, clearly above
all 7 decoy coarse candidates (0.6704–0.6735). This suggests native
verification applied earlier in the pipeline — on coarse candidates,
before the local scale/rotation refinement — could also rescue
`finfet_017`, without needing to fix Stage 2's window-search behavior
directly. Not implemented as a pipeline change yet; needs a subpixel
refinement step at native resolution added afterward to recover final
coordinate precision, and a full 30-pair A/B before being trusted as a
real improvement rather than a 1-sample anecdote.

`finfet_023` remains unrescued by any variant tried so far (its true
site isn't in the coarse top-100 either) — still the open research
problem the original handoff identified it as.

## 10. Honest current status (V1 → V2 → V3)

| version | self-eval coverage/accuracy | notes |
|---|---|---|
| V1 | 27/30 (90%) | single-argmax; 3 FinFET misses, all self-flagged low-confidence |
| V2 pool coverage | 28/30 | `finfet_021` newly covered; `finfet_017`/`finfet_023` still absent; ~24x slower than V1 at K=40 |
| V3 (native re-rank, applied to V2 pool) | rescues `finfet_021` (1/3 remaining failures) | `finfet_017`/`finfet_023` need candidate generation before Stage-2 or a coarse-stage variant (see §9 follow-up), not yet built/tested at scale |

No full 30-pair V2/V3 accuracy number is reported here yet (only the
3 diagnostic cases + a 30-pair *pool coverage* check) — that A/B run,
plus the OOD holdout, is the next step before claiming an overall
accuracy improvement over V1's 90%.
