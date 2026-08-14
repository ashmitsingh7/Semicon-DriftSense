# V4 experiment: local phase-correlation reranking of the V2 candidate pool

**Status: experimental, evaluated, NOT adopted for the primary submission.**
V1 (`localizer.localize`) remains the submitted algorithm. This document
records an honest negative-with-caveats result so the "next steps" /
failure-analysis story is backed by a real 40-pair run rather than three
hand-picked cases.

## 1. Implementation

New, isolated file: `src/phase_reranker.py`. Does not import or modify
`localizer.py` beyond calling the existing, unchanged `localize_topk()`.
Also added: `src/bench_phase_reranker.py` (full-benchmark runner) and
`src/make_v4_diagnostics.py` (figure generator). V1/V2/V3 source, the
existing ground truth, and the submitted PDF were not touched.

## 2. Exact algorithm

```
V2 candidate pool (unchanged: coarse NCC -> NMS -> Stage-2 refine -> dedup)
        |
for each candidate (x, y):
    extract a nominal_side x nominal_side window from the same
    denoised search image V2 already uses, centered on (x, y)
    Hann-window it and the nominal-scale reference template
    cv2.phaseCorrelate(template, window) -> response
        |
sort candidates by response (descending)
        |
V4 prediction = top-1 by phase response
```

`nominal_side` is fixed per sample by the existing 1/10 magnification
convention (30px for DRAM in this dataset, 100px for FinFET) — no
per-candidate scale/rotation search is redone; this is a single, cheap,
fixed probe applied identically to every candidate. No parameter here was
tuned against the 40-pair set: Hann-window-on and a single fixed template
size were decided before the full run (see §9 for what changing the
window does, checked *after* the fact as a diagnostic, not as tuning).

## 3. Origin note

A prior, uncommitted script (outside this repo) reported finfet_023
phase-correlation response ≈0.83 for GT vs. 0.03–0.31 for its top-20
false candidates, and a near-perfect rescue of finfet_021. That script's
code could not be located in either delivered ZIP — only its reported
numbers and 3 PNGs/a CSV. This experiment is a fresh implementation of
the same *method*, run for the first time against the real V2 pipeline's
actual (much smaller, already-deduped, ~9–12 candidate) pool rather than
a hand-built pool of the top 20 of 60 raw NMS peaks. The two pools are
not the same population — see §9 for why that matters.

## 4. Benchmark table — V2 vs V4, full 40 pairs (30 self_eval + 10 OOD)

| | Success ≤5px | ≤4px | ≤2px | ≤1px | Mean err | Median err |
|---|---:|---:|---:|---:|---:|---:|
| **V2 (baseline)** | 92.5% | 90.0% | 90.0% | 90.0% | 33.11px | 0.09px |
| **V4 (phase reranked)** | 87.5% | 85.0% | 85.0% | 85.0% | 50.78px | 0.085px |

By style:

| Style (n) | V2 @5px | V4 @5px | V2 mean err | V4 mean err |
|---|---:|---:|---:|---:|
| DRAM (15) | 100.0% | 80.0% | 0.17px | 52.38px |
| FinFET (15) | 80.0% | 86.7% | 88.10px | 82.99px |
| mixed_logic OOD (10) | 100.0% | 100.0% | 0.06px | 0.06px |

**Net: -2 successes (-5.0 pts) overall.** FinFET improves (+1 case,
80%→86.7%); DRAM regresses hard (-3 cases, 100%→80%); OOD is untouched.

## 5. Candidate recall (unaffected by reranking, as expected)

V2's own candidate pool contains a point within 5px of GT for 38/40
samples (95.0%) — reranking can only reorder within that pool, and this
number is identical for V2 and V4 by construction. The 2 samples where
GT is absent from the pool are `finfet_017` and `finfet_023`, matching
the original handoff's diagnosis.

## 6. finfet_017

- GT was **NOT** in the V2 candidate pool this run (closest pool member:
  213.8px away — consistent with the design notes' account of this being
  a Stage-2-window-search failure, not simple candidate absence, but the
  point stands: no candidate is close).
- V2 top-1: 266.8px error (score 0.6871).
- V4 top-1: 896.2px error (phase 0.2548) — **worse**.
- Reranking picked a different, more distant decoy: among this sample's
  9 candidates, the phase response of the *closest-to-GT* one (213.8px,
  phase=0.2142) was not the pool's highest — a different decoy scored
  phase=0.2548. No clean separation here, unlike the isolated finfet_023
  audit's broader/noisier pool (see §9).

## 7. finfet_021

- GT **was** in the V2 pool (candidate at 0.0px error existed, ranked
  #2 of 10 by V2's own NCC score, V2score=0.8020 vs. the winning decoy's
  0.8043 — essentially tied).
- V2 top-1: 181.8px error (a different, higher-NCC-scoring decoy).
- V4 top-1: **0.0px error** — phase response 0.8236 vs. next-best 0.3250,
  a real, wide margin. **Rescued.**
- This is the case the phase signal was built for: intensity NCC and its
  runner-up are near-tied (0.8043 vs 0.8020) but phase correlation is
  not close (0.325 vs 0.824) because the true candidate uniquely
  reproduces the reference's landmark blob (visible directly in
  `docs/figures/v4_diagnostic_cases.png`, row 2).

## 8. finfet_023

- GT is **confirmed absent** from the V2 candidate pool (closest pool
  member: 111.9px away).
- V2 top-1: 867.7px error. V4 top-1: 343.4px error — closer, but still a
  clear miss at any reasonable threshold.
- **Phase correlation does not solve this case.** As expected: a
  reranker cannot promote a candidate that candidate generation never
  produced. The best it can do is pick a *less-wrong* wrong answer.

## 9. Anti-artifact / validation checks

- **Hann window vs. none**, on the 3 regressed DRAM cases: turning the
  window off fixes `dram_008` and `dram_026` (both flip back to ~0.1px)
  but **not** `dram_012` (still 466.6px wrong). So windowing choice is
  part of the problem but not the whole story — the underlying issue is
  template size, not just the window function (see below).
- **Root cause of the DRAM regression** (inspected directly): DRAM's
  `nominal_side` is 30px vs. FinFET's 100px (set by the 10:1 magnification
  ratio and the dataset's `REF_SIZE` choice, not by this experiment).
  On `dram_008`/`dram_012`/`dram_026`, V2's own NCC already separates GT
  from the best decoy by a wide, confident margin (e.g. 0.852 vs. 0.610).
  But at a 30x30 window, `cv2.phaseCorrelate`'s estimate is visibly
  noisier: a decoy occasionally scores fractionally higher phase response
  than GT (e.g. `dram_012`: decoy phase=0.2679 vs GT's 0.2316), which is
  enough to overturn an already-confident, already-correct V2 decision.
  **The phase signal is unreliable at small window sizes and should not
  be trusted to overrule a high-margin V2 decision** — it is a real
  signal but a noisy one at 30px, not a bug in the windowing per se.
- **Correlation between phase response and V2's own NCC score**, pooled
  across ~400 candidates from 20 samples: Pearson r = 0.141. Low —
  phase response is carrying substantially independent information from
  intensity NCC, not restating it. This supports the signal being real
  (not a redundant proxy) even though it isn't reliable enough to trust
  blindly.
- **Why finfet_023's isolated audit looked cleaner than this run**: the
  original 3-case audit compared GT against the *top 20 of 60 raw NMS
  peaks before V2's dedup* — a broader, more heterogeneous pool that
  includes many clearly-bad candidates, which pads out the "false
  candidate" distribution and makes GT's margin look larger. This
  benchmark reranks V2's actual final pool (~9–12 candidates, already
  filtered to the *best* decoys) — a harder, more realistic test, and
  the margin narrows or disappears in cases like `finfet_017`. This
  distinction matters and should be stated plainly rather than let the
  cleaner 3-case number stand unchallenged.
- Border/edge effects: not the dominant cause here — the regressed DRAM
  cases were not edge-of-image; the effect was window-size noise (above).

## 10. Runtime

Mean V2 time/pair (this run, mixed DRAM/FinFET/OOD): 2.99s. Mean phase
reranking overhead: 0.014s/pair — **0.47% of V2's own runtime**. The
reranker itself is essentially free; the accuracy question, not cost, is
what rules it out below.

## 11. Failure analysis summary

Reranker behaviour across all 40 pairs:

| | count |
|---|---:|
| unchanged, already correct | 34 |
| unchanged, already wrong | 0 |
| changed, correct → correct | 0 |
| changed, correct → **wrong** | 3 (`dram_008`, `dram_012`, `dram_026`) |
| changed, wrong → **correct** | 1 (`finfet_021`) |
| changed, wrong → wrong | 2 (`finfet_017`, `finfet_023`) |

The reranker never left a correct prediction correct when it touched it —
every time it changed the top-1 pick on an already-correct sample, it
broke it. Its only genuine win is `finfet_021`.

## 12. Decision

**V4 should NOT replace V2 or be used as the default reranker in the
final submission.** On this benchmark it is a net regression
(92.5%→87.5% @5px), concentrated entirely in DRAM-style samples where the
30px window makes the phase estimate too noisy to trust over an
already-confident V2 decision.

**It should also not be discarded outright.** The FinFET-only picture is
mildly positive (80.0%→86.7%) and `finfet_021`'s rescue is decisive and
mechanistically well-understood (visible in the landmark blob, not a
fluke — see figure). A defensible middle path, **not implemented here**,
would gate the reranker by template size and by V2's own top1-vs-top2
score margin (e.g. only rerank when `nominal_side` is large enough for a
stable phase estimate, and only when V2's own top-2 candidates are
within some closeness threshold of each other) — but that is untested,
would need its own parameter-before-data discipline, and should not be
presented as validated. As stated in the acceptable-outcomes list, this
is Outcome B (helps hard FinFET cases, hurts others) trending toward
Outcome C at the current, ungated, one-size-fits-all setting — keep V1
as primary and report this honestly as a partial/negative result with a
concrete, evidence-backed idea for a follow-up experiment.
