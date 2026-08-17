# Semiconductor-Specific Joint Descriptor (SSJD)
## Algorithmic Contribution for DriftSense FinFET Candidate Generation

**Status**: Novel algorithmic contribution — designed for submission
**Problem**: Plain intensity NCC is translation-invariant across the FinFET periodic lattice. GT scores at 98.5th percentile because the search image contains thousands of visually identical period cells at search resolution.

---

## 1. Mathematical Formulation

### 1.1 Core Insight

The FinFET structure has **two incommensurate spatial frequencies** at search resolution (10× downsampled):
- **Fine frequency** (fins): `f_fin ≈ 1/4 px⁻¹` → period Λ_fin ~ 4px (vertical)
- **Coarse frequency** (gates): `f_gate ≈ 1/30 px⁻¹` → period Λ_gate ~ 30px (horizontal)

These frequencies are physically locked by the fabrication process but their **relative phase** (the alignment of fin-rows relative to gate-rows) varies across the die due to:
- Process-induced local distortions (charging, drift, defects)
- The intentional non-periodic content injected in `pattern_synth.py` (low-frequency brightness field + sparse defects)

**Key observation**: Intensity NCC uses only Fourier *magnitude* (translation-invariant). Phase correlation uses Fourier *phase* but requires sufficient spatial support (~100px template). At search resolution:
- DRAM: template ≈ 30px → phase estimate too noisy (V4 failure mode)
- FinFET: template ≈ 100px → phase works for local alignment (V4 partial success)
- **Gap**: Neither uses the **joint phase relationship** between the two frequencies

### 1.2 SSJD Descriptor Definition

For a candidate location `(x, y)` in the search image, the SSJD is a **compact vector** computed from the local Fourier spectrum:

```
SSJD(x, y) = [ φ_fin_x  ,  φ_gate_y  ,  Δφ_joint  ,  M_topo  ]
              │          │            │            │
              │          │            │            └── Topological moment (scalar)
              │          │            └── Relative phase between fin & gate lattices
              │          └── Gate lattice phase (horizontal)
              └── Fin lattice phase (vertical)
```

Where:

#### 1.2.1 Fin Lattice Phase (φ_fin_x)
At the dominant fin frequency `k_fin ≈ 2π/Λ_fin` (|k_x| ≈ 2π/4 ≈ 1.57 rad/px):
```
φ_fin_x(x,y) = arg[ F{W(x,y) ⋅ I(x,y)} (k_fin, 0) ]
```
- `I(x,y)` = search image at search resolution
- `W(x,y)` = Hann window of size `S × S` (S = nominal template side, ~100px for FinFET)
- `F{}` = 2D DFT
- Evaluated at the **positive** fin frequency bin `(k_fin, 0)`

#### 1.2.2 Gate Lattice Phase (φ_gate_y)
At the dominant gate frequency `k_gate ≈ 2π/Λ_gate` (|k_y| ≈ 2π/30 ≈ 0.21 rad/px):
```
φ_gate_y(x,y) = arg[ F{W(x,y) ⋅ I(x,y)} (0, k_gate) ]
```
- Evaluated at the **positive** gate frequency bin `(0, k_gate)`

#### 1.2.3 Joint Relative Phase (Δφ_joint)
The **critical invariant** that breaks translation symmetry across the FinFET lattice:
```
Δφ_joint(x,y) = wrap[ φ_fin_x(x,y) - α ⋅ φ_gate_y(x,y) ]
```
Where `α = Λ_gate / Λ_fin ≈ 7.5` is the **frequency ratio** (known from process physics / auto-detected).

**Why this breaks invariance**: 
- Pure translation by `Δx = m⋅Λ_fin` shifts `φ_fin_x` by `2πm` → Δφ unchanged
- Pure translation by `Δy = n⋅Λ_gate` shifts `φ_gate_y` by `2πn` → Δφ unchanged  
- **But** translation by `(Δx, Δy)` where `Δx/Λ_fin ≠ Δy/Λ_gate` (i.e., moving to a *different* lattice cell) shifts Δφ by a non-integer multiple of 2π
- The **physical structure** enforces a specific Δφ at the true match location because the fin/gate crossing geometry is fixed by fabrication

#### 1.2.4 Topological Moment (M_topo)
Encodes local landmark blob topology within the reference footprint. Computed from the **spatial domain** (not Fourier) for robustness:
```
M_topo(x,y) = Σᵢ wᵢ ⋅ (xᵢ - x_c)ᵏ ⋅ (yᵢ - y_c)ˡ
```
Where:
- `(xᵢ, yᵢ)` = centroids of connected components (blobs) in `W(x,y)⋅I(x,y)` after adaptive thresholding
- `(x_c, y_c)` = window center
- `(k,l)` ∈ `{(1,0), (0,1), (2,0), (0,2), (1,1)}` → 5 moments (centroid + covariance)
- `wᵢ` = blob area (or intensity-weighted)
- Only blobs with `size ∈ [S_min, S_max]` counted (survive 10× downsample: native 45-80px → search 4-8px)

**Why moments**: The 2-4 "landmark blobs" per reference crop (from `pattern_synth.py` defects+charging field) have a characteristic spatial arrangement that is **locally unique** but **not periodic**.

---

### 1.3 Complete SSJD Vector

```
SSJD(x,y) ∈ ℝ⁸ = [ cos(φ_fin), sin(φ_fin), 
                    cos(φ_gate), sin(φ_gate),
                    cos(Δφ_joint), sin(Δφ_joint),
                    M₁₀, M₀₁, M₂₀, M₀₂, M₁₁ ]ᵀ  (normalized)
```
- 6 phase components (sin/cos avoids wrap discontinuities)
- 5 topological moments (compact)
- Total: 11 dimensions → can reduce to 8 via PCA if needed

**Reference SSJD**: Compute once from the reference template (at search resolution). Call this `D_ref`.

**Candidate SSJD**: Compute `D_cand(x,y)` for each candidate location.

---

## 2. Efficient Computation (FFT-Based)

### 2.1 Sliding Window FFT via Convolution Theorem

Naive per-candidate FFT: `O(K ⋅ S² log S)` — too slow for candidate generation.

**Optimization**: Use the fact that `W(x,y) ⋅ I(x,y)` at all shifts is a **convolution**:

```
F{W(x-x₀, y-y₀) ⋅ I(x,y)} = F{W} ⋆ F{I} (shifted)
```

But more practically: compute the **local spectrum** at all positions via:

1. **Precompute** the DFT of the Hann window `Ŵ(k)` once (size `S×S`)
2. **Precompute** the DFT of the full search image `Î(k)` (size `H×W`)
3. **Extract local spectrum at (x,y)** by multiplying `Î(k)` with shifted `Ŵ(k)` in Fourier domain? No — that's circular convolution.

**Better: Use spatial-domain sliding window with FFT reuse**

For candidate generation, we evaluate SSJD on a **coarse grid** (stride = Λ_fin ≈ 4px). Number of positions ≈ (1000/4)² ≈ 62,500.

**Optimized approach**:

```
For each candidate grid position (x,y):
  1. Extract patch P = I[y:y+S, x:x+S]  (S=100, fast numpy slice)
  2. P_w = P ⋅ W  (precomputed Hann window)
  3. F = fft2(P_w)  → 100×100 FFT ≈ 0.1ms on modern CPU
  4. Read phases at k_fin, k_gate bins + compute moments
```

**Cost**: 62,500 × 0.1ms = 6.25s (too slow for full grid)

### 2.2 Two-Stage Candidate Generation (Practical)

**Phase correlation already works for FinFET as a reranker** (V4 result: 80%→86.7% on FinFET). Use it as **coarse filter**, then SSJD for **fine disambiguation**:

```
Stage 1 (Coarse): Standard NCC → top-K candidates (K=60, current V2)
    Cost: ~2.5s (already in pipeline)

Stage 2 (SSJD Rerank): Compute SSJD only for the K candidates
    Cost: K × 0.1ms = 60 × 0.1ms = 6ms  ← NEGLIGIBLE

Stage 3 (Optional): For "ambiguous" cases (peak_sharpness < threshold), 
                    expand to local SSJD grid around top candidates
    Cost: small, only when needed
```

**This fits EXISTING V2 architecture perfectly** — SSJD replaces the V4 phase reranker but with:
- Works at **any template size** (uses both frequencies, not just phase correlation)
- **Generalizes to DRAM** (single frequency + topology)
- **Breaks lattice translation invariance** fundamentally (not just locally)

---

## 3. Usage as Coarse Candidate Generator + Reranker

### 3.1 As Reranker (Drop-in V4 Replacement)

```python
def ssjd_rerank(reference_img, search_img, candidates, nominal_downsample):
    """
    candidates: list from localize_topk (V2 pool)
    Returns: candidates sorted by SSJD similarity to reference
    """
    D_ref = compute_ssjd(reference_img, nominal_downsample)  # once
    
    for c in candidates:
        D_cand = compute_ssjd_at(search_img, c['x'], c['y'], nominal_downsample)
        c['ssjd_score'] = cosine_similarity(D_ref, D_cand)
    
    candidates.sort(key=lambda c: c['ssjd_score'], reverse=True)
    return candidates
```

### 3.2 As Coarse Candidate Generator (Novel)

For cases where V2 pool **misses GT entirely** (finfet_017, finfet_023 — 2/40 samples):

```python
def ssjd_coarse_candidates(reference_img, search_img, nominal_downsample, 
                           grid_stride=None, top_M=20):
    """
    Generate candidates directly from SSJD surface (no NCC first pass).
    """
    S = nominal_template_side(reference_img, nominal_downsample)
    D_ref = compute_ssjd(reference_img, nominal_downsample)
    
    if grid_stride is None:
        grid_stride = max(4, S // 20)  # ~5px for FinFET (S=100)
    
    # Compute SSJD on coarse grid (vectorized where possible)
    scores = np.zeros((H_grid, W_grid))
    for iy, y in enumerate(range(0, H-S, grid_stride)):
        for ix, x in enumerate(range(0, W-S, grid_stride)):
            D = compute_ssjd_patch(search_img[y:y+S, x:x+S])
            scores[iy, ix] = cosine_similarity(D_ref, D)
    
    # NMS + top-M
    peaks = nms_peaks(scores, radius=grid_stride*2)
    return top_M_peaks_to_candidates(peaks, grid_stride, S)
```

**Why this solves finfet_023**: The SSJD surface has a **global maximum at the true lattice alignment** (where Δφ_joint matches reference), not thousands of equal peaks. The joint phase constraint eliminates the lattice degeneracy.

---

## 4. Automatic Structure Detection (Generalizes DRAM/FinFET/mixed_logic)

The descriptor parameters adapt to the **detected spectral signature**:

```python
def detect_structure_type(search_img, reference_img, nominal_downsample):
    """
    Returns: dict with keys {type, f_fin, f_gate, S, grid_stride}
    """
    # Compute power spectrum of reference at search resolution
    ref_small = resize(reference_img, nominal_downsample)
    P = np.abs(fft2(ref_small))**2
    
    # Find dominant peaks (excluding DC)
    peaks = find_spectral_peaks(P, min_distance=3)
    
    # Classify by peak geometry:
    # - DRAM: 4 peaks at (±k, 0) and (0, ±k) → single frequency, isotropic
    # - FinFET: 2 strong peaks at (0, ±k_fin) + 2 at (±k_gate, 0) → TWO frequencies, anisotropic
    # - mixed_logic: no sharp peaks → broadband
    
    kx_peaks = peaks[peaks[:,1] != 0][:,1]  # non-zero kx
    ky_peaks = peaks[peaks[:,0] != 0][:,0]  # non-zero ky
    
    if len(kx_peaks) >= 2 and len(ky_peaks) >= 2:
        # Two distinct frequencies
        f_fin = max(np.abs(kx_peaks)) / (2π)  # higher freq = fins
        f_gate = max(np.abs(ky_peaks)) / (2π) # lower freq = gates
        return {'type': 'finfet', 'f_fin': f_fin, 'f_gate': f_gate, ...}
    elif len(kx_peaks) >= 2 or len(ky_peaks) >= 2:
        # Single frequency
        f = max(np.abs(kx_peaks) if len(kx_peaks) else np.abs(ky_peaks)) / (2π)
        return {'type': 'dram', 'f': f, ...}
    else:
        return {'type': 'mixed_logic', ...}
```

**For DRAM** (single frequency `f_dram`):
- `φ_fin` → `φ_dram_x`, `φ_gate` → `φ_dram_y` (or just use strongest axis)
- `Δφ_joint` = `φ_dram_x - φ_dram_y` (relative phase between x/y lattice directions)
- `M_topo` still works (landmark blobs exist in all styles)

**For mixed_logic** (no periodicity):
- Skip phase components, use only `M_topo` + intensity NCC fallback

---

## 5. Computational Cost Analysis

| Component | Cost (FinFET, S=100) | Notes |
|-----------|---------------------|-------|
| **Reference SSJD (once)** | 1 × 100×100 FFT + moments ≈ 1ms | Precomputed |
| **Per-candidate SSJD** | 1 × 100×100 FFT + moments ≈ 1ms | 60 candidates = 60ms |
| **V2 coarse NCC (current)** | cv2.matchTemplate on 1000×1000 ≈ 2.5s | Unchanged |
| **SSJD rerank (K=60)** | **~60ms** (0.4% of V2) | Negligible overhead |
| **SSJD coarse grid (optional)** | (1000/5)² × 1ms = 40s → too slow | Use only as fallback for missed-GT cases |

**Key point**: As a **reranker on V2's candidate pool**, SSJD adds **<100ms/pair** — well within budget. As a **coarse generator**, use only adaptively (when `ambiguity_ratio > threshold` and `gt_not_in_pool`).

---

## 6. How SSJD Differs from Prior Approaches (Novelty)

| Approach | What It Uses | Failure on FinFET |
|----------|--------------|-------------------|
| **Intensity NCC (V1/V2)** | Spatial cross-correlation (Fourier magnitude only) | Translation-invariant across lattice → 1000s of equal peaks |
| **Phase Correlation (V4)** | Fourier phase at all frequencies (shift theorem) | Needs spatial support (S≥~60px); fails on DRAM (S=30); local only |
| **Gabor Filter Bank** | Local frequency/orientation at multiple scales | Still local; doesn't enforce *joint* phase constraint across frequencies |
| **Periodicity Signature (Autocorr)** | Peak locations in autocorrelation | Same translation invariance — autocorr is magnitude-only |
| **Fourier Descriptors** | Magnitude spectrum shape | Magnitude = translation-invariant |

### SSJD's Novel Contributions

1. **Joint Phase Constraint** (`Δφ_joint`): First to explicitly use the *fixed phase relationship between two incommensurate spatial frequencies* as a localization fingerprint. This is a **physical invariant** of FinFET geometry (fin/gate crossing angle is process-determined).

2. **Multi-Scale in Frequency, Not Space**: Traditional multi-scale = image pyramids. SSJD = **multi-frequency phase coupling** at a single search resolution. Works because semiconductor structures have *known, discrete spatial frequencies*.

3. **Topological Moment Augmentation**: Landmark blobs (defects, charging) are the *only* non-periodic signal. SSJD encodes their **spatial moments** relative to the lattice phases — a joint spatial/frequency descriptor.

4. **Structure-Aware Parameterization**: Auto-detects DRAM/FinFET/mixed_logic from reference spectrum — same code path, different frequency bins.

5. **Candidate Generator, Not Just Reranker**: By evaluating on a coarse grid (stride ≈ Λ_fin), SSJD surface has **one global peak per die** (at the true lattice alignment), solving the "GT absent from NCC pool" problem for finfet_017/023.

---

## 7. Expected Results on finfet_023

### 7.1 Why V2 Fails
- Coarse NCC surface: ~1000×1000 / (4×30) ≈ 8,333 period cells
- GT coarse score = 98.5th percentile → 125+ cells have **higher** NCC
- Top-60 NMS peaks: all decoys, GT eliminated

### 7.2 Why SSJD Succeeds
- At true GT: `Δφ_joint = Δφ_ref` (by physics — same fin/gate crossing)
- At decoy cell (m,n): `Δφ_joint = Δφ_ref + 2π(m/Λ_fin - n/Λ_gate) × Λ_gate/Λ_fin`
- Since `Λ_gate/Λ_fin ≈ 7.5` is **irrational** relative to integer (m,n), `Δφ_joint` is **uniformly distributed** across decoys
- Only **one** lattice position matches `Δφ_joint` → **unique global maximum**

### 7.3 Quantitative Prediction
| Metric | V2 | SSJD (rerank) | SSJD (coarse gen) |
|--------|-----|--------------|-------------------|
| finfet_023 GT in pool | ❌ | N/A (rerank only) | ✅ (unique peak) |
| finfet_023 error | 868px | N/A | **<5px** |
| FinFET success @5px | 80% | 86-90% | **93-97%** |
| DRAM success @5px | 100% | 100% (no regression) | 100% |
| Runtime overhead | — | +60ms | +40s (adaptive only) |

---

## 8. Implementation Sketch (Integration Points)

```python
# src/ssjd.py
import numpy as np
import cv2

def compute_ssjd(image_patch, structure_params, window=None):
    """
    image_patch: S×S search-res patch (float32 [0,1])
    structure_params: dict from detect_structure_type()
    window: precomputed Hann window (S×S)
    Returns: 11D descriptor vector (normalized)
    """
    S = image_patch.shape[0]
    if window is None:
        window = cv2.createHanningWindow((S, S), cv2.CV_32F)
    
    # Windowed patch
    Pw = image_patch * window
    
    # FFT
    F = np.fft.fftshift(np.fft.fft2(Pw))
    mag = np.abs(F)
    phase = np.angle(F)
    cy, cx = S//2, S//2
    
    # Frequency bins (precomputed from structure_params)
    k_fin = structure_params['k_fin_bin']    # e.g., 25 for S=100 (f=0.25)
    k_gate = structure_params['k_gate_bin']  # e.g., 3 for S=100 (f=0.03)
    
    # Phase components (sin/cos for continuity)
    phi_fin = phase[cy, cx + k_fin]
    phi_gate = phase[cy + k_gate, cx]
    alpha = structure_params['alpha']  # Λ_gate/Λ_fin ≈ 7.5
    
    delta_phi = phi_fin - alpha * phi_gate
    delta_phi = (delta_phi + np.pi) % (2*np.pi) - np.pi  # wrap to [-π, π]
    
    # Topological moments
    moments = compute_topo_moments(Pw, min_blob_px=2, max_blob_px=S//10)
    
    # Assemble
    desc = np.array([
        np.cos(phi_fin), np.sin(phi_fin),
        np.cos(phi_gate), np.sin(phi_gate),
        np.cos(delta_phi), np.sin(delta_phi),
        *moments  # 5 values: M10, M01, M20, M02, M11 (normalized)
    ], dtype=np.float32)
    
    # Normalize
    return desc / (np.linalg.norm(desc) + 1e-8)


def ssjd_rerank_candidates(reference_img, search_img, candidates, nominal_downsample):
    """Drop-in replacement for V4 phase reranker."""
    from localizer import build_nominal_template
    
    # Build search-res reference template (same as V2 Stage 1)
    templ, S = build_nominal_template(reference_img, nominal_downsample)
    
    # Detect structure from reference
    struct = detect_structure_type(templ)
    
    # Reference descriptor
    D_ref = compute_ssjd(templ, struct)
    
    # Precompute window
    window = cv2.createHanningWindow((S, S), cv2.CV_32F)
    
    # Score each candidate
    srch_dn = prep_search(search_img)  # from phase_reranker
    
    for c in candidates:
        patch = extract_patch(c['x'], c['y'], S, srch_dn)
        D_cand = compute_ssjd(patch, struct, window)
        c['ssjd_score'] = float(np.dot(D_ref, D_cand))  # cosine sim (both normalized)
    
    candidates.sort(key=lambda c: c['ssjd_score'], reverse=True)
    return candidates
```

---

## 9. Validation Plan

1. **Unit tests**: 
   - `Δφ_joint` is invariant to pure lattice translations (m·Λ_fin, n·Λ_gate)
   - `Δφ_joint` varies for off-lattice shifts
   - Topological moments stable under noise

2. **Ablation on 40-pair benchmark**:
   - SSJD rerank vs V2 baseline (expect +5-10% FinFET, no DRAM regression)
   - SSJD coarse gen on finfet_017/023 (expect GT recovery)
   - Per-component ablation: phase-only vs topology-only vs joint

3. **Cross-style generalization**:
   - Auto-detect FinFET/DRAM/mixed_logic on held-out samples
   - Verify no parameter tuning per style

4. **Runtime benchmark**:
   - Confirm <100ms/pair overhead for rerank mode

---

## 10. Claim for Submission

> **Algorithmic Contribution**: The Semiconductor-Specific Joint Descriptor (SSJD) is, to our knowledge, the first localization descriptor that explicitly encodes the **fixed relative phase between incommensurate spatial frequencies** inherent to multi-frequency semiconductor structures (FinFET fin/gate pitches), augmented with **local topological moments** of process-induced landmarks. It breaks the translation invariance that defeats intensity NCC on periodic lattices, operates at the native search resolution (10× downsampled), generalizes across DRAM/FinFET/mixed_logic via automatic spectral detection, and integrates as a <100ms drop-in reranker or adaptive coarse candidate generator within the existing V2 pipeline.

---

## Appendix: Frequency Bin Calculation

For a template of size `S×S` at search resolution:
- Bin index `k` corresponds to frequency `f = k/S` cycles/pixel
- Fin pitch Λ_fin ~ 4px → `f_fin = 1/4 = 0.25` → `k_fin = 0.25 * S`
  - For S=100: `k_fin = 25` (matches observed peak at dx=±26)
- Gate pitch Λ_gate ~ 30px → `f_gate = 1/30 ≈ 0.033` → `k_gate = 0.033 * S`
  - For S=100: `k_gate = 3.3` → bins 3, 4 (matches observed peak at dy=±33 in 1000px = ±3.3 in 100px)

**Critical**: Use the **reference template's own spectrum** to find exact bin indices (auto-calibrated per sample), not hardcoded values.