"""
pattern_synth.py
-----------------
Procedurally generates large-canvas synthetic "die" layouts that mimic the
repeating geometry of DRAM memory arrays and FinFET gate structures.

A single large native-resolution canvas is generated per sample. The
Reference Image is a native-resolution crop taken directly from that
canvas; the Search Image is the *entire* canvas downsampled 10x. Because
the reference crop is pixel-sourced from the same canvas that the search
image is downsampled from, the reference pattern is guaranteed to be
genuinely present (at 10x lower effective resolution) inside the search
image, and its true location is known exactly (this is the ground truth).

Structural choices (line pitch, via/contact placement, fin pitch, gate
crossing geometry) follow standard descriptions of DRAM 1T1C array layout
and FinFET multi-gate transistor layout geometry:

  [1] S. M. Sze & K. K. Ng, "Physics of Semiconductor Devices", 3rd ed.,
      Wiley, 2007 -- DRAM cell-array word-line/bit-line crossing geometry
      and 1T1C storage-node placement.
  [2] International Roadmap for Devices and Systems (IRDS), "More Moore"
      chapter, 2022 -- FinFET fin pitch, gate pitch and multi-fin layout
      conventions used in logic standard cells.
"""

import numpy as np
import cv2


def _dram_grid_vectorized(h, w, pitch, line_width, via_radius, phase_x, phase_y):
    """Fully vectorized (broadcast, no Python loops) periodic DRAM grid:
    horizontal word-lines + vertical bit-lines crossing at right angles,
    with a contact/via dot at every intersection ([1])."""
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]

    y_mod = np.mod(y - phase_y, pitch)
    x_mod = np.mod(x - phase_x, pitch)
    half = line_width / 2.0

    horiz = (y_mod < half) | (y_mod > pitch - half)
    vert = (x_mod < half) | (x_mod > pitch - half)
    lines = horiz | vert

    y_dist = np.minimum(y_mod, pitch - y_mod)
    x_dist = np.minimum(x_mod, pitch - x_mod)
    via = (y_dist * y_dist + x_dist * x_dist) < (via_radius * via_radius)

    img = np.where(lines | via, 1.0, 0.0).astype(np.float32)
    return img


def _finfet_grid_vectorized(h, w, fin_pitch, fin_width, gate_pitch,
                             gate_width, phase_x, phase_y):
    """Fully vectorized dense parallel vertical fins crossed by horizontal
    gate bars, per multi-fin logic layout conventions ([2])."""
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]

    x_mod = np.mod(x - phase_x, fin_pitch)
    fin_half = fin_width / 2.0
    fins = (x_mod < fin_half) | (x_mod > fin_pitch - fin_half)

    y_mod = np.mod(y - phase_y, gate_pitch)
    gate_half = gate_width / 2.0
    gates = (y_mod < gate_half) | (y_mod > gate_pitch - gate_half)

    img = np.zeros((h, w), dtype=np.float32)
    img[np.broadcast_to(fins, (h, w))] = 0.85
    img[np.broadcast_to(gates, (h, w))] = 1.0
    return img


def _mixed_logic_grid_vectorized(h, w, row_pitch, rng):
    """Irregular standard-cell logic layout: rows of alternating-width
    rectangular cell blocks (unlike the strictly periodic DRAM/FinFET
    grids, cell width varies within a row), following the general
    row-based standard-cell layout convention used in digital logic
    macros ([2]). This is a deliberately different texture statistic from
    DRAM/FinFET, held out entirely from any tuning of the localizer --
    used only to report out-of-distribution generalization, mirroring how
    the hackathon's own hidden test set includes unseen distributions."""
    img = np.zeros((h, w), dtype=np.float32)
    n_rows = int(h / row_pitch) + 2
    row_gap = max(2, int(row_pitch * 0.12))
    for r in range(n_rows):
        y0 = int(r * row_pitch)
        y1 = int(min(h, y0 + row_pitch - row_gap))
        if y0 >= h:
            break
        x = 0
        toggle = bool(rng.integers(0, 2))
        while x < w:
            cell_w = int(rng.integers(14, 42))
            val = 0.95 if toggle else 0.15
            img[y0:y1, x:min(w, x + cell_w)] = val
            toggle = not toggle
            x += cell_w
    return img


def synth_canvas(style, canvas_size, seed):
    """Generate one large synthetic die-layout canvas at native resolution.

    style: 'dram', 'finfet', or 'mixed_logic'
    canvas_size: (h, w)
    seed: int, controls structural jitter/phase for this die instance
    """
    rng = np.random.default_rng(seed)
    h, w = canvas_size
    phase_x = rng.uniform(0, 1000)
    phase_y = rng.uniform(0, 1000)

    if style == "dram":
        # pitch/line-width chosen so the periodic signal survives the 10x
        # area-average downsample used to build the Search Image (a duty
        # cycle that is too low vanishes into the noise floor after
        # downsampling -- see docs/design_notes.md).
        pitch = float(rng.integers(38, 48))
        img = _dram_grid_vectorized(
            h, w, pitch=pitch, line_width=pitch * 0.28,
            via_radius=pitch * 0.22, phase_x=phase_x, phase_y=phase_y,
        )
    elif style == "finfet":
        fin_pitch = float(rng.integers(36, 46))
        gate_pitch = float(rng.integers(260, 340))
        img = _finfet_grid_vectorized(
            h, w, fin_pitch=fin_pitch, fin_width=fin_pitch * 0.3,
            gate_pitch=gate_pitch, gate_width=gate_pitch * 0.12,
            phase_x=phase_x, phase_y=phase_y,
        )
    elif style == "mixed_logic":
        row_pitch = float(rng.integers(45, 60))
        img = _mixed_logic_grid_vectorized(h, w, row_pitch=row_pitch, rng=rng)
    else:
        raise ValueError(f"unknown style {style}")

    substrate = 0.18 + 0.03 * rng.random()
    img = np.clip(img + substrate * (1 - img), 0, 1)

    # A perfectly periodic pattern is, by construction, indistinguishable
    # from any translate of itself by an integer number of pitches -- the
    # true match location would not actually be unique. Real wafers are
    # never perfectly periodic at large scale: local charging/contrast
    # drift and rare process-induced feature irregularities (dust, missing
    # via, grain-boundary contrast) break the symmetry, which is exactly
    # what lets a real navigation-error-recovery algorithm (and this one)
    # disambiguate between otherwise-identical periodic sites. We inject
    # both effects so every location in the canvas is locally unique:
    #   - a smooth low-frequency brightness field, representing non-uniform
    #     secondary-electron yield / detector gain / charging drift across
    #     the field of view ([3],[9]);
    #   - sparse small random defects (missing via / bright particle),
    #     representing realistic process defects and surface contamination.
    #
    #   [9] "Sample surface structure measuring method" (SEM patent
    #       disclosure) -- SE brightness varies with local beam-incidence
    #       geometry, so absolute brightness is not translation-invariant
    #       across a real sample even for a geometrically periodic layout.
    field_lores = rng.normal(0, 1, size=(24, 24)).astype(np.float32)
    field = cv2.resize(field_lores, (w, h), interpolation=cv2.INTER_CUBIC)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    img = np.clip(img + 0.16 * (field - 0.5), 0, 1)

    n_defects = int((h * w) / (55 * 55))
    dxs = rng.integers(0, w, size=n_defects)
    dys = rng.integers(0, h, size=n_defects)
    radii = rng.integers(2, 7, size=n_defects)
    signs = rng.choice([-1.0, 1.0], size=n_defects)
    for dx, dy, r, s in zip(dxs, dys, radii, signs):
        cv2.circle(img, (int(dx), int(dy)), int(r), float(np.clip(
            (0.6 if s > 0 else 0.05), 0, 1)), thickness=-1)

    return img.astype(np.float32)


def apply_edge_brightening(img, gain=0.6, ksize=5):
    """Mimic the real SEM secondary-electron edge effect: excess SEs are
    generated where the beam interaction volume intersects a topographic
    edge, producing bright rims around features ([3],[4],[5]).

      [3] ETH Zurich, Electron Microscopy teaching notes, "Secondary
          Electron Imaging" -- edge effect defined as increased SE escape
          probability near edges, causing increased brightness there.
      [4] Nanoscience Instruments, "Secondary Electrons in SEM: Unlocking
          Surface Insights at the Nanoscale" -- edge enhancement / brighter
          edges when the primary beam interacts with an edge.
      [5] St. Cloud State Univ. Center for Microscopy & Imaging, "SEM A to
          Z: Basic Knowledge for Using the SEM" -- edge effect: edges of
          steps/protrusions appear bright with a characteristic width.
    """
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    edge_mag = cv2.magnitude(grad_x, grad_y)
    edge_mag = edge_mag / (edge_mag.max() + 1e-6)
    brightened = np.clip(img + gain * edge_mag, 0, 1)
    return brightened.astype(np.float32)


def apply_sensor_noise(img, rng, dose_scale=1.0):
    """Physically-motivated mixed Poisson-Gaussian SEM noise model.

    SEM signal-to-noise is dominated by Poisson-distributed electron shot
    noise from the primary/secondary electron emission process, with an
    additional signal-independent Gaussian contribution from detector /
    amplifier electronics ([6],[7],[8]).

      [6] "Poisson shot noise parameter estimation from a single scanning
          electron microscopy image" -- SEM noise is Poisson shot noise
          from electron production, plus additive white Gaussian noise
          (AWGN) from detection electronics.
      [7] Mulapudi & Joy (2003); Cizmar et al. (2008), summarized in
          "Scanning Electron Microscope Image SNR Monitoring" -- final SEM
          image noise modeled as Poisson (electron emission) + Gaussian
          (readout); Gaussian is a good approximation to Poisson at high
          mean signal.
      [8] "M-Denoiser: Unsupervised image denoising for real-world optical
          and electron microscopy data" -- microscopy noise = signal-
          dependent shot noise (Poisson) + signal-independent read noise
          (Gaussian).

    A fresh, independently-seeded `rng` must be passed per image so the
    reference and search images never share a noise realization.
    """
    peak = 220.0 * dose_scale  # effective electron-count scale -> controls SNR
    poisson_component = rng.poisson(np.clip(img, 0, 1) * peak).astype(np.float32) / peak
    read_noise_sigma = 0.02 / np.sqrt(dose_scale)
    gaussian_component = rng.normal(0, read_noise_sigma, size=img.shape).astype(np.float32)
    noisy = poisson_component + gaussian_component
    # Intentionally NOT clipped to [0,1]: consistent with the KLA webinar's
    # dataset note that noisy/search-side images may fall slightly outside
    # [0,1] due to additive read noise -- a feature of real sensor data,
    # not a bug, and downstream algorithms should account for it.
    return noisy


def apply_geometric_degradation(img, rng, max_blur_sigma=1.2,
                                 max_rot_deg=5.0, scale_jitter=0.05,
                                 return_matrix=False):
    """Blur + small rotation + small scale jitter, applied independently to
    reference and search so the pair is not related by a pure crop+resize
    (mandatory requirement: realistic blur / rotation / scaling
    variation).

    If return_matrix=True, also returns the 2x3 affine matrix used, so
    callers that need to track a specific point's true location through
    the transform (e.g. the dataset builder updating ground truth after
    warping the full search image) can do so exactly.
    """
    h, w = img.shape[:2]
    sigma = float(rng.uniform(0.15, max_blur_sigma))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

    angle = float(rng.uniform(-max_rot_deg, max_rot_deg))
    scale = 1.0 + float(rng.uniform(-scale_jitter, scale_jitter))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    warped = cv2.warpAffine(blurred, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    if return_matrix:
        return warped, M
    return warped


def transform_point(x, y, M):
    """Apply a 2x3 affine matrix M to a point (x, y)."""
    vec = np.array([x, y, 1.0])
    tx = M[0, 0] * vec[0] + M[0, 1] * vec[1] + M[0, 2]
    ty = M[1, 0] * vec[0] + M[1, 1] * vec[1] + M[1, 2]
    return float(tx), float(ty)
