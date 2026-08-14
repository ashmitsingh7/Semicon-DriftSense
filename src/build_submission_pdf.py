"""
build_submission_pdf.py
------------------------
Assembles the idea-submission PDF required by the hackathon form
(Round 1: problem understanding, approach, tech stack, results).
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, HRFlowable,
)

# Use relative paths from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "submission", "DriftSense_Submission.pdf")
FIGDIR = os.path.join(REPO_ROOT, "docs", "figures")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"],
                           fontSize=16, spaceAfter=10, spaceBefore=4,
                           textColor=colors.HexColor("#1a3a5c")))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"],
                           fontSize=12.5, spaceAfter=6, spaceBefore=14,
                           textColor=colors.HexColor("#1a3a5c")))
styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"],
                           fontSize=10.2, leading=14.5, spaceAfter=8))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"],
                           fontSize=8.5, leading=11.5, textColor=colors.grey))
styles.add(ParagraphStyle(name="Cover", parent=styles["Title"],
                           fontSize=22, leading=26,
                           textColor=colors.HexColor("#1a3a5c")))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["BodyText"],
                           fontSize=12, alignment=1, textColor=colors.grey,
                           spaceAfter=6))
styles.add(ParagraphStyle(name="CoverInfo", parent=styles["Body"], alignment=1, fontSize=11))

story = []

# ---------------- Cover ----------------
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("Drift-Sense", styles["Cover"]))
story.append(Paragraph("AI-Powered Navigation-Error Recovery for Wafer Inspection Tools",
                        styles["CoverSub"]))
story.append(Spacer(1, 0.15 * inch))
story.append(Paragraph("Semicon India Hackathon 2026 — Applied Materials Problem Statement",
                        styles["CoverSub"]))
story.append(Spacer(1, 0.5 * inch))
story.append(HRFlowable(width="60%", thickness=1, color=colors.HexColor("#1a3a5c"),
                         hAlign="CENTER"))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph(
    "Team: [TEAM NAME]<br/>Members: [NAMES]<br/>"
    "Institution: Vellore Institute of Technology (VIT)<br/>"
    "GitHub repository: [INSERT REPO LINK]<br/>"
    "Video walkthrough: [INSERT VIDEO LINK]",
    styles["CoverInfo"]))
story.append(PageBreak())

# ---------------- 1. Problem understanding ----------------
story.append(Paragraph("1. Problem Understanding", styles["H1"]))
story.append(Paragraph(
    "A wafer inspection tool must return to the exact same die site thousands of times a "
    "day, with measurements comparable across visits and across tools. In practice, motion "
    "stages accumulate small errors between visits &mdash; thermal expansion, vibration, and "
    "mechanical slack &mdash; so a revisit can land the tool several pixels away from the "
    "intended site. Because every die on a wafer carries the same repeating circuit layout, "
    "the landed image looks almost identical to the correct one: the core difficulty is not "
    "recognizing the pattern, but pinpointing <i>which</i> occurrence of a highly repetitive "
    "pattern is the correct one.", styles["Body"]))
story.append(Paragraph(
    "<b>Formal task:</b> given a Reference Image (the known-correct site, native resolution) "
    "and a Search Image (a ~10x lower-magnification view captured around where the tool "
    "actually landed), output the pixel center (x, y) in the Search Image where the reference "
    "pattern appears, choosing the match closest to the image center if more than one region "
    "qualifies.", styles["Body"]))
story.append(Paragraph(
    "No dataset is provided for this problem statement &mdash; generating a realistic, "
    "citation-backed synthetic dataset is itself graded (30% of the score per the problem "
    "statement) alongside the localization accuracy and inference throughput.", styles["Body"]))

# ---------------- 2. Approach & tech stack ----------------
story.append(Paragraph("2. Approach & Technology Stack", styles["H1"]))
story.append(Paragraph(
    "The submission has two parts: a synthetic dataset generator that produces realistic, "
    "physically-grounded DRAM-style and FinFET-style die imagery with recorded ground truth, "
    "and a template-localization algorithm that solves the stated task on that data.",
    styles["Body"]))
story.append(Paragraph("Tech stack:", styles["H2"]))
story.append(ListFlowable([
    ListItem(Paragraph("Python 3.12, NumPy, OpenCV (cv2) for all image synthesis, "
                        "degradation, and matching &mdash; no GPU/training dependency required "
                        "to run end to end.", styles["Body"])),
    ListItem(Paragraph("Matplotlib for result visualization (correlation-surface heatmaps, "
                        "ablation plots).", styles["Body"])),
    ListItem(Paragraph("Fully vectorized (broadcast, no per-pixel Python loops) pattern "
                        "generation for tractable runtime at 10,000x10,000 native-resolution "
                        "canvases.", styles["Body"])),
], bulletType="bullet"))

story.append(Paragraph("2.1 Dataset generation", styles["H2"]))
story.append(Paragraph(
    "Each sample starts from one large native-resolution synthetic die canvas. The Reference "
    "Image is a native-resolution crop from that canvas; the Search Image is the <i>entire</i> "
    "canvas downsampled 10x. Because the reference crop is sourced from the same canvas the "
    "search image is downsampled from, the reference pattern's presence inside the search "
    "image &mdash; and its exact location &mdash; is true by construction rather than asserted, "
    "which also makes the ground truth verifiably correct.", styles["Body"]))
story.append(Paragraph(
    "Two hackathon-required styles are generated: <b>DRAM-style</b> (word-lines &times; "
    "bit-lines crossing at right angles with a via/contact dot at every intersection, per "
    "standard 1T1C DRAM cell-array conventions) and <b>FinFET-style</b> (dense parallel fins "
    "crossed by gate bars, per standard multi-fin logic layout conventions). A third style, "
    "<b>mixed_logic</b> (irregular row-based standard-cell blocks), was built purely as a "
    "held-out generalization check and was never used to tune the algorithm &mdash; see "
    "Section 4.", styles["Body"]))
story.append(Paragraph(
    "A perfectly periodic pattern is mathematically ambiguous under translation by any "
    "multiple of its pitch &mdash; we found this directly, as an early version of the "
    "generator produced a dataset with literally zero localization signal beyond one period. "
    "Real wafers are never perfectly periodic at the SEM-charging/contamination scale even "
    "when the drawn layout is, so every canvas also receives a smooth low-frequency "
    "brightness field (non-uniform secondary-electron yield / charging drift), sparse "
    "background defects, and &mdash; guaranteed inside every reference footprint &mdash; a "
    "few larger high-contrast landmark features sized to survive the 10x downsample. This is "
    "what makes each site locally unique and the task solvable, while intentionally leaving "
    "some sites genuinely hard (the problem statement's own requirement: \"include at least "
    "one highly periodic array region where correct localization is genuinely difficult\").",
    styles["Body"]))
story.append(Paragraph(
    "Reference and Search images are degraded <b>independently</b> (separate RNGs per image, "
    "never a shared noise realization): Gaussian blur, small rotation/scale jitter "
    "(representing residual stage drift), a physically-motivated SEM edge-brightening pass, "
    "and a mixed Poisson-Gaussian sensor-noise model. The Search side uses a lower effective "
    "electron dose than the Reference side, so it is measurably noisier, matching the stated "
    "test-time behavior. Search-image pixel values are intentionally left unclipped outside "
    "[0, 1] after adding read noise, matching the KLA webinar's own note that this is a "
    "\"feature of the dataset, not a bug.\"", styles["Body"]))

story.append(Paragraph("2.2 Localization algorithm", styles["H2"]))
story.append(Paragraph(
    "The localizer uses a two-stage multi-scale, small-rotation <b>normalized "
    "cross-correlation</b> (NCC) approach: (1) one coarse, global NCC pass at nominal scale "
    "and zero rotation over the entire Search Image, which also yields an ambiguity signal "
    "(peak score vs. best score elsewhere in the whole image); then (2) a fine pass, "
    "restricted to a small window around the coarse peak, that searches a grid of 5 candidate "
    "scales &times; 7 candidate rotations around the nominal pose, with parabolic sub-pixel "
    "peak refinement. Confining the expensive grid search to a small local window (rather than "
    "the full image) is the single largest lever on inference throughput.", styles["Body"]))
story.append(Paragraph(
    "The algorithm also reports an <b>ambiguity ratio</b> and a <b>low-confidence flag</b>, "
    "directly targeting the \"failure-mode awareness\" requirement in the problem statement: "
    "when a site really is periodic/ambiguous, the algorithm should say so rather than "
    "silently returning a wrong answer.", styles["Body"]))

story.append(PageBreak())

# ---------------- 3. Results ----------------
story.append(Paragraph("3. Self-Evaluation Results", styles["H1"]))
story.append(Paragraph(
    "Evaluated on a freshly-generated 30-pair self-eval set (15 DRAM-style, 15 FinFET-style), "
    "against recorded ground truth:", styles["Body"]))

data = [
    ["Metric", "Value"],
    ["Success rate (error ≤ 5 px)", "90.0% (27/30)"],
    ["Median error on successful matches", "0.10 px"],
    ["DRAM-style success rate", "100% (15/15)"],
    ["FinFET-style success rate", "80% (12/15)"],
    ["Misses correctly self-flagged low-confidence", "3/3 (100%)"],
    ["Held-out mixed_logic style (never tuned on)", "100% (10/10), median error 0.06 px"],
    ["End-to-end throughput (CPU, sandbox)", "~230 ms/sample, ~4.3 samples/s"],
    ["Two-stage search speed-up vs. brute-force grid", "21.9s → 7.2s on 30 pairs (~3x), same accuracy"],
]
tbl = Table(data, colWidths=[3.4 * inch, 2.6 * inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.3),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(tbl)
story.append(Spacer(1, 12))
story.append(Paragraph(
    "The three misses are all FinFET cases where the correlation surface has many near-tied "
    "periodic peaks &mdash; a genuine, by-design difficulty, not a bug. All three are correctly "
    "flagged as low-confidence by the ambiguity ratio. Figure 1 shows the worst case "
    "(finfet_023, error 868px) directly: the reference pattern, the search image with "
    "ground-truth vs. predicted location marked, and the raw NCC correlation surface showing "
    "the periodic banding responsible for the ambiguity.", styles["Body"]))

fig1 = os.path.join(FIGDIR, "ambiguity_finfet_023.png")
if os.path.exists(fig1):
    story.append(Spacer(1, 8))
    story.append(Image(fig1, width=6.4 * inch, height=6.4 * inch * (692 / 2050)))
    story.append(Paragraph(
        "Figure 1. Failure-mode visualization for the hardest self-eval case. Ambiguity ratio "
        "≈ 1.0 (essentially tied peaks) is correctly flagged as low-confidence rather than "
        "silently mis-reported.", styles["Small"]))

story.append(Spacer(1, 14))
story.append(Paragraph(
    "The held-out <b>mixed_logic</b> style &mdash; a third layout the algorithm was never "
    "adjusted for &mdash; scores <i>higher</i> (100%) than the tuned DRAM/FinFET styles, "
    "because its irregular, non-periodic structure carries inherently less positional "
    "ambiguity. This is presented as evidence of generalization rather than overfitting to "
    "one synthetic distribution, mirroring how the hackathon's own hidden test set will mix "
    "in-distribution and out-of-distribution samples.", styles["Body"]))

story.append(Paragraph("3.1 Robustness sweep", styles["H2"]))
story.append(Paragraph(
    "Beyond the single fixed self-eval dataset, we swept sensor-noise severity and rotation-"
    "drift severity beyond the main dataset's settings to characterize the operating envelope "
    "of the approach (Figure 2). Success degrades gradually under increased noise; degradation "
    "under rotation is milder because the guaranteed landmark features used for disambiguation "
    "are themselves rotation-invariant blobs.", styles["Body"]))

fig2 = os.path.join(FIGDIR, "ablation.png")
if os.path.exists(fig2):
    story.append(Spacer(1, 8))
    story.append(Image(fig2, width=6.4 * inch, height=6.4 * inch * (572 / 1525)))
    story.append(Paragraph(
        "Figure 2. Success rate vs. sensor-noise severity (left) and rotation-drift severity "
        "(right), generated on a smaller companion canvas for tractable sweep runtime.",
        styles["Small"]))

story.append(PageBreak())

# ---------------- 4. Evaluation-criteria mapping ----------------
story.append(Paragraph("4. Mapping to Stated Evaluation Criteria", styles["H1"]))
story.append(Paragraph(
    "The problem statement scores submissions on three axes: accuracy, throughput, and "
    "training/code hygiene. This is how the submission addresses each:", styles["Body"]))
story.append(ListFlowable([
    ListItem(Paragraph("<b>Accuracy</b> &mdash; 90% localization success with sub-pixel "
                        "median error on hits; explicit in-distribution vs. out-of-"
                        "distribution reporting (DRAM/FinFET vs. held-out mixed_logic).",
                        styles["Body"])),
    ListItem(Paragraph("<b>Throughput</b> &mdash; a dedicated directory-in/directory-out "
                        "inference script (<code>run_inference.py</code>) with a measured "
                        "timing report; a two-stage coarse-then-local search structure that "
                        "gave a measured 3x speedup with no accuracy loss. GPU (H100) "
                        "profiling is flagged as not yet done rather than claimed.",
                        styles["Body"])),
    ListItem(Paragraph("<b>Training/code hygiene</b> &mdash; deterministic, seeded, "
                        "reproducible dataset generation; no hidden state; every "
                        "augmentation/physics choice backed by a specific citation in "
                        "<code>docs/design_notes.md</code>; a documented failure case found "
                        "and fixed during development (ground truth not tracked through a "
                        "geometric transform) is described honestly rather than omitted.",
                        styles["Body"])),
], bulletType="bullet"))

# ---------------- 5. Limitations ----------------
story.append(Paragraph("5. Known Limitations & Next Steps", styles["H1"]))
story.append(ListFlowable([
    ListItem(Paragraph("The NCC baseline is a simple, fast, fully-explainable starting point "
                        "with no training dependency; it can be beaten on the hardest periodic "
                        "sub-regions by approaches using more global context (e.g. a learned "
                        "embedding trained with a contrastive/triplet loss on this same "
                        "generator).", styles["Body"])),
    ListItem(Paragraph("Throughput has been measured and optimized on CPU only; "
                        "<code>cv2.matchTemplate</code>/<code>cv2.warpAffine</code> have "
                        "direct <code>cv2.cuda</code> equivalents, expected to give further "
                        "speedup, not yet independently verified.", styles["Body"])),
    ListItem(Paragraph("Noise/edge-effect parameters are citation-backed but validation "
                        "against a real public-domain SEM image was started (candidates "
                        "identified on Wikimedia Commons) and not completed before "
                        "submission.", styles["Body"])),
], bulletType="bullet"))

story.append(Spacer(1, 16))
story.append(Paragraph(
    "Full code, the generated self-eval dataset, ground truth, figures, and this document "
    "are available at: <b>[INSERT GITHUB REPO LINK]</b>", styles["Body"]))

doc = SimpleDocTemplate(OUT, pagesize=letter,
                         topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                         leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                         title="Drift-Sense Submission")
doc.build(story)
print(f"Wrote {OUT}")
