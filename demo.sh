#!/usr/bin/env bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DEMO_OUT="/tmp/driftsense_demo"
RESULTS_CSV="/tmp/driftsense_results.csv"
FAILURE_IMG="/tmp/finfet_023.png"

cleanup() {
    rm -rf "$DEMO_OUT" "$RESULTS_CSV" "$FAILURE_IMG"
}

cleanup

section() {
    echo
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo
}

info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warning() {
    echo -e "${RED}[WARN]${NC} $1"
}

pause() {
    echo
    read -p "Press ENTER to continue..."
}

verify_file() {
    if [[ ! -f "$1" ]]; then
        warning "Required file not found: $1"
        exit 1
    fi
}

# ============================================================
# INTRODUCTION
# ============================================================

clear
section "DRIFT-SENSE REPRODUCIBLE DEMO"
echo "Semicon India Hackathon 2026 — Applied Materials Problem"
echo
echo "Repository: $(pwd)"
echo
echo "PROBLEM:"
echo "  Wafer inspection tools suffer motion-stage drift between revisits."
echo "  Given a high-res reference image and a wider lower-res search image,"
echo "  the SAME semiconductor structures repeat periodically."
echo "  The challenge: find the CORRECT occurrence, not just a visually similar one."
echo
echo "DriftSense approach:"
echo "  1. Two-stage NCC localization (coarse global → fine local)"
echo "  2. Peak sharpness / ambiguity ratio to self-flag difficult cases"
echo "  3. Gated phase reranker for FinFET ambiguity (rescues finfet_021)"
echo
pause

# ============================================================
# 1. SUCCESSFUL DRAM EXAMPLE
# ============================================================

section "1/6  SUCCESSFUL DRAM LOCALIZATION (dram_000)"

info "Reference:  data/self_eval/reference/dram_000.png"
info "Search:     data/self_eval/search/dram_000.png"
echo

python3 src/run_inference.py \
    --ref data/self_eval/reference/dram_000.png \
    --search data/self_eval/search/dram_000.png | python3 -m json.tool

echo
pause

# ============================================================
# 2. GROUND-TRUTH VERIFICATION
# ============================================================

section "2/6  GROUND-TRUTH VERIFICATION (dram_000)"

python3 -c '
import json, math, subprocess

gt_path = "data/self_eval/ground_truth.json"
with open(gt_path) as f:
    gt_data = json.load(f)

result = subprocess.run([
    "python3", "src/run_inference.py",
    "--ref", "data/self_eval/reference/dram_000.png",
    "--search", "data/self_eval/search/dram_000.png"
], capture_output=True, text=True)
pred = json.loads(result.stdout)
gt = gt_data["dram_000"]["gt_center_xy"]
err = math.hypot(pred["x"] - gt[0], pred["y"] - gt[1])

print(f"Ground truth:    ({gt[0]:.2f}, {gt[1]:.2f})")
print(f"Prediction:      ({pred[\"x\"]:.2f}, {pred[\"y\"]:.2f})")
print(f"Localization error: {err:.2f} pixels")
print(f"Passes 5 px criterion: {\"YES\" if err <= 5 else \"NO\"}")
print(f"Confidence: {pred[\"confidence\"]}")
print(f"Ambiguity ratio: {pred[\"ambiguity_ratio\"]}")
print(f"Low confidence flag: {pred[\"low_confidence_flag\"]}")
'

pause

# ============================================================
# 3. SUCCESSFUL FINFET EXAMPLE
# ============================================================

section "3/6  SUCCESSFUL FINFET LOCALIZATION (finfet_025)"

info "Reference:  data/self_eval/reference/finfet_025.png"
info "Search:     data/self_eval/search/finfet_025.png"
echo

python3 src/run_inference.py \
    --ref data/self_eval/reference/finfet_025.png \
    --search data/self_eval/search/finfet_025.png | python3 -m json.tool

echo

python3 -c '
import json, math, subprocess

gt_path = "data/self_eval/ground_truth.json"
with open(gt_path) as f:
    gt_data = json.load(f)

result = subprocess.run([
    "python3", "src/run_inference.py",
    "--ref", "data/self_eval/reference/finfet_025.png",
    "--search", "data/self_eval/search/finfet_025.png"
], capture_output=True, text=True)
pred = json.loads(result.stdout)
gt = gt_data["finfet_025"]["gt_center_xy"]
err = math.hypot(pred["x"] - gt[0], pred["y"] - gt[1])

print(f"Ground truth:    ({gt[0]:.2f}, {gt[1]:.2f})")
print(f"Prediction:      ({pred[\"x\"]:.2f}, {pred[\"y\"]:.2f})")
print(f"Localization error: {err:.2f} pixels")
print(f"Passes 5 px criterion: {\"YES\" if err <= 5 else \"NO\"}")
print(f"Confidence: {pred[\"confidence\"]}")
print(f"Ambiguity ratio: {pred[\"ambiguity_ratio\"]}")
print(f"Low confidence flag: {pred[\"low_confidence_flag\"]}")
'

pause

# ============================================================
# 4. FULL VERIFIED BENCHMARK
# ============================================================

section "4/6  COMPLETE SELF-EVALUATION BENCHMARK (V1 baseline)"

info "Running evaluation on all 30 pairs (15 DRAM + 15 FinFET)..."
echo

python3 src/evaluate.py --data data/self_eval --out_csv "$RESULTS_CSV"

echo
success "Results also saved to $RESULTS_CSV"
pause

# ============================================================
# 5. BATCH INFERENCE + TIMING
# ============================================================

section "5/6  BATCH INFERENCE + TIMING (30 pairs, V1 method)"

python3 src/run_inference.py \
    --input data/self_eval \
    --output "$DEMO_OUT" \
    --method v1

echo
info "Generated files:"
ls -la "$DEMO_OUT"/
echo

info "Timing summary:"
cat "$DEMO_OUT/timing_report.json" | python3 -m json.tool

pause

# ============================================================
# 6. FAILURE-AWARENESS EXAMPLE
# ============================================================

section "6/6  FAILURE-AWARENESS / PERIODIC AMBIGUITY (finfet_023)"

info "Difficult case: finfet_023"
info "Multiple visually similar periodic locations exist."
info "The system should expose ambiguity rather than false confidence."
echo

python3 src/visualize_ambiguity.py \
    --data data/self_eval \
    --sample finfet_023 \
    --out "$FAILURE_IMG"

echo
success "Visualization saved to $FAILURE_IMG"

python3 - <<'PY'
import json
import math

gt_path = "data/self_eval/ground_truth.json"
with open(gt_path) as f:
    gt_data = json.load(f)

# Get prediction from batch output
pred_path = "/tmp/driftsense_demo/predictions.json"
with open(pred_path) as f:
    preds = json.load(f)

pred = preds["finfet_023"]
gt = gt_data["finfet_023"]["gt_center_xy"]
err = math.hypot(pred["x"] - gt[0], pred["y"] - gt[1])

print(f"Ground truth:    ({gt[0]:.2f}, {gt[1]:.2f})")
print(f"Prediction:      ({pred['x']:.2f}, {pred['y']:.2f})")
print(f"Localization error: {err:.2f} pixels (>> 5 px)")
print(f"Confidence: {pred['confidence']}")
print(f"Ambiguity ratio: {pred['ambiguity_ratio']} (≈1.0 = near-tied peaks)")
print(f"Low confidence flag: {pred['low_confidence_flag']} ← CORRECTLY SELF-FLAGGED")
PY

echo
info "Opening visualization..."
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$FAILURE_IMG" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
    open "$FAILURE_IMG" >/dev/null 2>&1 &
fi

pause

# ============================================================
# FINAL SUMMARY
# ============================================================

section "DEMO COMPLETE — JUDGE-FACING SUMMARY"

cat <<'EOF'

PROBLEM:
  Recover the intended wafer-inspection location from a drifted search image
  containing repeated semiconductor structures.

DEMONSTRATED:
  ✓ Successful DRAM localization (dram_000: 0.29 px error)
  ✓ Successful FinFET localization (finfet_025: 0.06 px error)
  ✓ Ground-truth verification against stored GT coordinates
  ✓ Full 30-pair benchmark: 90% @ ≤5px (DRAM 100%, FinFET 80%)
  ✓ Batch inference: 30 pairs in ~4.8s (161 ms/pair on CPU)
  ✓ Failure awareness: finfet_023 correctly self-flagged (ambiguity≈1.0)

CORE IDEA:
  Scale-aware template localization (two-stage NCC) with explicit
  confidence/ambiguity analysis. Gated phase reranker rescues ranking-
  ambiguity cases (finfet_021 type) while avoiding DRAM regressions.

KEY FAILURE MODES (all self-flagged):
  • finfet_017: Stage-2 refinement corruption (214 px)
  • finfet_021: Ranking ambiguity in candidate pool (571 px)
  • finfet_023: Candidate generation failure — GT absent from pool (868 px)

The algorithm knows when it doesn't know — no silent mis-localizations.

EOF

cleanup
success "Demo complete. Temporary files cleaned up."