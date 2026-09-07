#!/usr/bin/env bash
# Runs the CUDA (lightning.gpu) benchmark while sampling nvidia-smi power
# draw, and combines both into one JSON result. Unlike the macOS script,
# nvidia-smi doesn't need sudo, so this can run fully unattended.
#
# Written to run directly on a Linux box with `uv`, the quantum_metal_bench
# package, and nvidia-smi on PATH. Verified on bare-metal Linux.
#
# KNOWN CAVEAT: running this inside a WSL2+Docker+nvidia-container-toolkit
# stack has been observed to produce unreliable nvidia-smi power.draw and
# clocks.sm readings — power pinned at idle levels and clocks.sm returning
# physically impossible values (up to 47,385 MHz on a GPU that tops out
# near 2,700). This script's own telemetry-sanity check below will flag
# that failure mode (`telemetry_suspect: true`) if it recurs, but for
# trustworthy power numbers, run this on bare metal rather than inside
# that virtualization stack.

set -euo pipefail

QUBITS="${1:-28}"
DEPTH="${2:-4}"
REPS="${3:-5}"
SEED="${4:-0}"
SAMPLE_INTERVAL_SEC="${5:-0.3}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/benchmarks-results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
POWER_CSV="$RESULTS_DIR/power-raw-cuda-${TIMESTAMP}.csv"
BENCH_LOG="$RESULTS_DIR/bench-raw-cuda-${TIMESTAMP}.json"
COMBINED="$RESULTS_DIR/cuda-power-benchmark-${TIMESTAMP}.json"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found on PATH." >&2
  exit 1
fi

echo "timestamp_epoch,power_draw_w,utilization_gpu_pct,clocks_sm_mhz" > "$POWER_CSV"

sampler() {
  while true; do
    ts=$(date +%s.%N)
    reading=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null || echo "0,0,0")
    echo "${ts},${reading}" >> "$POWER_CSV"
    sleep "$SAMPLE_INTERVAL_SEC"
  done
}

sampler &
SAMPLER_PID=$!
cleanup() { kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

sleep 1  # idle baseline before load

echo "Running CUDA benchmark: qubits=$QUBITS depth=$DEPTH reps=$REPS seed=$SEED"
BENCH_START_EPOCH=$(date +%s.%N)
uv run --project "$REPO_ROOT" qmb single --circuit qaoa --qubits "$QUBITS" --depth "$DEPTH" --reps "$REPS" --backend cuda --seed "$SEED" | tee "$BENCH_LOG"
BENCH_END_EPOCH=$(date +%s.%N)

sleep 1  # trailing idle sample
cleanup
trap - EXIT

echo "Parsing power samples from $POWER_CSV..."
python3 "$REPO_ROOT/scripts/parse_nvidia_smi_csv.py" \
  --power-csv "$POWER_CSV" \
  --bench-log "$BENCH_LOG" \
  --start-epoch "$BENCH_START_EPOCH" \
  --end-epoch "$BENCH_END_EPOCH" \
  --platform "cuda" \
  --out "$COMBINED"

echo "Combined result written to: $COMBINED"
echo "Raw power samples kept at: $POWER_CSV"
echo ""
echo "REMINDER: if this ran inside WSL2/Docker rather than bare metal, treat"
echo "the power/clock numbers as unverified until re-run on bare metal."
