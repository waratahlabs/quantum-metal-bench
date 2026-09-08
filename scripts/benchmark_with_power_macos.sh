#!/usr/bin/env bash
# Runs the Metal (qe-bench) benchmark while sampling real power draw via
# powermetrics, and combines both into one JSON result.
#
# powermetrics requires sudo and cannot be run non-interactively without a
# cached credential — this script must be run directly by a human in a
# real terminal (`./scripts/benchmark_with_power_macos.sh`), not from an
# automated/headless context.
#
# 2026-09-06 addition: added the `tasks` sampler so the log includes a
# per-process CPU breakdown (a first real run showed all 10 CPU cores
# simultaneously >20% active residency for 95% of the benchmark window --
# far more than this single-threaded Metal-dispatch binary should need --
# suggesting background process contention, not confirmed which process).
#
# 2026-09-06, same day: `--show-process-gpu` (added alongside `tasks` in
# the same commit, flagged as unverified at the time) crashed powermetrics
# with SIGABRT on a real run, right as it started printing the "Running
# tasks" table -- one line in (a DEAD_TASKS catch-all bucket, not a real
# process name), then aborted. Removed. `tasks` alone is untested against
# real hardware as of this comment -- if it also crashes, drop it and
# fall back to `--samplers cpu_power,gpu_power` (the combination that has
# run cleanly twice) and check Activity Monitor manually during a run
# instead.
#
# Validated against a real powermetrics log on 2026-09-06 (macOS
# "26A5425a" / MacBookPro18,1). Two things were fixed after that real run:
#   1. The header line has a second parenthetical group
#      ("*** Sampled system activity (<date>) (208.10ms elapsed) ***") that
#      broke the original timestamp regex — fixed in parse_powermetrics.py.
#   2. The shutdown sequence used `sudo kill "$PID" 2>/dev/null || true`
#      followed by an unbounded `wait`. Redirecting stderr hid a sudo
#      password re-prompt, so `wait` blocked SILENTLY for ~13 minutes
#      (powermetrics kept sampling the whole time, badly diluting the
#      averages) instead of failing fast or succeeding quickly. Rewritten
#      below with a bounded wait, a visible SIGTERM/SIGKILL escalation,
#      and no suppressed sudo prompts.

set -euo pipefail

QUBITS="${1:-28}"
DEPTH="${2:-4}"
REPS="${3:-5}"
SEED="${4:-0}"
SAMPLE_INTERVAL_MS="${5:-200}"

# Chip auto-detection — must not be hardcoded (an m1pro label shipped on an
# M3 once). Override with PLATFORM="..." if detection is wrong.
PLATFORM="metal-$(sysctl -n machdep.cpu.brand_string | sed -E 's/^Apple //; s/ /-/g' | tr '[:upper:]' '[:lower:]')"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QE_BENCH="$REPO_ROOT/metal/.build/release/qe-bench"
RESULTS_DIR="$REPO_ROOT/benchmarks-results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
POWER_LOG="$RESULTS_DIR/power-raw-${TIMESTAMP}.txt"
BENCH_LOG="$RESULTS_DIR/bench-raw-${TIMESTAMP}.json"
COMBINED="$RESULTS_DIR/metal-power-benchmark-${TIMESTAMP}.json"

if [[ ! -x "$QE_BENCH" ]]; then
  echo "error: $QE_BENCH not found or not executable." >&2
  echo "Build it first: (cd \"$REPO_ROOT/metal\" && swift build -c release)" >&2
  exit 1
fi

echo "Requesting sudo up front (needed for powermetrics)..."
sudo -v

echo "Starting powermetrics in the background (sampling every ${SAMPLE_INTERVAL_MS}ms)..."
sudo powermetrics -i "$SAMPLE_INTERVAL_MS" --samplers cpu_power,gpu_power,tasks > "$POWER_LOG" 2>&1 &
POWERMETRICS_PID=$!

# Bounded, visible shutdown — never suppress a sudo prompt, never wait
# unboundedly. This is what a ~13-minute silent hang on 2026-09-06 turned
# out to be caused by: a hidden sudo re-prompt during an unbounded `wait`.
stop_powermetrics() {
  kill -0 "$POWERMETRICS_PID" 2>/dev/null || return 0  # already gone
  echo "Stopping powermetrics (pid $POWERMETRICS_PID)..."
  sudo -v   # refresh credential now, visibly, before we need it
  sudo kill -TERM "$POWERMETRICS_PID" 2>/dev/null || true
  local waited=0
  while kill -0 "$POWERMETRICS_PID" 2>/dev/null && [ "$waited" -lt 10 ]; do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$POWERMETRICS_PID" 2>/dev/null; then
    echo "warning: powermetrics still running after ${waited}s, forcing SIGKILL" >&2
    sudo -v
    sudo kill -KILL "$POWERMETRICS_PID" 2>/dev/null || true
    sleep 1
  fi
  wait "$POWERMETRICS_PID" 2>/dev/null || true
}

cleanup() {
  stop_powermetrics
}
trap cleanup EXIT

# Let powermetrics establish a baseline sample before load starts.
sleep 2

echo "Running qe-bench: qubits=$QUBITS depth=$DEPTH reps=$REPS seed=$SEED"
BENCH_START_EPOCH=$(date +%s)
"$QE_BENCH" bench --circuit qaoa --qubits "$QUBITS" --depth "$DEPTH" --reps "$REPS" --seed "$SEED" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); d.pop('statevector_real',None); d.pop('statevector_imag',None); print(json.dumps(d, indent=2))" \
  | tee "$BENCH_LOG"
BENCH_END_EPOCH=$(date +%s)

# A trailing sample after load ends, then stop (bounded — see above).
sleep 2
stop_powermetrics
trap - EXIT

echo "Parsing power samples from $POWER_LOG..."
python3 "$REPO_ROOT/scripts/parse_powermetrics.py" \
  --power-log "$POWER_LOG" \
  --bench-log "$BENCH_LOG" \
  --start-epoch "$BENCH_START_EPOCH" \
  --end-epoch "$BENCH_END_EPOCH" \
  --platform "$PLATFORM" \
  --out "$COMBINED"

echo "Combined result written to: $COMBINED"
echo "Raw powermetrics log kept at: $POWER_LOG (for manual spot-check if numbers look off)"
