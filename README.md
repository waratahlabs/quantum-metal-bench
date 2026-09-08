# quantum-metal-bench

Cross-platform quantum statevector simulation benchmark: Apple Silicon Metal vs NVIDIA CUDA/cuQuantum vs CPU.

Built by Waratah Labs (which also builds [Quantum Edge](https://apps.apple.com/us/app/quantum-edge/id6770526340)) — see `PRD.md` §2 for the conflict-of-interest disclosure and how this benchmark is designed to survive scrutiny from framework maintainers rather than just make Metal look good.

## Status

CPU (`lightning.qubit`) and Metal backends both working end to end for the QAOA circuit, with real cross-backend fidelity comparison. Clifford+T is CPU-only — neither GPU backend has a cross-language/RNG-parity mirror for it yet (reports `SKIPPED_CIRCUIT_TYPE`, not silently run wrong).

**CUDA (`lightning.gpu`) adapter: bare-metal verified.** Earlier draft runs were over a WSL2/Docker/nvidia-container-toolkit stack, where `nvidia-smi` showed zero measurable GPU utilization/memory/process despite plausible results — a known observability gap in that virtualization stack. Re-run on a real bare-metal Linux box (RTX 5070 Ti, no WSL2/Docker in the path): telemetry now shows real load-correlated power draw (idle ~4W → peak ~76-96W under load) and zero implausible clock readings. Treat these numbers as real.

**Accuracy gate: fixed.** `fidelity_threshold()` used to collapse to 0.0 for n≥13, making the accuracy check unable to reject anything past that point. Root cause was the analytic error bound scaling with statevector dimension (`2**n_qubits`) instead of gate count — the wrong model for unitary, norm-preserving gate application (every gate in this circuit preserves vector norm, so accumulated float32 error scales with the number of chained gates, not with Hilbert space dimension — a unitary matrix has condition number 1). Fixed and verified against real hardware on both CPU and CUDA backends at n=28: threshold now reports a real, non-trivial `0.9994325637817383` that both measured fidelities clear comfortably, instead of the old always-passing 0.0.

**Metal/CUDA backends: fidelity comparison skipped above n=16, fixed a real OOM.** Both backends used to compare their full output statevector against a CPU ground truth on every run, at every qubit count. That's fine at small n (and how cross-language parity between the Swift Metal kernel and the Python ground truth was originally verified, to ~1e-7 agreement) but is O(2**n) memory — at n=28 the Metal path's JSON round-trip (statevector → JSON text → re-parsed Python list → numpy array, several copies coexisting) reached ~18GB and got OOM-killed on real hardware. Fidelity comparison is now skipped above n=16 on both backends (`accuracy_status: "SKIPPED_FIDELITY_LARGE_N"`, timing still reported normally) — parity is already established at small n and doesn't need re-proving on every large-n timing run. Verified on real hardware: the Metal binary's peak memory at n=28 dropped from ~18GB (OOM-killed) to ~2.1GB (just the statevector itself, no JSON multiplication).

### metal/ (QuantumEdgeKit)

```
cd metal && swift build -c release && swift test
.build/release/qe-bench validate                        # quick per-qubit-count timing sweep
.build/release/qe-bench bench --circuit qaoa --qubits 8 --depth 4 --reps 5 --seed 0   # JSON, used by the Python adapter
```

### ios/ (A-series chips — A18, A18 Pro, etc.)

The macOS CLI can't reach real iPhone/iPad silicon. `ios/` wraps the same `QuantumEdgeKit` kernel in a minimal standalone app for exactly that — see `ios/README.md` for setup and the tethered-Instruments workflow needed to get real power data (iOS has no `powermetrics` equivalent for third-party apps).

### Running the full sweep (CPU + Metal)

```
uv run qmb sweep --circuit qaoa --qubit-min 4 --qubit-max 16
uv run qmb single --circuit qaoa --qubits 10 --backend metal
```

Note: at small qubit counts (n<16 on the hardware tested), CPU is faster than Metal — subprocess spawn overhead dominates below the crossover point found in the `quantum-edge` app's own benchmarks. This isn't hidden; report exactly what the data shows (PRD §8).

## Install

```
uv sync --extra dev
```

## Usage

```
uv run qmb single --circuit qaoa --qubits 10 --depth 4 --reps 5
uv run qmb sweep --circuit qaoa --qubit-min 4 --qubit-max 16
```

## Methodology

See `PRD.md` — versioned, hardened against the credibility gaps in the original draft (baked-in conclusion, missing NVIDIA leg, single-shot timing, fixed fidelity threshold, macOS-only OOM detection, no independent review gate).

## Power-normalized benchmarking

Two combined benchmark+power-capture scripts, for the energy-per-result comparison (not just wall-clock):

```
# macOS/Metal — requires you to run it yourself (interactive sudo for powermetrics)
./scripts/benchmark_with_power_macos.sh [qubits] [depth] [reps] [seed]

# Linux/CUDA — no sudo needed, portable to bare-metal Linux
./scripts/benchmark_with_power_cuda.sh [qubits] [depth] [reps] [seed]
```

Both write a combined JSON (benchmark timing + power stats) to `benchmarks-results/`. The combined JSON and the CUDA raw-sample CSV are tracked in git (small, useful as data); the macOS `powermetrics` raw text logs are large (10-15MB) and gitignored — keep them locally. The CUDA script's parser automatically flags physically-impossible clock readings (`telemetry_suspect: true`) rather than letting a bad number pass silently. The macOS script's powermetrics text parsing hasn't been validated against a real log (no sudo access in the environment that wrote it) — only against a synthetic one built to match the documented format; if numbers look wrong, check the raw log path the script prints.

## Contributing a device benchmark

We want this dataset to grow past what one machine can produce — if you have hardware we don't (M3/M4 Macs, other NVIDIA cards, AMD, whatever), a PR adding your numbers is genuinely useful.

To keep results comparable across contributors, every submitted run must:

1. Use the fixed protocol: `--circuit qaoa --qubits 28 --depth 4 --reps 5 --seed 0` (n=28 is the current cross-device reference point — open an issue first if you want to propose a different reference n).
2. Run through the power-capture script for your platform, not the bare `qmb` command, so timing and power are captured together:
   ```
   ./scripts/benchmark_with_power_macos.sh 28 4 5 0   # Apple Silicon
   ./scripts/benchmark_with_power_cuda.sh 28 4 5 0    # NVIDIA/Linux
   ```
3. Report `reps=5` minimum — single-shot numbers aren't accepted. Run-to-run variance at n=28 has been observed to span roughly 8.8s-13.3s on the same hardware, so N<5 is noise, not a result.
4. Include the device/OS/driver info the script's combined JSON doesn't capture automatically: chip model (e.g. "M3 Pro, 14-core GPU"), macOS/Linux version, and for CUDA runs the driver + CUDA version (`nvidia-smi` header).
5. Commit the combined JSON (`*-power-benchmark-*.json`) and, for CUDA runs, the raw sample CSV — both are tracked in git and small. macOS raw `powermetrics` logs are large and gitignored; keep yours locally, don't force-add them.
6. If the run's telemetry gets flagged (`telemetry_suspect: true` on the CUDA leg, or anything that looks like idle-pinned power on macOS), say so in the PR rather than omitting the run — a flagged run with an honest caveat is more useful to this dataset than a clean-looking number that might not be real.

PR title convention: `Add <device> benchmark` (e.g. "Add M3 Pro benchmark"). No results table to update yet — that comes once there are enough rows to make one meaningful; for now, state the headline numbers (time, mean/max power) in the PR body.

## Tests

```
uv run pytest
```

## License

MIT — see `LICENSE`. Chosen deliberately: the Metal statevector kernel here is a candidate for upstreaming into Qiskit or PennyLane (both Apache 2.0), and MIT code is freely includable in an Apache 2.0 project with no license friction.
