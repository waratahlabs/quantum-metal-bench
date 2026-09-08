# QMBench (iOS)

A minimal standalone iOS wrapper around `QuantumEdgeKit` (the same Metal kernel used by the macOS `qe-bench` CLI) for getting real timing and power data from A-series chips (A18, A18 Pro, etc.) that the macOS-only benchmark scripts can't reach.

Not distributed anywhere — build and run it yourself via Xcode. It runs the same locked reference protocol as every other device in this dataset: QAOA, qubits=28, depth=4, reps=5, seed=0.

## What this does and doesn't measure

- **Timing:** real, measured on-device via `Date()` around each rep, same convention as the macOS CLI.
- **Power: not measured by the app itself.** iOS gives third-party apps no API for instantaneous GPU/CPU wattage the way macOS's `powermetrics` does. To get real power data, you need Instruments attached — see below.
- **Fidelity/correctness:** not checked by this app. Cross-language circuit parity (this Swift kernel vs the Python ground truth) is already established once, at small n, by the CLI's own test suite — this app exists purely to measure device timing/power, not to re-prove correctness.

## Setup

```
cd ios
xcodegen generate
open QMBench.xcodeproj
```

(`xcodegen` — `brew install xcodegen` if you don't have it. The `.xcodeproj` itself isn't committed; regenerate it from `project.yml`.)

## Getting real timing + power data (tethered Instruments workflow)

This requires a real device — the simulator runs on your Mac's own silicon via Metal's translation layer, so its numbers mean nothing for A-series comparison.

1. Connect a real iPhone/iPad via cable (not wireless — Instruments' Energy Log needs the wired connection for accurate power sampling).
2. In Xcode, select your device as the run destination (not a simulator).
3. Product → Profile (⌘I) instead of a normal Run — this launches Instruments.
4. Choose the **Energy Log** template.
5. Hit Record, then tap "Run Benchmark" in the app.
6. Let it run to completion (the UI shows "Done — 5 reps" when finished).
7. Stop the Instruments recording. The Energy Log gives you CPU/GPU/ANE power over the exact time window, plus overall energy impact.
8. Report: device model (Settings → General → About, or `Mac15,3`-style identifier isn't applicable here — use the marketing name, e.g. "iPhone 16 Pro, A18 Pro"), iOS version, the app's reported mean/min/max timing, and the Energy Log's power readings — same fields as the macOS/Linux contribution protocol in the main README, adapted for what's actually measurable on iOS.

## Known limitations

- No background-load control the way `./scripts/benchmark_with_power_macos.sh` documents for macOS — closing other apps and disabling background refresh before profiling is on you.
- Battery percentage is a coarse fallback if Instruments isn't available (Settings → Battery, before/after — expect this to be too imprecise to register a single ~10-25s run; only useful for repeated back-to-back runs).
- `QMBenchUITests` exercises the tap-to-result pipeline end-to-end but is a simulator-only correctness check (proves the UI/Task/QuantumEdgeKit wiring works), not a source of real device numbers.
