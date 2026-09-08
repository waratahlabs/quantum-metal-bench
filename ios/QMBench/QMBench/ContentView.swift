import SwiftUI
import QuantumEdgeKit

/// Minimal standalone benchmark app — no UI polish intended. Its only job is
/// to run the SAME locked reference protocol used across every other device
/// in quantum-metal-bench's dataset (qubits=28, depth=4, reps=5, seed=0) on
/// real A-series silicon, so results are directly comparable.
///
/// Deliberately does NOT call `stateAsComplexArray()` or attempt any
/// fidelity comparison — this app has no CPU ground-truth to compare
/// against (that lives in the Python harness) and no need to hold a full
/// statevector in memory just to report timing. Circuit correctness is
/// already established via the CLI's own cross-language check at small n;
/// this app exists purely to measure real device timing/power.
///
/// Timing alone is measurable programmatically. Power is NOT — iOS gives
/// third-party apps no API for instantaneous GPU/CPU wattage the way
/// macOS's powermetrics does. To get real power data, run this app from
/// Xcode on a real (not simulator) device with Instruments' Energy Log
/// template attached — see ios/README.md.
struct ContentView: View {
    static let qubits = 28
    static let depth = 4
    static let reps = 5
    static let seed = 0

    @State private var status = "Idle"
    @State private var results: [Double] = []
    @State private var isRunning = false

    var body: some View {
        VStack(spacing: 16) {
            Text("quantum-metal-bench")
                .font(.headline)
            Text("QAOA — qubits=\(Self.qubits), depth=\(Self.depth), reps=\(Self.reps), seed=\(Self.seed)")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text(status)
                .font(.body)
                .multilineTextAlignment(.center)
                .padding()
                .accessibilityIdentifier("statusText")

            if !results.isEmpty {
                let mean = results.reduce(0, +) / Double(results.count)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Mean: \(String(format: "%.3f", mean))s")
                    Text("Min: \(String(format: "%.3f", results.min() ?? 0))s")
                    Text("Max: \(String(format: "%.3f", results.max() ?? 0))s")
                }
                .font(.system(.body, design: .monospaced))
            }

            Button(isRunning ? "Running…" : "Run Benchmark") {
                runBenchmark()
            }
            .disabled(isRunning)
            .buttonStyle(.borderedProminent)
            .accessibilityIdentifier("runBenchmarkButton")

            Text("For power data: run from Xcode on a real device with Instruments' Energy Log attached, not the simulator. See ios/README.md.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }

    private func runBenchmark() {
        isRunning = true
        status = "Allocating statevector (\(Self.qubits) qubits)…"
        results = []

        Task.detached(priority: .userInitiated) {
            do {
                let sv = try MetalStatevector(nQubits: Self.qubits)

                // Warmup, discarded — same convention as the macOS CLI
                // (metal/Sources/QuantumEdgeCLI/main.swift runBench).
                sv.reset()
                QAOACircuit.run(on: sv, nQubits: Self.qubits, depth: Self.depth, seed: Self.seed)

                var samples: [Double] = []
                for _ in 0..<Self.reps {
                    sv.reset()
                    let t0 = Date()
                    QAOACircuit.run(on: sv, nQubits: Self.qubits, depth: Self.depth, seed: Self.seed)
                    samples.append(Date().timeIntervalSince(t0))
                }

                await MainActor.run {
                    results = samples
                    status = "Done — \(samples.count) reps"
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    status = "Allocation failed: \(error)"
                    isRunning = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
