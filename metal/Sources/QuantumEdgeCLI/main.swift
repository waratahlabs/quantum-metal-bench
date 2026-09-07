import QuantumEdgeKit
import Foundation

func parseArgs(_ args: [String]) -> [String: String] {
    var result: [String: String] = [:]
    var i = 0
    while i < args.count {
        if args[i].hasPrefix("--"), i + 1 < args.count {
            result[String(args[i].dropFirst(2))] = args[i + 1]
            i += 2
        } else {
            i += 1
        }
    }
    return result
}

func runValidateSweep() {
    let H: [SIMD2<Float>] = [
        SIMD2(Float(1 / 2.squareRoot() as Double), 0),
        SIMD2(Float(1 / 2.squareRoot() as Double), 0),
        SIMD2(Float(1 / 2.squareRoot() as Double), 0),
        SIMD2(Float(-1 / 2.squareRoot() as Double), 0),
    ]
    for n in stride(from: 4, through: 20, by: 2) {
        guard let sv = try? MetalStatevector(nQubits: n) else { continue }
        sv.reset()
        let t0 = Date()
        for _ in 0..<100 {
            sv.reset()
            sv.applyGate1q(H, target: 0)
        }
        let ms = Date().timeIntervalSince(t0) * 10
        print(String(format: "n=%02d  %.2f ms/op", n, ms))
    }
}

/// `qe-bench bench --circuit qaoa --qubits N --depth D --reps R --seed S`
/// Emits JSON matching the PRD-§5 timing/statevector fields so the Python
/// harness (quantum_metal_bench.backends.metal) can parse it directly.
func runBench(_ opts: [String: String]) {
    let circuit = opts["circuit"] ?? "qaoa"
    guard circuit == "qaoa" else {
        // Only QAOA has a Swift mirror today (cross-language RNG parity for
        // clifford_t isn't built) — fail loud rather than silently running
        // the wrong circuit and reporting a bogus fidelity comparison.
        FileHandle.standardError.write("qe-bench bench: circuit '\(circuit)' not supported on Metal yet (only 'qaoa')\n".data(using: .utf8)!)
        exit(2)
    }
    let nQubits = Int(opts["qubits"] ?? "10") ?? 10
    let depth = Int(opts["depth"] ?? "4") ?? 4
    let reps = Int(opts["reps"] ?? "5") ?? 5
    let seed = Int(opts["seed"] ?? "0") ?? 0

    guard let sv = try? MetalStatevector(nQubits: nQubits) else {
        print("{\"status\": \"FAILED_ALLOC\"}")
        exit(1)
    }

    // Warmup, discarded (PRD §3.B)
    sv.reset()
    QAOACircuit.run(on: sv, nQubits: nQubits, depth: depth, seed: seed)

    var samples: [Double] = []
    for _ in 0..<reps {
        sv.reset()
        let t0 = Date()
        QAOACircuit.run(on: sv, nQubits: nQubits, depth: depth, seed: seed)
        samples.append(Date().timeIntervalSince(t0))
    }

    let mean = samples.reduce(0, +) / Double(samples.count)
    let variance = samples.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(max(samples.count - 1, 1))
    let stddev = variance.squareRoot()

    let state = sv.stateAsComplexArray()
    let realParts = state.map { Double($0.0) }
    let imagParts = state.map { Double($0.1) }

    let payload: [String: Any] = [
        "n_reps": reps,
        "execution_time_sec_mean": mean,
        "execution_time_sec_stddev": stddev,
        "execution_time_sec_min": samples.min() ?? mean,
        "execution_time_sec_max": samples.max() ?? mean,
        "gate_count": QAOACircuit.gateCount(nQubits: nQubits, depth: depth),
        "statevector_real": realParts,
        "statevector_imag": imagParts,
    ]

    let data = try! JSONSerialization.data(withJSONObject: payload)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

let args = Array(CommandLine.arguments.dropFirst())
if args.first == "bench" {
    runBench(parseArgs(Array(args.dropFirst())))
} else {
    runValidateSweep()
}
