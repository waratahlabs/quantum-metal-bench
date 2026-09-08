import Foundation

/// Swift mirror of `quantum_metal_bench.circuits.qaoa_circuit` /
/// `qaoa_angles` (Python). MUST stay byte-identical to that formula — a
/// fidelity comparison between the CPU and Metal backends is only valid if
/// both run the same circuit. If you change the angle formula on one side,
/// change it here too.
///
/// Lives in QuantumEdgeKit (not QuantumEdgeCLI) so it's importable by any
/// consumer of the library — the macOS qe-bench CLI, the iOS benchmark app
/// (ios/), and QuantumEdge itself all share this one implementation rather
/// than each carrying their own copy.
public enum QAOACircuit {

    /// deterministic angle schedule, same formula as circuits.py:qaoa_angles
    public static func angles(depth: Int, seed: Int) -> (gammas: [Double], betas: [Double]) {
        var gammas: [Double] = []
        var betas: [Double] = []
        for layer in 0..<depth {
            let g = Double((seed * 37 + layer * 17 + 1) % 100) / 100.0 * (Double.pi / 2)
            let b = Double((seed * 53 + layer * 11 + 1) % 100) / 100.0 * (Double.pi / 4)
            gammas.append(g)
            betas.append(b)
        }
        return (gammas, betas)
    }

    public static func hadamard() -> [SIMD2<Float>] {
        let s = Float(1 / 2.squareRoot() as Double)
        return [SIMD2(s, 0), SIMD2(s, 0), SIMD2(s, 0), SIMD2(-s, 0)]
    }

    /// CNOT, control=qLo, target=qHi — same convention as StatevectorTests.
    public static func cnot() -> [SIMD2<Float>] {
        [
            SIMD2(1, 0), SIMD2(0, 0), SIMD2(0, 0), SIMD2(0, 0),
            SIMD2(0, 0), SIMD2(0, 0), SIMD2(0, 0), SIMD2(1, 0),
            SIMD2(0, 0), SIMD2(0, 0), SIMD2(1, 0), SIMD2(0, 0),
            SIMD2(0, 0), SIMD2(1, 0), SIMD2(0, 0), SIMD2(0, 0),
        ]
    }

    /// RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2}) — PennyLane convention.
    public static func rz(_ theta: Double) -> [SIMD2<Float>] {
        let half = theta / 2
        let a = SIMD2(Float(cos(-half)), Float(sin(-half)))
        let d = SIMD2(Float(cos(half)), Float(sin(half)))
        return [a, SIMD2(0, 0), SIMD2(0, 0), d]
    }

    /// RX(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]] — PennyLane convention.
    public static func rx(_ theta: Double) -> [SIMD2<Float>] {
        let half = theta / 2
        let c = SIMD2(Float(cos(half)), Float(0))
        let s = SIMD2(Float(0), Float(-sin(half)))
        return [c, s, s, c]
    }

    /// Builds and runs the QAOA circuit on `sv`, mirroring circuits.py:qaoa_circuit
    /// gate-for-gate (ring topology, CNOT-RZ-CNOT cost layer, RX mixer layer).
    public static func run(on sv: MetalStatevector, nQubits: Int, depth: Int, seed: Int) {
        let edges = (0..<nQubits).map { ($0, ($0 + 1) % nQubits) }
        let (gammas, betas) = angles(depth: depth, seed: seed)
        let H = hadamard()
        let CNOT = cnot()

        for q in 0..<nQubits {
            sv.applyGate1q(H, target: q)
        }
        for layer in 0..<depth {
            for (i, j) in edges {
                sv.applyGate2q(CNOT, qLo: i, qHi: j)
                sv.applyGate1q(rz(gammas[layer]), target: j)
                sv.applyGate2q(CNOT, qLo: i, qHi: j)
            }
            for q in 0..<nQubits {
                sv.applyGate1q(rx(2 * betas[layer]), target: q)
            }
        }
    }

    public static func gateCount(nQubits: Int, depth: Int) -> Int {
        nQubits + depth * (nQubits * 3 + nQubits)
    }
}
