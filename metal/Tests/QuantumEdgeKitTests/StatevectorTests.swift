import XCTest
@testable import QuantumEdgeKit

final class StatevectorTests: XCTestCase {
    let H: [SIMD2<Float>] = [
        SIMD2(Float(1/2.squareRoot()), 0),
        SIMD2(Float(1/2.squareRoot()), 0),
        SIMD2(Float(1/2.squareRoot()), 0),
        SIMD2(Float(-1/2.squareRoot()), 0)
    ]

    // CNOT control=q_lo(0), target=q_hi(1): |01>->|11>, |11>->|01>
    let CNOT: [SIMD2<Float>] = [
        SIMD2(1,0), SIMD2(0,0), SIMD2(0,0), SIMD2(0,0),
        SIMD2(0,0), SIMD2(0,0), SIMD2(0,0), SIMD2(1,0),
        SIMD2(0,0), SIMD2(0,0), SIMD2(1,0), SIMD2(0,0),
        SIMD2(0,0), SIMD2(1,0), SIMD2(0,0), SIMD2(0,0)
    ]

    func testInitIsZeroState() throws {
        let sv = try MetalStatevector(nQubits: 2)
        let state = sv.stateAsComplexArray()
        XCTAssertEqual(state.count, 4)
        XCTAssertEqual(state[0].0, 1.0, accuracy: 1e-6)
        XCTAssertEqual(state[0].1, 0.0, accuracy: 1e-6)
        for i in 1..<4 {
            XCTAssertEqual(state[i].0, 0.0, accuracy: 1e-6)
            XCTAssertEqual(state[i].1, 0.0, accuracy: 1e-6)
        }
    }

    func testHSquaredIsIdentity() throws {
        let sv = try MetalStatevector(nQubits: 2)
        sv.reset()
        sv.applyGate1q(H, target: 0)
        sv.applyGate1q(H, target: 0)
        let state = sv.stateAsComplexArray()
        XCTAssertEqual(state[0].0, 1.0, accuracy: 1e-5)
        XCTAssertEqual(state[0].1, 0.0, accuracy: 1e-5)
        for i in 1..<4 {
            XCTAssertEqual(state[i].0, 0.0, accuracy: 1e-5)
            XCTAssertEqual(state[i].1, 0.0, accuracy: 1e-5)
        }
    }

    func testBellState() throws {
        let sv = try MetalStatevector(nQubits: 2)
        sv.reset()
        sv.applyGate1q(H, target: 0)
        sv.applyGate2q(CNOT, qLo: 0, qHi: 1)
        let state = sv.stateAsComplexArray()
        let inv2 = Float(1 / 2.squareRoot())
        XCTAssertEqual(state[0].0, inv2, accuracy: 1e-5)
        XCTAssertEqual(state[1].0, 0.0, accuracy: 1e-5)
        XCTAssertEqual(state[2].0, 0.0, accuracy: 1e-5)
        XCTAssertEqual(state[3].0, inv2, accuracy: 1e-5)
    }

    /// H on qubit 1 (the high bit) from |00⟩ produces (|00⟩ + |10⟩)/√2.
    /// State indices: |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3 (qubit 1 = bit 1).
    /// This exercises the kernel's stride/mask index math for target > 0.
    func testHOnQubit1() throws {
        let sv = try MetalStatevector(nQubits: 2)
        sv.reset()
        sv.applyGate1q(H, target: 1)
        let state = sv.stateAsComplexArray()
        let inv2 = Float(1 / 2.squareRoot())
        XCTAssertEqual(state[0].0, inv2, accuracy: 1e-5, "state[00] should be 1/√2")
        XCTAssertEqual(state[1].0, 0.0,  accuracy: 1e-5, "state[01] should be 0")
        XCTAssertEqual(state[2].0, inv2, accuracy: 1e-5, "state[10] should be 1/√2")
        XCTAssertEqual(state[3].0, 0.0,  accuracy: 1e-5, "state[11] should be 0")
        // All imaginary parts zero
        for i in 0..<4 { XCTAssertEqual(state[i].1, 0.0, accuracy: 1e-5) }
    }

    /// n=20 requires two scratch-buffer chunks (4 MB / 8 bytes = 524,288 amps/chunk;
    /// 2^20 = 1,048,576 amps needs 2 passes). Exercises the chunked blit loop in
    /// both reset() and stateAsComplexArray().
    func testChunkedBlit() throws {
        let n = 20
        let sv = try MetalStatevector(nQubits: n)
        // After reset: |0⟩ state — amplitude 1 at index 0, zero everywhere else.
        let afterReset = sv.stateAsComplexArray()
        XCTAssertEqual(afterReset.count, 1 << n)
        XCTAssertEqual(afterReset[0].0, 1.0, accuracy: 1e-5, "index 0 should be 1 after reset")
        XCTAssertEqual(afterReset[0].1, 0.0, accuracy: 1e-5)
        // Spot-check that the second chunk arrived correctly (non-zero is a blit offset bug).
        XCTAssertEqual(afterReset[524288].0, 0.0, accuracy: 1e-5,
                       "first amp of second chunk should be 0 — blit offset regression")

        // Apply H on qubit 0 and verify superposition survives the chunked readback.
        sv.applyGate1q(H, target: 0)
        let afterH = sv.stateAsComplexArray()
        let inv2 = Float(1 / 2.squareRoot())
        XCTAssertEqual(afterH[0].0, inv2, accuracy: 1e-5, "|0…0⟩ should be 1/√2")
        XCTAssertEqual(afterH[1].0, inv2, accuracy: 1e-5, "|0…1⟩ should be 1/√2")
        XCTAssertEqual(afterH[2].0, 0.0,  accuracy: 1e-5, "|0…10⟩ should be 0")
    }

    /// n=1 single-qubit system: init, reset, and H gate should work correctly.
    func testNEquals1() throws {
        let sv = try MetalStatevector(nQubits: 1)
        sv.reset()
        let init_state = sv.stateAsComplexArray()
        XCTAssertEqual(init_state.count, 2)
        XCTAssertEqual(init_state[0].0, 1.0, accuracy: 1e-5)
        XCTAssertEqual(init_state[1].0, 0.0, accuracy: 1e-5)

        sv.applyGate1q(H, target: 0)
        let after_H = sv.stateAsComplexArray()
        let inv2 = Float(1 / 2.squareRoot())
        XCTAssertEqual(after_H[0].0, inv2, accuracy: 1e-5)
        XCTAssertEqual(after_H[1].0, inv2, accuracy: 1e-5)
    }

    /// Allocation of an unreasonably large statevector must throw budgetExceeded,
    /// not crash or silently succeed.
    func testOversizedNThrowsBudgetExceeded() {
        // n=40 = 8TB statevector — no device has this budget.
        XCTAssertThrowsError(try MetalStatevector(nQubits: 40)) { error in
            guard case MetalMemoryError.budgetExceeded = error else {
                XCTFail("Expected budgetExceeded for n=40, got \(error)")
                return
            }
        }
    }
}
