import XCTest
import QuantumEdgeBridge

/// Tests for the C-ABI bridge layer.
///
/// The bridge applies PennyLane's MSB-first wire convention:
///   metalBit = n - 1 - plWire
///
/// For n=2:
///   PL wire 0 (MSB) → Metal bit 1 (high bit, stride=2)
///   PL wire 1 (LSB) → Metal bit 0 (low bit, stride=1)
///
/// State index layout: bit 1 is the high bit, bit 0 is the low bit.
///   index 0 = |00⟩, index 1 = |01⟩, index 2 = |10⟩, index 3 = |11⟩
///
/// H on PL wire 0 (MSB, Metal bit 1) from |00⟩:
///   produces (|00⟩ + |10⟩)/√2 → amps at indices 0 and 2
///
/// H on PL wire 1 (LSB, Metal bit 0) from |00⟩:
///   produces (|00⟩ + |01⟩)/√2 → amps at indices 0 and 1
final class BridgeTests: XCTestCase {

    private var stateBuffer = [Float](repeating: 0, count: 8)  // 4 amps × 2 floats, n≤2

    override func setUp() {
        super.setUp()
        // Re-initialise the global bridge statevector before each test.
        XCTAssertEqual(qe_init(2), 0, "qe_init should return 0 on success")
        qe_reset()
    }

    // MARK: - Helpers

    private func H_matrix() -> [Float] {
        let inv2 = Float(1 / 2.squareRoot())
        // Row-major complex64: [re,im, re,im, re,im, re,im] for 2×2 matrix
        return [inv2, 0,  inv2, 0,
                inv2, 0, -inv2, 0]
    }

    private func CNOT_matrix() -> [Float] {
        // 4×4 CNOT for control=q_hi (Metal bit 1), target=q_lo (Metal bit 0).
        // Kernel basis ordering: (q_lo=0,q_hi=0), (q_lo=1,q_hi=0), (q_lo=0,q_hi=1), (q_lo=1,q_hi=1)
        // = identity on {00,01}, swap {10↔11}:
        //   [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        // Distinct from the StatevectorTests CNOT which has control=q_lo, target=q_hi.
        var m = [Float](repeating: 0, count: 32)
        func set(_ r: Int, _ c: Int) { m[(r*4+c)*2] = 1.0 }
        set(0,0); set(1,1); set(2,3); set(3,2)
        return m
    }

    private func readState(nQubits: Int = 2) -> [Float] {
        let count = (1 << nQubits) * 2
        var buf = [Float](repeating: 0, count: count)
        let rc = buf.withUnsafeMutableBufferPointer { ptr in
            qe_copy_statevector(ptr.baseAddress)
        }
        XCTAssertEqual(rc, 0, "qe_copy_statevector should succeed")
        return buf
    }

    // MARK: - Init / reset

    func testQEInitSetsNQubits() {
        XCTAssertEqual(qe_n_qubits(), 2)
    }

    func testQEResetProducesZeroState() {
        qe_reset()
        let buf = readState()
        // Amplitude 0: re=1, im=0
        XCTAssertEqual(buf[0], 1.0, accuracy: 1e-5, "amp[0].re should be 1 after reset")
        XCTAssertEqual(buf[1], 0.0, accuracy: 1e-5, "amp[0].im should be 0")
        // All other amplitudes: zero
        for i in 1..<4 {
            XCTAssertEqual(buf[i*2],   0.0, accuracy: 1e-5, "amp[\(i)].re should be 0")
            XCTAssertEqual(buf[i*2+1], 0.0, accuracy: 1e-5, "amp[\(i)].im should be 0")
        }
    }

    func testQEStatevectorPointerReturnsNil() {
        // Private storage has no CPU mapping — bridge always returns nil.
        XCTAssertNil(qe_statevector_pointer())
    }

    // MARK: - Wire convention: H on PL wire 0 (MSB)

    /// PL wire 0 → Metal bit 1. H from |00⟩ produces (|00⟩+|10⟩)/√2.
    /// Non-zero amplitudes are at indices 0 and 2 (bit 1 flipped).
    func testHOnPLWire0ProducesCorrectSuperposition() {
        var hMat = H_matrix()
        hMat.withUnsafeMutableBufferPointer { ptr in
            qe_apply_1q(ptr.baseAddress, 0)
        }
        let buf = readState()
        let inv2 = Float(1 / 2.squareRoot())

        XCTAssertEqual(buf[0], inv2, accuracy: 1e-5, "amp[0].re = 1/√2  (|00⟩)")
        XCTAssertEqual(buf[1], 0.0,  accuracy: 1e-5, "amp[0].im = 0")
        XCTAssertEqual(buf[2], 0.0,  accuracy: 1e-5, "amp[1].re = 0     (|01⟩)")
        XCTAssertEqual(buf[4], inv2, accuracy: 1e-5, "amp[2].re = 1/√2  (|10⟩)")
        XCTAssertEqual(buf[5], 0.0,  accuracy: 1e-5, "amp[2].im = 0")
        XCTAssertEqual(buf[6], 0.0,  accuracy: 1e-5, "amp[3].re = 0     (|11⟩)")
    }

    // MARK: - Wire convention: H on PL wire 1 (LSB)

    /// PL wire 1 → Metal bit 0. H from |00⟩ produces (|00⟩+|01⟩)/√2.
    /// Non-zero amplitudes are at indices 0 and 1 (bit 0 flipped).
    func testHOnPLWire1ProducesCorrectSuperposition() {
        var hMat = H_matrix()
        hMat.withUnsafeMutableBufferPointer { ptr in
            qe_apply_1q(ptr.baseAddress, 1)
        }
        let buf = readState()
        let inv2 = Float(1 / 2.squareRoot())

        XCTAssertEqual(buf[0], inv2, accuracy: 1e-5, "amp[0].re = 1/√2  (|00⟩)")
        XCTAssertEqual(buf[2], inv2, accuracy: 1e-5, "amp[1].re = 1/√2  (|01⟩)")
        XCTAssertEqual(buf[4], 0.0,  accuracy: 1e-5, "amp[2].re = 0     (|10⟩)")
        XCTAssertEqual(buf[6], 0.0,  accuracy: 1e-5, "amp[3].re = 0     (|11⟩)")
    }

    // MARK: - 2q gate via bridge: Bell state

    /// H on PL wire 0, CNOT(control=PL wire 0, target=PL wire 1) → Bell state.
    /// PL wire convention: wire 0=MSB, wire 1=LSB.
    /// Expected: (|00⟩ + |11⟩)/√2 → amps at indices 0 and 3.
    func testBellStateViaBridge() {
        var hMat = H_matrix()
        hMat.withUnsafeMutableBufferPointer { ptr in
            qe_apply_1q(ptr.baseAddress, 0)  // H on PL wire 0 (MSB)
        }

        var cnotMat = CNOT_matrix()
        cnotMat.withUnsafeMutableBufferPointer { ptr in
            qe_apply_2q(ptr.baseAddress, 0, 1)  // CNOT: control=PL wire 0, target=PL wire 1
        }

        let buf = readState()
        let inv2 = Float(1 / 2.squareRoot())

        XCTAssertEqual(buf[0], inv2, accuracy: 1e-4, "amp[0].re = 1/√2  (|00⟩)")
        XCTAssertEqual(buf[2], 0.0,  accuracy: 1e-4, "amp[1].re = 0     (|01⟩)")
        XCTAssertEqual(buf[4], 0.0,  accuracy: 1e-4, "amp[2].re = 0     (|10⟩)")
        XCTAssertEqual(buf[6], inv2, accuracy: 1e-4, "amp[3].re = 1/√2  (|11⟩)")
    }

    // MARK: - H² = I via bridge

    func testHSquaredIsIdentityViaBridge() {
        var hMat = H_matrix()
        hMat.withUnsafeMutableBufferPointer { ptr in
            qe_apply_1q(ptr.baseAddress, 0)
            qe_apply_1q(ptr.baseAddress, 0)
        }
        let buf = readState()
        XCTAssertEqual(buf[0], 1.0, accuracy: 1e-5, "H² should return to |0⟩")
        for i in 1..<4 { XCTAssertEqual(buf[i*2], 0.0, accuracy: 1e-5) }
    }
}
