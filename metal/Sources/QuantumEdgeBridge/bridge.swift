import QuantumEdgeKit
import Foundation

nonisolated(unsafe) private var _sv: MetalStatevector? = nil

@_cdecl("qe_init")
public func qe_init(_ nQubits: Int32) -> Int32 {
    do {
        _sv = try MetalStatevector(nQubits: Int(nQubits))
        return 0
    } catch {
        return -1
    }
}

@_cdecl("qe_reset")
public func qe_reset() {
    _sv?.reset()
}

@_cdecl("qe_n_qubits")
public func qe_n_qubits() -> Int32 {
    Int32(_sv?.nQubits ?? 0)
}

// qe_statevector_pointer: removed — svBuffer uses .storageModePrivate and has no CPU mapping.
// Calling contents() on a private buffer is a GPU fault.
// Use qe_copy_statevector instead for CPU readback via blit.
@_cdecl("qe_statevector_pointer")
public func qe_statevector_pointer() -> UnsafeMutableRawPointer? {
    // Always returns nil — private storage has no CPU pointer.
    // Callers must use qe_copy_statevector for readback.
    return nil
}

/// Copy the statevector into a caller-supplied buffer via blit (chunked, synchronous).
/// `outPtr` must point to at least `2 * 2^nQubits` floats (real, imag interleaved).
/// Returns 0 on success, -1 if no statevector is initialised.
@_cdecl("qe_copy_statevector")
public func qe_copy_statevector(_ outPtr: UnsafeMutablePointer<Float>?) -> Int32 {
    guard let sv = _sv, let out = outPtr else { return -1 }
    let state = sv.stateAsComplexArray()
    for (i, (re, im)) in state.enumerated() {
        out[i * 2]     = re
        out[i * 2 + 1] = im
    }
    return 0
}

// wire indices entering these functions use PennyLane convention:
// wire 0 = MSB (bit n-1 of state index). The Metal kernel uses LSB (wire 0 = bit 0).
// Conversion: metalBit = n - 1 - plWire

@_cdecl("qe_apply_1q")
public func qe_apply_1q(_ uPtr: UnsafePointer<Float>?, _ plWire: Int32) {
    guard let sv = _sv, let uPtr = uPtr else { return }
    let metalTarget = sv.nQubits - 1 - Int(plWire)
    let U = (0..<4).map { SIMD2<Float>(uPtr[$0*2], uPtr[$0*2+1]) }
    sv.applyGate1q(U, target: metalTarget)
}

@_cdecl("qe_apply_2q")
public func qe_apply_2q(_ uPtr: UnsafePointer<Float>?, _ plWire0: Int32, _ plWire1: Int32) {
    guard let sv = _sv, let uPtr = uPtr else { return }
    let n = sv.nQubits
    let b0 = n - 1 - Int(plWire0)
    let b1 = n - 1 - Int(plWire1)
    var raw = (0..<16).map { SIMD2<Float>(uPtr[$0*2], uPtr[$0*2+1]) }
    // If PL wire0 maps to a higher bit than PL wire1, the matrix row/col ordering
    // must be permuted [0,2,1,3] to match the kernel's q_lo/q_hi (ascending bit) convention.
    if b0 < b1 {
        let p = [0, 2, 1, 3]
        raw = p.flatMap { r in p.map { c in raw[r*4+c] } }
    }
    let (q_lo, q_hi) = b0 < b1 ? (b0, b1) : (b1, b0)
    sv.applyGate2q(raw, qLo: q_lo, qHi: q_hi)
}
