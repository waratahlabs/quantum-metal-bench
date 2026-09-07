import Metal
import Foundation

/// GPU-accelerated quantum statevector using a private Metal heap.
///
/// ## Storage architecture
///
/// The statevector lives in a `.storageModePrivate` buffer sub-allocated from
/// an `MTLHeap`. Private storage means:
///
/// - The CPU has NO direct access to the buffer contents. Calling `.contents()`
///   on `svBuffer` would be a GPU fault, so no CPU-pointer accessor is exposed
///   for the statevector; use the chunked blit path below instead.
/// - The allocation does NOT count against Jetsam RSS, so the process is not
///   killed for large GPU allocations. This is what enables 8 GiB statevectors
///   on a 12 GB device.
///
/// ## CPU ↔ GPU data movement
///
/// `reset()` and `stateAsComplexArray()` use a small shared scratch buffer
/// (≤ 4 MB) and blit encoders to move data in chunks. The scratch buffer is
/// the ONLY shared allocation; making it full-statevector-sized would reintroduce
/// the Jetsam problem for large n.
///
/// ## Gate application
///
/// `applyGate1q` and `applyGate2q` bind `svBuffer` (private) directly to the
/// compute pipeline at index 0. Metal compute kernels can read and write private
/// buffers without any staging — only the CPU needs the blit path.
public final class MetalStatevector {
    public let nQubits: Int

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let sv1qPipeline: MTLComputePipelineState
    private let sv2qPipeline: MTLComputePipelineState

    /// The statevector heap — private storage, GPU-addressable only.
    private let svHeap: MTLHeap
    /// The statevector buffer, sub-allocated from svHeap with .storageModePrivate.
    private let svBuffer: MTLBuffer

    /// Shared scratch buffer for chunked CPU↔GPU blits. Max 4 MB.
    /// NOT sized to the full statevector — that would reintroduce Jetsam pressure.
    private let scratchBuffer: MTLBuffer
    /// Number of SIMD2<Float> amplitudes that fit in one scratch buffer chunk.
    private let scratchChunkCount: Int

    /// The memory manager used for this statevector's heap allocation.
    public let manager: MetalMemoryManager

    // MARK: - Init

    /// Create a statevector for `nQubits` qubits backed by a GPU-private heap.
    ///
    /// Primary designated initializer. Uses the default 0.65 GPU budget ceiling.
    ///
    /// - Parameter nQubits: Number of qubits. State space = 2^nQubits complex amplitudes.
    /// - Throws: `MetalMemoryError` if the budget ceiling is exceeded or if the
    ///   Metal runtime returns nil for heap or buffer allocation.
    ///   Does NOT fall back silently to shared storage on failure.
    public convenience init(nQubits: Int) throws {
        try self.init(nQubits: nQubits, ceilingFraction: 0.65)
    }

    /// Create a statevector for `nQubits` qubits with an explicit GPU budget ceiling.
    ///
    /// - Parameters:
    ///   - nQubits: Number of qubits. State space = 2^nQubits complex amplitudes.
    ///   - ceilingFraction: GPU budget ceiling as fraction of device RAM (default 0.65).
    ///     Use 0.70 for aggressive 8 GiB experiments on 12 GB devices.
    ///
    /// - Throws: `MetalMemoryError` if the budget ceiling is exceeded or if the
    ///   Metal runtime returns nil for heap or buffer allocation.
    ///   Does NOT fall back silently to shared storage on failure.
    public init(nQubits: Int, ceilingFraction: Double = 0.65) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "MetalStatevector", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No Metal device"])
        }
        device = dev

        guard let q = dev.makeCommandQueue() else {
            throw NSError(domain: "MetalStatevector", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "No command queue"])
        }
        queue = q

        let opts = MTLCompileOptions()
        let lib = try dev.makeLibrary(source: metalSrc, options: opts)
        guard let fn1 = lib.makeFunction(name: "apply_single_qubit_gate"),
              let fn2 = lib.makeFunction(name: "apply_two_qubit_gate") else {
            throw NSError(domain: "MetalStatevector", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Missing kernel function"])
        }
        sv1qPipeline = try dev.makeComputePipelineState(function: fn1)
        sv2qPipeline = try dev.makeComputePipelineState(function: fn2)

        // Heap allocation — throws MetalMemoryError on failure.
        // No silent fallback to shared storage.
        let ampCount = 1 << nQubits
        let stride = MemoryLayout<SIMD2<Float>>.stride
        let totalBytes = ampCount * stride

        let mgr = MetalMemoryManager(device: dev, ceilingFraction: ceilingFraction)
        manager = mgr

        let heap = try mgr.makeHeap(bytes: totalBytes)
        svHeap = heap

        guard let svBuf = heap.makeBuffer(length: totalBytes, options: .storageModePrivate) else {
            throw MetalMemoryError.bufferAllocationFailed
        }
        svBuffer = svBuf

        // Scratch buffer: capped at 4 MB, shared storage (CPU-accessible).
        let scratchBytes = min(totalBytes, 4 * 1024 * 1024)
        guard let scratch = dev.makeBuffer(length: scratchBytes, options: .storageModeShared) else {
            throw MetalMemoryError.bufferAllocationFailed
        }
        scratchBuffer = scratch
        scratchChunkCount = scratchBytes / stride

        self.nQubits = nQubits

        // Initialise to |0⟩ via chunked blit.
        reset()
    }

    // MARK: - State Initialisation

    /// Reset statevector to computational basis state |0⟩ (amplitude 1 at index 0).
    ///
    /// Uses chunked blit: fills the 4 MB scratch buffer in CPU memory, then
    /// blits each chunk to the private svBuffer via a blit command encoder.
    public func reset() {
        let ampCount = 1 << nQubits
        let stride = MemoryLayout<SIMD2<Float>>.stride
        let scratchPtr = scratchBuffer.contents()
            .bindMemory(to: SIMD2<Float>.self, capacity: scratchChunkCount)

        var offset = 0
        while offset < ampCount {
            let chunkSize = min(scratchChunkCount, ampCount - offset)

            // Write |0⟩ chunk into CPU-accessible scratch.
            scratchPtr.initialize(repeating: .zero, count: chunkSize)
            if offset == 0 {
                scratchPtr[0] = SIMD2<Float>(1, 0)
            }

            // Blit scratch → svBuffer at the current offset.
            guard let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeBlitCommandEncoder() else { break }
            enc.copy(from: scratchBuffer,
                     sourceOffset: 0,
                     to: svBuffer,
                     destinationOffset: offset * stride,
                     size: chunkSize * stride)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            offset += chunkSize
        }
    }

    // MARK: - State Readback

    /// Read the full statevector to the CPU as an array of (real, imag) pairs.
    ///
    /// Uses chunked blit: blits each chunk from the private svBuffer to the shared
    /// scratch buffer, then copies the CPU-accessible scratch contents.
    ///
    /// For large statevectors (e.g. n=30, 8 GiB), this is intentionally slow —
    /// the round-trip blit is the only correct path for private-storage buffers.
    public func stateAsComplexArray() -> [(Float, Float)] {
        let ampCount = 1 << nQubits
        let stride = MemoryLayout<SIMD2<Float>>.stride
        let scratchPtr = scratchBuffer.contents()
            .bindMemory(to: SIMD2<Float>.self, capacity: scratchChunkCount)

        var result = [(Float, Float)](repeating: (0, 0), count: ampCount)
        var offset = 0
        while offset < ampCount {
            let chunkSize = min(scratchChunkCount, ampCount - offset)

            // Blit svBuffer chunk → scratch.
            guard let cb = queue.makeCommandBuffer(),
                  let enc = cb.makeBlitCommandEncoder() else { break }
            enc.copy(from: svBuffer,
                     sourceOffset: offset * stride,
                     to: scratchBuffer,
                     destinationOffset: 0,
                     size: chunkSize * stride)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            // Copy CPU-accessible scratch into result.
            for i in 0..<chunkSize {
                let amp = scratchPtr[i]
                result[offset + i] = (amp.x, amp.y)
            }
            offset += chunkSize
        }
        return result
    }

    // MARK: - Gate Application

    /// Apply a single-qubit gate U (4 complex elements, row-major) to `target` qubit.
    ///
    /// Binds `svBuffer` (private) at index 0. Metal compute kernels address private
    /// buffers directly — no staging required here.
    public func applyGate1q(_ U: [SIMD2<Float>], target: Int) {
        precondition(U.count == 4)
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(sv1qPipeline)
        enc.setBuffer(svBuffer, offset: 0, index: 0)   // svBuffer, not scratchBuffer
        var uCopy = U
        enc.setBytes(&uCopy, length: 4 * MemoryLayout<SIMD2<Float>>.size, index: 1)
        var tgt = UInt32(target)
        enc.setBytes(&tgt, length: 4, index: 2)
        var nq = UInt32(nQubits)
        enc.setBytes(&nq, length: 4, index: 3)
        let half = 1 << (nQubits - 1)
        let tgWidth = min(256, half)
        let tgSize = MTLSize(width: tgWidth, height: 1, depth: 1)
        let groups = MTLSize(width: (half + tgWidth - 1) / tgWidth, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Apply a two-qubit gate U (16 complex elements, row-major) to qubits `qLo` and `qHi`.
    ///
    /// Binds `svBuffer` (private) at index 0. No staging required for compute.
    public func applyGate2q(_ U: [SIMD2<Float>], qLo: Int, qHi: Int) {
        precondition(U.count == 16)
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(sv2qPipeline)
        enc.setBuffer(svBuffer, offset: 0, index: 0)   // svBuffer, not scratchBuffer
        var uCopy = U
        enc.setBytes(&uCopy, length: 16 * MemoryLayout<SIMD2<Float>>.size, index: 1)
        var lo = UInt32(qLo), hi = UInt32(qHi)
        enc.setBytes(&lo, length: 4, index: 2)
        enc.setBytes(&hi, length: 4, index: 3)
        var nq = UInt32(nQubits)
        enc.setBytes(&nq, length: 4, index: 4)
        let quarter = 1 << (nQubits - 2)
        let tgWidth = min(256, quarter)
        let tgSize = MTLSize(width: tgWidth, height: 1, depth: 1)
        let groups = MTLSize(width: (quarter + tgWidth - 1) / tgWidth, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}
