import Metal
import Foundation
import os.log
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Error Types

/// Errors thrown by MetalMemoryManager allocation operations.
public enum MetalMemoryError: Error, CustomStringConvertible {
    /// GPU budget ceiling would be exceeded by the requested allocation.
    case budgetExceeded(requested: Int, available: Int)
    /// The Metal runtime returned nil for the heap allocation.
    case heapAllocationFailed(bytes: Int)
    /// A buffer sub-allocation from a heap returned nil.
    case bufferAllocationFailed

    public var description: String {
        switch self {
        case .budgetExceeded(let req, let avail):
            let reqMB = req / 1_048_576
            let availMB = avail / 1_048_576
            return "MetalMemoryError: budget exceeded — requested \(reqMB) MB, available \(availMB) MB under soft ceiling"
        case .heapAllocationFailed(let bytes):
            return "MetalMemoryError: device.makeHeap returned nil for \(bytes / 1_048_576) MB heap"
        case .bufferAllocationFailed:
            return "MetalMemoryError: heap.makeBuffer returned nil"
        }
    }
}

// MARK: - MetalMemoryManager

/// iOS-safe Metal memory budget manager.
///
/// Manages GPU-heap allocation for large Metal buffers on Apple Silicon devices.
/// All APIs used are iOS-safe public SDK symbols — no `API_UNAVAILABLE(ios)`
/// symbols. The macOS-only working-set-size query is deliberately not used; the
/// budget ceiling here is derived from `ProcessInfo.physicalMemory` instead.
///
/// The GPU heap accounting is separate from CPU Jetsam RSS tracking:
/// `.storageModePrivate` buffers allocated from a heap do NOT count against the
/// process's `phys_footprint`, so Jetsam does not kill the process for GPU
/// allocations. This is the mechanism that enables large statevectors.
///
/// Memory pressure signals:
/// - `gpuBytesInUse` / `softCeilingBytes` — GPU-side budget (the real constraint)
/// - `cpuBytesAvailable` — CPU Jetsam headroom (useful for staging buffer sizing)
/// - `onMemoryPressure` — fires on `UIApplication.didReceiveMemoryWarningNotification`,
///   which is a CPU-memory signal. Useful for shedding CPU-side staging buffers.
///   It will NOT fire for GPU-only pressure; poll `gpuBytesInUse` for that.
public final class MetalMemoryManager {

    // MARK: - Properties

    public let device: MTLDevice

    /// Environment probe — platform, RAM, GPU budget, and ceiling values computed
    /// once at init. Used for ceiling decisions and display.
    public let environment: MemoryEnvironment

    /// Fraction of total device RAM to use as GPU allocation ceiling.
    /// Default 0.65 (conservative). Use 0.70 for aggressive experiments targeting
    /// 8 GiB on a 12 GB device (0.70 × 12 GB = 8.4 GB ceiling).
    public let ceilingFraction: Double

    /// Soft ceiling in bytes, sourced from the environment probe computed at init.
    /// The manager refuses new heap allocations that would push `gpuBytesInUse`
    /// above this value. This is a self-imposed limit — there is no public iOS API
    /// that exposes the per-process GPU budget.
    public var softCeilingBytes: Int {
        environment.softCeilingBytes
    }

    /// Current GPU bytes allocated by this process via Metal.
    /// Backed by `device.currentAllocatedSize`, the only public iOS API for this.
    /// Includes ALL storage modes (private, shared, managed) for this process.
    public var gpuBytesInUse: Int {
        device.currentAllocatedSize
    }

    /// CPU-side Jetsam headroom in bytes.
    /// Backed by `os_proc_available_memory()` on iOS (iOS 13+).
    /// Returns `Int.max` on macOS (where Jetsam does not apply).
    ///
    /// NOTE: Private Metal buffers do NOT count toward CPU phys_footprint,
    /// so this value does NOT reflect GPU pressure. Use `gpuBytesInUse` /
    /// `softCeilingBytes` for GPU budget awareness.
    public var cpuBytesAvailable: Int {
#if canImport(UIKit)
        Int(os_proc_available_memory())
#else
        Int.max  // macOS — no Jetsam; use GPU budget instead
#endif
    }

    /// Optional closure invoked when the system sends a CPU memory warning.
    /// Useful for shedding `.storageModeShared` staging buffers to reduce
    /// CPU phys_footprint. Will NOT fire for GPU-only pressure.
    public var onMemoryPressure: (() -> Void)?

    private let log = OSLog(subsystem: "com.waratahlabs.quantum-edge", category: "MetalMemory")

    // MARK: - Init

    public init(device: MTLDevice, ceilingFraction: Double = 0.65) {
        self.device = device
        self.ceilingFraction = ceilingFraction
        self.environment = MemoryEnvironment.probe(device: device, ceilingFraction: ceilingFraction)

        let totalRAMGB = Double(environment.totalRAMBytes) / 1_073_741_824.0
        let ceilingGB = Double(environment.softCeilingBytes) / 1_073_741_824.0
        let initLog = OSLog(subsystem: "com.waratahlabs.quantum-edge", category: "MetalMemory")
        os_log(.info, log: initLog,
               "MetalMemoryManager init — device: %{public}@, platform: %{public}@, RAM: %.1f GB, ceiling: %.1f GB (%{public}@), GPU in use: %d MB",
               device.name,
               environment.platform.rawValue,
               totalRAMGB,
               ceilingGB,
               environment.gpuBudgetSource,
               environment.gpuBytesInUse / 1_048_576)

        registerMemoryPressureObserver()
    }

    // MARK: - Budget

    /// Returns true if allocating `bytes` more would remain under `softCeilingBytes`.
    public func canAllocate(bytes: Int) -> Bool {
        (gpuBytesInUse + bytes) < softCeilingBytes
    }

    // MARK: - Heap Factory

    /// Creates a `MTLHeap` with `.storageModePrivate` for the requested byte count.
    ///
    /// Private storage means:
    /// - The allocation does NOT count against CPU Jetsam RSS
    /// - The CPU cannot access buffer contents directly (no `contents()` pointer)
    /// - CPU↔GPU data movement requires blit command encoders
    ///
    /// - Throws: `MetalMemoryError.budgetExceeded` if `canAllocate` returns false.
    /// - Throws: `MetalMemoryError.heapAllocationFailed` if `device.makeHeap` returns nil.
    public func makeHeap(bytes: Int) throws -> MTLHeap {
        guard canAllocate(bytes: bytes) else {
            let available = softCeilingBytes - gpuBytesInUse
            os_log(.error, log: log,
                   "makeHeap: budget exceeded — requested %d MB, available %d MB",
                   bytes / 1_048_576, available / 1_048_576)
            throw MetalMemoryError.budgetExceeded(requested: bytes, available: available)
        }

        let descriptor = MTLHeapDescriptor()
        descriptor.size = bytes
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked

        guard let heap = device.makeHeap(descriptor: descriptor) else {
            os_log(.error, log: log,
                   "makeHeap: device.makeHeap returned nil for %d MB", bytes / 1_048_576)
            throw MetalMemoryError.heapAllocationFailed(bytes: bytes)
        }

        os_log(.info, log: log,
               "makeHeap: allocated %d MB private heap — GPU now %d MB / ceiling %d MB",
               bytes / 1_048_576,
               gpuBytesInUse / 1_048_576,
               softCeilingBytes / 1_048_576)

        return heap
    }

    // MARK: - Description

    /// Human-readable summary of the current memory state.
    public var description: String {
        let totalRAMGB = Double(environment.totalRAMBytes) / 1_073_741_824.0
        let gpuMB = gpuBytesInUse / 1_048_576
        let ceilMB = softCeilingBytes / 1_048_576
        let cpuMB = cpuBytesAvailable / 1_048_576
        return """
        MetalMemoryManager
          Device:       \(device.name)
          Platform:     \(environment.platform.rawValue)
          Total RAM:    \(String(format: "%.1f", totalRAMGB)) GB
          GPU in use:   \(gpuMB) MB
          Ceiling:      \(ceilMB) MB (\(environment.gpuBudgetSource))
          Max qubits:   \(environment.qubitSummary)
          CPU headroom: \(cpuMB) MB (Jetsam budget, not GPU)
        """
    }

    // MARK: - Private

    private func registerMemoryPressureObserver() {
#if canImport(UIKit)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            os_log(.error, log: self.log,
                   "CPU memory warning received — CPU headroom: %d MB, GPU in use: %d MB",
                   self.cpuBytesAvailable / 1_048_576,
                   self.gpuBytesInUse / 1_048_576)
            self.onMemoryPressure?()
        }
#endif
    }
}

// MARK: - MemoryEnvironment

/// Environment probe result — computed once, used for display and ceiling decisions.
public struct MemoryEnvironment {
    public enum Platform: String {
        case iOS = "iOS"
        case visionOS = "visionOS"
        case macOS = "macOS"
    }

    /// Detected platform.
    public let platform: Platform
    /// Total device RAM from ProcessInfo.physicalMemory.
    public let totalRAMBytes: Int
    /// GPU budget source: macOS uses recommendedMaxWorkingSetSize; others use physicalMemory.
    public let gpuBudgetBytes: Int
    /// Soft ceiling the memory manager will enforce (gpuBudget × fraction or physicalMemory × fraction).
    public let softCeilingBytes: Int
    /// Estimated maximum qubits: floor(log2(softCeilingBytes / 8)).
    public let estimatedMaxQubits: Int
    /// GPU bytes in use at probe time (device.currentAllocatedSize).
    public let gpuBytesInUse: Int
    /// Human-readable description of how gpuBudgetBytes was derived.
    public let gpuBudgetSource: String
    /// The ceiling fraction used.
    public let ceilingFraction: Double

    /// e.g. "~30 qubits  (~8.0 GB state vector)"
    public var qubitSummary: String {
        let svBytes = Int(pow(2.0, Double(estimatedMaxQubits))) * 8
        let svGB = Double(svBytes) / 1_073_741_824.0
        return "~\(estimatedMaxQubits) qubits  (~\(String(format: "%.1f", svGB)) GB state vector)"
    }

    /// Conservative ceiling fraction (default). Balances OS headroom vs. qubit count.
    public static let conservativeCeilingFraction: Double = 0.65
    /// Aggressive ceiling fraction for experiments targeting maximum qubit count.
    /// 0.70 × 12 GB = 8.4 GB ceiling — just enough for n=30 (8.0 GiB) on a 12 GB device.
    public static let aggressiveCeilingFraction: Double = 0.70

    /// Probe the current device and compute all memory budget values.
    public static func probe(device: MTLDevice, ceilingFraction: Double = conservativeCeilingFraction) -> MemoryEnvironment {
        let totalRAM = Int(ProcessInfo.processInfo.physicalMemory)

        #if os(macOS)
        // macOS: recommendedMaxWorkingSetSize is the real GPU budget — use 85% of it.
        // This API is API_UNAVAILABLE(ios) and must stay inside this #if block.
        let gpuBudget = Int(device.recommendedMaxWorkingSetSize)
        let effectiveFraction = 0.85
        let ceiling = Int(Double(gpuBudget) * effectiveFraction)
        let budgetSource = "recommendedMaxWorkingSetSize × 0.85"
        let platform = Platform.macOS
        #elseif os(visionOS)
        // visionOS: Metal shares budget with the compositor; use a conservative 0.55
        // regardless of the caller's ceilingFraction — visionOS terminates more aggressively
        // than iOS, so the 0.70 "aggressive" iOS value is unsafe here.
        let gpuBudget = totalRAM
        let effectiveFraction = min(ceilingFraction, 0.55)
        let ceiling = Int(Double(totalRAM) * effectiveFraction)
        let budgetSource = "physicalMemory × \(String(format: "%.2f", effectiveFraction)) (visionOS cap)"
        let platform = Platform.visionOS
        #else
        // iOS: private heap escapes Jetsam; use the caller's fraction (0.65 conservative, 0.70 aggressive).
        let gpuBudget = totalRAM
        let effectiveFraction = ceilingFraction
        let ceiling = Int(Double(totalRAM) * effectiveFraction)
        let budgetSource = "physicalMemory × \(String(format: "%.2f", effectiveFraction))"
        let platform = Platform.iOS
        #endif

        let maxAmplitudes = ceiling / 8
        let estimatedMaxQubits = maxAmplitudes > 0 ? Int(log2(Double(maxAmplitudes))) : 0

        return MemoryEnvironment(
            platform: platform,
            totalRAMBytes: totalRAM,
            gpuBudgetBytes: gpuBudget,
            softCeilingBytes: ceiling,
            estimatedMaxQubits: estimatedMaxQubits,
            gpuBytesInUse: device.currentAllocatedSize,
            gpuBudgetSource: budgetSource,
            ceilingFraction: ceilingFraction
        )
    }
}
