import XCTest
import Metal
@testable import QuantumEdgeKit

final class MemoryEnvironmentTests: XCTestCase {

    var device: MTLDevice!

    override func setUpWithError() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        device = dev
    }

    // MARK: - Platform detection

    func testProbePlatformIsMacOS() {
        let env = MemoryEnvironment.probe(device: device)
        // The test runner always runs on macOS
        XCTAssertEqual(env.platform, .macOS)
    }

    // MARK: - Budget source

    func testMacOSBudgetSourceContainsRecommendedMaxWorkingSetSize() {
        let env = MemoryEnvironment.probe(device: device)
        XCTAssertTrue(
            env.gpuBudgetSource.contains("recommendedMaxWorkingSetSize"),
            "macOS budget source should reference recommendedMaxWorkingSetSize, got: \(env.gpuBudgetSource)"
        )
    }

    func testMacOSCeilingIs85PercentOfBudget() {
        let env = MemoryEnvironment.probe(device: device)
        let expected = Int(Double(env.gpuBudgetBytes) * 0.85)
        // Allow ±1 byte for floating-point truncation
        XCTAssertEqual(env.softCeilingBytes, expected, accuracy: 1,
                       "macOS ceiling should be gpuBudgetBytes × 0.85")
    }

    // MARK: - Ceiling fraction constants

    func testConservativeFractionValue() {
        XCTAssertEqual(MemoryEnvironment.conservativeCeilingFraction, 0.65, accuracy: 1e-10)
    }

    func testAggressiveFractionValue() {
        XCTAssertEqual(MemoryEnvironment.aggressiveCeilingFraction, 0.70, accuracy: 1e-10)
    }

    func testCustomFractionIsRespected() {
        // On macOS the fraction is overridden to 0.85 — custom fraction applies to iOS/visionOS.
        // We can still verify that the probe doesn't crash with arbitrary fractions.
        let env1 = MemoryEnvironment.probe(device: device, ceilingFraction: 0.50)
        let env2 = MemoryEnvironment.probe(device: device, ceilingFraction: 0.80)
        // Both should produce a positive ceiling
        XCTAssertGreaterThan(env1.softCeilingBytes, 0)
        XCTAssertGreaterThan(env2.softCeilingBytes, 0)
        // On macOS both use the fixed 0.85 — ceilings should be equal
        XCTAssertEqual(env1.softCeilingBytes, env2.softCeilingBytes,
                       "macOS ceiling ignores caller fraction — should be equal regardless")
    }

    // MARK: - Qubit estimate

    func testEstimatedMaxQubitsFormula() {
        let env = MemoryEnvironment.probe(device: device)
        let maxAmplitudes = env.softCeilingBytes / 8
        let expected = maxAmplitudes > 0 ? Int(log2(Double(maxAmplitudes))) : 0
        XCTAssertEqual(env.estimatedMaxQubits, expected)
    }

    func testEstimatedMaxQubitsIsPositive() {
        let env = MemoryEnvironment.probe(device: device)
        XCTAssertGreaterThan(env.estimatedMaxQubits, 0)
    }

    func testEstimatedMaxQubitsIsReasonable() {
        // Any real device should support at least 20 qubits (8 MB) and at most 40 qubits
        let env = MemoryEnvironment.probe(device: device)
        XCTAssertGreaterThanOrEqual(env.estimatedMaxQubits, 20,
                                    "Expected at least 20 qubits on any real device")
        XCTAssertLessThanOrEqual(env.estimatedMaxQubits, 40,
                                 "More than 40 qubits would require >8 TB — unexpected")
    }

    // MARK: - qubitSummary

    func testQubitSummaryContainsEstimate() {
        let env = MemoryEnvironment.probe(device: device)
        XCTAssertTrue(env.qubitSummary.contains("~\(env.estimatedMaxQubits) qubits"),
                      "qubitSummary should contain the estimated qubit count")
    }

    func testQubitSummaryContainsGB() {
        let env = MemoryEnvironment.probe(device: device)
        XCTAssertTrue(env.qubitSummary.contains("GB"),
                      "qubitSummary should include a GB size estimate")
    }

    // MARK: - totalRAMBytes

    func testTotalRAMMatchesProcessInfo() {
        let env = MemoryEnvironment.probe(device: device)
        let expected = Int(ProcessInfo.processInfo.physicalMemory)
        XCTAssertEqual(env.totalRAMBytes, expected)
    }

    // MARK: - canAllocate / budgetExceeded

    func testCanAllocateSmallBuffer() throws {
        let mgr = MetalMemoryManager(device: device)
        XCTAssertTrue(mgr.canAllocate(bytes: 1024 * 1024),  // 1 MB — always fits
                      "Should be able to allocate 1 MB on any device")
    }

    func testCanAllocateReturnsFalseWhenOverCeiling() {
        let mgr = MetalMemoryManager(device: device)
        // Request more than the entire ceiling — must be rejected
        XCTAssertFalse(mgr.canAllocate(bytes: mgr.softCeilingBytes + 1))
    }

    func testMakeHeapThrowsBudgetExceededWhenOverCeiling() {
        let mgr = MetalMemoryManager(device: device)
        let oversized = mgr.softCeilingBytes + 1
        XCTAssertThrowsError(try mgr.makeHeap(bytes: oversized)) { error in
            guard case MetalMemoryError.budgetExceeded(let requested, _) = error else {
                XCTFail("Expected budgetExceeded, got \(error)")
                return
            }
            XCTAssertEqual(requested, oversized)
        }
    }

    func testMakeHeapSucceedsForSmallAllocation() throws {
        let mgr = MetalMemoryManager(device: device)
        // 16 MB — tiny relative to any real GPU budget
        let heap = try mgr.makeHeap(bytes: 16 * 1024 * 1024)
        XCTAssertGreaterThan(heap.size, 0)
    }
}
