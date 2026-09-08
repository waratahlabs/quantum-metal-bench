import XCTest

/// End-to-end proof the tap-to-result pipeline actually works, not just that
/// the app compiles and launches. Runs the full locked-protocol benchmark
/// (n=28) via the real UI, same as a human tapping the button — on the
/// simulator this exercises the host Mac's own GPU through Metal's
/// simulator translation layer, which is enough to prove the SwiftUI/Task/
/// MainActor plumbing and the QuantumEdgeKit call path are wired correctly.
/// It is NOT a source of real device timing/power data — see ios/README.md.
final class QMBenchUITests: XCTestCase {
    func testRunBenchmarkCompletesAndReportsDone() throws {
        let app = XCUIApplication()
        app.launch()

        let statusText = app.staticTexts["statusText"]
        XCTAssertTrue(statusText.waitForExistence(timeout: 5))
        XCTAssertEqual(statusText.label, "Idle")

        let button = app.buttons["runBenchmarkButton"]
        XCTAssertTrue(button.exists)
        button.tap()

        // n=28, depth=4, reps=5 takes tens of seconds on real Apple Silicon
        // (11-24s measured across devices this session) -- generous timeout
        // for the simulator's Metal translation layer.
        let donePredicate = NSPredicate(format: "label BEGINSWITH 'Done'")
        let doneExpectation = XCTNSPredicateExpectation(predicate: donePredicate, object: statusText)
        let result = XCTWaiter().wait(for: [doneExpectation], timeout: 120)

        XCTAssertEqual(result, .completed, "Benchmark did not report Done within 120s")
        XCTAssertTrue(statusText.label.hasPrefix("Done"), "Unexpected final status: \(statusText.label)")
    }
}
