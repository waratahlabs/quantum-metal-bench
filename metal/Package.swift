// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "QuantumEdge",
    platforms: [.macOS(.v14), .iOS(.v16), .visionOS(.v1)],
    products: [
        .library(name: "QuantumEdgeKit", targets: ["QuantumEdgeKit"]),
        .library(name: "QuantumEdgeBridge", type: .dynamic, targets: ["QuantumEdgeBridge"]),
        .executable(name: "qe-bench", targets: ["QuantumEdgeCLI"]),
    ],
    targets: [
        .target(name: "QuantumEdgeKit"),
        .target(
            name: "QuantumEdgeBridge",
            dependencies: ["QuantumEdgeKit"],
            swiftSettings: [
                .unsafeFlags(["-strict-concurrency=minimal"])
            ]
        ),
        .executableTarget(name: "QuantumEdgeCLI", dependencies: ["QuantumEdgeKit"]),
        .testTarget(name: "QuantumEdgeKitTests", dependencies: ["QuantumEdgeKit"]),
        .testTarget(
            name: "QuantumEdgeBridgeTests",
            dependencies: ["QuantumEdgeBridge", "QuantumEdgeKit"],
            swiftSettings: [
                .unsafeFlags(["-strict-concurrency=minimal"])
            ]
        ),
    ]
)
