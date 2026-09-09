// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "SyntheticResamplingLab",
    platforms: [.macOS(.v27)],
    products: [.executable(name: "SyntheticResamplingLab", targets: ["SyntheticResamplingLab"])],
    targets: [
        .executableTarget(name: "SyntheticResamplingLab"),
        .testTarget(name: "SyntheticResamplingLabTests", dependencies: ["SyntheticResamplingLab"]),
    ]
)
