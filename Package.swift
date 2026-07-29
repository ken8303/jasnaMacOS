// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "JasnaMetalPoC",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "JasnaMetalPoC", targets: ["JasnaMetalPoC"]),
    ],
    targets: [
        .executableTarget(name: "JasnaMetalPoC"),
        .testTarget(
            name: "JasnaMetalPoCTests",
            dependencies: ["JasnaMetalPoC"]
        ),
    ]
)
