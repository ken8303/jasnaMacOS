// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "JasnaMetalPoC",
    platforms: [.macOS(.v27)],
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
