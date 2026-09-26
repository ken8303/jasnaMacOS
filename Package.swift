// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "JasnaMetalPoC",
    platforms: [.macOS(.v27)],
    products: [
        .executable(name: "JasnaMetalPoC", targets: ["JasnaMetalPoC"]),
        .executable(name: "JasnaMacApp", targets: ["JasnaMacApp"]),
        .library(name: "JasnaAppSupport", targets: ["JasnaAppSupport"]),
    ],
    targets: [
        .executableTarget(name: "JasnaMetalPoC"),
        .target(name: "JasnaAppSupport"),
        .executableTarget(
            name: "JasnaMacApp",
            dependencies: ["JasnaAppSupport"]
        ),
        .testTarget(
            name: "JasnaMetalPoCTests",
            dependencies: ["JasnaMetalPoC"]
        ),
        .testTarget(
            name: "JasnaAppSupportTests",
            dependencies: ["JasnaAppSupport"]
        ),
    ]
)
