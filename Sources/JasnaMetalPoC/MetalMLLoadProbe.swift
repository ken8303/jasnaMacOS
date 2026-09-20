import Foundation
import Metal

struct MetalMLPipelineLoadMeasurement: Sendable {
    let package: String
    let cacheHit: Bool
    let libraryMilliseconds: Double
    let compilerMilliseconds: Double
    let specializationMilliseconds: Double
    let totalMilliseconds: Double

    static func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = start.duration(to: .now).components
        return Double(elapsed.seconds) * 1_000
            + Double(elapsed.attoseconds) / 1_000_000_000_000_000
    }
}

enum MetalMLLoadProbe {
    // Match the order of construction in the full graph. These eight packages
    // bracket the four long gaps seen in the first-use graph setup trace.
    static let packages = ["backward_1", "forward_1", "backward_2", "forward_2"].flatMap {
        ["offset_\($0)", "backbone_\($0)"]
    }

    static func run(
        load: (String) throws -> MetalMLPipelineLoadMeasurement,
        report: (String) -> Void
    ) throws -> [MetalMLPipelineLoadMeasurement] {
        var measurements = [MetalMLPipelineLoadMeasurement]()
        for pass in 1...2 {
            report("Load-only pass \(pass)/2: \(pass == 1 ? "first process load" : "in-process cache reuse")")
            for package in packages {
                report("Load-only pass \(pass)/2: starting \(package)")
                let value = try load(package)
                guard value.package == package else {
                    throw DeformConvError.commandFailed("load probe received a different package")
                }
                report(
                    "Load-only pass \(pass)/2: \(package), cache \(value.cacheHit ? "hit" : "miss"), "
                        + "library \(String(format: "%.3f", value.libraryMilliseconds)) ms, "
                        + "compiler \(String(format: "%.3f", value.compilerMilliseconds)) ms, "
                        + "specialization \(String(format: "%.3f", value.specializationMilliseconds)) ms, "
                        + "total \(String(format: "%.3f", value.totalMilliseconds)) ms"
                )
                guard pass != 2 || value.cacheHit else {
                    throw DeformConvError.commandFailed("load-only second pass missed the pipeline cache")
                }
                measurements.append(value)
            }
        }
        return measurements
    }
}

@available(macOS 27.0, *)
func runMetalMLLoadOnly(commandLine: JasnaCommandLine, index: Int) throws {
    guard index == 1, commandLine.arguments.count == 3 else {
        throw DeformConvError.commandFailed("--metal-ml-load-only requires one MetalML directory and no other modes")
    }
    let directory = URL(fileURLWithPath: commandLine[index + 1], isDirectory: true)
    for package in MetalMLLoadProbe.packages {
        let url = directory.appendingPathComponent("\(package).mtlpackage")
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw DeformConvError.commandFailed("missing Metal ML package: \(url.path)")
        }
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw DeformConvError.metalUnavailable
    }
    SideBySideRestoration.report("Metal ML load-only device: \(device.name)")
    SideBySideRestoration.report("No video decode, detection, restoration, or GPU inference is dispatched")
    SideBySideRestoration.report("First process load may use Apple's existing persistent caches; no caches are cleared")
    let measurements = try MetalMLLoadProbe.run(load: { package in
        var measurement: MetalMLPipelineLoadMeasurement?
        _ = try makeMetalMLPipeline(
            device: device,
            packageURL: directory.appendingPathComponent("\(package).mtlpackage"),
            reportPhase: { phase in
                SideBySideRestoration.report("Pipeline \(package): \(phase)")
            },
            reportMeasurement: { measurement = $0 }
        )
        guard let measurement else {
            throw DeformConvError.commandFailed("missing pipeline-load measurement")
        }
        return measurement
    }, report: SideBySideRestoration.report)
    let count = MetalMLLoadProbe.packages.count
    let first = measurements.prefix(count).reduce(0) { $0 + $1.totalMilliseconds }
    let reused = measurements.suffix(count).reduce(0) { $0 + $1.totalMilliseconds }
    SideBySideRestoration.report(
        "Metal ML load-only: PASS, \(count) packages, first-load total "
            + "\(String(format: "%.3f", first)) ms, cached total "
            + "\(String(format: "%.3f", reused)) ms"
    )
}
