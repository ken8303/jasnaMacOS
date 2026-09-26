import Foundation
import Metal

enum CopyError: Error { case failed(String) }
struct Sample: Codable {
    let trial: Int
    let mode: String
    let position: Int
    let submitWaitMS: Double
    let gpuMS: Double?
    let copyMS: Double
    let readMS: Double
    let totalMS: Double
}
struct Case: Codable {
    let width: Int
    let height: Int
    let reusableDestinationBytes: Int
    let samples: [Sample]
}
struct Report: Encodable {
    let status = "PASS"
    let device: String
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let scope = "Generated GPU float4 pattern only; no video, model or application code. 8 warmup pairs, 64 measured pairs per size, alternating order. Pipeline/buffer/reusable destination setup and reference validation excluded. Fresh allocation included; destination release excluded. Full CPU checksum consumption included. GPU time overlaps submit/wait and must not be added to it."
    let cases: [Case]
}
func ms(_ start: UInt64, _ end: UInt64) -> Double { Double(end - start) / 1_000_000 }
func p(_ values: [Double], _ fraction: Double = 0.5) -> Double {
    let sorted = values.sorted()
    return sorted[Int(Double(sorted.count - 1) * fraction)]
}
@inline(never)
func consume(_ pixels: [SIMD4<Float>]) -> SIMD4<Double> {
    var sum = SIMD4<Double>.zero
    for pixel in pixels { sum += SIMD4<Double>(pixel) }
    return sum
}

do {
    guard CommandLine.arguments.count == 2 else { throw CopyError.failed("REPORT.json required") }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw CopyError.failed("Metal unavailable")
    }
    let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void generated_output(device float4 *output [[buffer(0)]],
            constant uint2 &args [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if (i >= args.x) return;
            float v = float((i + args.y * 17u) % 251u) / 256.0f;
            output[i] = float4(v, v / 2.0f, 1.0f - v, 1.0f);
        }
        """, options: nil)
    guard let function = library.makeFunction(name: "generated_output") else {
        throw CopyError.failed("Kernel missing")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    print("Synthetic GPU-written output copy on \(device.name)")
    print("64 balanced pairs after 8 warmups; each partner gets its own completed GPU write.")
    print("Timings include fresh allocation, copying and full CPU consumption; setup, validation and release excluded.")
    var cases = [Case]()
    for (width, height) in [(96, 80), (256, 256), (512, 512), (640, 512)] {
        let count = width * height, bytes = count * MemoryLayout<SIMD4<Float>>.stride
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw CopyError.failed("Buffer allocation failed")
        }
        var reused = [SIMD4<Float>](repeating: .zero, count: count)
        var samples = [Sample]()
        for trial in -8..<64 {
            let seed = trial + 8
            let expected = (0..<count).map { index -> SIMD4<Float> in
                let v = Float((index + seed * 17) % 251) / 256
                return SIMD4(v, v / 2, 1 - v, 1)
            }
            let expectedSum = consume(expected)
            for (position, mode) in (trial % 2 == 0 ? ["fresh", "reuse"] : ["reuse", "fresh"]).enumerated() {
                try autoreleasepool {
                    let start = DispatchTime.now().uptimeNanoseconds
                    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                        throw CopyError.failed("Command allocation failed")
                    }
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(buffer, offset: 0, index: 0)
                    var args = SIMD2<UInt32>(UInt32(count), UInt32(seed))
                    encoder.setBytes(&args, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
                    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    guard command.status == .completed else {
                        throw CopyError.failed("GPU failed: \(String(describing: command.error))")
                    }
                    let completed = DispatchTime.now().uptimeNanoseconds
                    // No CPU read of the shared output occurs before this point.
                    var fresh = [SIMD4<Float>]()
                    if mode == "fresh" {
                        fresh = Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: count))
                    } else {
                        reused.withUnsafeMutableBytes { $0.baseAddress!.copyMemory(from: buffer.contents(), byteCount: bytes) }
                    }
                    let copied = DispatchTime.now().uptimeNanoseconds
                    let checksum = mode == "fresh" ? consume(fresh) : consume(reused)
                    let read = DispatchTime.now().uptimeNanoseconds
                    guard checksum == expectedSum, (mode == "fresh" ? fresh == expected : reused == expected) else {
                        throw CopyError.failed("Pixel or checksum mismatch at \(width)x\(height), trial \(trial), \(mode)")
                    }
                    let gpuDuration = command.gpuEndTime - command.gpuStartTime
                    if trial >= 0 {
                        samples.append(Sample(trial: trial, mode: mode, position: position,
                            submitWaitMS: ms(start, completed),
                            gpuMS: command.gpuStartTime > 0 && gpuDuration > 0 && gpuDuration.isFinite ? gpuDuration * 1000 : nil,
                            copyMS: ms(completed, copied), readMS: ms(copied, read), totalMS: ms(start, read)))
                    }
                }
            }
        }
        for mode in ["fresh", "reuse"] {
            let selected = samples.filter { $0.mode == mode }
            print(String(format: "%dx%d %@: total %.3f ms; submit/wait %.3f; copy %.3f; CPU read %.3f",
                width, height, mode, p(selected.map(\.totalMS)), p(selected.map(\.submitWaitMS)),
                p(selected.map(\.copyMS)), p(selected.map(\.readMS))))
        }
        let differences = (0..<64).map { trial -> Double in
            let pair = samples.filter { $0.trial == trial }
            return pair.first { $0.mode == "reuse" }!.totalMS - pair.first { $0.mode == "fresh" }!.totalMS
        }
        print(String(format: "  Paired total reuse−fresh %.3f ms [P10 %.3f–P90 %.3f]; faster %d/64",
            p(differences), p(differences, 0.1), p(differences, 0.9), differences.filter { $0 < 0 }.count))
        cases.append(Case(width: width, height: height, reusableDestinationBytes: bytes, samples: samples))
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(Report(device: device.name, cases: cases)).write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
    print("PASS GPU-written output copy: every pixel and checksum matched the CPU reference")
} catch {
    print("INCOMPLETE or FAILED: \(error)")
    exit(1)
}
