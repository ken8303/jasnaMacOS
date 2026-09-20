import Foundation
import Metal

enum Failure: Error { case invalid(String) }
struct Sample: Encodable {
    let trial: Int
    let mode: String
    let position: Int
    let submissions: Int
    let waits: Int
    let outputs = 4
    let submitWaitMS: Double
    let copyMS: Double
    let readMS: Double
    let totalMS: Double
}
struct Measurement: Encodable {
    let width: Int
    let height: Int
    let sharedOutputBytes: Int
    let samples: [Sample]
}
struct Report: Encodable {
    let status = "PASS"
    let device: String
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let scope = "Generated float4 GPU writes only. Four independent shared buffers, four distinct patterns per trial, all outputs copied to fresh arrays and consumed. 12 warmup trials and 60 measured trials per mode; all six mode permutations balanced. Separate submits/waits four times; queued commits four commands on one queue then waits on the last and checks all completed; grouped submits/waits once. CPU copies occur after all four writes. Setup, reference validation and destination release excluded. Total is four outputs, not per-image latency."
    let measurements: [Measurement]
}
func ms(_ start: UInt64, _ end: UInt64) -> Double { Double(end - start) / 1_000_000 }
func percentile(_ xs: [Double], _ fraction: Double = 0.5) -> Double {
    xs.sorted()[Int(Double(xs.count - 1) * fraction)]
}
@inline(never) func consume(_ pixels: [SIMD4<Float>]) -> SIMD4<Double> {
    var sum = SIMD4<Double>.zero
    for pixel in pixels { sum += SIMD4<Double>(pixel) }
    return sum
}
do {
    guard CommandLine.arguments.count == 2 else { throw Failure.invalid("Report path required") }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw Failure.invalid("Metal unavailable")
    }
    let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void generate(device float4 *out [[buffer(0)]],
            constant uint2 &args [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if (i >= args.x) return;
            float v = float((i + args.y * 17u) % 251u) / 256.0f;
            out[i] = float4(v, v / 2, 1 - v, 1);
        }
        """, options: nil)
    guard let function = library.makeFunction(name: "generate") else { throw Failure.invalid("Kernel missing") }
    let pipeline = try device.makeComputePipelineState(function: function)
    var measurements = [Measurement]()
    let orders = [["separate", "queued", "grouped"], ["separate", "grouped", "queued"],
                  ["queued", "separate", "grouped"], ["queued", "grouped", "separate"],
                  ["grouped", "separate", "queued"], ["grouped", "queued", "separate"]]
    print("Synthetic four-output submission comparison on \(device.name)")
    print("60 balanced trials/mode after 12 warmups; all six mode orders. Times are per four outputs, all copied and consumed.")
    for (width, height) in [(96, 80), (256, 256), (512, 512), (640, 512)] {
        let count = width * height, bytes = count * MemoryLayout<SIMD4<Float>>.stride
        let buffers: [MTLBuffer] = try (0..<4).map { _ in
            guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw Failure.invalid("Shared output allocation failed")
            }
            return b
        }
        guard Set(buffers.map { ObjectIdentifier($0) }).count == 4 else { throw Failure.invalid("Aliased outputs") }
        var samples = [Sample]()
        for trial in -12..<60 {
            let seeds = (0..<4).map { (trial + 12) * 4 + $0 }
            let expected = seeds.map { seed in
                (0..<count).map { index -> SIMD4<Float> in
                    let v = Float((index + seed * 17) % 251) / 256
                    return SIMD4(v, v / 2, 1 - v, 1)
                }
            }
            let sums = expected.map(consume)
            for (position, mode) in orders[(trial + 12) % orders.count].enumerated() {
                try autoreleasepool {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let groups = mode == "grouped" ? [[0, 1, 2, 3]] : [[0], [1], [2], [3]]
                    var written = [Int]()
                    var commands = [MTLCommandBuffer]()
                    var waits = 0
                    for group in groups {
                        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                            throw Failure.invalid("Command allocation failed")
                        }
                        encoder.setComputePipelineState(pipeline)
                        for slot in group {
                            encoder.setBuffer(buffers[slot], offset: 0, index: 0)
                            var args = SIMD2<UInt32>(UInt32(count), UInt32(seeds[slot]))
                            encoder.setBytes(&args, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
                            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            written.append(slot)
                        }
                        encoder.endEncoding()
                        commands.append(command)
                        command.commit()
                        if mode != "queued" {
                            command.waitUntilCompleted()
                            waits += 1
                        }
                    }
                    if mode == "queued" {
                        commands.last!.waitUntilCompleted()
                        waits += 1
                    }
                    // Validate every command, not just the final command. Never
                    // read partially written outputs after an earlier failure.
                    for command in commands {
                        guard command.status == .completed else { throw Failure.invalid("GPU incomplete/failed: \(String(describing: command.error))") }
                    }
                    let completed = DispatchTime.now().uptimeNanoseconds
                    let outputs = buffers.map { buffer in
                        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: SIMD4<Float>.self), count: count))
                    }
                    let copied = DispatchTime.now().uptimeNanoseconds
                    let checksums = outputs.map(consume)
                    let read = DispatchTime.now().uptimeNanoseconds
                    guard written == [0, 1, 2, 3], commands.count == (mode == "grouped" ? 1 : 4),
                          waits == (mode == "separate" ? 4 : 1), outputs == expected, checksums == sums else {
                        throw Failure.invalid("Work count, pixel or checksum mismatch")
                    }
                    if trial >= 0 {
                        samples.append(Sample(trial: trial, mode: mode, position: position,
                            submissions: commands.count, waits: waits, submitWaitMS: ms(start, completed),
                            copyMS: ms(completed, copied), readMS: ms(copied, read), totalMS: ms(start, read)))
                    }
                }
            }
        }
        for mode in ["separate", "queued", "grouped"] {
            let selected = samples.filter { $0.mode == mode }
            print(String(format: "%dx%d %@: total %.3f ms [P10 %.3f–P90 %.3f]; submit/wait %.3f; copy %.3f; CPU read %.3f",
                width, height, mode, percentile(selected.map(\.totalMS)), percentile(selected.map(\.totalMS), 0.1),
                percentile(selected.map(\.totalMS), 0.9), percentile(selected.map(\.submitWaitMS)),
                percentile(selected.map(\.copyMS)), percentile(selected.map(\.readMS))))
        }
        for (candidate, control) in [("queued", "separate"), ("grouped", "queued"), ("grouped", "separate")] {
        let paired = (0..<60).map { trial -> Double in
            let pair = samples.filter { $0.trial == trial }
            return pair.first { $0.mode == candidate }!.totalMS - pair.first { $0.mode == control }!.totalMS
        }
        print(String(format: "  Paired %@−%@ %.3f ms [P10 %.3f–P90 %.3f]; faster %d/60",
            candidate, control, percentile(paired), percentile(paired, 0.1), percentile(paired, 0.9), paired.filter { $0 < 0 }.count))
        }
        measurements.append(Measurement(width: width, height: height, sharedOutputBytes: bytes * 4, samples: samples))
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(Report(device: device.name, measurements: measurements)).write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
    print("PASS grouped submission: all four outputs and checksums matched on every trial")
} catch {
    print("INCOMPLETE or FAILED: \(error)")
    exit(1)
}
