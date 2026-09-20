import Foundation
import Metal

enum Failure: Error { case invalid(String) }
func pixel(_ index: Int, _ seed: Int) -> SIMD4<Float> {
    let v = Float((index + seed * 17) % 251) / 256
    return SIMD4(v, v / 2, 1 - v, 1)
}
func ms(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000 }
func percentile(_ xs: [Double], _ f: Double = 0.5) -> Double { xs.sorted()[Int(Double(xs.count - 1) * f)] }
struct Sample: Encodable {
    let trial: Int
    let depth: Int
    let position: Int
    let totalMS: Double
    let latencyMS: [Double]
    let maxPending: Int
    let outputs: Int
    let poolBytes: Int
    let cpuCopyPayloadBytes: Int
}
struct SizeResult: Encodable { let size: Int; let samples: [Sample] }
struct Report: Encodable {
    let status = "PASS"
    let device: String
    let scope = "Synthetic generated float4 writes only. Fixed ring of 1/2/4 slots, four buffers and one command per batch; 12 batches/48 outputs per trial. 6 warmup trials and 12 measured trials per depth, all six orders balanced. Total includes submission, wait, fresh CPU copy/release and full checksum reading; excludes pool/pipeline setup, reference generation and full-pixel validation. Batch latency is submission-start to CPU-consumption completion. Memory is allocated Metal pool bytes plus one image CPU payload, not peak process RSS. Pre/post full-pixel checks and every timed checksum required."
    let results: [SizeResult]
}
final class Slot {
    let buffers: [MTLBuffer]
    var command: MTLCommandBuffer?
    var batch = -1
    var started: UInt64 = 0
    init(device: MTLDevice, bytes: Int) throws {
        buffers = try (0..<4).map { _ in
            guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else { throw Failure.invalid("Allocation failed") }
            return b
        }
    }
}

do {
    guard CommandLine.arguments.count == 2, let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else { throw Failure.invalid("Report path and Metal device required") }
    let library = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void generate(device float4 *out [[buffer(0)]], constant uint2 &args [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            if (i >= args.x) return;
            float v = float((i + args.y * 17u) % 251u) / 256.0f;
            out[i] = float4(v, v / 2, 1 - v, 1);
        }
        """, options: nil)
    guard let function = library.makeFunction(name: "generate") else { throw Failure.invalid("Kernel missing") }
    let pipeline = try device.makeComputePipelineState(function: function)
    let orders = [[1,2,4], [1,4,2], [2,1,4], [2,4,1], [4,1,2], [4,2,1]]
    var results = [SizeResult]()
    print("Synthetic bounded pipeline on \(device.name): 12 batches × 4 outputs/trial")
    print("Depths 1/2/4; 12 measured trials each after 6 warmups; pool memory is not process RSS.")
    for size in [96, 256, 512] {
        let count = size * size, bytes = count * MemoryLayout<SIMD4<Float>>.stride
        var samples = [Sample]()
        // Each depth gets a separate bounded pool, allocated/released outside its
        // trial. No pool grows with video length or number of completed batches.
        func run(depth: Int, trial: Int, position: Int, validate: Bool) throws -> Sample {
            let slots = try (0..<depth).map { _ in try Slot(device: device, bytes: bytes) }
            let bufferIDs = slots.flatMap { $0.buffers.map { ObjectIdentifier($0) } }
            guard Set(bufferIDs).count == depth * 4 else { throw Failure.invalid("Aliased pool buffers") }
            let seedBase = (trial + 20) * 48
            // Calculate exact expected sums using the pattern's 251-pixel period.
            let sums = (0..<48).map { output -> SIMD4<Double> in
                var sum = SIMD4<Double>.zero
                for i in 0..<251 { sum += SIMD4<Double>(pixel(i, seedBase + output)) * Double(count / 251) }
                for i in 0..<(count % 251) { sum += SIMD4<Double>(pixel(i, seedBase + output)) }
                return sum
            }
            var submitted = 0, consumed = 0, maxPending = 0
            var latencies = [Double]()
            let start = DispatchTime.now().uptimeNanoseconds
            func submit(_ batch: Int) throws {
                let slot = slots[batch % depth]
                guard slot.command == nil else { throw Failure.invalid("Reusing a live slot") }
                slot.started = DispatchTime.now().uptimeNanoseconds
                guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { throw Failure.invalid("Command allocation failed") }
                encoder.setComputePipelineState(pipeline)
                for output in 0..<4 {
                    encoder.setBuffer(slot.buffers[output], offset: 0, index: 0)
                    var args = SIMD2<UInt32>(UInt32(count), UInt32(seedBase + batch * 4 + output))
                    encoder.setBytes(&args, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
                    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                encoder.endEncoding()
                slot.command = command
                slot.batch = batch
                command.commit()
                submitted += 1
                maxPending = max(maxPending, submitted - consumed)
            }
            for batch in 0..<depth { try submit(batch) }
            for batch in 0..<12 {
                try autoreleasepool {
                    let slot = slots[batch % depth]
                    guard slot.batch == batch, let command = slot.command else { throw Failure.invalid("Out-of-order consumption") }
                    command.waitUntilCompleted()
                    guard command.status == .completed else { throw Failure.invalid("GPU incomplete/failed: \(String(describing: command.error))") }
                    for output in 0..<4 {
                        let copy = Array(UnsafeBufferPointer(start: slot.buffers[output].contents().assumingMemoryBound(to: SIMD4<Float>.self), count: count))
                        var sum = SIMD4<Double>.zero
                        for value in copy { sum += SIMD4<Double>(value) }
                        guard sum == sums[batch * 4 + output] else { throw Failure.invalid("Stale or corrupted output checksum") }
                        if validate {
                            for i in copy.indices where copy[i] != pixel(i, seedBase + batch * 4 + output) {
                                throw Failure.invalid("Pixel mismatch")
                            }
                        }
                    }
                    latencies.append(ms(slot.started))
                    slot.command = nil
                    consumed += 1
                }
                if submitted < 12 { try submit(submitted) }
            }
            let elapsed = ms(start)
            guard submitted == 12, consumed == 12, maxPending == depth, slots.allSatisfy({ $0.command == nil }) else {
                throw Failure.invalid("Pool bound or completion count failed")
            }
            return Sample(trial: trial, depth: depth, position: position, totalMS: elapsed,
                latencyMS: latencies, maxPending: maxPending, outputs: consumed * 4,
                poolBytes: slots.reduce(0) { $0 + $1.buffers.reduce(0) { $0 + $1.allocatedSize } }, cpuCopyPayloadBytes: bytes)
        }
        for depth in [1,2,4] { _ = try run(depth: depth, trial: -10, position: 0, validate: true) }
        for trial in -6..<12 {
            for (position, depth) in orders[(trial + 6) % 6].enumerated() {
                let sample = try run(depth: depth, trial: trial, position: position, validate: false)
                if trial >= 0 { samples.append(sample) }
            }
        }
        for depth in [1,2,4] {
            _ = try run(depth: depth, trial: 12, position: 0, validate: true)
            let selected = samples.filter { $0.depth == depth }
            let times = selected.map(\.totalMS), latency = selected.flatMap(\.latencyMS)
            print(String(format: "%dx%d depth %d: trial %.3f ms [P10 %.3f–P90 %.3f]; %.0f outputs/s; latency median/P90 %.3f/%.3f ms; pool %.2f MiB",
                size, size, depth, percentile(times), percentile(times, 0.1), percentile(times, 0.9), 48000 / percentile(times),
                percentile(latency), percentile(latency, 0.9), Double(selected[0].poolBytes) / 1048576))
        }
        results.append(SizeResult(size: size, samples: samples))
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(Report(device: device.name, results: results)).write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
    print("PASS bounded pipeline: pixels, checksums, slot ownership and completion counts")
} catch { print("INCOMPLETE or FAILED: \(error)"); exit(1) }
