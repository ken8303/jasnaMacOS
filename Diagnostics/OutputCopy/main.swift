import Foundation
import Metal

// Standalone output-copy experiment. No application imports or media inputs.
// The source models an already completed GPU output in shared memory.
struct Result: Codable {
    let width: Int
    let height: Int
    let freshMedianMS: Double
    let reuseMedianMS: Double
    let pairedMedianMS: Double
    let pairedP10MS: Double
    let pairedP90MS: Double
    let reuseFasterPairs: Int
    let samples: Int
    let retainedBytes: Int
}
func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted()
    return sorted[Int(Double(sorted.count - 1) * fraction)]
}
func milliseconds(_ start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
}
enum Failure: Error { case invalid(String) }

do {
    guard CommandLine.arguments.count == 2,
          let device = MTLCreateSystemDefaultDevice() else {
        throw Failure.invalid("Usage: output-copy REPORT.json; Metal device required")
    }
    print("Synthetic shared-buffer output copy on \(device.name)")
    print("Allocation + copy + full-output consumption included; source fill and reusable allocation excluded.")
    print("64 balanced pairs per size after 8 warmups; no GPU execution or video processing.")
    var results = [Result]()
    for (width, height) in [(96, 80), (256, 256), (512, 512), (640, 512)] {
        let count = width * height
        let bytes = count * MemoryLayout<SIMD4<Float>>.stride
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw Failure.invalid("Shared buffer allocation failed")
        }
        let source = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
        var reused = [SIMD4<Float>](repeating: .zero, count: count)
        var freshTimes = [Double](), reuseTimes = [Double]()
        for trial in -8..<64 {
            // Alternating contents detect stale-output reuse; exact binary fractions
            // keep full checksums deterministic and exactly representable in Double.
            var expected = SIMD4<Double>.zero
            for index in 0..<count {
                let value = Float((index + (trial + 8) * 17) % 251) / 256
                let pixel = SIMD4(value, value / 2, 1 - value, 1)
                source[index] = pixel
                expected += SIMD4<Double>(pixel)
            }
            for mode in (trial % 2 == 0 ? [0, 1] : [1, 0]) {
                let start = DispatchTime.now().uptimeNanoseconds
                var checksum = SIMD4<Double>.zero
                if mode == 0 {
                    let fresh = Array(UnsafeBufferPointer(start: source, count: count))
                    for pixel in fresh { checksum += SIMD4<Double>(pixel) }
                } else {
                    reused.withUnsafeMutableBytes {
                        $0.baseAddress!.copyMemory(from: source, byteCount: bytes)
                    }
                    for pixel in reused { checksum += SIMD4<Double>(pixel) }
                }
                let elapsed = milliseconds(start)
                guard checksum == expected else { throw Failure.invalid("Output checksum mismatch") }
                if trial >= 0 {
                    if mode == 0 { freshTimes.append(elapsed) } else { reuseTimes.append(elapsed) }
                }
            }
            // Validate only after both partners so neither receives an extra
            // source-cache warm-up between their paired measurements.
            let referenceCopy = Array(UnsafeBufferPointer(start: source, count: count))
            guard referenceCopy == reused else { throw Failure.invalid("Full pixel mismatch") }
        }
        let paired = zip(reuseTimes, freshTimes).map { $0 - $1 }
        let result = Result(width: width, height: height,
            freshMedianMS: percentile(freshTimes, 0.5), reuseMedianMS: percentile(reuseTimes, 0.5),
            pairedMedianMS: percentile(paired, 0.5), pairedP10MS: percentile(paired, 0.1),
            pairedP90MS: percentile(paired, 0.9), reuseFasterPairs: paired.filter { $0 < 0 }.count,
            samples: paired.count, retainedBytes: bytes)
        results.append(result)
        print(String(format: "%dx%d: fresh %.3f ms; reusable %.3f ms; paired %.3f ms [P10 %.3f–P90 %.3f]; faster %d/64",
            width, height, result.freshMedianMS, result.reuseMedianMS, result.pairedMedianMS,
            result.pairedP10MS, result.pairedP90MS, result.reuseFasterPairs))
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    // Runner supplies a unique result directory. Avoid atomic auxiliary-file
    // creation, which the Documents file provider rejects on this machine.
    try encoder.encode(results).write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
    print("PASS output copy: all changing-input checks completed")
} catch {
    print("INCOMPLETE or FAILED: \(error)")
    exit(1)
}
