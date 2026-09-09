import Foundation
import Dispatch

func cpuChunkRanges(count: Int, workers: Int) throws -> [Range<Int>] {
    guard count > 0 && [1, 4].contains(workers) else {
        throw LabError.invalid("CPU sampling requires nonempty output and one or four worker chunks")
    }
    let active = min(count, workers), chunk = count / active, remainder = count % active
    return (0..<active).map { worker in
        let start = worker * chunk + min(worker, remainder)
        return start..<(start + chunk + (worker < remainder ? 1 : 0))
    }
}

// The owner keeps all three arrays alive until synchronous concurrentPerform
// returns. Sources/coordinates are read-only; tested, disjoint chunk ranges
// give every output element exactly one writer. Pointers never escape the call.
private struct CPUStorage: @unchecked Sendable {
    let pixels: UnsafeRawPointer
    let points: UnsafePointer<SIMD2<Float>>
    let output: UnsafeMutablePointer<SIMD4<Float>>
    let width: Int
    let height: Int

    @inline(__always)
    private func rgba(x: Int, y: Int) -> SIMD4<Float> {
        let bytes = pixels.loadUnaligned(fromByteOffset: (y * width + x) * 4, as: SIMD4<UInt8>.self)
        return SIMD4<Float>(bytes)
    }

    @inline(__always)
    func sample(_ range: Range<Int>) {
        for index in range {
            let point = points[index]
            let x = min(max(point.x, 0), Float(width - 1))
            let y = min(max(point.y, 0), Float(height - 1))
            // Integer clamps also protect the pointer loads when large integer
            // dimensions round upward on conversion to Float.
            let x0 = min(Int(x), width - 1), y0 = min(Int(y), height - 1)
            let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
            let fx = x - Float(x0), fy = y - Float(y0)
            let a = rgba(x: x0, y: y0), b = rgba(x: x1, y: y0)
            let c = rgba(x: x0, y: y1), d = rgba(x: x1, y: y1)
            let value = ((a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy) / 255
            output.advanced(by: index).initialize(to: value)
        }
    }
}

// Includes input validation, output allocation, and (when requested) worker
// scheduling. No coordinate-plan cache or preconverted pixel buffer is hidden
// outside the stopwatch. This is a SIMD implementation, not Accelerate/vImage.
func optimizedCPUSample(_ raster: Raster, coordinates: [SIMD2<Float>], workers: Int = 1) throws -> [SIMD4<Float>] {
    guard raster.width > 0 && raster.height > 0 else { throw LabError.invalid("Invalid CPU source dimensions") }
    let (pixels, pixelOverflow) = raster.width.multipliedReportingOverflow(by: raster.height)
    let (bytes, byteOverflow) = pixels.multipliedReportingOverflow(by: 4)
    guard !pixelOverflow && !byteOverflow && bytes == raster.bytes.count,
          !coordinates.isEmpty && coordinates.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
        throw LabError.invalid("Invalid CPU source bytes or coordinates")
    }
    let ranges = try cpuChunkRanges(count: coordinates.count, workers: workers)
    return raster.bytes.withUnsafeBytes { source in
        coordinates.withUnsafeBufferPointer { points in
            Array<SIMD4<Float>>(unsafeUninitializedCapacity: coordinates.count) { output, initialized in
                let storage = CPUStorage(pixels: source.baseAddress!, points: points.baseAddress!,
                                         output: output.baseAddress!, width: raster.width, height: raster.height)
                if ranges.count == 1 {
                    storage.sample(ranges[0])
                } else {
                    DispatchQueue.concurrentPerform(iterations: ranges.count) { worker in
                        storage.sample(ranges[worker])
                    }
                }
                initialized = coordinates.count
            }
        }
    }
}
