import Foundation

enum LabError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String { switch self { case .invalid(let message): message } }
}

struct Raster {
    let width: Int
    let height: Int
    let bytes: [UInt8]

    // Distinct channels catch swaps; the hard checkerboard challenges filtering.
    static func generated(width: Int, height: Int, frame: Int) -> Raster {
        precondition(width > 1 && height > 1 && frame >= 0)
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let index = (y * width + x) * 4
                bytes[index] = UInt8(((x + frame * 2) % width) * 255 / (width - 1))
                bytes[index + 1] = UInt8(((y + frame) % height) * 255 / (height - 1))
                bytes[index + 2] = ((x + frame * 3) / 16 + (y + frame * 2) / 16) % 2 == 0 ? 32 : 224
            }
        }
        return Raster(width: width, height: height, bytes: bytes)
    }
}

func movingCoordinates(source: Raster, width: Int, height: Int, frame: Int) -> [SIMD2<Float>] {
    let angle = Float(frame) * 0.002
    let cosine = cos(angle), sine = sin(angle)
    return (0..<(width * height)).map { index in
        // Integer coordinates denote source-pixel centres, not pixel edges.
        let x = (Float(index % width) + 0.5) * Float(source.width) / Float(width) - Float(source.width) / 2
        let y = (Float(index / width) + 0.5) * Float(source.height) / Float(height) - Float(source.height) / 2
        return SIMD2(
            cosine * x - sine * y + Float(source.width - 1) / 2 - 7 + Float(frame) * 0.25,
            sine * x + cosine * y + Float(source.height - 1) / 2 - 5 + Float(frame) * 0.125
        )
    }
}

// Deliberately plain, serial Swift Float implementation; not Accelerate/vImage.
func cpuSample(_ raster: Raster, coordinates: [SIMD2<Float>]) -> [SIMD4<Float>] {
    coordinates.map { point in
        let x = min(max(point.x, 0), Float(raster.width - 1))
        let y = min(max(point.y, 0), Float(raster.height - 1))
        let x0 = Int(x), y0 = Int(y)
        let x1 = min(x0 + 1, raster.width - 1), y1 = min(y0 + 1, raster.height - 1)
        let fx = x - Float(x0), fy = y - Float(y0)
        var result = SIMD4<Float>.zero
        for channel in 0..<4 {
            let a = Float(raster.bytes[(y0 * raster.width + x0) * 4 + channel])
            let b = Float(raster.bytes[(y0 * raster.width + x1) * 4 + channel])
            let c = Float(raster.bytes[(y1 * raster.width + x0) * 4 + channel])
            let d = Float(raster.bytes[(y1 * raster.width + x1) * 4 + channel])
            result[channel] = ((a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy) / 255
        }
        return result
    }
}

// Independent double-precision weighted-sum oracle, including clamp-to-edge.
func referenceSample(_ raster: Raster, coordinates: [SIMD2<Float>]) -> [SIMD4<Float>] {
    coordinates.map { point in
        let x = min(max(Double(point.x), 0), Double(raster.width - 1))
        let y = min(max(Double(point.y), 0), Double(raster.height - 1))
        let lowX = Int(floor(x)), lowY = Int(floor(y))
        let fractions = (x - floor(x), y - floor(y))
        var sum = SIMD4<Double>.zero
        for dy in 0...1 {
            for dx in 0...1 {
                let weight = (dx == 0 ? 1 - fractions.0 : fractions.0)
                    * (dy == 0 ? 1 - fractions.1 : fractions.1)
                let offset = (min(lowY + dy, raster.height - 1) * raster.width
                    + min(lowX + dx, raster.width - 1)) * 4
                for channel in 0..<4 { sum[channel] += Double(raster.bytes[offset + channel]) * weight / 255 }
            }
        }
        return SIMD4<Float>(sum)
    }
}

func maximumError(_ actual: [SIMD4<Float>], _ expected: [SIMD4<Float>]) throws -> Double {
    guard !actual.isEmpty && actual.count == expected.count else { throw LabError.invalid("Mismatched or empty output") }
    var maximum: Double = 0
    for (a, b) in zip(actual, expected) {
        for channel in 0..<4 {
            guard a[channel].isFinite && b[channel].isFinite else { throw LabError.invalid("Non-finite output") }
            maximum = max(maximum, Double(abs(a[channel] - b[channel])))
        }
    }
    return maximum
}

struct TimingSummary: Codable {
    let rawMS: [Double]
    let samples: Int
    let medianMS: Double
    let p10MS: Double
    let p90MS: Double
    init(_ values: [Double]) throws {
        guard !values.isEmpty && values.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw LabError.invalid("Invalid timing samples")
        }
        let sorted = values.sorted()
        func percentile(_ fraction: Double) -> Double {
            let position = Double(sorted.count - 1) * fraction
            let lower = Int(position), upper = min(lower + 1, sorted.count - 1)
            return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
        }
        rawMS = values
        samples = values.count
        medianMS = percentile(0.5); p10MS = percentile(0.1); p90MS = percentile(0.9)
    }
}

func timed<T>(_ operation: () throws -> T) rethrows -> (value: T, milliseconds: Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let result = try operation()
    return (result, Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
}
