import CoreVideo
import Foundation

enum TilePixelPipeline {
    static func extractPlanarRGB(
        from pixelBuffer: CVPixelBuffer,
        tile: VideoTile
    ) throws -> [Float16] {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              tile.x >= 0,
              tile.y >= 0,
              tile.x + tile.width <= CVPixelBufferGetWidth(pixelBuffer),
              tile.y + tile.height <= CVPixelBufferGetHeight(pixelBuffer)
        else { throw DeformConvError.invalidShape }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DeformConvError.commandFailed("pixel buffer has no base address")
        }
        let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let plane = tile.width * tile.height
        var result = [Float16](repeating: 0, count: 3 * plane)
        for localY in 0..<tile.height {
            let sourceRow = (tile.y + localY) * bytesPerRow
            for localX in 0..<tile.width {
                let source = sourceRow + (tile.x + localX) * 4
                let destination = localY * tile.width + localX
                result[destination] = Float16(Float(bytes[source + 2]) / 255)
                result[plane + destination] = Float16(Float(bytes[source + 1]) / 255)
                result[2 * plane + destination] = Float16(Float(bytes[source]) / 255)
            }
        }
        return result
    }
}

struct TileFrameAccumulator {
    let dimensions: VideoDimensions
    private var colors: [Float]
    private var weights: [Float]

    init(dimensions: VideoDimensions) throws {
        guard dimensions.width > 0, dimensions.height > 0 else {
            throw DeformConvError.invalidShape
        }
        self.dimensions = dimensions
        colors = [Float](repeating: 0, count: 3 * dimensions.pixelCount)
        weights = [Float](repeating: 0, count: dimensions.pixelCount)
    }

    mutating func accumulate(tile: VideoTile, planarRGB: [Float16]) throws {
        let tilePlane = tile.width * tile.height
        guard planarRGB.count == 3 * tilePlane,
              tile.x >= 0,
              tile.y >= 0,
              tile.x + tile.width <= dimensions.width,
              tile.y + tile.height <= dimensions.height
        else { throw DeformConvError.invalidShape }

        for localY in 0..<tile.height {
            let vertical = Self.axisWeight(
                position: localY,
                length: tile.height,
                leadingOverlap: tile.topOverlap,
                trailingOverlap: tile.bottomOverlap
            )
            for localX in 0..<tile.width {
                let horizontal = Self.axisWeight(
                    position: localX,
                    length: tile.width,
                    leadingOverlap: tile.leftOverlap,
                    trailingOverlap: tile.rightOverlap
                )
                let weight = horizontal * vertical
                let source = localY * tile.width + localX
                let destination = (tile.y + localY) * dimensions.width + tile.x + localX
                weights[destination] += weight
                for channel in 0..<3 {
                    colors[channel * dimensions.pixelCount + destination]
                        += Float(planarRGB[channel * tilePlane + source]) * weight
                }
            }
        }
    }

    func makeBGRABytes() throws -> [UInt8] {
        guard weights.allSatisfy({ $0 > 0 }) else {
            throw DeformConvError.commandFailed("tile blend left uncovered output pixels")
        }
        var output = [UInt8](repeating: 255, count: dimensions.pixelCount * 4)
        for pixel in 0..<dimensions.pixelCount {
            let inverseWeight = 1 / weights[pixel]
            let red = colors[pixel] * inverseWeight
            let green = colors[dimensions.pixelCount + pixel] * inverseWeight
            let blue = colors[2 * dimensions.pixelCount + pixel] * inverseWeight
            output[4 * pixel] = Self.byte(blue)
            output[4 * pixel + 1] = Self.byte(green)
            output[4 * pixel + 2] = Self.byte(red)
        }
        return output
    }

    func writeBGRA(to pixelBuffer: CVPixelBuffer) throws {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(pixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(pixelBuffer) == dimensions.height
        else { throw DeformConvError.invalidShape }
        let bytes = try makeBGRABytes()
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DeformConvError.commandFailed("output pixel buffer has no base address")
        }
        let destination = baseAddress.assumingMemoryBound(to: UInt8.self)
        let destinationRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let packedRowBytes = dimensions.width * 4
        bytes.withUnsafeBytes { source in
            guard let sourceBase = source.baseAddress else { return }
            for row in 0..<dimensions.height {
                destination.advanced(by: row * destinationRowBytes).update(
                    from: sourceBase.advanced(by: row * packedRowBytes)
                        .assumingMemoryBound(to: UInt8.self),
                    count: packedRowBytes
                )
            }
        }
    }

    var accumulatedWeightRange: ClosedRange<Float>? {
        guard let minimum = weights.min(), let maximum = weights.max() else { return nil }
        return minimum...maximum
    }

    private static func axisWeight(
        position: Int,
        length: Int,
        leadingOverlap: Int,
        trailingOverlap: Int
    ) -> Float {
        var result: Float = 1
        if leadingOverlap > 0, position < leadingOverlap {
            result = Float(position + 1) / Float(leadingOverlap + 1)
        }
        if trailingOverlap > 0, position >= length - trailingOverlap {
            result = min(result, Float(length - position) / Float(trailingOverlap + 1))
        }
        return result
    }

    private static func byte(_ value: Float) -> UInt8 {
        UInt8(clamping: Int((min(max(value, 0), 1) * 255).rounded()))
    }
}
