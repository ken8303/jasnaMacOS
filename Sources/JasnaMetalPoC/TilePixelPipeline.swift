import CoreVideo
import Foundation

enum VRMosaicProjection: String, CaseIterable, Sendable {
    case raw
    case fisheye
}

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

    static func extractResizedPlanarRGB(
        from pixelBuffer: CVPixelBuffer,
        region: MosaicRegion,
        projection: VRMosaicProjection = .raw
    ) throws -> [Float16] {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              region.x >= 0,
              region.y >= 0,
              region.x + region.width <= CVPixelBufferGetWidth(pixelBuffer),
              region.y + region.height <= CVPixelBufferGetHeight(pixelBuffer)
        else { throw DeformConvError.invalidShape }
        let samplingMap = MosaicCropSamplingMap(
            region: region,
            eyeWidth: CVPixelBufferGetWidth(pixelBuffer),
            eyeHeight: CVPixelBufferGetHeight(pixelBuffer),
            projection: projection
        )
        return try samplingMap.extractPlanarRGB(from: pixelBuffer)
    }
}

struct MosaicCompositeSample: Sendable {
    var modelX: Float
    var modelY: Float
    var alpha: Float
    var padding: Float = 0
}

struct MosaicCropSamplingMap: Sendable {
    private struct Sample: Sendable {
        let x0: Int32
        let y0: Int32
        let x1: Int32
        let y1: Int32
        let fx: Float
        let fy: Float
    }

    let eyeWidth: Int
    let eyeHeight: Int
    let modelSize: Int
    let compositeSamples: [MosaicCompositeSample]
    private let samples: [Sample]

    init(
        region: MosaicRegion,
        eyeWidth: Int,
        eyeHeight: Int,
        modelSize: Int = SideBySideVideoPlan.modelTileSize,
        projection: VRMosaicProjection = .raw
    ) {
        self.eyeWidth = eyeWidth
        self.eyeHeight = eyeHeight
        self.modelSize = modelSize
        let rawTransform = MosaicCropTransform(region: region, modelSize: modelSize)
        let fisheyeTransform = projection == .fisheye
            ? FisheyeMosaicCropTransform(
                region: region,
                eyeWidth: eyeWidth,
                eyeHeight: eyeHeight,
                modelSize: modelSize
            )
            : nil
        var generated = [Sample]()
        generated.reserveCapacity(modelSize * modelSize)
        for modelY in 0..<modelSize {
            for modelX in 0..<modelSize {
                let source = fisheyeTransform?.sourceCoordinate(
                    modelX: modelX, modelY: modelY
                ) ?? rawTransform.sourceCoordinate(modelX: modelX, modelY: modelY)
                let clampedX = min(max(source.x, 0), Float(eyeWidth - 1))
                let clampedY = min(max(source.y, 0), Float(eyeHeight - 1))
                let x0 = Int(floor(clampedX))
                let y0 = Int(floor(clampedY))
                generated.append(
                    Sample(
                        x0: Int32(x0),
                        y0: Int32(y0),
                        x1: Int32(min(x0 + 1, eyeWidth - 1)),
                        y1: Int32(min(y0 + 1, eyeHeight - 1)),
                        fx: clampedX - Float(x0),
                        fy: clampedY - Float(y0)
                    )
                )
            }
        }
        samples = generated
        if let fisheyeTransform {
            var generatedComposite = [MosaicCompositeSample]()
            generatedComposite.reserveCapacity(region.width * region.height)
            let left = region.effectiveBlendX
            let right = left + region.effectiveBlendWidth - 1
            let top = region.effectiveBlendY
            let bottom = top + region.effectiveBlendHeight - 1
            for pixelY in region.y..<(region.y + region.height) {
                for pixelX in region.x..<(region.x + region.width) {
                    let model = fisheyeTransform.modelCoordinate(
                        pixelX: pixelX, pixelY: pixelY
                    )
                    let outside = max(
                        max(left - pixelX, pixelX - right),
                        max(top - pixelY, pixelY - bottom)
                    )
                    let alpha: Float
                    if outside > 0 {
                        alpha = max(0, 0.5 * (1 - Float(outside) / 12))
                    } else {
                        let inside = min(
                            min(pixelX - left, right - pixelX),
                            min(pixelY - top, bottom - pixelY)
                        )
                        alpha = min(1, 0.5 + 0.5 * Float(inside + 1) / 12)
                    }
                    generatedComposite.append(
                        MosaicCompositeSample(
                            modelX: model.x, modelY: model.y, alpha: alpha
                        )
                    )
                }
            }
            compositeSamples = generatedComposite
        } else {
            compositeSamples = []
        }
    }

    func extractPlanarRGB(from pixelBuffer: CVPixelBuffer) throws -> [Float16] {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(pixelBuffer) == eyeWidth,
              CVPixelBufferGetHeight(pixelBuffer) == eyeHeight
        else { throw DeformConvError.invalidShape }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DeformConvError.commandFailed("pixel buffer has no base address")
        }
        let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let plane = modelSize * modelSize
        var result = [Float16](repeating: 0, count: 3 * plane)
        for (destination, sample) in samples.enumerated() {
            let x0 = Int(sample.x0)
            let y0 = Int(sample.y0)
            let x1 = Int(sample.x1)
            let y1 = Int(sample.y1)
            let topLeft = y0 * bytesPerRow + x0 * 4
            let topRight = y0 * bytesPerRow + x1 * 4
            let bottomLeft = y1 * bytesPerRow + x0 * 4
            let bottomRight = y1 * bytesPerRow + x1 * 4
            let topWeight = 1 - sample.fy
            let leftWeight = 1 - sample.fx
            let w00 = leftWeight * topWeight
            let w10 = sample.fx * topWeight
            let w01 = leftWeight * sample.fy
            let w11 = sample.fx * sample.fy
            @inline(__always) func channel(_ offset: Int) -> Float16 {
                let value = Float(bytes[topLeft + offset]) * w00
                    + Float(bytes[topRight + offset]) * w10
                    + Float(bytes[bottomLeft + offset]) * w01
                    + Float(bytes[bottomRight + offset]) * w11
                return Float16(value / 255)
            }
            result[destination] = channel(2)
            result[plane + destination] = channel(1)
            result[2 * plane + destination] = channel(0)
        }
        return result
    }
}

struct FisheyeMosaicCropTransform: Equatable, Sendable {
    let region: MosaicRegion
    let eyeWidth: Int
    let eyeHeight: Int
    let modelSize: Int
    let fisheyeMinU: Double
    let fisheyeMaxU: Double
    let fisheyeMinV: Double
    let fisheyeMaxV: Double

    init(
        region: MosaicRegion,
        eyeWidth: Int,
        eyeHeight: Int,
        modelSize: Int = SideBySideVideoPlan.modelTileSize,
        margin: Double = 1.06
    ) {
        self.region = region
        self.eyeWidth = eyeWidth
        self.eyeHeight = eyeHeight
        self.modelSize = modelSize
        let u1 = Double(region.x) / Double(eyeWidth)
        let v1 = Double(region.y) / Double(eyeHeight)
        let u2 = Double(region.x + region.width) / Double(eyeWidth)
        let v2 = Double(region.y + region.height) / Double(eyeHeight)
        var us = [Double]()
        var vs = [Double]()
        us.reserveCapacity(132)
        vs.reserveCapacity(132)
        for index in 0..<33 {
            let position = Double(index) / 32
            let u = u1 + (u2 - u1) * position
            let v = v1 + (v2 - v1) * position
            for point in [(u, v1), (u1, v), (u2, v), (u, v2)] {
                let mapped = Self.hequirectToFisheye(u: point.0, v: point.1)
                us.append(mapped.u)
                vs.append(mapped.v)
            }
        }
        let minU = us.min() ?? 0.5
        let maxU = us.max() ?? 0.5
        let minV = vs.min() ?? 0.5
        let maxV = vs.max() ?? 0.5
        let centerU = (minU + maxU) * 0.5
        let centerV = (minV + maxV) * 0.5
        let half = max(maxU - minU, maxV - minV) * 0.5 * margin
        fisheyeMinU = centerU - half
        fisheyeMaxU = centerU + half
        fisheyeMinV = centerV - half
        fisheyeMaxV = centerV + half
    }

    func sourceCoordinate(modelX: Int, modelY: Int) -> (x: Float, y: Float) {
        let patchU = (Double(modelX) + 0.5) / Double(modelSize)
        let patchV = (Double(modelY) + 0.5) / Double(modelSize)
        let fisheyeU = fisheyeMinU + (fisheyeMaxU - fisheyeMinU) * patchU
        let fisheyeV = fisheyeMinV + (fisheyeMaxV - fisheyeMinV) * patchV
        let px = fisheyeU * 2 - 1
        let py = fisheyeV * 2 - 1
        let radius = hypot(px, py)
        let theta = radius * .pi * 0.5
        let sinTheta = sin(theta)
        let x = radius > 1e-12 ? sinTheta * px / radius : 0
        let y = radius > 1e-12 ? -sinTheta * py / radius : 0
        let z = cos(theta)
        let longitude = atan2(x, z)
        let latitude = atan2(y, hypot(x, z))
        let u = longitude / .pi + 0.5
        let v = 0.5 - latitude / .pi
        return (
            Float(u * Double(eyeWidth - 1)),
            Float(v * Double(eyeHeight - 1))
        )
    }

    func modelCoordinate(pixelX: Int, pixelY: Int) -> (x: Float, y: Float) {
        let u = (Double(pixelX) + 0.5) / Double(eyeWidth)
        let v = (Double(pixelY) + 0.5) / Double(eyeHeight)
        let fisheye = Self.hequirectToFisheye(u: u, v: v)
        let patchU = (fisheye.u - fisheyeMinU) / (fisheyeMaxU - fisheyeMinU)
        let patchV = (fisheye.v - fisheyeMinV) / (fisheyeMaxV - fisheyeMinV)
        return (
            Float(patchU * Double(modelSize - 1)),
            Float(patchV * Double(modelSize - 1))
        )
    }

    private static func hequirectToFisheye(u: Double, v: Double) -> (u: Double, v: Double) {
        let longitude = (u - 0.5) * .pi
        let latitude = (0.5 - v) * .pi
        let cosLatitude = cos(latitude)
        let x = cosLatitude * sin(longitude)
        let y = sin(latitude)
        let z = cosLatitude * cos(longitude)
        let theta = acos(min(max(z, -1), 1))
        let radius = theta / (.pi * 0.5)
        let sinTheta = sin(theta)
        let projectedX = sinTheta > 1e-12 ? x * radius / sinTheta : 0
        let projectedY = sinTheta > 1e-12 ? -y * radius / sinTheta : 0
        return ((projectedX + 1) * 0.5, (projectedY + 1) * 0.5)
    }
}

struct MosaicCropTransform: Equatable, Sendable {
    let region: MosaicRegion
    let modelSize: Int
    let resizedWidth: Int
    let resizedHeight: Int
    let padX: Int
    let padY: Int

    init(region: MosaicRegion, modelSize: Int = SideBySideVideoPlan.modelTileSize) {
        self.region = region
        self.modelSize = modelSize
        let scale = min(
            Double(modelSize) / Double(region.width),
            Double(modelSize) / Double(region.height)
        )
        resizedWidth = max(1, min(modelSize, Int(Double(region.width) * scale)))
        resizedHeight = max(1, min(modelSize, Int(Double(region.height) * scale)))
        padX = (modelSize - resizedWidth) / 2
        padY = (modelSize - resizedHeight) / 2
    }

    func sourceCoordinate(modelX: Int, modelY: Int) -> (x: Float, y: Float) {
        let contentX = Self.reflected(modelX - padX, length: resizedWidth)
        let contentY = Self.reflected(modelY - padY, length: resizedHeight)
        return (
            Float(region.x) + (Float(contentX) + 0.5) * Float(region.width) / Float(resizedWidth) - 0.5,
            Float(region.y) + (Float(contentY) + 0.5) * Float(region.height) / Float(resizedHeight) - 0.5
        )
    }

    func modelCoordinate(pixelX: Int, pixelY: Int) -> (x: Float, y: Float) {
        (
            Float(padX) + (Float(pixelX - region.x) + 0.5)
                * Float(resizedWidth) / Float(region.width) - 0.5,
            Float(padY) + (Float(pixelY - region.y) + 0.5)
                * Float(resizedHeight) / Float(region.height) - 0.5
        )
    }

    private static func reflected(_ value: Int, length: Int) -> Int {
        guard length > 1 else { return 0 }
        var result = value
        while result < 0 || result >= length {
            if result < 0 {
                result = -result - 1
            } else {
                result = 2 * length - result - 1
            }
        }
        return result
    }
}

struct MosaicRegionFrameAccumulator {
    let dimensions: VideoDimensions
    private var bytes: [UInt8]

    init(basePixelBuffer: CVPixelBuffer, dimensions: VideoDimensions) throws {
        guard CVPixelBufferGetPixelFormatType(basePixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(basePixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(basePixelBuffer) == dimensions.height
        else { throw DeformConvError.invalidShape }
        self.dimensions = dimensions
        CVPixelBufferLockBaseAddress(basePixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(basePixelBuffer, .readOnly) }
        guard let source = CVPixelBufferGetBaseAddress(basePixelBuffer) else {
            throw DeformConvError.commandFailed("base pixel buffer has no base address")
        }
        let packedRowBytes = dimensions.width * 4
        let sourceRowBytes = CVPixelBufferGetBytesPerRow(basePixelBuffer)
        bytes = [UInt8](repeating: 0, count: packedRowBytes * dimensions.height)
        bytes.withUnsafeMutableBytes { destination in
            guard let destinationBase = destination.baseAddress else { return }
            for row in 0..<dimensions.height {
                destinationBase.advanced(by: row * packedRowBytes).copyMemory(
                    from: source.advanced(by: row * sourceRowBytes),
                    byteCount: packedRowBytes
                )
            }
        }
    }

    mutating func composite(
        region: MosaicRegion,
        planarRGB: [Float16],
        originalPlanarRGB: [Float16]? = nil,
        projection: VRMosaicProjection = .raw
    ) throws {
        let modelSize = SideBySideVideoPlan.modelTileSize
        let plane = modelSize * modelSize
        guard planarRGB.count == 3 * plane else { throw DeformConvError.invalidShape }
        guard projection == .raw || originalPlanarRGB?.count == planarRGB.count else {
            throw DeformConvError.invalidShape
        }
        let rawTransform = MosaicCropTransform(region: region, modelSize: modelSize)
        let fisheyeTransform = projection == .fisheye
            ? FisheyeMosaicCropTransform(
                region: region,
                eyeWidth: dimensions.width,
                eyeHeight: dimensions.height,
                modelSize: modelSize
            )
            : nil
        let startX = projection == .fisheye ? region.x : region.effectiveBlendX
        let endX = projection == .fisheye
            ? region.x + region.width
            : region.effectiveBlendX + region.effectiveBlendWidth
        let startY = projection == .fisheye ? region.y : region.effectiveBlendY
        let endY = projection == .fisheye
            ? region.y + region.height
            : region.effectiveBlendY + region.effectiveBlendHeight
        for pixelY in startY..<endY {
            for pixelX in startX..<endX {
                let alpha = projection == .fisheye
                    ? Self.expandedFeatherAlpha(
                        region: region, x: pixelX, y: pixelY, feather: 12
                    )
                    : region.featherAlpha(x: pixelX, y: pixelY, feather: 12)
                guard alpha > 0 else { continue }
                let model = fisheyeTransform?.modelCoordinate(
                    pixelX: pixelX, pixelY: pixelY
                ) ?? rawTransform.modelCoordinate(pixelX: pixelX, pixelY: pixelY)
                let destination = 4 * (pixelY * dimensions.width + pixelX)
                let baseBlue = Float(bytes[destination]) / 255
                let baseGreen = Float(bytes[destination + 1]) / 255
                let baseRed = Float(bytes[destination + 2]) / 255
                func restored(_ offset: Int, base: Float) -> Float {
                    let value = Self.bilinearPlane(
                        planarRGB, offset: offset, x: model.x, y: model.y
                    )
                    guard let originalPlanarRGB else { return value }
                    let original = Self.bilinearPlane(
                        originalPlanarRGB, offset: offset, x: model.x, y: model.y
                    )
                    return base + value - original
                }
                let red = restored(0, base: baseRed)
                let green = restored(plane, base: baseGreen)
                let blue = restored(2 * plane, base: baseBlue)
                bytes[destination] = Self.blend(base: bytes[destination], restored: blue, alpha: alpha)
                bytes[destination + 1] = Self.blend(
                    base: bytes[destination + 1], restored: green, alpha: alpha
                )
                bytes[destination + 2] = Self.blend(
                    base: bytes[destination + 2], restored: red, alpha: alpha
                )
            }
        }
    }

    private static func expandedFeatherAlpha(
        region: MosaicRegion,
        x: Int,
        y: Int,
        feather: Int
    ) -> Float {
        guard feather > 0 else {
            return region.contains(x: x, y: y) ? 1 : 0
        }
        let left = region.effectiveBlendX
        let right = left + region.effectiveBlendWidth - 1
        let top = region.effectiveBlendY
        let bottom = top + region.effectiveBlendHeight - 1
        let outside = max(max(left - x, x - right), max(top - y, y - bottom))
        if outside > 0 {
            return max(0, 0.5 * (1 - Float(outside) / Float(feather)))
        }
        let inside = min(min(x - left, right - x), min(y - top, bottom - y))
        return min(1, 0.5 + 0.5 * Float(inside + 1) / Float(feather))
    }

    func writeBGRA(to pixelBuffer: CVPixelBuffer) throws {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(pixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(pixelBuffer) == dimensions.height
        else { throw DeformConvError.invalidShape }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let destination = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DeformConvError.commandFailed("output pixel buffer has no base address")
        }
        let destinationRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let packedRowBytes = dimensions.width * 4
        bytes.withUnsafeBytes { source in
            guard let sourceBase = source.baseAddress else { return }
            for row in 0..<dimensions.height {
                destination.advanced(by: row * destinationRowBytes).copyMemory(
                    from: sourceBase.advanced(by: row * packedRowBytes),
                    byteCount: packedRowBytes
                )
            }
        }
    }

    private static func bilinearPlane(
        _ values: [Float16], offset: Int, x: Float, y: Float
    ) -> Float {
        let size = SideBySideVideoPlan.modelTileSize
        let clampedX = min(max(x, 0), Float(size - 1))
        let clampedY = min(max(y, 0), Float(size - 1))
        let x0 = Int(floor(clampedX))
        let y0 = Int(floor(clampedY))
        let x1 = min(x0 + 1, size - 1)
        let y1 = min(y0 + 1, size - 1)
        let fx = clampedX - Float(x0)
        let fy = clampedY - Float(y0)
        func value(_ px: Int, _ py: Int) -> Float {
            Float(values[offset + py * size + px])
        }
        let top = value(x0, y0) * (1 - fx) + value(x1, y0) * fx
        let bottom = value(x0, y1) * (1 - fx) + value(x1, y1) * fx
        return top * (1 - fy) + bottom * fy
    }

    private static func blend(base: UInt8, restored: Float, alpha: Float) -> UInt8 {
        let restoredByte = min(max(restored, 0), 1) * 255
        return UInt8(clamping: Int((Float(base) * (1 - alpha) + restoredByte * alpha).rounded()))
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
        let outputPlane = dimensions.pixelCount
        guard planarRGB.count == 3 * tilePlane,
              tile.x >= 0,
              tile.y >= 0,
              tile.x + tile.width <= dimensions.width,
              tile.y + tile.height <= dimensions.height
        else { throw DeformConvError.invalidShape }

        let horizontalWeights = (0..<tile.width).map {
            Self.axisWeight(
                position: $0,
                length: tile.width,
                leadingOverlap: tile.leftOverlap,
                trailingOverlap: tile.rightOverlap
            )
        }
        let verticalWeights = (0..<tile.height).map {
            Self.axisWeight(
                position: $0,
                length: tile.height,
                leadingOverlap: tile.topOverlap,
                trailingOverlap: tile.bottomOverlap
            )
        }
        for localY in 0..<tile.height {
            let vertical = verticalWeights[localY]
            for localX in 0..<tile.width {
                let weight = horizontalWeights[localX] * vertical
                let source = localY * tile.width + localX
                let destination = (tile.y + localY) * dimensions.width + tile.x + localX
                weights[destination] += weight
                colors[destination] += Float(planarRGB[source]) * weight
                colors[outputPlane + destination]
                    += Float(planarRGB[tilePlane + source]) * weight
                colors[2 * outputPlane + destination]
                    += Float(planarRGB[2 * tilePlane + source]) * weight
            }
        }
    }

    func makeBGRABytes() throws -> [UInt8] {
        var output = [UInt8](repeating: 255, count: dimensions.pixelCount * 4)
        for pixel in 0..<dimensions.pixelCount {
            guard weights[pixel] > 0 else {
                throw DeformConvError.commandFailed("tile blend left uncovered output pixels")
            }
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

struct SparseTileFrameAccumulator {
    let dimensions: VideoDimensions
    private let baseBGRA: [UInt8]
    private var colors: [Float]
    private var weights: [Float]

    init(basePixelBuffer: CVPixelBuffer, dimensions: VideoDimensions) throws {
        guard CVPixelBufferGetPixelFormatType(basePixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(basePixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(basePixelBuffer) == dimensions.height
        else { throw DeformConvError.invalidShape }
        self.dimensions = dimensions
        baseBGRA = try Self.copyPackedBGRA(from: basePixelBuffer, dimensions: dimensions)
        colors = [Float](repeating: 0, count: 3 * dimensions.pixelCount)
        weights = [Float](repeating: 0, count: dimensions.pixelCount)
    }

    mutating func accumulate(tile: VideoTile, planarRGB: [Float16]) throws {
        let tilePlane = tile.width * tile.height
        let outputPlane = dimensions.pixelCount
        guard planarRGB.count == 3 * tilePlane,
              tile.x >= 0,
              tile.y >= 0,
              tile.x + tile.width <= dimensions.width,
              tile.y + tile.height <= dimensions.height
        else { throw DeformConvError.invalidShape }

        let horizontalWeights = (0..<tile.width).map {
            Self.axisWeight(
                position: $0,
                length: tile.width,
                leadingOverlap: tile.leftOverlap,
                trailingOverlap: tile.rightOverlap
            )
        }
        let verticalWeights = (0..<tile.height).map {
            Self.axisWeight(
                position: $0,
                length: tile.height,
                leadingOverlap: tile.topOverlap,
                trailingOverlap: tile.bottomOverlap
            )
        }
        for localY in 0..<tile.height {
            for localX in 0..<tile.width {
                let weight = horizontalWeights[localX] * verticalWeights[localY]
                let source = localY * tile.width + localX
                let destination = (tile.y + localY) * dimensions.width + tile.x + localX
                weights[destination] += weight
                colors[destination] += Float(planarRGB[source]) * weight
                colors[outputPlane + destination]
                    += Float(planarRGB[tilePlane + source]) * weight
                colors[2 * outputPlane + destination]
                    += Float(planarRGB[2 * tilePlane + source]) * weight
            }
        }
    }

    func writeBGRA(
        to pixelBuffer: CVPixelBuffer,
        regions: [MosaicRegion],
        feather: Int = 12
    ) throws {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              CVPixelBufferGetWidth(pixelBuffer) == dimensions.width,
              CVPixelBufferGetHeight(pixelBuffer) == dimensions.height
        else { throw DeformConvError.invalidShape }

        var bytes = baseBGRA
        for pixelY in 0..<dimensions.height {
            for pixelX in 0..<dimensions.width {
                let pixel = pixelY * dimensions.width + pixelX
                guard weights[pixel] > 0 else { continue }
                let alpha = regions.reduce(Float(0)) {
                    max($0, $1.featherAlpha(x: pixelX, y: pixelY, feather: feather))
                }
                guard alpha > 0 else { continue }
                let inverseWeight = 1 / weights[pixel]
                let red = colors[pixel] * inverseWeight
                let green = colors[dimensions.pixelCount + pixel] * inverseWeight
                let blue = colors[2 * dimensions.pixelCount + pixel] * inverseWeight
                let byteOffset = 4 * pixel
                bytes[byteOffset] = Self.blend(base: bytes[byteOffset], restored: blue, alpha: alpha)
                bytes[byteOffset + 1] = Self.blend(
                    base: bytes[byteOffset + 1], restored: green, alpha: alpha
                )
                bytes[byteOffset + 2] = Self.blend(
                    base: bytes[byteOffset + 2], restored: red, alpha: alpha
                )
            }
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let destination = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DeformConvError.commandFailed("output pixel buffer has no base address")
        }
        let destinationRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let packedRowBytes = dimensions.width * 4
        bytes.withUnsafeBytes { source in
            guard let sourceBase = source.baseAddress else { return }
            for row in 0..<dimensions.height {
                destination.advanced(by: row * destinationRowBytes).copyMemory(
                    from: sourceBase.advanced(by: row * packedRowBytes),
                    byteCount: packedRowBytes
                )
            }
        }
    }

    private static func copyPackedBGRA(
        from pixelBuffer: CVPixelBuffer,
        dimensions: VideoDimensions
    ) throws -> [UInt8] {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let source = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DeformConvError.commandFailed("base pixel buffer has no base address")
        }
        let packedRowBytes = dimensions.width * 4
        let sourceRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var result = [UInt8](repeating: 0, count: packedRowBytes * dimensions.height)
        result.withUnsafeMutableBytes { destination in
            guard let destinationBase = destination.baseAddress else { return }
            for row in 0..<dimensions.height {
                destinationBase.advanced(by: row * packedRowBytes).copyMemory(
                    from: source.advanced(by: row * sourceRowBytes),
                    byteCount: packedRowBytes
                )
            }
        }
        return result
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

    private static func blend(base: UInt8, restored: Float, alpha: Float) -> UInt8 {
        let restoredByte = min(max(restored, 0), 1) * 255
        return UInt8(clamping: Int((Float(base) * (1 - alpha) + restoredByte * alpha).rounded()))
    }
}
