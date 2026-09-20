import CoreVideo
import Foundation
import Testing
@testable import JasnaMetalPoC

// Analytic ramps and smooth checkerboards only: no media, weights or inference.
// A ramp converts resampling error directly into source-pixel displacement;
// a smooth checkerboard exercises detail without a Nyquist/aliasing ambiguity.
func interpolateSynthetic(
    x: Double, y: Double, sample: (Int, Int) -> Double
) -> Double {
    let x0 = Int(floor(x))
    let y0 = Int(floor(y))
    let fx = x - Double(x0)
    let fy = y - Double(y0)
    return (sample(x0, y0) * (1 - fx) + sample(x0 + 1, y0) * fx) * (1 - fy)
        + (sample(x0, y0 + 1) * (1 - fx) + sample(x0 + 1, y0 + 1) * fx) * fy
}

private func movingSyntheticRegion(frame: Int, scale: Int) -> MosaicRegion {
    MosaicRegion(
        startFrame: frame, endFrame: frame + 1,
        x: (140 + frame * 2) * scale, y: (220 - frame) * scale,
        width: 160 * scale, height: 128 * scale, confidence: 1
    )
}

@Test func syntheticMovingRawRampKeepsPixelCentres() {
    var maximumError: Double = 0
    for frame in 0..<12 {
        let region = movingSyntheticRegion(frame: frame, scale: 8)
        let transform = MosaicCropTransform(region: region)
        for dy in stride(from: 16, through: 112, by: 16) {
            for dx in stride(from: 16, through: 144, by: 16) {
                let x = region.x + dx * 8
                let y = region.y + dy * 8
                let model = transform.modelCoordinate(pixelX: x, pixelY: y)
                for axis in 0..<2 {
                    let ramp = interpolateSynthetic(x: Double(model.x), y: Double(model.y)) {
                        let source = transform.sourceCoordinate(modelX: $0, modelY: $1)
                        return Double(axis == 0 ? source.x : source.y)
                    }
                    maximumError = max(maximumError, abs(ramp - Double(axis == 0 ? x : y)))
                }
            }
        }
    }
    #expect(maximumError < 0.001)
    print("Synthetic raw moving ramp: maximum displacement \(maximumError) source pixels")
}

@Test(arguments: [1, 8])
func syntheticMovingFisheyeRampKeepsPixelCentres(scale: Int) {
    var maximumError: Double = 0
    var referenceError: Double = 0
    var maximumTemporalDrift: Double = 0
    var previousErrors = [Double]()
    for frame in 0..<12 {
        let region = movingSyntheticRegion(frame: frame, scale: scale)
        let transform = FisheyeMosaicCropTransform(
            region: region, eyeWidth: 512 * scale, eyeHeight: 512 * scale
        )
        var errors = [Double]()
        for dy in stride(from: 16, through: 112, by: 16) {
            for dx in stride(from: 16, through: 144, by: 16) {
                let x = region.x + dx * scale
                let y = region.y + dy * scale
                let model = transform.modelCoordinate(pixelX: x, pixelY: y)
                let referenceModel = pixelCentreReferenceModel(x: Double(x), y: Double(y), transform: transform)
                for axis in 0..<2 {
                    let expected = Double(axis == 0 ? x : y)
                    let ramp = interpolateSynthetic(x: Double(model.x), y: Double(model.y)) {
                        let source = transform.sourceCoordinate(modelX: $0, modelY: $1)
                        return Double(axis == 0 ? source.x : source.y)
                    }
                    let reference = interpolateSynthetic(x: referenceModel.x, y: referenceModel.y) {
                        let source = pixelCentreReferenceSource(modelX: Double($0), modelY: Double($1), transform: transform)
                        return axis == 0 ? source.x : source.y
                    }
                    errors.append(ramp - expected)
                    maximumError = max(maximumError, abs(ramp - expected))
                    referenceError = max(referenceError, abs(reference - expected))
                }
            }
        }
        if !previousErrors.isEmpty {
            for (current, previous) in zip(errors, previousErrors) {
                maximumTemporalDrift = max(maximumTemporalDrift, abs(current - previous))
            }
        }
        previousErrors = errors
    }
    print("Synthetic fisheye \(512 * scale)px moving ramp: displacement \(maximumError), reference \(referenceError), frame-to-frame bias change \(maximumTemporalDrift) source pixels")
    // Reference interpolation/Float coordinate precision must be comfortably
    // below the defect threshold, otherwise this fixture cannot diagnose it.
    #expect(referenceError < 0.01)
    #expect(maximumError < 0.05)
}

private func movingPatternByte(x: Int, y: Int, frame: Int, channel: Int) -> UInt8 {
    let translatedX = Double(x - frame * 3)
    let translatedY = Double(y - frame * 2)
    let value: Double
    switch channel {
    case 0: value = 0.25 + translatedX / 512
    case 1: value = 0.25 + translatedY / 512
    default:
        value = 0.5 + 0.3 * sin(translatedX * 2 * .pi / 64) * cos(translatedY * 2 * .pi / 64)
    }
    return UInt8(max(0, min(255, (value * 255).rounded())))
}

@Test(arguments: [VRMosaicProjection.raw, .fisheye], [0, 64, 128])
func syntheticMovingDetailExtractionMatchesBilinearOracle(projection: VRMosaicProjection, origin: Int) throws {
    let size = 256
    let modelSize = 64
    let region = MosaicRegion(
        startFrame: 0, endFrame: 12, x: origin, y: origin, width: 128, height: 128, confidence: 1
    )
    let map = MosaicCropSamplingMap(
        region: region, eyeWidth: size, eyeHeight: size, modelSize: modelSize, projection: projection
    )
    var boundarySamples = 0
    for y in 0..<modelSize {
        for x in 0..<modelSize {
            let coordinate = map.sourceCoordinate(modelX: x, modelY: y)
            try #require(coordinate.x.isFinite && coordinate.y.isFinite)
            try #require(coordinate.x >= 0 && coordinate.x <= Float(size - 1))
            try #require(coordinate.y >= 0 && coordinate.y <= Float(size - 1))
            if coordinate.x == 0 || coordinate.y == 0
                || coordinate.x == Float(size - 1) || coordinate.y == Float(size - 1) {
                boundarySamples += 1
            }
        }
    }
    if projection == .fisheye && origin != 64 {
        // Confirm these cases actually exercise raster-edge clamping.
        #expect(boundarySamples > 0)
    }
    var optionalBuffer: CVPixelBuffer?
    // CPU-only raster: no IOSurface, Metal device or model packages required.
    let status = CVPixelBufferCreate(
        nil, size, size, kCVPixelFormatType_32BGRA, nil, &optionalBuffer
    )
    try #require(status == kCVReturnSuccess)
    let buffer = try #require(optionalBuffer)
    var maximumError: Double = 0
    var maximumTemporalError: Double = 0
    var largestFrameChange: Double = 0
    var previousActual = [Double]()
    var previousExpected = [Double]()
    for frame in 0..<12 {
        try #require(CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess)
        do {
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            let bytes = try #require(CVPixelBufferGetBaseAddress(buffer)).assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<size {
                for x in 0..<size {
                    for channel in 0..<3 {
                        bytes[y * stride + x * 4 + 2 - channel] = movingPatternByte(
                            x: x, y: y, frame: frame, channel: channel
                        )
                    }
                    bytes[y * stride + x * 4 + 3] = 255
                }
            }
        }
        let actual = try map.extractPlanarRGB(from: buffer).map(Double.init)
        var expected = [Double](repeating: 0, count: actual.count)
        for channel in 0..<3 {
            for y in 0..<modelSize {
                for x in 0..<modelSize {
                    let coordinate = map.sourceCoordinate(modelX: x, modelY: y)
                    let index = channel * modelSize * modelSize + y * modelSize + x
                    expected[index] = interpolateSynthetic(x: Double(coordinate.x), y: Double(coordinate.y)) {
                        Double(movingPatternByte(
                            x: min(size - 1, max(0, $0)), y: min(size - 1, max(0, $1)),
                            frame: frame, channel: channel
                        )) / 255
                    }
                    maximumError = max(maximumError, abs(actual[index] - expected[index]))
                    if !previousActual.isEmpty {
                        let actualChange = actual[index] - previousActual[index]
                        let expectedChange = expected[index] - previousExpected[index]
                        maximumTemporalError = max(maximumTemporalError, abs(actualChange - expectedChange))
                        largestFrameChange = max(largestFrameChange, abs(expectedChange))
                    }
                }
            }
        }
        previousActual = actual
        previousExpected = expected
    }
    // Half-precision rounding: half an ULP spatially, two roundings temporally.
    #expect(maximumError <= 0.000_25)
    #expect(maximumTemporalError <= 0.000_5)
    #expect(largestFrameChange > 0.05)
    print("Synthetic \(projection.rawValue) moving detail at \(origin): \(boundarySamples) boundary samples, max sample error \(maximumError), temporal error \(maximumTemporalError)")
}
