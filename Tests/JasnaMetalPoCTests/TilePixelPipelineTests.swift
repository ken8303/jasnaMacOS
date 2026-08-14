import CoreVideo
import Metal
import Testing
@testable import JasnaMetalPoC

private func makeMetalPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    var optionalBuffer: CVPixelBuffer?
    let attributes: [String: Any] = [
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ]
    let status = CVPixelBufferCreate(
        nil, width, height, kCVPixelFormatType_32BGRA,
        attributes as CFDictionary, &optionalBuffer
    )
    #expect(status == kCVReturnSuccess)
    return try #require(optionalBuffer)
}

private func fillPixelBuffer(_ pixelBuffer: CVPixelBuffer, color: (UInt8, UInt8, UInt8, UInt8)) throws {
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    let base = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
        .assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    for y in 0..<CVPixelBufferGetHeight(pixelBuffer) {
        for x in 0..<CVPixelBufferGetWidth(pixelBuffer) {
            let offset = y * rowBytes + x * 4
            base[offset] = color.0
            base[offset + 1] = color.1
            base[offset + 2] = color.2
            base[offset + 3] = color.3
        }
    }
}

@available(macOS 27.0, *)
@Test func fusedMetalStereoCompositePlacesBothEyesWithoutCPUAssembly() throws {
    let eyeWidth = 8
    let height = 4
    let left = try makeMetalPixelBuffer(width: eyeWidth, height: height)
    let right = try makeMetalPixelBuffer(width: eyeWidth, height: height)
    let output = try makeMetalPixelBuffer(width: eyeWidth * 2, height: height)
    try fillPixelBuffer(left, color: (11, 22, 33, 255))
    try fillPixelBuffer(right, color: (44, 55, 66, 255))

    let device = try #require(MTLCreateSystemDefaultDevice())
    let compositor = try MetalMosaicCompositor(device: device)
    try compositor.compositeStereo(
        leftPixelBuffer: left,
        rightPixelBuffer: right,
        outputPixelBuffer: output,
        dimensions: VideoDimensions(width: eyeWidth * 2, height: height),
        inputs: []
    )

    CVPixelBufferLockBaseAddress(output, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(output, .readOnly) }
    let base = try #require(CVPixelBufferGetBaseAddress(output))
        .assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(output)
    for y in 0..<height {
        for x in 0..<(eyeWidth * 2) {
            let offset = y * rowBytes + x * 4
            let expected: [UInt8] = x < eyeWidth
                ? [11, 22, 33, 255] : [44, 55, 66, 255]
            #expect(Array(UnsafeBufferPointer(start: base + offset, count: 4)) == expected)
        }
    }
}

@Test func decodedTilesRoundTripThroughFeatherBlend() throws {
    let width = 960
    let height = 256
    var optionalBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        nil,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        [kCVPixelBufferMetalCompatibilityKey as String: true] as CFDictionary,
        &optionalBuffer
    )
    let pixelBuffer = try #require(optionalBuffer)
    #expect(status == kCVReturnSuccess)

    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    let base = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
        .assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    var expected = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
        for x in 0..<width {
            let blue = UInt8((x * 7 + y * 3) % 256)
            let green = UInt8((x * 5 + y * 11) % 256)
            let red = UInt8((x * 13 + y * 17) % 256)
            let source = y * rowBytes + x * 4
            base[source] = blue
            base[source + 1] = green
            base[source + 2] = red
            base[source + 3] = 255
            let packed = (y * width + x) * 4
            expected[packed] = blue
            expected[packed + 1] = green
            expected[packed + 2] = red
        }
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

    let plan = try SideBySideVideoPlan(
        width: width,
        height: height,
        sourceFramesPerSecond: 30,
        durationSeconds: 1
    )
    #expect(plan.tiles.count == 4)
    var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
    for tile in plan.tiles {
        let planar = try TilePixelPipeline.extractPlanarRGB(from: pixelBuffer, tile: tile)
        try accumulator.accumulate(tile: tile, planarRGB: planar)
    }
    let actual = try accumulator.makeBGRABytes()
    let maximumByteError = zip(expected, actual).map {
        abs(Int($0) - Int($1))
    }.max() ?? 0

    #expect(maximumByteError <= 1)
    #expect(abs((accumulator.accumulatedWeightRange?.lowerBound ?? 0) - 1) < 0.000_01)
    #expect(abs((accumulator.accumulatedWeightRange?.upperBound ?? 0) - 1) < 0.000_01)
}
