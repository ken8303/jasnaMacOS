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

@Test func mosaicDetailResidualLimitIsBoundedAndConfigurable() {
    #expect(MosaicCompositeQuality.detailResidualLimit(environment: [:]) == 0.03)
    #expect(MosaicCompositeQuality.detailResidualLimit(environment: [
        "JASNA_MOSAIC_DETAIL_RESIDUAL_LIMIT": "0"
    ]) == 0)
    #expect(MosaicCompositeQuality.detailResidualLimit(environment: [
        "JASNA_MOSAIC_DETAIL_RESIDUAL_LIMIT": "2"
    ]) == 1)
}

@Test func fisheyeDetailCropFadesInFromItsRectangularEdge() {
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 0,
        y: 0,
        width: 100,
        height: 100,
        confidence: 1,
        blendX: 10,
        blendY: 10,
        blendWidth: 80,
        blendHeight: 80,
        subdivisionGroup: 1,
        detailBlendFeather: 20
    )
    let map = MosaicCropSamplingMap(
        region: region,
        eyeWidth: 100,
        eyeHeight: 100,
        projection: .fisheye
    )

    let edge = map.compositeSamples[50 * 100 + 10].alpha
    let inside = map.compositeSamples[50 * 100 + 29].alpha
    let outside = map.compositeSamples[50 * 100 + 9].alpha

    #expect(edge < 0.05)
    #expect(inside > 0.95)
    #expect(outside == 0)
}

@available(macOS 27.0, *)
@Test func rightEyeTranslationPreservesDetailCropFeather() {
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 576,
        height: 576,
        confidence: 1,
        subdivisionGroup: 3,
        detailBlendFeather: 72
    )

    let translated = SideBySideRestoration.RestoredFrameWriter.translated(
        region,
        xOffset: 4_096
    )

    #expect(translated.x == 4_196)
    #expect(translated.subdivisionGroup == 1_000_003)
    #expect(translated.detailBlendFeather == 72)
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

@available(macOS 27.0, *)
@Test func groupedDetailCropDoesNotIncreaseRectangularCoverage() throws {
    let width = 32
    let height = 32
    let base = try makeMetalPixelBuffer(width: width, height: height)
    let primaryOnly = try makeMetalPixelBuffer(width: width, height: height)
    let withDetail = try makeMetalPixelBuffer(width: width, height: height)
    try fillPixelBuffer(base, color: (0, 0, 0, 255))
    try fillPixelBuffer(primaryOnly, color: (0, 0, 0, 255))
    try fillPixelBuffer(withDetail, color: (0, 0, 0, 255))

    let primaryRegion = MosaicRegion(
        startFrame: 0,
        endFrame: 1,
        x: 0,
        y: 0,
        width: width,
        height: height,
        confidence: 1,
        blendX: 4,
        blendY: 4,
        blendWidth: 24,
        blendHeight: 24,
        subdivisionGroup: 1
    )
    let detailRegion = MosaicRegion(
        startFrame: 0,
        endFrame: 1,
        x: 0,
        y: 0,
        width: width,
        height: height,
        confidence: 1,
        blendX: 4,
        blendY: 4,
        blendWidth: 24,
        blendHeight: 24,
        subdivisionGroup: 1,
        detailBlendFeather: 8
    )
    let modelSize = SideBySideVideoPlan.modelTileSize
    let restored = [Float16](repeating: 0.5, count: 3 * modelSize * modelSize)
    let original = [Float16](repeating: 0, count: restored.count)
    func input(_ region: MosaicRegion) -> MetalMosaicCompositeInput {
        let map = MosaicCropSamplingMap(
            region: region,
            eyeWidth: width,
            eyeHeight: height,
            projection: .fisheye
        )
        return MetalMosaicCompositeInput(
            region: region,
            restored: restored,
            original: original,
            samples: map.compositeSamples
        )
    }

    let compositor = try MetalMosaicCompositor(
        device: try #require(MTLCreateSystemDefaultDevice())
    )
    let dimensions = VideoDimensions(width: width, height: height)
    try compositor.composite(
        basePixelBuffer: base,
        outputPixelBuffer: primaryOnly,
        dimensions: dimensions,
        inputs: [input(primaryRegion)]
    )
    try compositor.composite(
        basePixelBuffer: base,
        outputPixelBuffer: withDetail,
        dimensions: dimensions,
        inputs: [input(primaryRegion), input(detailRegion)]
    )

    func blue(atX x: Int, y: Int, in pixelBuffer: CVPixelBuffer) throws -> UInt8 {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let bytes = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
            .assumingMemoryBound(to: UInt8.self)
        return bytes[y * CVPixelBufferGetBytesPerRow(pixelBuffer) + x * 4]
    }
    let primaryBlue = try blue(atX: 5, y: 16, in: primaryOnly)
    let detailBlue = try blue(atX: 5, y: 16, in: withDetail)
    #expect(primaryBlue > 0)
    #expect(abs(Int(primaryBlue) - Int(detailBlue)) <= 1)
}

@available(macOS 27.0, *)
@Test func metalCompositeCapsLargeSourceResiduals() throws {
    let width = 32
    let height = 32
    let base = try makeMetalPixelBuffer(width: width, height: height)
    let output = try makeMetalPixelBuffer(width: width, height: height)
    try fillPixelBuffer(base, color: (0, 0, 0, 255))
    try fillPixelBuffer(output, color: (0, 0, 0, 255))
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 1,
        x: 0,
        y: 0,
        width: width,
        height: height,
        confidence: 1
    )
    let modelSize = SideBySideVideoPlan.modelTileSize
    let restored = [Float16](repeating: 0.5, count: 3 * modelSize * modelSize)
    let original = [Float16](repeating: 0.5, count: restored.count)
    let map = MosaicCropSamplingMap(
        region: region,
        eyeWidth: width,
        eyeHeight: height,
        projection: .fisheye
    )
    let compositor = try MetalMosaicCompositor(
        device: try #require(MTLCreateSystemDefaultDevice())
    )
    try compositor.composite(
        basePixelBuffer: base,
        outputPixelBuffer: output,
        dimensions: VideoDimensions(width: width, height: height),
        inputs: [MetalMosaicCompositeInput(
            region: region,
            restored: restored,
            original: original,
            samples: map.compositeSamples
        )]
    )

    CVPixelBufferLockBaseAddress(output, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(output, .readOnly) }
    let bytes = try #require(CVPixelBufferGetBaseAddress(output))
        .assumingMemoryBound(to: UInt8.self)
    let center = 16 * CVPixelBufferGetBytesPerRow(output) + 16 * 4
    #expect(bytes[center] > 95)
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
