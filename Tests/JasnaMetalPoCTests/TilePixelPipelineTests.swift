import CoreVideo
import Testing
@testable import JasnaMetalPoC

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
