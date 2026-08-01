import Testing
@testable import JasnaMetalPoC

@Test func mosaicManifestSelectsOnlyIntersectingTilesAndWindows() throws {
    let plan = try SideBySideVideoPlan(
        width: 1_024,
        height: 1_024,
        sourceFramesPerSecond: 30,
        durationSeconds: 2,
        eyeLayout: .singleEye
    )
    let manifest = MosaicRegionManifest(
        version: 1,
        width: 1_024,
        height: 1_024,
        framesPerSecond: 30,
        frameCount: 60,
        regions: [
            MosaicRegion(
                startFrame: 0,
                endFrame: 30,
                x: 256,
                y: 256,
                width: 256,
                height: 256,
                confidence: 0.9
            )
        ]
    )

    try manifest.validate(for: plan)
    let first = manifest.regions(intersecting: 0..<30)
    let second = manifest.regions(intersecting: 30..<60)
    let tiles = manifest.tiles(from: plan, intersecting: first)

    #expect(first.count == 1)
    #expect(second.isEmpty)
    #expect(!tiles.isEmpty)
    #expect(tiles.count < plan.tiles.count)
    #expect(tiles.allSatisfy { first[0].intersects(tile: $0) })
}

@Test func mosaicRegionFeathersAtEdgesAndKeepsTheCenterOpaque() {
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 100,
        height: 100,
        confidence: 1
    )

    #expect(region.featherAlpha(x: 99, y: 250, feather: 12) == 0)
    #expect(region.featherAlpha(x: 100, y: 250, feather: 12) > 0)
    #expect(region.featherAlpha(x: 150, y: 250, feather: 12) == 1)
}

@available(macOS 27.0, *)
@Test func sparseCacheVariantIsStableForTheSameTiles() throws {
    let plan = try SideBySideVideoPlan(
        width: 512,
        height: 512,
        sourceFramesPerSecond: 30,
        durationSeconds: 1,
        eyeLayout: .singleEye
    )

    let first = SideBySideRestoration.sparseCacheVariant(tiles: Array(plan.tiles.prefix(2)))
    let repeated = SideBySideRestoration.sparseCacheVariant(tiles: Array(plan.tiles.prefix(2)))
    let different = SideBySideRestoration.sparseCacheVariant(tiles: Array(plan.tiles.suffix(2)))

    #expect(first == repeated)
    #expect(first != different)
}

@available(macOS 27.0, *)
@Test func sparseRegionCacheSeparatesVRProjectionModes() {
    let regions = [
        MosaicRegion(
            startFrame: 0,
            endFrame: 30,
            x: 1_200,
            y: 2_300,
            width: 280,
            height: 280,
            confidence: 0.9
        )
    ]

    let raw = SideBySideRestoration.sparseRegionCacheVariant(
        regions: regions, projection: .raw
    )
    let fisheye = SideBySideRestoration.sparseRegionCacheVariant(
        regions: regions, projection: .fisheye
    )
    let shiftedRange = SideBySideRestoration.sparseRegionCacheVariant(
        regions: [
            MosaicRegion(
                startFrame: 6,
                endFrame: 12,
                x: 1_200,
                y: 2_300,
                width: 280,
                height: 280,
                confidence: 0.9
            )
        ],
        projection: .fisheye
    )

    #expect(raw != fisheye)
    #expect(fisheye != shiftedRange)
    #expect(raw.contains("raw"))
    #expect(fisheye.contains("fisheye"))
}

@Test func mosaicCropUsesAspectFitAndReflectPadding() {
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 512,
        height: 256,
        confidence: 1,
        blendX: 120,
        blendY: 220,
        blendWidth: 472,
        blendHeight: 216
    )
    let transform = MosaicCropTransform(region: region)

    #expect(transform.resizedWidth == 256)
    #expect(transform.resizedHeight == 128)
    #expect(transform.padX == 0)
    #expect(transform.padY == 64)
    let center = transform.modelCoordinate(pixelX: 356, pixelY: 328)
    #expect(abs(center.x - 127.75) < 0.01)
    #expect(abs(center.y - 127.75) < 0.01)
}

@Test func fisheyeMosaicCropApproximatelyRoundTripsModelCoordinates() {
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 3_149,
        y: 2_524,
        width: 276,
        height: 275,
        confidence: 1
    )
    let transform = FisheyeMosaicCropTransform(
        region: region,
        eyeWidth: 4_096,
        eyeHeight: 4_096
    )

    for coordinate in [(32, 32), (128, 128), (224, 224)] {
        let source = transform.sourceCoordinate(
            modelX: coordinate.0, modelY: coordinate.1
        )
        let roundTrip = transform.modelCoordinate(
            pixelX: Int(source.x.rounded()),
            pixelY: Int(source.y.rounded())
        )
        #expect(abs(roundTrip.x - Float(coordinate.0)) < 2)
        #expect(abs(roundTrip.y - Float(coordinate.1)) < 2)
    }
    #expect(
        abs(
            (transform.fisheyeMaxU - transform.fisheyeMinU)
                - (transform.fisheyeMaxV - transform.fisheyeMinV)
        ) < 0.000_001
    )
}
