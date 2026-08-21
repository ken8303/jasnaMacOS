import Foundation
import Testing
@testable import JasnaMetalPoC

private let subdivisionConfiguration = MosaicRegionSubdivisionConfiguration(
    maximumBlendDimension: 800,
    overlap: 96,
    splitLimit: 1,
    maximumAxisCrops: 3,
    maskGrowthFraction: 0.05,
    maskFeatherFraction: 0.025,
    blockResidualGrowthFraction: 0.04,
    maskTemporalRadius: 1,
    detailCropDimension: 576,
    detailCropCount: 1
)

@Test func largeMosaicRegionBecomesOverlappingModelCrops() throws {
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 1_800,
        height: 1_600,
        confidence: 0.9,
        blendX: 200,
        blendY: 300,
        blendWidth: 1_600,
        blendHeight: 1_400,
        maskWidth: 4,
        maskHeight: 4,
        maskData: Data((0..<16).map { UInt8($0 * 16) }),
        maskKeyframes: [
            MosaicMaskKeyframe(frame: 0, maskData: Data(repeating: 32, count: 16)),
            MosaicMaskKeyframe(frame: 29, maskData: Data(repeating: 224, count: 16)),
        ]
    )

    let children = MosaicRegionSubdivision.subdivide(
        region, configuration: subdivisionConfiguration
    )

    #expect(children.count == 4)
    #expect(children[0].effectiveBlendX == region.effectiveBlendX)
    #expect(children[0].effectiveBlendY == region.effectiveBlendY)
    #expect(children[1].effectiveBlendX < children[0].effectiveBlendX
        + children[0].effectiveBlendWidth)
    #expect(children[2].effectiveBlendY < children[0].effectiveBlendY
        + children[0].effectiveBlendHeight)
    #expect(children.map { $0.effectiveBlendX + $0.effectiveBlendWidth }.max()
        == region.effectiveBlendX + region.effectiveBlendWidth)
    #expect(children.map { $0.effectiveBlendY + $0.effectiveBlendHeight }.max()
        == region.effectiveBlendY + region.effectiveBlendHeight)
    #expect(children.allSatisfy {
        $0.x >= region.x && $0.y >= region.y
            && $0.x + $0.width <= region.x + region.width
            && $0.y + $0.height <= region.y + region.height
    })
    #expect(children.allSatisfy { $0.maskData?.count == 16 })
    #expect(children.allSatisfy { $0.maskKeyframes?.count == 2 })
    #expect(children.allSatisfy { $0.subdivisionGroup == 1 })
    #expect(children[0].maskData != children[1].maskData)

    let manifest = MosaicRegionManifest(
        version: 1,
        width: 2_048,
        height: 2_048,
        framesPerSecond: 30,
        frameCount: 30,
        regions: children
    )
    try manifest.validate()
}

@Test func subdivisionOnlyExpandsLargestConfiguredRegion() {
    let largest = MosaicRegion(
        startFrame: 0, endFrame: 30, x: 0, y: 0,
        width: 1_600, height: 1_400, confidence: 1
    )
    let second = MosaicRegion(
        startFrame: 0, endFrame: 30, x: 1_700, y: 0,
        width: 1_100, height: 1_100, confidence: 1
    )

    let result = MosaicRegionSubdivision.expand(
        [second, largest], configuration: subdivisionConfiguration
    )

    #expect(result.splitRegionCount == 1)
    #expect(result.addedModelCropCount == 3)
    #expect(result.regions.count == 5)
    #expect(result.regions.contains(second))
    #expect(!result.regions.contains(largest))
}

@Test func subdivisionConfigurationCanBeDisabled() {
    let defaults = MosaicRegionSubdivisionConfiguration.fromEnvironment([:])
    let disabled = MosaicRegionSubdivisionConfiguration.fromEnvironment([
        "JASNA_LARGE_REGION_MAX_BLEND": "0"
    ])

    #expect(defaults.maximumBlendDimension == 768)
    #expect(defaults.overlap == 96)
    #expect(defaults.splitLimit == 1)
    #expect(defaults.maximumAxisCrops == 4)
    #expect(defaults.maskGrowthFraction == 0.05)
    #expect(defaults.maskFeatherFraction == 0.025)
    #expect(defaults.blockResidualGrowthFraction == 0.04)
    #expect(defaults.maskTemporalRadius == 1)
    #expect(defaults.detailCropDimension == 576)
    #expect(defaults.detailCropCount == 2)
    #expect(disabled == .disabled)
}

@Test func blockResidualGrowthReachesSquareMaskCorners() throws {
    var source = [UInt8](repeating: 0, count: 11 * 11)
    source[5 * 11 + 5] = 255

    let expanded = try #require(MosaicRegionSubdivision.expandedMask(
        Data(source),
        width: 11,
        height: 11,
        growthFraction: 0.10,
        featherFraction: 0,
        blockResidualGrowthFraction: 0.20
    ))
    let values = [UInt8](expanded)

    #expect(values[2 * 11 + 2] == 255)
    #expect(values[0] == 0)
}

@Test func adjacentMaskKeyframesProtectMovingEdgesAtLowWeight() {
    var first = [UInt8](repeating: 0, count: 9)
    var second = [UInt8](repeating: 0, count: 9)
    first[1] = 255
    second[7] = 255
    let stabilized = MosaicRegionSubdivision.temporallyStabilizedKeyframes(
        [
            MosaicMaskKeyframe(frame: 0, maskData: Data(first)),
            MosaicMaskKeyframe(frame: 2, maskData: Data(second)),
        ],
        expectedByteCount: 9,
        radius: 1
    )

    #expect(stabilized[0].maskData[1] == 255)
    #expect(stabilized[0].maskData[7] == 127)
    #expect(stabilized[1].maskData[1] == 127)
    #expect(stabilized[1].maskData[7] == 255)
}

@Test func adaptiveLargeRegionMaskGrowthAddsSoftOuterCoverage() throws {
    var source = [UInt8](repeating: 0, count: 9 * 9)
    source[4 * 9 + 4] = 255

    let expanded = try #require(MosaicRegionSubdivision.expandedMask(
        Data(source),
        width: 9,
        height: 9,
        growthFraction: 0.20,
        featherFraction: 0.20
    ))
    let values = [UInt8](expanded)

    #expect(values[4 * 9 + 4] == 255)
    #expect(values[4 * 9 + 2] == 255)
    #expect(values[4 * 9 + 1] > 0)
    #expect(values[4 * 9] == 0)
    #expect(values[2 * 9 + 2] > 0)
}

@Test func adaptiveLargeRegionMaskGrowthAlsoExpandsTheBlendEnvelope() throws {
    let mask = Data(repeating: 255, count: 64 * 64)
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 1_200,
        height: 1_000,
        confidence: 1,
        blendX: 250,
        blendY: 350,
        blendWidth: 900,
        blendHeight: 700,
        maskWidth: 64,
        maskHeight: 64,
        maskData: mask
    )

    let result = MosaicRegionSubdivision.expand(
        [region], configuration: subdivisionConfiguration
    )

    let blendLeft = try #require(result.regions.map(\.effectiveBlendX).min())
    let blendTop = try #require(result.regions.map(\.effectiveBlendY).min())
    let blendRight = try #require(result.regions.map {
        $0.effectiveBlendX + $0.effectiveBlendWidth
    }.max())
    let blendBottom = try #require(result.regions.map {
        $0.effectiveBlendY + $0.effectiveBlendHeight
    }.max())
    #expect(blendLeft < region.effectiveBlendX)
    #expect(blendTop < region.effectiveBlendY)
    #expect(blendRight > region.effectiveBlendX + region.effectiveBlendWidth)
    #expect(blendBottom > region.effectiveBlendY + region.effectiveBlendHeight)
    #expect(blendLeft >= region.x)
    #expect(blendTop >= region.y)
    #expect(blendRight <= region.x + region.width)
    #expect(blendBottom <= region.y + region.height)
}

@Test func maskGrowthAlsoCoversRegionsBelowTheSubdivisionThreshold() throws {
    var mask = [UInt8](repeating: 0, count: 9 * 9)
    mask[4 * 9 + 4] = 255
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 400,
        height: 360,
        confidence: 1,
        blendX: 180,
        blendY: 280,
        blendWidth: 240,
        blendHeight: 200,
        maskWidth: 9,
        maskHeight: 9,
        maskData: Data(mask)
    )

    let result = MosaicRegionSubdivision.expand(
        [region], configuration: subdivisionConfiguration
    )
    let expanded = try #require(result.regions.first)
    let expandedMask = try #require(expanded.maskData)

    #expect(result.splitRegionCount == 0)
    #expect(result.addedModelCropCount == 0)
    #expect(result.regions.count == 1)
    #expect(expanded.effectiveBlendX < region.effectiveBlendX)
    #expect(expanded.effectiveBlendY < region.effectiveBlendY)
    #expect(expanded.effectiveBlendWidth > region.effectiveBlendWidth)
    #expect(expanded.effectiveBlendHeight > region.effectiveBlendHeight)
    #expect(expandedMask != region.maskData)
    #expect(expandedMask[4 * 9 + 3] > 0)
}

@Test func lowerResidualDetailCropFollowsTheSemanticMaskBottom() throws {
    var mask = [UInt8](repeating: 0, count: 8 * 8)
    for y in 2...5 {
        for x in 2...6 { mask[y * 8 + x] = 255 }
    }
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 1_600,
        height: 1_400,
        confidence: 1,
        blendX: 200,
        blendY: 300,
        blendWidth: 1_400,
        blendHeight: 1_200,
        maskWidth: 8,
        maskHeight: 8,
        maskData: Data(mask)
    )

    let details = MosaicRegionSubdivision.lowerResidualDetailCrops(
        from: region,
        focusReference: region,
        configuration: subdivisionConfiguration,
        subdivisionGroup: 7
    )
    let detail = try #require(details.first)
    let maskBottom = region.y + 6 * region.height / 8

    #expect(details.count == 1)
    #expect(detail.effectiveBlendWidth == 576)
    #expect(detail.effectiveBlendHeight == 576)
    #expect(detail.effectiveBlendY <= maskBottom)
    #expect(detail.effectiveBlendY + detail.effectiveBlendHeight >= maskBottom)
    #expect(detail.subdivisionGroup == 7)
    #expect(detail.detailBlendFeather == 72)
    #expect(detail.maskData?.count == 64)
}

@Test func longMovingBoundaryReceivesTwoDistributedDetailCrops() throws {
    var mask = [UInt8](repeating: 0, count: 16 * 8)
    for y in 4...6 {
        for x in 1...14 { mask[y * 16 + x] = 255 }
    }
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 0,
        y: 0,
        width: 4_096,
        height: 1_400,
        confidence: 1,
        blendX: 64,
        blendY: 500,
        blendWidth: 3_968,
        blendHeight: 800,
        maskWidth: 16,
        maskHeight: 8,
        maskData: Data(mask)
    )
    let configuration = MosaicRegionSubdivisionConfiguration(
        maximumBlendDimension: 768,
        overlap: 96,
        splitLimit: 1,
        maximumAxisCrops: 4,
        maskGrowthFraction: 0.05,
        maskFeatherFraction: 0.025,
        blockResidualGrowthFraction: 0.04,
        maskTemporalRadius: 1,
        detailCropDimension: 576,
        detailCropCount: 2
    )

    let details = MosaicRegionSubdivision.lowerResidualDetailCrops(
        from: region,
        focusReference: region,
        configuration: configuration,
        subdivisionGroup: 1
    )

    #expect(details.count == 2)
    #expect(details[0].effectiveBlendX < details[1].effectiveBlendX)
    #expect(details.allSatisfy { $0.effectiveBlendWidth == 576 })
    #expect(details.allSatisfy { $0.detailBlendFeather == 72 })
}

@Test func detailCropUsesADeepEdgeFadeWithoutChangingGridChildren() throws {
    let mask = Data(repeating: 255, count: 8 * 8)
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 100,
        y: 200,
        width: 1_600,
        height: 1_400,
        confidence: 1,
        blendX: 200,
        blendY: 300,
        blendWidth: 1_400,
        blendHeight: 1_200,
        maskWidth: 8,
        maskHeight: 8,
        maskData: mask
    )

    let result = MosaicRegionSubdivision.expand(
        [region], configuration: subdivisionConfiguration
    )
    let detail = try #require(result.regions.first { $0.detailBlendFeather != nil })
    let gridChildren = result.regions.filter { $0.detailBlendFeather == nil }

    #expect(detail.detailBlendFeather == 72)
    #expect(detail.featherAlpha(
        x: detail.effectiveBlendX,
        y: detail.effectiveBlendY + detail.effectiveBlendHeight / 2,
        feather: detail.detailBlendFeather!
    ) < 0.02)
    #expect(detail.featherAlpha(
        x: detail.effectiveBlendX + detail.detailBlendFeather! - 1,
        y: detail.effectiveBlendY + detail.effectiveBlendHeight / 2,
        feather: detail.detailBlendFeather!
    ) > 0.98)
    #expect(!gridChildren.isEmpty)
    #expect(gridChildren.allSatisfy { $0.subdivisionGroup == 1 })
}

@Test func adaptiveBlendGrowthDoesNotIncreaseTheModelGridDimensions() {
    let mask = Data(repeating: 255, count: 64 * 64)
    let region = MosaicRegion(
        startFrame: 0,
        endFrame: 30,
        x: 0,
        y: 0,
        width: 1_800,
        height: 1_800,
        confidence: 1,
        blendX: 100,
        blendY: 100,
        blendWidth: 1_500,
        blendHeight: 1_500,
        maskWidth: 64,
        maskHeight: 64,
        maskData: mask
    )

    let result = MosaicRegionSubdivision.expand(
        [region], configuration: subdivisionConfiguration
    )

    #expect(result.regions.count == 5)
    #expect(result.addedModelCropCount == 4)
}
