import CoreVideo
import Foundation
import Metal
import Testing
@testable import JasnaMetalPoC

// Generated geometry and solid colours only. No media, detector, or ML packages.
private func syntheticGrid(_ maximum: Int) -> MosaicRegionSubdivisionConfiguration {
    MosaicRegionSubdivisionConfiguration(
        maximumBlendDimension: maximum, overlap: 96, splitLimit: 1,
        maximumAxisCrops: 4, maskGrowthFraction: 0, maskFeatherFraction: 0,
        blockResidualGrowthFraction: 0, maskTemporalRadius: 0,
        maskTemporalStrength: 0, detailCropDimension: 0, detailCropCount: 0
    )
}

private func syntheticParent(keyframes: [MosaicMaskKeyframe]? = nil) -> MosaicRegion {
    MosaicRegion(
        startFrame: 0, endFrame: 30, x: 512, y: 1_536,
        width: 2_048, height: 1_536, confidence: 1,
        blendX: 608, blendY: 1_632, blendWidth: 1_856, blendHeight: 1_344,
        maskWidth: keyframes == nil ? nil : 64,
        maskHeight: keyframes == nil ? nil : 32,
        maskData: keyframes?.first?.maskData, maskKeyframes: keyframes
    )
}

@Test(arguments: [768, 1_024])
func syntheticGridHasNoCoverageGaps(maximum: Int) throws {
    let parents = [
        syntheticParent(),
        MosaicRegion(
            startFrame: 0, endFrame: 30, x: 1_792, y: 2_560,
            width: 2_304, height: 1_536, confidence: 1,
            blendX: 1_888, blendY: 2_656, blendWidth: 2_208, blendHeight: 1_440
        ),
        MosaicRegion(
            startFrame: 0, endFrame: 30, x: 147, y: 23,
            width: 2_777, height: 993, confidence: 1,
            blendX: 243, blendY: 119, blendWidth: 2_593, blendHeight: 801
        ),
    ]
    for parent in parents {
        let children = MosaicRegionSubdivision.subdivide(
            parent, configuration: syntheticGrid(maximum)
        )
        #expect(children.count > 1)
        #expect(children.allSatisfy {
            $0.x >= parent.x && $0.y >= parent.y
                && $0.x + $0.width <= parent.x + parent.width
                && $0.y + $0.height <= parent.y + parent.height
        })
        try MosaicRegionManifest(
            version: 1, width: 4_096, height: 4_096,
            framesPerSecond: 30, frameCount: 30, regions: children
        ).validate()

        // Merge intervals on every source row: checking only outer bounds can
        // miss a one-pixel internal gap from integer partition rounding.
        var uncoveredRows = 0
        let right = parent.effectiveBlendX + parent.effectiveBlendWidth
        for y in parent.effectiveBlendY..<(parent.effectiveBlendY + parent.effectiveBlendHeight) {
            let intervals = children.filter {
                y >= $0.effectiveBlendY && y < $0.effectiveBlendY + $0.effectiveBlendHeight
            }.sorted { $0.effectiveBlendX < $1.effectiveBlendX }
            var coveredUntil = parent.effectiveBlendX
            for child in intervals {
                if child.effectiveBlendX > coveredUntil { break }
                coveredUntil = max(coveredUntil, child.effectiveBlendX + child.effectiveBlendWidth)
            }
            if coveredUntil < right { uncoveredRows += 1 }
        }
        #expect(uncoveredRows == 0)
    }
}

@Test(arguments: [768, 1_024])
func syntheticMovingMaskSurvivesSubdivision(maximum: Int) throws {
    let centres = (0..<30).map { frame in (x: 8 + frame * 46 / 29, y: 8 + frame * 14 / 29) }
    let keyframes = centres.enumerated().map { frame, centre in
        var bytes = [UInt8](repeating: 0, count: 64 * 32)
        for y in (centre.y - 3)...(centre.y + 3) {
            for x in (centre.x - 3)...(centre.x + 3) { bytes[y * 64 + x] = 255 }
        }
        return MosaicMaskKeyframe(frame: frame, maskData: Data(bytes))
    }
    let parent = syntheticParent(keyframes: keyframes)
    let children = MosaicRegionSubdivision.subdivide(parent, configuration: syntheticGrid(maximum))
    #expect(children.count == (maximum == 768 ? 6 : 4))
    #expect(children.allSatisfy { $0.maskKeyframes?.count == 30 })
    var lostSamples = 0
    var backgroundLeaks = 0
    for (frame, centre) in centres.enumerated() {
        let resolved = children.map { $0.resolvingSegmentationMask(at: frame) }
        for dy in -1...1 {
            for dx in -1...1 {
                let x = parent.x + Int((Double(centre.x + dx) * Double(parent.width - 1) / 63).rounded())
                let y = parent.y + Int((Double(centre.y + dy) * Double(parent.height - 1) / 31).rounded())
                if !resolved.contains(where: {
                    $0.contains(x: x, y: y) && $0.segmentationMaskAlpha(x: x, y: y) >= 0.99
                }) { lostSamples += 1 }
            }
        }
        // Fixed background point, well outside the square throughout its motion.
        let backgroundX = parent.x + parent.width - 200
        let backgroundY = parent.y + 200
        if resolved.contains(where: {
            $0.contains(x: backgroundX, y: backgroundY)
                && $0.segmentationMaskAlpha(x: backgroundX, y: backgroundY) > 0.01
        }) { backgroundLeaks += 1 }
    }
    #expect(lostSamples == 0)
    #expect(backgroundLeaks == 0)
}

private func syntheticPixelBuffer(width: Int, height: Int, rgb: [UInt8]) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let attributes: [String: Any] = [
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ]
    let status = CVPixelBufferCreate(
        nil, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer
    )
    #expect(status == kCVReturnSuccess)
    let result = try #require(buffer)
    let lock = CVPixelBufferLockBaseAddress(result, [])
    #expect(lock == kCVReturnSuccess)
    defer { CVPixelBufferUnlockBaseAddress(result, []) }
    let bytes = try #require(CVPixelBufferGetBaseAddress(result)).assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(result)
    for y in 0..<height {
        for x in 0..<width {
            let offset = y * stride + x * 4
            bytes[offset] = rgb[2]
            bytes[offset + 1] = rgb[1]
            bytes[offset + 2] = rgb[0]
            bytes[offset + 3] = 255
        }
    }
    return result
}

@available(macOS 27.0, *)
@Test func syntheticStereoOverlapMatchesColourOracle() throws {
    let eyeWidth = 128
    let height = 96
    let baseRGB: [[UInt8]] = [[96, 80, 64], [80, 64, 48]]
    let restoredRGB: [[Float16]] = [[0.75, 0.5, 0.25], [0.25, 0.5, 0.75]]
    let left = try syntheticPixelBuffer(width: eyeWidth, height: height, rgb: baseRGB[0])
    let right = try syntheticPixelBuffer(width: eyeWidth, height: height, rgb: baseRGB[1])
    let output = try syntheticPixelBuffer(width: eyeWidth * 2, height: height, rgb: [0, 0, 0])
    let regions = [
        MosaicRegion(
            startFrame: 0, endFrame: 1, x: 8, y: 8, width: 80, height: 80,
            confidence: 1, blendX: 16, blendY: 16, blendWidth: 64, blendHeight: 64,
            subdivisionGroup: 1
        ),
        MosaicRegion(
            startFrame: 0, endFrame: 1, x: 40, y: 8, width: 80, height: 80,
            confidence: 1, blendX: 48, blendY: 16, blendWidth: 64, blendHeight: 64,
            subdivisionGroup: 1
        ),
    ]
    let maps = regions.map {
        MosaicCropSamplingMap(region: $0, eyeWidth: eyeWidth, eyeHeight: height, projection: .fisheye)
    }
    let plane = SideBySideVideoPlan.modelTileSize * SideBySideVideoPlan.modelTileSize
    var inputs = [MetalMosaicCompositeInput]()
    for eye in 0..<2 {
        let restored = restoredRGB[eye].flatMap { [Float16](repeating: $0, count: plane) }
        let original = baseRGB[eye].flatMap {
            [Float16](repeating: Float16(Float($0) / 255), count: plane)
        }
        for (region, map) in zip(regions, maps) {
            inputs.append(MetalMosaicCompositeInput(
                region: SideBySideRestoration.RestoredFrameWriter.translated(
                    region, xOffset: eye * eyeWidth
                ),
                restored: restored, original: original, samples: map.compositeSamples
            ))
        }
    }
    let compositor = try MetalMosaicCompositor(
        device: try #require(MTLCreateSystemDefaultDevice()), ordinaryMaskRecoveryEnabled: false
    )
    try compositor.compositeStereo(
        leftPixelBuffer: left, rightPixelBuffer: right, outputPixelBuffer: output,
        dimensions: VideoDimensions(width: eyeWidth * 2, height: height), inputs: inputs
    )

    let lock = CVPixelBufferLockBaseAddress(output, .readOnly)
    #expect(lock == kCVReturnSuccess)
    defer { CVPixelBufferUnlockBaseAddress(output, .readOnly) }
    let bytes = try #require(CVPixelBufferGetBaseAddress(output)).assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(output)
    let residualLimit = MosaicCompositeQuality.detailResidualLimit()
    var maximumByteError = 0
    var changedBackgroundChannels = 0
    var changedAlphaPixels = 0
    var overlappingPixels = 0
    for y in 0..<height {
        for x in 0..<(eyeWidth * 2) {
            let eye = x / eyeWidth
            let localX = x % eyeWidth
            var coverage: Float = 0
            var contributingCrops = 0
            for (region, map) in zip(regions, maps) where
                localX >= region.x && localX < region.x + region.width
                    && y >= region.y && y < region.y + region.height {
                let sample = map.compositeSamples[(y - region.y) * region.width + localX - region.x]
                coverage += sample.alpha
                if sample.alpha > 0 { contributingCrops += 1 }
            }
            if contributingCrops > 1 { overlappingPixels += 1 }
            coverage = min(coverage, 1)
            let offset = y * stride + x * 4
            for channel in 0..<3 {
                let base = Float(baseRGB[eye][channel]) / 255
                let original = Float(Float16(base))
                let detail = min(residualLimit, max(-residualLimit, base - original))
                let expected = base * (1 - coverage)
                    + (Float(restoredRGB[eye][channel]) + detail) * coverage
                let actual = Int(bytes[offset + 2 - channel])
                maximumByteError = max(maximumByteError, abs(actual - Int((expected * 255).rounded())))
                if coverage == 0 && actual != Int(baseRGB[eye][channel]) {
                    changedBackgroundChannels += 1
                }
            }
            if bytes[offset + 3] != 255 { changedAlphaPixels += 1 }
        }
    }
    #expect(overlappingPixels > 0)
    #expect(maximumByteError <= 2)
    #expect(changedBackgroundChannels == 0)
    #expect(changedAlphaPixels == 0)
    print("Synthetic stereo: \(overlappingPixels) overlapping pixels, maximum byte error \(maximumByteError)")
}
