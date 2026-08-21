import Foundation

enum MosaicCompositeQuality {
    static func detailResidualLimit(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Float {
        guard let text = environment["JASNA_MOSAIC_DETAIL_RESIDUAL_LIMIT"],
              let value = Float(text), value.isFinite
        else { return 0.03 }
        return min(1, max(0, value))
    }
}

struct MosaicMaskKeyframe: Codable, Equatable, Sendable {
    let frame: Int
    let maskData: Data
}

struct MosaicRegion: Codable, Equatable, Sendable {
    let startFrame: Int
    let endFrame: Int
    let x: Int
    let y: Int
    let width: Int
    let height: Int
    let confidence: Double
    let blendX: Int?
    let blendY: Int?
    let blendWidth: Int?
    let blendHeight: Int?
    let maskWidth: Int?
    let maskHeight: Int?
    let maskData: Data?
    let maskKeyframes: [MosaicMaskKeyframe]?
    let subdivisionGroup: Int?
    let detailBlendFeather: Int?

    init(
        startFrame: Int,
        endFrame: Int,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        confidence: Double,
        blendX: Int? = nil,
        blendY: Int? = nil,
        blendWidth: Int? = nil,
        blendHeight: Int? = nil,
        maskWidth: Int? = nil,
        maskHeight: Int? = nil,
        maskData: Data? = nil,
        maskKeyframes: [MosaicMaskKeyframe]? = nil,
        subdivisionGroup: Int? = nil,
        detailBlendFeather: Int? = nil
    ) {
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.blendX = blendX
        self.blendY = blendY
        self.blendWidth = blendWidth
        self.blendHeight = blendHeight
        self.maskWidth = maskWidth
        self.maskHeight = maskHeight
        self.maskData = maskData
        self.maskKeyframes = maskKeyframes
        self.subdivisionGroup = subdivisionGroup
        self.detailBlendFeather = detailBlendFeather
    }

    var frameRange: Range<Int> { startFrame..<endFrame }
    var effectiveBlendX: Int { blendX ?? x }
    var effectiveBlendY: Int { blendY ?? y }
    var effectiveBlendWidth: Int { blendWidth ?? width }
    var effectiveBlendHeight: Int { blendHeight ?? height }
    var recommendedFeather: Int {
        max(12, min(64, max(effectiveBlendWidth, effectiveBlendHeight) / 32))
    }
    var hasSegmentationMask: Bool {
        guard let maskWidth, let maskHeight, let maskData else { return false }
        return maskWidth > 1 && maskHeight > 1 && maskData.count == maskWidth * maskHeight
    }

    func segmentationMaskAlpha(x pixelX: Int, y pixelY: Int) -> Float {
        guard hasSegmentationMask,
              let maskWidth, let maskHeight, let maskData,
              pixelX >= x, pixelX < x + width,
              pixelY >= y, pixelY < y + height
        else { return 1 }
        let maskX = Float(pixelX - x) * Float(maskWidth - 1) / Float(max(width - 1, 1))
        let maskY = Float(pixelY - y) * Float(maskHeight - 1) / Float(max(height - 1, 1))
        let x0 = Int(floor(maskX))
        let y0 = Int(floor(maskY))
        let x1 = min(x0 + 1, maskWidth - 1)
        let y1 = min(y0 + 1, maskHeight - 1)
        let fx = maskX - Float(x0)
        let fy = maskY - Float(y0)
        return maskData.withUnsafeBytes { bytes in
            let values = bytes.bindMemory(to: UInt8.self)
            let top = Float(values[y0 * maskWidth + x0]) * (1 - fx)
                + Float(values[y0 * maskWidth + x1]) * fx
            let bottom = Float(values[y1 * maskWidth + x0]) * (1 - fx)
                + Float(values[y1 * maskWidth + x1]) * fx
            return (top * (1 - fy) + bottom * fy) / 255
        }
    }

    func resolvingSegmentationMask(at frame: Int) -> MosaicRegion {
        let resolvedData = interpolatedSegmentationMask(at: frame) ?? maskData
        return MosaicRegion(
            startFrame: startFrame,
            endFrame: endFrame,
            x: x,
            y: y,
            width: width,
            height: height,
            confidence: confidence,
            blendX: blendX,
            blendY: blendY,
            blendWidth: blendWidth,
            blendHeight: blendHeight,
            maskWidth: maskWidth,
            maskHeight: maskHeight,
            maskData: resolvedData,
            subdivisionGroup: subdivisionGroup,
            detailBlendFeather: detailBlendFeather
        )
    }

    /// Diagnostic coverage mode that applies the restored delta throughout the
    /// complete detected crop. This isolates model quality from segmentation-
    /// mask coverage; normal restoration should retain the softer mask path.
    func usingFullDetectedRegionBlend() -> MosaicRegion {
        MosaicRegion(
            startFrame: startFrame,
            endFrame: endFrame,
            x: x,
            y: y,
            width: width,
            height: height,
            confidence: confidence,
            blendX: x,
            blendY: y,
            blendWidth: width,
            blendHeight: height,
            subdivisionGroup: subdivisionGroup,
            detailBlendFeather: detailBlendFeather
        )
    }

    private func interpolatedSegmentationMask(at frame: Int) -> Data? {
        guard let maskKeyframes, !maskKeyframes.isEmpty,
              let maskWidth, let maskHeight
        else { return nil }
        let expectedCount = maskWidth * maskHeight
        let upperIndex = maskKeyframes.firstIndex { $0.frame >= frame }
            ?? maskKeyframes.endIndex
        if upperIndex == maskKeyframes.startIndex {
            return maskKeyframes[upperIndex].maskData
        }
        if upperIndex == maskKeyframes.endIndex {
            return maskKeyframes[maskKeyframes.index(before: upperIndex)].maskData
        }
        let upper = maskKeyframes[upperIndex]
        if upper.frame == frame { return upper.maskData }
        let lower = maskKeyframes[maskKeyframes.index(before: upperIndex)]
        guard lower.maskData.count == expectedCount,
              upper.maskData.count == expectedCount,
              upper.frame > lower.frame
        else { return nil }
        let weight = Float(frame - lower.frame) / Float(upper.frame - lower.frame)
        var result = Data(count: expectedCount)
        result.withUnsafeMutableBytes { outputBytes in
            lower.maskData.withUnsafeBytes { lowerBytes in
                upper.maskData.withUnsafeBytes { upperBytes in
                    let output = outputBytes.bindMemory(to: UInt8.self)
                    let left = lowerBytes.bindMemory(to: UInt8.self)
                    let right = upperBytes.bindMemory(to: UInt8.self)
                    for index in 0..<expectedCount {
                        output[index] = UInt8(clamping: Int(
                            (Float(left[index]) * (1 - weight)
                                + Float(right[index]) * weight).rounded()
                        ))
                    }
                }
            }
        }
        return result
    }

    func intersects(_ range: Range<Int>) -> Bool {
        frameRange.overlaps(range)
    }

    func contains(x pixelX: Int, y pixelY: Int) -> Bool {
        pixelX >= effectiveBlendX && pixelX < effectiveBlendX + effectiveBlendWidth
            && pixelY >= effectiveBlendY && pixelY < effectiveBlendY + effectiveBlendHeight
    }

    func intersects(tile: VideoTile) -> Bool {
        x < tile.x + tile.width && x + width > tile.x
            && y < tile.y + tile.height && y + height > tile.y
    }

    func featherAlpha(x pixelX: Int, y pixelY: Int, feather: Int) -> Float {
        guard contains(x: pixelX, y: pixelY) else { return 0 }
        guard feather > 0 else { return 1 }
        let distance = min(
            pixelX - effectiveBlendX,
            effectiveBlendX + effectiveBlendWidth - 1 - pixelX,
            pixelY - effectiveBlendY,
            effectiveBlendY + effectiveBlendHeight - 1 - pixelY
        )
        return min(1, Float(distance + 1) / Float(feather + 1))
    }
}

struct MosaicRegionManifest: Codable, Equatable, Sendable {
    let version: Int
    let width: Int
    let height: Int
    let framesPerSecond: Double
    let frameCount: Int
    let regions: [MosaicRegion]

    static func load(from url: URL) throws -> MosaicRegionManifest {
        let manifest = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        try manifest.validate()
        return manifest
    }

    func validate() throws {
        guard version == 1,
              width > 0,
              height > 0,
              frameCount > 0,
              framesPerSecond > 0,
              framesPerSecond.isFinite
        else {
            throw DeformConvError.commandFailed("invalid mosaic-region manifest header")
        }
        for region in regions {
            guard region.startFrame >= 0,
                  region.endFrame > region.startFrame,
                  region.endFrame <= frameCount,
                  region.x >= 0,
                  region.y >= 0,
                  region.width > 0,
                  region.height > 0,
                  region.x + region.width <= width,
                  region.y + region.height <= height,
                  region.effectiveBlendX >= region.x,
                  region.effectiveBlendY >= region.y,
                  region.effectiveBlendWidth > 0,
                  region.effectiveBlendHeight > 0,
                  region.effectiveBlendX + region.effectiveBlendWidth <= region.x + region.width,
                  region.effectiveBlendY + region.effectiveBlendHeight <= region.y + region.height,
                  region.confidence.isFinite,
                  region.subdivisionGroup.map({ $0 > 0 }) ?? true,
                  (region.maskWidth == nil && region.maskHeight == nil && region.maskData == nil)
                    || region.hasSegmentationMask,
                  validMaskKeyframes(region)
            else {
                throw DeformConvError.commandFailed("mosaic-region manifest contains an invalid region")
            }
        }
    }

    private func validMaskKeyframes(_ region: MosaicRegion) -> Bool {
        guard let keyframes = region.maskKeyframes else { return true }
        guard !keyframes.isEmpty,
              let maskWidth = region.maskWidth,
              let maskHeight = region.maskHeight
        else { return false }
        let byteCount = maskWidth * maskHeight
        var previousFrame: Int?
        for keyframe in keyframes {
            guard keyframe.frame >= region.startFrame,
                  keyframe.frame < region.endFrame,
                  keyframe.maskData.count == byteCount,
                  previousFrame.map({ keyframe.frame > $0 }) ?? true
            else { return false }
            previousFrame = keyframe.frame
        }
        return true
    }

    func validate(for plan: SideBySideVideoPlan) throws {
        try validate()
        guard width == plan.dimensions.width,
              height == plan.dimensions.height,
              abs(framesPerSecond - SideBySideVideoPlan.outputFramesPerSecond) < 0.01,
              abs(frameCount - plan.frameRate.outputFrameCount) <= 1
        else {
            throw DeformConvError.commandFailed(
                "mosaic-region manifest does not match the eye video"
            )
        }
    }

    func regions(intersecting frameRange: Range<Int>) -> [MosaicRegion] {
        regions.filter { $0.intersects(frameRange) }
    }

    func tiles(
        from plan: SideBySideVideoPlan,
        intersecting activeRegions: [MosaicRegion]
    ) -> [VideoTile] {
        guard !activeRegions.isEmpty else { return [] }
        return plan.tiles.filter { tile in
            activeRegions.contains { $0.intersects(tile: tile) }
        }
    }
}
