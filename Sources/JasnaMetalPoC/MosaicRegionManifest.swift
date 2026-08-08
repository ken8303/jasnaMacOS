import Foundation

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
        blendHeight: Int? = nil
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
    }

    var frameRange: Range<Int> { startFrame..<endFrame }
    var effectiveBlendX: Int { blendX ?? x }
    var effectiveBlendY: Int { blendY ?? y }
    var effectiveBlendWidth: Int { blendWidth ?? width }
    var effectiveBlendHeight: Int { blendHeight ?? height }

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
                  region.confidence.isFinite
            else {
                throw DeformConvError.commandFailed("mosaic-region manifest contains an invalid region")
            }
        }
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
