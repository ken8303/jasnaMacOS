import Foundation

struct VideoDimensions: Equatable, Sendable {
    let width: Int
    let height: Int

    var pixelCount: Int { width * height }
}

struct VideoTile: Equatable, Sendable {
    let eyeIndex: Int
    let x: Int
    let y: Int
    let width: Int
    let height: Int
    let leftOverlap: Int
    let rightOverlap: Int
    let topOverlap: Int
    let bottomOverlap: Int

    init(
        eyeIndex: Int,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        leftOverlap: Int = 0,
        rightOverlap: Int = 0,
        topOverlap: Int = 0,
        bottomOverlap: Int = 0
    ) {
        self.eyeIndex = eyeIndex
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.leftOverlap = leftOverlap
        self.rightOverlap = rightOverlap
        self.topOverlap = topOverlap
        self.bottomOverlap = bottomOverlap
    }
}

private struct TileAxisPlacement: Sendable {
    let origin: Int
    let leadingOverlap: Int
    let trailingOverlap: Int
}

struct ConstantFrameRatePlan: Sendable {
    let sourceFramesPerSecond: Double
    let outputFramesPerSecond: Double
    let durationSeconds: Double
    let sourceFrameCount: Int
    let outputFrameCount: Int

    init(
        sourceFramesPerSecond: Double,
        outputFramesPerSecond: Double = 30,
        durationSeconds: Double
    ) throws {
        guard sourceFramesPerSecond > 0,
              outputFramesPerSecond > 0,
              durationSeconds > 0,
              sourceFramesPerSecond.isFinite,
              outputFramesPerSecond.isFinite,
              durationSeconds.isFinite
        else { throw DeformConvError.invalidShape }

        self.sourceFramesPerSecond = sourceFramesPerSecond
        self.outputFramesPerSecond = outputFramesPerSecond
        self.durationSeconds = durationSeconds
        sourceFrameCount = max(1, Int((durationSeconds * sourceFramesPerSecond).rounded()))
        outputFrameCount = max(1, Int((durationSeconds * outputFramesPerSecond).rounded()))
    }

    func sourceFrameIndex(forOutputFrame outputFrame: Int) throws -> Int {
        guard (0..<outputFrameCount).contains(outputFrame) else {
            throw DeformConvError.invalidShape
        }
        let sourcePosition = Double(outputFrame) * sourceFramesPerSecond / outputFramesPerSecond
        return min(Int(sourcePosition.rounded()), sourceFrameCount - 1)
    }
}

struct SideBySideVideoPlan: Sendable {
    static let modelTileSize = 256
    static let defaultOverlap = 32
    static let outputFramesPerSecond = 30.0
    static let temporalWindowFrames = 30

    let dimensions: VideoDimensions
    let eyeDimensions: VideoDimensions
    let overlap: Int
    let tiles: [VideoTile]
    let frameRate: ConstantFrameRatePlan

    init(
        width: Int,
        height: Int,
        sourceFramesPerSecond: Double,
        durationSeconds: Double,
        overlap: Int = defaultOverlap
    ) throws {
        let tileSize = Self.modelTileSize
        guard width > 0,
              height > 0,
              width.isMultiple(of: 2),
              width / 2 >= tileSize,
              height >= tileSize,
              overlap >= 0,
              overlap < tileSize
        else { throw DeformConvError.invalidShape }

        dimensions = VideoDimensions(width: width, height: height)
        eyeDimensions = VideoDimensions(width: width / 2, height: height)
        self.overlap = overlap
        frameRate = try ConstantFrameRatePlan(
            sourceFramesPerSecond: sourceFramesPerSecond,
            outputFramesPerSecond: Self.outputFramesPerSecond,
            durationSeconds: durationSeconds
        )

        let xPlacements = try Self.axisPlacements(length: eyeDimensions.width, overlap: overlap)
        let yPlacements = try Self.axisPlacements(length: eyeDimensions.height, overlap: overlap)
        var generated = [VideoTile]()
        generated.reserveCapacity(2 * xPlacements.count * yPlacements.count)
        for eye in 0..<2 {
            let eyeOffset = eye * eyeDimensions.width
            for vertical in yPlacements {
                for horizontal in xPlacements {
                    generated.append(VideoTile(
                        eyeIndex: eye,
                        x: eyeOffset + horizontal.origin,
                        y: vertical.origin,
                        width: tileSize,
                        height: tileSize,
                        leftOverlap: horizontal.leadingOverlap,
                        rightOverlap: horizontal.trailingOverlap,
                        topOverlap: vertical.leadingOverlap,
                        bottomOverlap: vertical.trailingOverlap
                    ))
                }
            }
        }
        tiles = generated
    }

    var tilesPerEye: Int { tiles.count / 2 }

    var temporalWindowCount: Int {
        (frameRate.outputFrameCount + Self.temporalWindowFrames - 1)
            / Self.temporalWindowFrames
    }

    var modelGraphExecutions: Int { tiles.count * temporalWindowCount }

    var outputBGRABytesPerFrame: Int { dimensions.pixelCount * 4 }

    private static func axisPlacements(
        length: Int, overlap: Int
    ) throws -> [TileAxisPlacement] {
        let tileSize = modelTileSize
        guard length >= tileSize else { throw DeformConvError.invalidShape }
        if length == tileSize {
            return [TileAxisPlacement(origin: 0, leadingOverlap: 0, trailingOverlap: 0)]
        }

        let step = tileSize - overlap
        let finalOrigin = length - tileSize
        let intervalCount = Int(ceil(Double(finalOrigin) / Double(step)))
        let origins = (0...intervalCount).map { index in
            Int((Double(index) * Double(finalOrigin) / Double(intervalCount)).rounded())
        }
        return origins.indices.map { index in
            let leading = index == 0 ? 0 : origins[index - 1] + tileSize - origins[index]
            let trailing = index == origins.count - 1
                ? 0 : origins[index] + tileSize - origins[index + 1]
            return TileAxisPlacement(
                origin: origins[index],
                leadingOverlap: leading,
                trailingOverlap: trailing
            )
        }
    }
}
