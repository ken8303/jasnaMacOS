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

        let xOrigins = try Self.axisOrigins(length: eyeDimensions.width, overlap: overlap)
        let yOrigins = try Self.axisOrigins(length: eyeDimensions.height, overlap: overlap)
        var generated = [VideoTile]()
        generated.reserveCapacity(2 * xOrigins.count * yOrigins.count)
        for eye in 0..<2 {
            let eyeOffset = eye * eyeDimensions.width
            for y in yOrigins {
                for localX in xOrigins {
                    generated.append(VideoTile(
                        eyeIndex: eye,
                        x: eyeOffset + localX,
                        y: y,
                        width: tileSize,
                        height: tileSize
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

    private static func axisOrigins(length: Int, overlap: Int) throws -> [Int] {
        let tileSize = modelTileSize
        guard length >= tileSize else { throw DeformConvError.invalidShape }
        if length == tileSize { return [0] }

        let step = tileSize - overlap
        var origins = Array(Swift.stride(from: 0, through: length - tileSize, by: step))
        let finalOrigin = length - tileSize
        if origins.last != finalOrigin { origins.append(finalOrigin) }
        return origins
    }
}
