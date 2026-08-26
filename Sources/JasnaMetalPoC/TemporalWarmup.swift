import Foundation

struct TemporalWarmupConfiguration: Equatable, Sendable {
    // Five 4096×4096 BGRA frames per eye already retain roughly 640 MiB in the
    // direct stereo path. Keep this experiment tightly bounded on unified memory.
    static let maximumFrames = 5

    let frames: Int

    static func fromEnvironment(_ environment: [String: String]) -> Self {
        let requested = Int(environment["JASNA_TEMPORAL_WARMUP_FRAMES"] ?? "") ?? maximumFrames
        return Self(frames: min(maximumFrames, max(0, requested)))
    }
}

struct TemporalWindowSchedule: Equatable, Sendable {
    let outputStartFrame: Int
    let outputFrameCount: Int
    let decodedStartFrame: Int

    var warmupFrameCount: Int { outputStartFrame - decodedStartFrame }
    var decodedFrameCount: Int { warmupFrameCount + outputFrameCount }
    var outputFrameOffset: Int { warmupFrameCount }

    init(outputStartFrame: Int, outputFrameCount: Int, requestedWarmupFrames: Int) {
        precondition(outputStartFrame >= 0)
        precondition(outputFrameCount > 0)
        self.outputStartFrame = outputStartFrame
        self.outputFrameCount = outputFrameCount
        decodedStartFrame = max(0, outputStartFrame - max(0, requestedWarmupFrames))
    }
}

struct TemporalRegionSchedule: Equatable, Sendable {
    let outputLocalStart: Int
    let activeFrameCount: Int
    let decodedLocalRange: Range<Int>
    let restoredFrameOffset: Int

    init?(
        regionStartFrame: Int,
        regionEndFrame: Int,
        window: TemporalWindowSchedule
    ) {
        let outputEndFrame = window.outputStartFrame + window.outputFrameCount
        let targetStartFrame = max(window.outputStartFrame, regionStartFrame)
        let targetEndFrame = min(outputEndFrame, regionEndFrame)
        guard targetStartFrame < targetEndFrame else { return nil }

        let modelStartFrame = max(
            window.decodedStartFrame,
            targetStartFrame - window.warmupFrameCount
        )
        outputLocalStart = targetStartFrame - window.outputStartFrame
        activeFrameCount = targetEndFrame - targetStartFrame
        let decodedLocalStart = modelStartFrame - window.decodedStartFrame
        let decodedLocalEnd = targetEndFrame - window.decodedStartFrame
        decodedLocalRange = Range(uncheckedBounds: (
            lower: decodedLocalStart, upper: decodedLocalEnd
        ))
        restoredFrameOffset = targetStartFrame - modelStartFrame
    }
}
