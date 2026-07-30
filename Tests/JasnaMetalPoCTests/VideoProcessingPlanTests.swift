import Testing
@testable import JasnaMetalPoC

@Test func eightKSideBySidePlanKeepsTilesInsideEachEye() throws {
    let plan = try SideBySideVideoPlan(
        width: 7_680,
        height: 4_320,
        sourceFramesPerSecond: 60,
        durationSeconds: 10
    )

    #expect(plan.eyeDimensions == VideoDimensions(width: 3_840, height: 4_320))
    #expect(plan.tilesPerEye == 340)
    #expect(plan.tiles.count == 680)
    #expect(plan.tiles.filter { $0.eyeIndex == 0 }.allSatisfy { $0.x + $0.width <= 3_840 })
    #expect(plan.tiles.filter { $0.eyeIndex == 1 }.allSatisfy { $0.x >= 3_840 })
    #expect(plan.tiles.allSatisfy { $0.y + $0.height <= 4_320 })
    #expect(plan.tiles.contains { $0.eyeIndex == 0 && $0.x == 3_584 && $0.y == 4_064 })
    #expect(plan.tiles.contains { $0.eyeIndex == 1 && $0.x == 7_424 && $0.y == 4_064 })
    let positiveOverlaps = plan.tiles.flatMap {
        [$0.leftOverlap, $0.rightOverlap, $0.topOverlap, $0.bottomOverlap]
    }.filter { $0 > 0 }
    #expect(positiveOverlaps.min() == 32)
    #expect(positiveOverlaps.max() == 43)
}

@Test func frameRatePlanConvertsSixtyToThirtyWithoutChangingDuration() throws {
    let plan = try ConstantFrameRatePlan(
        sourceFramesPerSecond: 60,
        outputFramesPerSecond: 30,
        durationSeconds: 2
    )

    #expect(plan.sourceFrameCount == 120)
    #expect(plan.outputFrameCount == 60)
    #expect(try plan.sourceFrameIndex(forOutputFrame: 0) == 0)
    #expect(try plan.sourceFrameIndex(forOutputFrame: 1) == 2)
    #expect(try plan.sourceFrameIndex(forOutputFrame: 59) == 118)
}

@Test func frameRatePlanDuplicatesFramesWhenSourceIsSlower() throws {
    let plan = try ConstantFrameRatePlan(
        sourceFramesPerSecond: 24,
        outputFramesPerSecond: 30,
        durationSeconds: 1
    )
    let firstSix = try (0..<6).map(plan.sourceFrameIndex)

    #expect(plan.outputFrameCount == 30)
    #expect(firstSix == [0, 1, 2, 2, 3, 4])
}

@Test func oneSecondEightKPlanUsesOneTemporalWindowPerTile() throws {
    let plan = try SideBySideVideoPlan(
        width: 7_680,
        height: 4_320,
        sourceFramesPerSecond: 59.94,
        durationSeconds: 1
    )

    #expect(plan.frameRate.outputFrameCount == 30)
    #expect(plan.temporalWindowCount == 1)
    #expect(plan.modelGraphExecutions == 680)
    #expect(plan.outputBGRABytesPerFrame == 132_710_400)
}
