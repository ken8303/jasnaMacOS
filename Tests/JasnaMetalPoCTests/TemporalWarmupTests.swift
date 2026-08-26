import Testing
@testable import JasnaMetalPoC

@Test func firstTemporalWindowClampsWarmupAtSourceStart() {
    let schedule = TemporalWindowSchedule(
        outputStartFrame: 0, outputFrameCount: 30, requestedWarmupFrames: 5
    )

    #expect(schedule.decodedStartFrame == 0)
    #expect(schedule.warmupFrameCount == 0)
    #expect(schedule.decodedFrameCount == 30)
    #expect(schedule.outputFrameOffset == 0)
}

@Test func laterTemporalWindowPrependsWarmupWithoutChangingOutputCount() {
    let schedule = TemporalWindowSchedule(
        outputStartFrame: 60, outputFrameCount: 30, requestedWarmupFrames: 5
    )

    #expect(schedule.decodedStartFrame == 55)
    #expect(schedule.warmupFrameCount == 5)
    #expect(schedule.decodedFrameCount == 35)
    #expect(schedule.outputFrameOffset == 5)
    #expect(schedule.outputFrameCount == 30)
}

@Test func temporalWarmupEnvironmentDefaultsToValidatedFiveAndIsBounded() {
    #expect(
        TemporalWarmupConfiguration.fromEnvironment([:]).frames
            == TemporalWarmupConfiguration.maximumFrames
    )
    #expect(
        TemporalWarmupConfiguration.fromEnvironment([
            "JASNA_TEMPORAL_WARMUP_FRAMES": "5"
        ]).frames == 5
    )
    #expect(
        TemporalWarmupConfiguration.fromEnvironment([
            "JASNA_TEMPORAL_WARMUP_FRAMES": "999"
        ]).frames == TemporalWarmupConfiguration.maximumFrames
    )
    #expect(
        TemporalWarmupConfiguration.fromEnvironment([
            "JASNA_TEMPORAL_WARMUP_FRAMES": "-2"
        ]).frames == 0
    )
    #expect(
        TemporalWarmupConfiguration.fromEnvironment([
            "JASNA_TEMPORAL_WARMUP_FRAMES": "0"
        ]).frames == 0
    )
}

@Test func regionScheduleDiscardsWarmupBeforeWritingTargetFrames() throws {
    let window = TemporalWindowSchedule(
        outputStartFrame: 60, outputFrameCount: 30, requestedWarmupFrames: 5
    )
    let region = try #require(TemporalRegionSchedule(
        regionStartFrame: 60, regionEndFrame: 90, window: window
    ))

    #expect(region.outputLocalStart == 0)
    #expect(region.activeFrameCount == 30)
    #expect(region.decodedLocalRange == 0..<35)
    #expect(region.restoredFrameOffset == 5)
}

@Test func midWindowRegionUsesImmediateFramesAsWarmup() throws {
    let window = TemporalWindowSchedule(
        outputStartFrame: 60, outputFrameCount: 30, requestedWarmupFrames: 5
    )
    let region = try #require(TemporalRegionSchedule(
        regionStartFrame: 70, regionEndFrame: 80, window: window
    ))

    #expect(region.outputLocalStart == 10)
    #expect(region.activeFrameCount == 10)
    #expect(region.decodedLocalRange == 10..<25)
    #expect(region.restoredFrameOffset == 5)
}
