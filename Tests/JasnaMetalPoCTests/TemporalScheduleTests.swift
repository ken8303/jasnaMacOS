import Testing
@testable import JasnaMetalPoC

@Test func temporalScheduleMatchesBasicVSRTraversal() throws {
    let schedule = try TemporalSchedule(frameCount: 4)
    let backward = schedule.steps.filter { $0.branch.name == "backward_1" }
    let forward = schedule.steps.filter { $0.branch.name == "forward_1" }
    #expect(backward.map(\.frameIndex) == [3, 2, 1, 0])
    #expect(backward.map(\.flowIndex) == [nil, 2, 1, 0])
    #expect(forward.map(\.frameIndex) == [0, 1, 2, 3])
    #expect(forward.map(\.flowIndex) == [nil, 0, 1, 2])
    #expect(backward[2].secondOrderFrameIndex == 3)
    #expect(forward[2].secondOrderFrameIndex == 0)
}

@Test func temporalScheduleUsesExpectedBackboneWidths() throws {
    let schedule = try TemporalSchedule(frameCount: 3)
    #expect(schedule.branches.map(\.backboneInputChannels) == [128, 192, 256, 320])
    #expect(schedule.steps.count == 12)
    #expect(schedule.flowCountPerDirection == 2)
}

@Test func fiveFrameScheduleCarriesRollingSecondOrderHistory() throws {
    let schedule = try TemporalSchedule(frameCount: 5)
    let backward = schedule.steps.filter { $0.branch.name == "backward_1" }
    let forward = schedule.steps.filter { $0.branch.name == "forward_1" }
    #expect(backward.map(\.frameIndex) == [4, 3, 2, 1, 0])
    #expect(backward.map(\.flowIndex) == [nil, 3, 2, 1, 0])
    #expect(backward.map(\.secondOrderFrameIndex) == [nil, nil, 4, 3, 2])
    #expect(forward.map(\.frameIndex) == [0, 1, 2, 3, 4])
    #expect(forward.map(\.flowIndex) == [nil, 0, 1, 2, 3])
    #expect(forward.map(\.secondOrderFrameIndex) == [nil, nil, 0, 1, 2])
}
