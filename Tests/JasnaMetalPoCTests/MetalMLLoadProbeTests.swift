import Testing
@testable import JasnaMetalPoC

private func loadMeasurement(_ package: String, cacheHit: Bool) -> MetalMLPipelineLoadMeasurement {
    MetalMLPipelineLoadMeasurement(
        package: package, cacheHit: cacheHit,
        libraryMilliseconds: cacheHit ? 0 : 10,
        compilerMilliseconds: cacheHit ? 0 : 1,
        specializationMilliseconds: cacheHit ? 0 : 50,
        totalMilliseconds: cacheHit ? 0.01 : 61
    )
}

@Test func loadOnlyProbePreservesBranchOrderAndChecksSecondPassCache() throws {
    var seen = [String: Int]()
    var messages = [String]()
    let result = try MetalMLLoadProbe.run(load: { package in
        let count = seen[package, default: 0]
        seen[package] = count + 1
        return loadMeasurement(package, cacheHit: count > 0)
    }, report: { messages.append($0) })
    let expected = [
        "offset_backward_1", "backbone_backward_1",
        "offset_forward_1", "backbone_forward_1",
        "offset_backward_2", "backbone_backward_2",
        "offset_forward_2", "backbone_forward_2",
    ]
    #expect(result.map(\.package) == expected + expected)
    #expect(result.prefix(8).allSatisfy { !$0.cacheHit })
    #expect(result.suffix(8).allSatisfy { $0.cacheHit })
    #expect(messages.contains { $0.contains("specialization 50.000 ms") })
}

@Test func loadOnlyProbeRejectsSecondPassCacheMiss() {
    #expect(throws: DeformConvError.self) {
        try MetalMLLoadProbe.run(
            load: { loadMeasurement($0, cacheHit: false) }, report: { _ in }
        )
    }
}

@Test func loadOnlyProbeDoesNotRequireAClearedPersistentCache() throws {
    let result = try MetalMLLoadProbe.run(
        load: { loadMeasurement($0, cacheHit: true) }, report: { _ in }
    )
    #expect(result.count == 16)
}

@Test func loadOnlyProbeStopsOnPackageFailure() {
    struct InjectedFailure: Error {}
    var attempts = 0
    #expect(throws: InjectedFailure.self) {
        try MetalMLLoadProbe.run(load: { _ in
            attempts += 1
            throw InjectedFailure()
        }, report: { _ in })
    }
    #expect(attempts == 1)
}

@Test func loadOnlyProbeRejectsMismatchedMeasurement() {
    #expect(throws: DeformConvError.self) {
        try MetalMLLoadProbe.run(
            load: { _ in loadMeasurement("wrong", cacheHit: true) }, report: { _ in }
        )
    }
}
