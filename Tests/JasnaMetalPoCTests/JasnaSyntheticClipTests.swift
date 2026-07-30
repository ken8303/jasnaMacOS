import Testing
@testable import JasnaMetalPoC

@Test func syntheticClipFramesUseStableDistinctPatterns() {
    let first = makeJasnaSyntheticFrame(index: 0)
    let second = makeJasnaSyntheticFrame(index: 1)
    #expect(first.count == 3 * 256 * 256)
    #expect(second.count == first.count)
    #expect(first[0] == Float16(17.0 / 1020.0))
    #expect(second[0] == Float16(40.0 / 1020.0))
    #expect(first != second)
}

@Test func bicubicQuarterScalePreservesAHorizontalRamp() throws {
    var input = [Float16](repeating: 0, count: 3 * 256 * 256)
    for channel in 0..<3 {
        for y in 0..<256 {
            for x in 0..<256 {
                input[channel * 256 * 256 + y * 256 + x] = Float16(x)
            }
        }
    }
    let output = try bicubicDownsampleQuarterReference(input)
    for channel in 0..<3 {
        for y in [0, 31, 63] {
            for x in [0, 17, 63] {
                let expected = Float16(Float(x) * 4 + 1.5)
                #expect(output[channel * 64 * 64 + y * 64 + x] == expected)
            }
        }
    }
}
