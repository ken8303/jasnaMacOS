import Testing
@testable import JasnaMetalPoC

@Test func oneByOneConvolutionMatchesExpectedValues() throws {
    let shape = DeformConvShape(
        batch: 1,
        inputChannels: 1,
        inputHeight: 2,
        inputWidth: 2,
        outputChannels: 1,
        outputHeight: 2,
        outputWidth: 2,
        kernelHeight: 1,
        kernelWidth: 1,
        padHeight: 0,
        padWidth: 0,
        strideHeight: 1,
        strideWidth: 1,
        dilationHeight: 1,
        dilationWidth: 1,
        groups: 1,
        offsetGroups: 1
    )
    let output = try DeformConvCPU.run(
        shape: shape,
        input: [1, 2, 3, 4],
        offset: [Float](repeating: 0, count: shape.offsetCount),
        mask: [Float](repeating: 0.5, count: shape.maskCount),
        weight: [2],
        bias: [1]
    )
    #expect(output == [2, 3, 4, 5])
}
