import Metal
import Testing
@testable import JasnaMetalPoC

@Test func temporalPreparationBatchMatchesIndependentRuns() throws {
    guard #available(macOS 27.0, *), MTLCreateSystemDefaultDevice() != nil else { return }
    let runner = try MetalDeformConv()
    let width = 2
    let height = 2
    let plane = width * height

    func values(count: Int, scale: Float, offset: Float) -> [Float16] {
        (0..<count).map { Float16(Float($0 % 11) * scale + offset) }
    }
    let first = (
        prop: values(count: 64 * plane, scale: 0.01, offset: -0.2),
        current: values(count: 64 * plane, scale: 0.02, offset: -0.1),
        n2: values(count: 64 * plane, scale: -0.01, offset: 0.3),
        flow: values(count: 2 * plane, scale: 0.01, offset: -0.02),
        previous: values(count: 2 * plane, scale: -0.015, offset: 0.04)
    )
    let second = (
        prop: values(count: 64 * plane, scale: -0.02, offset: 0.4),
        current: values(count: 64 * plane, scale: 0.015, offset: -0.3),
        n2: values(count: 64 * plane, scale: 0.005, offset: 0.1),
        flow: values(count: 2 * plane, scale: -0.02, offset: 0.03),
        previous: values(count: 2 * plane, scale: 0.01, offset: -0.05)
    )
    let firstResult = try runner.runTemporalPreparation(
        width: width, height: height,
        featProp: first.prop, featCurrent: first.current, featN2: first.n2,
        flow1: first.flow, previousFlow: first.previous, hasSecondOrder: true
    )
    let secondResult = try runner.runTemporalPreparation(
        width: width, height: height,
        featProp: second.prop, featCurrent: second.current, featN2: second.n2,
        flow1: second.flow, previousFlow: second.previous, hasSecondOrder: true
    )
    let batched = try runner.runTemporalPreparation(
        width: width, height: height,
        featProp: first.prop + second.prop,
        featCurrent: first.current + second.current,
        featN2: first.n2 + second.n2,
        flow1: first.flow + second.flow,
        previousFlow: first.previous + second.previous,
        hasSecondOrder: true,
        batch: 2
    )
    #expect(batched.conditions == firstResult.conditions + secondResult.conditions)
    #expect(batched.deformInput == firstResult.deformInput + secondResult.deformInput)
    #expect(
        batched.secondOrderFlow
            == firstResult.secondOrderFlow + secondResult.secondOrderFlow
    )
}

@Test func dcnOffsetTransformBatchMatchesIndependentRuns() throws {
    guard #available(macOS 27.0, *), MTLCreateSystemDefaultDevice() != nil else { return }
    let runner = try MetalDeformConv()
    let plane = 4
    func values(count: Int, scale: Float, offset: Float) -> [Float16] {
        (0..<count).map { Float16(Float($0 % 17) * scale + offset) }
    }
    let firstRaw = values(count: 432 * plane, scale: 0.002, offset: -0.1)
    let secondRaw = values(count: 432 * plane, scale: -0.003, offset: 0.2)
    let firstFlow1 = values(count: 2 * plane, scale: 0.01, offset: -0.03)
    let secondFlow1 = values(count: 2 * plane, scale: -0.02, offset: 0.05)
    let firstFlow2 = values(count: 2 * plane, scale: -0.01, offset: 0.04)
    let secondFlow2 = values(count: 2 * plane, scale: 0.015, offset: -0.02)
    let first = try runner.runDCNOffsetTransform(
        plane: plane, raw: firstRaw, flow1: firstFlow1, flow2: firstFlow2
    )
    let second = try runner.runDCNOffsetTransform(
        plane: plane, raw: secondRaw, flow1: secondFlow1, flow2: secondFlow2
    )
    let batched = try runner.runDCNOffsetTransform(
        plane: plane,
        raw: firstRaw + secondRaw,
        flow1: firstFlow1 + secondFlow1,
        flow2: firstFlow2 + secondFlow2,
        batch: 2
    )
    #expect(batched.offset == first.offset + second.offset)
    #expect(batched.mask == first.mask + second.mask)
}
