import Foundation
import Metal

@available(macOS 26.0, *)
final class MetalResourceCache: @unchecked Sendable {
    static let shared = MetalResourceCache()

    private let lock = NSLock()
    private let machineLearningExecutionLock = NSLock()
    private var machineLearningPipelines = [String: any MTL4MachineLearningPipelineState]()
    private var shaderLibraries = [UInt64: MTLLibrary]()
    private var computePipelines = [String: MTLComputePipelineState]()
    private var deformConvBuffers = [String: (weight: MTLBuffer, bias: MTLBuffer)]()

    private init() {}

    func beginMachineLearningExecution() {
        machineLearningExecutionLock.lock()
    }

    func endMachineLearningExecution() {
        machineLearningExecutionLock.unlock()
    }

    func machineLearningPipeline(
        device: MTLDevice,
        packageURL: URL,
        create: () throws -> any MTL4MachineLearningPipelineState
    ) throws -> any MTL4MachineLearningPipelineState {
        let key = "\(device.registryID):\(packageURL.standardizedFileURL.path)"
        lock.lock()
        if let cached = machineLearningPipelines[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let created = try create()
        lock.lock()
        defer { lock.unlock() }
        if let cached = machineLearningPipelines[key] { return cached }
        machineLearningPipelines[key] = created
        return created
    }

    func shaderLibrary(
        device: MTLDevice,
        create: () throws -> MTLLibrary
    ) throws -> MTLLibrary {
        let key = device.registryID
        lock.lock()
        if let cached = shaderLibraries[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let created = try create()
        lock.lock()
        defer { lock.unlock() }
        if let cached = shaderLibraries[key] { return cached }
        shaderLibraries[key] = created
        return created
    }

    func computePipeline(
        device: MTLDevice,
        function: MTLFunction
    ) throws -> MTLComputePipelineState {
        let key = "\(device.registryID):\(function.name)"
        lock.lock()
        if let cached = computePipelines[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let created = try device.makeComputePipelineState(function: function)
        lock.lock()
        defer { lock.unlock() }
        if let cached = computePipelines[key] { return cached }
        computePipelines[key] = created
        return created
    }

    func deformConvWeightBuffers(
        device: MTLDevice,
        direction: String,
        url: URL
    ) throws -> (weight: MTLBuffer, bias: MTLBuffer) {
        let key = "\(device.registryID):\(url.standardizedFileURL.path)"
        lock.lock()
        if let cached = deformConvBuffers[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let created = try DeformConvWeightSet(direction: direction, url: url)
            .makeBuffers(device: device)
        lock.lock()
        defer { lock.unlock() }
        if let cached = deformConvBuffers[key] { return cached }
        deformConvBuffers[key] = created
        return created
    }
}
