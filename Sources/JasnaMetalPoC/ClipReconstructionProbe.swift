import Foundation
import Metal

struct ClipReconstructionResult {
    let statistics: BenchmarkStatistics
    let medianMilliseconds: Double
    let minimumMilliseconds: Double
    let maximumMilliseconds: Double
    let iterations: Int
    let elementCount: Int
    let maximumMagnitude: Float
    let residualMaximumError: Float
    let repeatMaximumError: Float
    let checksums: [Double]
}

@available(macOS 27.0, *)
func verifyReconstructedClip(
    device: MTLDevice,
    modelsURL: URL,
    spatialFrames: [[Float16]],
    branchFrames: [[[Float16]]],
    inputFrames: [[Float16]]
) throws -> ClipReconstructionResult {
    let featurePlane = 64 * 64
    let featureCount = 64 * featurePlane
    let frameCount = 3 * 256 * 256
    guard spatialFrames.count == 3,
          spatialFrames.allSatisfy({ $0.count == featureCount }),
          branchFrames.count == 4,
          branchFrames.allSatisfy({ branch in
              branch.count == 3 && branch.allSatisfy({ $0.count == featureCount })
          }),
          inputFrames.count == 3,
          inputFrames.allSatisfy({ $0.count == frameCount })
    else { throw DeformConvError.invalidShape }

    func makeBuffer(elements: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: elements * 2, options: .storageModeShared) else {
            throw DeformConvError.metalUnavailable
        }
        return buffer
    }
    func makeTensor(dimensions: [Int]) throws -> (any MTLTensor, MTLBuffer) {
        var strides = [Int]()
        var elements = 1
        for dimension in dimensions {
            strides.append(elements)
            elements *= dimension
        }
        let buffer = try makeBuffer(elements: elements)
        let descriptor = MTLTensorDescriptor()
        descriptor.dimensions = try tensorExtents(dimensions)
        descriptor.strides = try tensorExtents(strides)
        descriptor.dataType = .float16
        descriptor.usage = [.compute, .machineLearning]
        descriptor.storageMode = .shared
        let attachments = MTLTensorBufferAttachments()
        attachments.setBuffer(buffer, offset: 0, for: .data)
        return (try device.makeTensor(descriptor: descriptor, attachments: attachments), buffer)
    }
    func makeMLArguments(
        pipeline: any MTL4MachineLearningPipelineState,
        resources: [String: MTLResourceID]
    ) throws -> any MTL4ArgumentTable {
        guard let bindings = pipeline.reflection?.bindings else {
            throw DeformConvError.commandFailed("Metal ML pipeline has no reflected bindings")
        }
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = (bindings.map(\.index).max() ?? 0) + 1
        descriptor.initializeBindings = true
        let table = try device.makeArgumentTable(descriptor: descriptor)
        for (name, resource) in resources {
            guard let binding = bindings.first(where: { $0.name == name }) else {
                throw DeformConvError.commandFailed("missing Metal ML binding: \(name)")
            }
            table.setResource(resource, bufferIndex: binding.index)
        }
        return table
    }
    func makeComputeArguments(_ buffers: [MTLBuffer]) throws -> any MTL4ArgumentTable {
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = buffers.count
        let table = try device.makeArgumentTable(descriptor: descriptor)
        for (index, buffer) in buffers.enumerated() {
            table.setAddress(buffer.gpuAddress, index: index)
        }
        return table
    }
    func makeConstant<T>(_ value: inout T) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: 256, options: .storageModeShared) else {
            throw DeformConvError.metalUnavailable
        }
        withUnsafeBytes(of: &value) { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        return buffer
    }
    func makeHeap(size: Int) throws -> MTLHeap {
        let descriptor = MTLHeapDescriptor()
        descriptor.storageMode = .private
        descriptor.size = max(size, 4_096)
        guard let heap = device.makeHeap(descriptor: descriptor) else {
            throw DeformConvError.metalUnavailable
        }
        return heap
    }
    func copy(_ values: [Float16], to buffer: MTLBuffer) {
        let destination = buffer.contents().bindMemory(to: Float16.self, capacity: values.count)
        values.withUnsafeBufferPointer { source in
            destination.update(from: source.baseAddress!, count: source.count)
        }
    }

    let upsamplePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("upsample.mtlpackage")
    )
    let upsampleHeaps = try (0..<3).map { _ in
        try makeHeap(size: upsamplePipeline.intermediatesHeapSize)
    }
    let library = try device.makeLibrary(source: MetalShader.source, options: nil)
    guard let assembleFunction = library.makeFunction(name: "assemble_reconstruction_fp16"),
          let residualFunction = library.makeFunction(name: "add_frame_residual_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let assemblePipeline = try device.makeComputePipelineState(function: assembleFunction)
    let residualPipeline = try device.makeComputePipelineState(function: residualFunction)

    var planeValue = UInt32(featurePlane)
    var frameCountValue = UInt32(frameCount)
    let planeBuffer = try makeConstant(&planeValue)
    let frameCountBuffer = try makeConstant(&frameCountValue)
    var heldTensors = [any MTLTensor]()
    var spatialBuffers = [MTLBuffer]()
    var branchBuffers = [[MTLBuffer]](repeating: [], count: 4)
    var frameBuffers = [MTLBuffer]()
    var reconstructionBuffers = [MTLBuffer]()
    var predictedBuffers = [MTLBuffer]()
    var restoredBuffers = [MTLBuffer]()
    var assembleArguments = [any MTL4ArgumentTable]()
    var upsampleArguments = [any MTL4ArgumentTable]()
    var residualArguments = [any MTL4ArgumentTable]()
    for frameIndex in 0..<3 {
        let spatial = try makeBuffer(elements: featureCount)
        copy(spatialFrames[frameIndex], to: spatial)
        spatialBuffers.append(spatial)
        var frameBranches = [MTLBuffer]()
        for branchIndex in 0..<4 {
            let buffer = try makeBuffer(elements: featureCount)
            copy(branchFrames[branchIndex][frameIndex], to: buffer)
            branchBuffers[branchIndex].append(buffer)
            frameBranches.append(buffer)
        }
        let frame = try makeBuffer(elements: frameCount)
        copy(inputFrames[frameIndex], to: frame)
        frameBuffers.append(frame)
        let (reconstructionTensor, reconstruction) = try makeTensor(
            dimensions: [64, 64, 320, 1]
        )
        let (predictedTensor, predicted) = try makeTensor(dimensions: [256, 256, 3, 1])
        heldTensors += [reconstructionTensor, predictedTensor]
        reconstructionBuffers.append(reconstruction)
        predictedBuffers.append(predicted)
        let restored = try makeBuffer(elements: frameCount)
        restoredBuffers.append(restored)
        assembleArguments.append(try makeComputeArguments(
            [spatial] + frameBranches + [reconstruction, planeBuffer]
        ))
        upsampleArguments.append(try makeMLArguments(
            pipeline: upsamplePipeline,
            resources: [
                "features": reconstructionTensor.gpuResourceID,
                "output": predictedTensor.gpuResourceID,
            ]
        ))
        residualArguments.append(try makeComputeArguments([
            predicted, frame, restored, frameCountBuffer,
        ]))
    }

    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "three-frame fused reconstruction"
    residencyDescriptor.initialCapacity = 40
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    let residentBuffers = spatialBuffers + branchBuffers.flatMap { $0 } + frameBuffers
        + reconstructionBuffers + predictedBuffers + restoredBuffers
        + [planeBuffer, frameCountBuffer]
    for buffer in residentBuffers { residencySet.addAllocation(buffer) }
    for heap in upsampleHeaps { residencySet.addAllocation(heap) }
    residencySet.commit()
    guard let queue = device.makeMTL4CommandQueue() else { throw DeformConvError.metalUnavailable }

    func execute() throws -> (Double, [[Float16]]) {
        for buffer in reconstructionBuffers + predictedBuffers + restoredBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)

        guard let assembly = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        for arguments in assembleArguments {
            assembly.setComputePipelineState(assemblePipeline)
            assembly.setArgumentTable(arguments)
            assembly.dispatchThreads(
                threadsPerGrid: MTLSize(width: 320 * featurePlane, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: assemblePipeline.threadExecutionWidth, height: 1, depth: 1)
            )
        }
        assembly.barrier(
            afterStages: .dispatch, beforeQueueStages: .machineLearning, visibilityOptions: .device
        )
        assembly.endEncoding()

        for frameIndex in 0..<3 {
            guard let upsample = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            upsample.setPipelineState(upsamplePipeline)
            upsample.setArgumentTable(upsampleArguments[frameIndex])
            upsample.dispatchNetwork(intermediatesHeap: upsampleHeaps[frameIndex])
            upsample.endEncoding()
        }
        guard let residual = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        residual.barrier(
            afterQueueStages: .machineLearning, beforeStages: .dispatch, visibilityOptions: .device
        )
        for arguments in residualArguments {
            residual.setComputePipelineState(residualPipeline)
            residual.setArgumentTable(arguments)
            residual.dispatchThreads(
                threadsPerGrid: MTLSize(width: frameCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: residualPipeline.threadExecutionWidth, height: 1, depth: 1)
            )
        }
        residual.endEncoding()
        commandBuffer.endCommandBuffer()

        let semaphore = DispatchSemaphore(value: 0)
        let result = CommitResult()
        let options = MTL4CommitOptions()
        options.addFeedbackHandler { feedback in
            result.store(
                milliseconds: (feedback.gpuEndTime - feedback.gpuStartTime) * 1_000,
                error: feedback.error
            )
            semaphore.signal()
        }
        queue.commit([commandBuffer], options: options)
        semaphore.wait()
        let (milliseconds, error) = result.load()
        if let error { throw error }
        let outputs = restoredBuffers.map { buffer -> [Float16] in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: frameCount)
            return Array(UnsafeBufferPointer(start: pointer, count: frameCount))
        }
        return (milliseconds, outputs)
    }

    _ = try execute()
    _ = try execute()
    _ = try execute()
    let (firstMilliseconds, first) = try execute()
    var samples = [firstMilliseconds]
    var last = first
    for _ in 1..<20 {
        let (milliseconds, outputs) = try execute()
        samples.append(milliseconds)
        last = outputs
    }
    guard let statistics = BenchmarkStatistics(samples) else {
        throw DeformConvError.commandFailed("invalid reconstruction benchmark samples")
    }
    var maximumMagnitude: Float = 0
    var residualError: Float = 0
    var repeatError: Float = 0
    var checksums = [Double](repeating: 0, count: 3)
    for frameIndex in 0..<3 {
        let predicted = predictedBuffers[frameIndex].contents().bindMemory(
            to: Float16.self, capacity: frameCount
        )
        for index in 0..<frameCount {
            let value = Float(last[frameIndex][index])
            guard value.isFinite else {
                throw DeformConvError.commandFailed("clip reconstruction produced a non-finite value")
            }
            maximumMagnitude = max(maximumMagnitude, abs(value))
            residualError = max(
                residualError,
                abs(Float(predicted[index]) + Float(inputFrames[frameIndex][index]) - value)
            )
            repeatError = max(
                repeatError, abs(Float(first[frameIndex][index]) - value)
            )
            if index.isMultiple(of: 257) { checksums[frameIndex] += Double(value) }
        }
    }
    guard maximumMagnitude > 0.001, residualError <= 0.001, repeatError <= 0.001 else {
        throw DeformConvError.commandFailed(
            "clip reconstruction validation failed (max=\(maximumMagnitude), "
                + "residual=\(residualError), repeat=\(repeatError))"
        )
    }
    _ = heldTensors
    return ClipReconstructionResult(
        statistics: statistics,
        medianMilliseconds: statistics.median,
        minimumMilliseconds: statistics.minimum,
        maximumMilliseconds: statistics.maximum,
        iterations: statistics.samples.count,
        elementCount: 3 * frameCount,
        maximumMagnitude: maximumMagnitude,
        residualMaximumError: residualError,
        repeatMaximumError: repeatError,
        checksums: checksums
    )
}
