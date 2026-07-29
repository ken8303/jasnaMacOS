import Foundation
import Metal

struct ThreeFrameRecurrenceResult {
    let statistics: BenchmarkStatistics
    let medianMilliseconds: Double
    let minimumMilliseconds: Double
    let maximumMilliseconds: Double
    let iterations: Int
    let elementCount: Int
    let maximumMagnitude: Float
    let secondOrderFlowMaximum: Float
    let repeatMaximumError: Float
    let checksum: Double
    let propagatedFrames: [[Float16]]
    let spatialFrames: [[Float16]]
    let inputFrames: [[Float16]]
}

@available(macOS 27.0, *)
private struct ThreeFramePrepareShape {
    var width: UInt32 = 64
    var height: UInt32 = 64
    var hasSecondOrder: UInt32
}

@available(macOS 27.0, *)
func verifyThreeFrameRecurrence(
    device: MTLDevice,
    modelsURL: URL,
    weightsURL: URL,
    branchName: String = "backward_1",
    direction: PropagationDirection = .backward,
    flows: [[Float16]],
    priorBranchFrames: [[[Float16]]] = [],
    inputSpatialFrames: [[Float16]] = []
) throws -> ThreeFrameRecurrenceResult {
    let plane = 64 * 64
    let featureCount = 64 * plane
    let frameElements = 3 * 256 * 256
    let branchNames = ["backward_1", "forward_1", "backward_2", "forward_2"]
    guard let branchIndex = branchNames.firstIndex(of: branchName) else {
        throw DeformConvError.invalidShape
    }
    let backboneInputChannels = (branchIndex + 2) * 64
    guard flows.count == 2,
          flows.allSatisfy({ $0.count == 2 * plane }),
          priorBranchFrames.count == branchIndex,
          priorBranchFrames.allSatisfy({ branch in
              branch.count == 3 && branch.allSatisfy({ $0.count == featureCount })
          }),
          inputSpatialFrames.isEmpty || (
              inputSpatialFrames.count == 3
                  && inputSpatialFrames.allSatisfy({ $0.count == featureCount })
          )
    else { throw DeformConvError.invalidShape }

    func makeBuffer(elements: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: elements * 2, options: .storageModeShared) else {
            throw DeformConvError.metalUnavailable
        }
        return buffer
    }
    func makePrivateBuffer(bytes: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
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

    let featurePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("feature_extract.mtlpackage")
    )
    let offsetPipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("offset_\(branchName).mtlpackage")
    )
    let backbonePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("backbone_\(branchName).mtlpackage")
    )
    let featureHeaps = try (0..<3).map { _ in
        try makeHeap(size: featurePipeline.intermediatesHeapSize)
    }
    let offsetHeap = try makeHeap(size: offsetPipeline.intermediatesHeapSize)
    let backboneHeap = try makeHeap(size: backbonePipeline.intermediatesHeapSize)

    var heldTensors = [any MTLTensor]()
    var frameBuffers = [MTLBuffer]()
    var spatialBuffers = [MTLBuffer]()
    var featureArguments = [any MTL4ArgumentTable]()
    for _ in 0..<3 {
        let (frameTensor, frameBuffer) = try makeTensor(dimensions: [256, 256, 3, 1])
        let (spatialTensor, spatialBuffer) = try makeTensor(dimensions: [64, 64, 64, 1])
        heldTensors += [frameTensor, spatialTensor]
        frameBuffers.append(frameBuffer)
        spatialBuffers.append(spatialBuffer)
        featureArguments.append(try makeMLArguments(
            pipeline: featurePipeline,
            resources: ["frames": frameTensor.gpuResourceID, "output": spatialTensor.gpuResourceID]
        ))
    }

    let (conditionTensor, conditionBuffer) = try makeTensor(dimensions: [64, 64, 196, 1])
    let (rawTensor, rawBuffer) = try makeTensor(dimensions: [64, 64, 432, 1])
    let (backboneInputTensor, backboneInputBuffer) = try makeTensor(
        dimensions: [64, 64, backboneInputChannels, 1]
    )
    let (backboneOutputTensor, backboneOutputBuffer) = try makeTensor(dimensions: [64, 64, 64, 1])
    heldTensors += [conditionTensor, rawTensor, backboneInputTensor, backboneOutputTensor]
    let offsetArguments = try makeMLArguments(
        pipeline: offsetPipeline,
        resources: ["conditions": conditionTensor.gpuResourceID, "output": rawTensor.gpuResourceID]
    )
    let backboneArguments = try makeMLArguments(
        pipeline: backbonePipeline,
        resources: ["features": backboneInputTensor.gpuResourceID, "output": backboneOutputTensor.gpuResourceID]
    )

    let zeroFeatureBuffer = try makeBuffer(elements: featureCount)
    let priorBranchBuffers = try priorBranchFrames.map { branch in
        try (0..<3).map { frameIndex -> MTLBuffer in
            let buffer = try makeBuffer(elements: featureCount)
            let values = branch[frameIndex]
            let destination = buffer.contents().bindMemory(to: Float16.self, capacity: featureCount)
            values.withUnsafeBufferPointer { source in
                destination.update(from: source.baseAddress!, count: source.count)
            }
            return buffer
        }
    }
    let flowBuffers = try (0..<2).map { _ in try makeBuffer(elements: 2 * plane) }
    for (index, buffer) in flowBuffers.enumerated() {
        let flow = flows[index]
        let destination = buffer.contents().bindMemory(to: Float16.self, capacity: flow.count)
        flow.withUnsafeBufferPointer { source in
            destination.update(from: source.baseAddress!, count: source.count)
        }
    }
    let flow2Buffer = try makeBuffer(elements: 2 * plane)
    let deformInputBuffer = try makeBuffer(elements: 128 * plane)
    let offsetBuffer = try makeBuffer(elements: 288 * plane)
    let maskBuffer = try makeBuffer(elements: 144 * plane)
    let alignedBuffer = try makeBuffer(elements: featureCount)
    let gatheredBuffer = try makePrivateBuffer(bytes: plane * 128 * 9 * 2)
    let matrixOutputBuffer = try makePrivateBuffer(bytes: plane * 64 * 4)
    let propagationBuffers = try (0..<3).map { _ in try makeBuffer(elements: featureCount) }
    let checkpoint = try DeformConvWeightSet(
        direction: branchName,
        url: weightsURL.appendingPathComponent("\(branchName).dcnfp16")
    ).makeBuffers(device: device)

    var firstShape = ThreeFramePrepareShape(hasSecondOrder: 0)
    var secondShape = ThreeFramePrepareShape(hasSecondOrder: 1)
    var planeValue = UInt32(plane)
    var branchIndexValue = UInt32(branchIndex)
    var prefixChannelsValue = UInt32(64)
    var featureCountValue = UInt32(featureCount)
    var deformShape = PropagationDeformConvShape()
    let firstShapeBuffer = try makeConstant(&firstShape)
    let secondShapeBuffer = try makeConstant(&secondShape)
    let planeBuffer = try makeConstant(&planeValue)
    let branchIndexBuffer = try makeConstant(&branchIndexValue)
    let prefixChannelsBuffer = try makeConstant(&prefixChannelsValue)
    let featureCountBuffer = try makeConstant(&featureCountValue)
    let deformShapeBuffer = try makeConstant(&deformShape)

    let library = try device.makeLibrary(source: MetalShader.source, options: nil)
    guard let accumulateFunction = library.makeFunction(name: "accumulate_second_order_flow_fp16"),
          let prepareFunction = library.makeFunction(name: "assemble_temporal_alignment_fp16"),
          let transformFunction = library.makeFunction(name: "prepare_dcn_offsets_fp16"),
          let gatherFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_gather"),
          let gemmFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_simdgroup_gemm"),
          let gemmOutputFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_float_gemm_output"),
          let simpleBackboneAssemblyFunction = library.makeFunction(name: "assemble_propagation_backbone_fp16"),
          let temporalBackboneAssemblyFunction = library.makeFunction(name: "assemble_temporal_backbone_fp16"),
          let residualFunction = library.makeFunction(name: "add_propagation_residual_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let accumulatePipeline = try device.makeComputePipelineState(function: accumulateFunction)
    let preparePipeline = try device.makeComputePipelineState(function: prepareFunction)
    let transformPipeline = try device.makeComputePipelineState(function: transformFunction)
    let gatherPipeline = try device.makeComputePipelineState(function: gatherFunction)
    let gemmPipeline = try device.makeComputePipelineState(function: gemmFunction)
    let gemmOutputPipeline = try device.makeComputePipelineState(function: gemmOutputFunction)
    let simpleBackboneAssemblyPipeline = try device.makeComputePipelineState(
        function: simpleBackboneAssemblyFunction
    )
    let temporalBackboneAssemblyPipeline = try device.makeComputePipelineState(
        function: temporalBackboneAssemblyFunction
    )
    let backboneAssemblyPipeline = branchIndex == 0
        ? simpleBackboneAssemblyPipeline : temporalBackboneAssemblyPipeline
    let residualPipeline = try device.makeComputePipelineState(function: residualFunction)

    var transformArguments = [any MTL4ArgumentTable]()
    let gatherArguments = try makeComputeArguments([
        deformInputBuffer, offsetBuffer, maskBuffer, gatheredBuffer, deformShapeBuffer,
    ])
    let gemmArguments = try makeComputeArguments([
        gatheredBuffer, checkpoint.weight, matrixOutputBuffer,
    ])
    let gemmOutputArguments = try makeComputeArguments([
        matrixOutputBuffer, checkpoint.bias, alignedBuffer, deformShapeBuffer,
    ])
    let traversal = direction == .backward ? [2, 1, 0] : [0, 1, 2]
    func priorBuffer(_ priorIndex: Int, frame: Int) -> MTLBuffer {
        priorIndex < priorBranchBuffers.count
            ? priorBranchBuffers[priorIndex][frame] : zeroFeatureBuffer
    }
    let step0Frame = traversal[0]
    let step0AssemblyArguments = try makeComputeArguments(
        branchIndex == 0
            ? [spatialBuffers[step0Frame], zeroFeatureBuffer, backboneInputBuffer,
               planeBuffer, prefixChannelsBuffer]
            : [spatialBuffers[step0Frame], priorBuffer(0, frame: step0Frame),
               priorBuffer(1, frame: step0Frame), priorBuffer(2, frame: step0Frame),
               zeroFeatureBuffer,
               backboneInputBuffer, planeBuffer, branchIndexBuffer]
    )
    let step0ResidualArguments = try makeComputeArguments([
        zeroFeatureBuffer, backboneOutputBuffer, propagationBuffers[step0Frame], featureCountBuffer,
    ])
    var accumulateArguments = [any MTL4ArgumentTable]()
    var prepareArguments = [any MTL4ArgumentTable]()
    var backboneAssemblyArguments = [any MTL4ArgumentTable]()
    var residualArguments = [any MTL4ArgumentTable]()
    for step in 1...2 {
        let frameIndex = traversal[step]
        let priorFrameIndex = traversal[step - 1]
        let featProp = propagationBuffers[priorFrameIndex]
        let featN2 = step == 2 ? propagationBuffers[traversal[0]] : zeroFeatureBuffer
        let flowIndex = direction == .backward ? frameIndex : frameIndex - 1
        let previousFlowIndex = step == 2
            ? (direction == .backward ? traversal[step - 1] : traversal[step - 1] - 1)
            : flowIndex
        let flow1 = flowBuffers[flowIndex]
        let previousFlow = flowBuffers[previousFlowIndex]
        let shapeBuffer = step == 2 ? secondShapeBuffer : firstShapeBuffer
        accumulateArguments.append(try makeComputeArguments([
            flow1, previousFlow, flow2Buffer, shapeBuffer,
        ]))
        prepareArguments.append(try makeComputeArguments([
            featProp, spatialBuffers[frameIndex], featN2, flow1, flow2Buffer,
            conditionBuffer, deformInputBuffer, shapeBuffer,
        ]))
        transformArguments.append(try makeComputeArguments([
            rawBuffer, flow1, flow2Buffer, offsetBuffer, maskBuffer, planeBuffer,
        ]))
        backboneAssemblyArguments.append(try makeComputeArguments(
            branchIndex == 0
                ? [spatialBuffers[frameIndex], alignedBuffer, backboneInputBuffer,
                   planeBuffer, prefixChannelsBuffer]
                : [spatialBuffers[frameIndex], priorBuffer(0, frame: frameIndex),
                   priorBuffer(1, frame: frameIndex), priorBuffer(2, frame: frameIndex),
                   alignedBuffer,
                   backboneInputBuffer, planeBuffer, branchIndexBuffer]
        ))
        residualArguments.append(try makeComputeArguments([
            alignedBuffer, backboneOutputBuffer, propagationBuffers[frameIndex], featureCountBuffer,
        ]))
    }

    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "three-frame \(branchName) recurrence"
    residencyDescriptor.initialCapacity = 48
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    let residentBuffers = frameBuffers + spatialBuffers + [
        conditionBuffer, rawBuffer, backboneInputBuffer, backboneOutputBuffer,
        zeroFeatureBuffer, flow2Buffer, deformInputBuffer, offsetBuffer, maskBuffer,
        alignedBuffer, checkpoint.weight, checkpoint.bias, firstShapeBuffer,
        secondShapeBuffer, planeBuffer, branchIndexBuffer, prefixChannelsBuffer, featureCountBuffer,
        deformShapeBuffer,
    ] + flowBuffers + propagationBuffers + priorBranchBuffers.flatMap { $0 }
    residencySet.addAllocation(gatheredBuffer)
    residencySet.addAllocation(matrixOutputBuffer)
    for buffer in residentBuffers { residencySet.addAllocation(buffer) }
    for heap in featureHeaps { residencySet.addAllocation(heap) }
    residencySet.addAllocation(offsetHeap)
    residencySet.addAllocation(backboneHeap)
    residencySet.commit()
    guard let queue = device.makeMTL4CommandQueue() else { throw DeformConvError.metalUnavailable }

    let transientBuffers = spatialBuffers + propagationBuffers + [
        conditionBuffer, rawBuffer, backboneInputBuffer, backboneOutputBuffer,
        zeroFeatureBuffer, flow2Buffer, deformInputBuffer, offsetBuffer, maskBuffer, alignedBuffer,
    ]
    func initializeBuffers() {
        for buffer in transientBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        for (frameIndex, buffer) in frameBuffers.enumerated() {
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: frameElements)
            for index in 0..<frameElements {
                pointer[index] = Float16(Float((index * (29 + frameIndex * 6) + 17 + frameIndex * 23) % 1021) / 1020)
            }
        }
        if !inputSpatialFrames.isEmpty {
            for frameIndex in 0..<3 {
                let values = inputSpatialFrames[frameIndex]
                let destination = spatialBuffers[frameIndex].contents().bindMemory(
                    to: Float16.self, capacity: featureCount
                )
                values.withUnsafeBufferPointer { source in
                    destination.update(from: source.baseAddress!, count: source.count)
                }
            }
        }
    }

    func dispatch1D(
        _ encoder: any MTL4ComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        arguments: any MTL4ArgumentTable,
        count: Int,
        threads: Int? = nil,
        threadgroups: Bool = false
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setArgumentTable(arguments)
        if threadgroups {
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads ?? 128, height: 1, depth: 1)
            )
        } else {
            encoder.dispatchThreads(
                threadsPerGrid: MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads ?? pipeline.threadExecutionWidth, height: 1, depth: 1)
            )
        }
    }

    func execute() throws -> (Double, [[Float16]]) {
        initializeBuffers()
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)

        if inputSpatialFrames.isEmpty {
            for (index, arguments) in featureArguments.enumerated() {
                guard let encoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                encoder.setPipelineState(featurePipeline)
                encoder.setArgumentTable(arguments)
                encoder.dispatchNetwork(intermediatesHeap: featureHeaps[index])
                encoder.endEncoding()
            }
        }

        guard let initialAssembly = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        if inputSpatialFrames.isEmpty {
            initialAssembly.barrier(
                afterQueueStages: .machineLearning,
                beforeStages: .dispatch,
                visibilityOptions: .device
            )
        }
        dispatch1D(
            initialAssembly, pipeline: backboneAssemblyPipeline,
            arguments: step0AssemblyArguments, count: backboneInputChannels * plane
        )
        initialAssembly.barrier(
            afterStages: .dispatch, beforeQueueStages: .machineLearning, visibilityOptions: .device
        )
        initialAssembly.endEncoding()

        guard let initialBackbone = commandBuffer.makeMachineLearningCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        initialBackbone.setPipelineState(backbonePipeline)
        initialBackbone.setArgumentTable(backboneArguments)
        initialBackbone.dispatchNetwork(intermediatesHeap: backboneHeap)
        initialBackbone.barrier(
            afterStages: .machineLearning, beforeQueueStages: .dispatch, visibilityOptions: .device
        )
        initialBackbone.endEncoding()

        guard let initialResidual = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        dispatch1D(
            initialResidual, pipeline: residualPipeline,
            arguments: step0ResidualArguments, count: featureCount
        )
        initialResidual.barrier(
            afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device
        )
        initialResidual.endEncoding()

        for stepIndex in 0..<2 {
            guard let preparation = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            preparation.barrier(
                afterQueueStages: .dispatch, beforeStages: .dispatch, visibilityOptions: .device
            )
            dispatch1D(
                preparation, pipeline: accumulatePipeline,
                arguments: accumulateArguments[stepIndex], count: 2 * plane
            )
            preparation.barrier(
                afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device
            )
            dispatch1D(
                preparation, pipeline: preparePipeline,
                arguments: prepareArguments[stepIndex], count: 196 * plane
            )
            preparation.barrier(
                afterStages: .dispatch, beforeQueueStages: .machineLearning, visibilityOptions: .device
            )
            preparation.endEncoding()

            guard let offsetEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            offsetEncoder.setPipelineState(offsetPipeline)
            offsetEncoder.setArgumentTable(offsetArguments)
            offsetEncoder.dispatchNetwork(intermediatesHeap: offsetHeap)
            offsetEncoder.barrier(
                afterStages: .machineLearning, beforeQueueStages: .dispatch, visibilityOptions: .device
            )
            offsetEncoder.endEncoding()

            guard let alignment = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            dispatch1D(
                alignment, pipeline: transformPipeline,
                arguments: transformArguments[stepIndex], count: 432 * plane
            )
            alignment.barrier(
                afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device
            )
            dispatch1D(
                alignment, pipeline: gatherPipeline,
                arguments: gatherArguments, count: plane,
                threads: 128, threadgroups: true
            )
            alignment.barrier(
                afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device
            )
            dispatch1D(
                alignment, pipeline: gemmPipeline,
                arguments: gemmArguments, count: plane / 8,
                threads: 8 * gemmPipeline.threadExecutionWidth, threadgroups: true
            )
            alignment.barrier(
                afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device
            )
            dispatch1D(
                alignment, pipeline: gemmOutputPipeline,
                arguments: gemmOutputArguments, count: featureCount
            )
            alignment.barrier(
                afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device
            )
            dispatch1D(
                alignment, pipeline: backboneAssemblyPipeline,
                arguments: backboneAssemblyArguments[stepIndex], count: backboneInputChannels * plane
            )
            alignment.barrier(
                afterStages: .dispatch, beforeQueueStages: .machineLearning, visibilityOptions: .device
            )
            alignment.endEncoding()

            guard let backboneEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            backboneEncoder.setPipelineState(backbonePipeline)
            backboneEncoder.setArgumentTable(backboneArguments)
            backboneEncoder.dispatchNetwork(intermediatesHeap: backboneHeap)
            backboneEncoder.barrier(
                afterStages: .machineLearning, beforeQueueStages: .dispatch, visibilityOptions: .device
            )
            backboneEncoder.endEncoding()

            guard let residual = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            dispatch1D(
                residual, pipeline: residualPipeline,
                arguments: residualArguments[stepIndex], count: featureCount
            )
            if stepIndex == 0 {
                residual.barrier(
                    afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device
                )
            }
            residual.endEncoding()
        }
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
        let outputs = propagationBuffers.map { buffer -> [Float16] in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: featureCount)
            return Array(UnsafeBufferPointer(start: pointer, count: featureCount))
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
        let (milliseconds, output) = try execute()
        samples.append(milliseconds)
        last = output
    }
    guard let statistics = BenchmarkStatistics(samples) else {
        throw DeformConvError.commandFailed("invalid three-frame benchmark samples")
    }
    var maximumMagnitude: Float = 0
    var repeatMaximumError: Float = 0
    var checksum = 0.0
    for frameIndex in 0..<3 {
        for index in last[frameIndex].indices {
            let value = Float(last[frameIndex][index])
            guard value.isFinite else {
                throw DeformConvError.commandFailed("three-frame recurrence produced a non-finite feature")
            }
            maximumMagnitude = max(maximumMagnitude, abs(value))
            repeatMaximumError = max(
                repeatMaximumError,
                abs(Float(first[frameIndex][index]) - value)
            )
            if frameIndex == traversal.last, index.isMultiple(of: 257) {
                checksum += Double(value)
            }
        }
    }
    let secondOrderPointer = flow2Buffer.contents().bindMemory(to: Float16.self, capacity: 2 * plane)
    var secondOrderMaximum: Float = 0
    for index in 0..<(2 * plane) {
        secondOrderMaximum = max(secondOrderMaximum, abs(Float(secondOrderPointer[index])))
    }
    guard maximumMagnitude > 0.001,
          secondOrderMaximum > 0.001,
          repeatMaximumError <= 0.001
    else {
        throw DeformConvError.commandFailed(
            "three-frame recurrence validation failed (max=\(maximumMagnitude), "
                + "flow2=\(secondOrderMaximum), repeat=\(repeatMaximumError))"
        )
    }
    _ = heldTensors
    return ThreeFrameRecurrenceResult(
        statistics: statistics,
        medianMilliseconds: statistics.median,
        minimumMilliseconds: statistics.minimum,
        maximumMilliseconds: statistics.maximum,
        iterations: statistics.samples.count,
        elementCount: featureCount,
        maximumMagnitude: maximumMagnitude,
        secondOrderFlowMaximum: secondOrderMaximum,
        repeatMaximumError: repeatMaximumError,
        checksum: checksum,
        propagatedFrames: last,
        spatialFrames: spatialBuffers.map { buffer in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: featureCount)
            return Array(UnsafeBufferPointer(start: pointer, count: featureCount))
        },
        inputFrames: frameBuffers.map { buffer in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: frameElements)
            return Array(UnsafeBufferPointer(start: pointer, count: frameElements))
        }
    )
}
