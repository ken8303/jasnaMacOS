import Foundation
import Metal

struct ZeroCopyFrameGraphResult {
    let gpuMilliseconds: Double
    let minimumMilliseconds: Double
    let maximumMilliseconds: Double
    let iterations: Int
    let elementCount: Int
    let allocatedBytes: Int
    let maximumMagnitude: Float
    let residualMaximumError: Float
    let repeatMaximumError: Float
    let checksum: Double
}

@available(macOS 27.0, *)
private struct ZeroCopyBranchRuntime {
    let direction: String
    let backboneInputChannels: Int
    let offsetPipeline: any MTL4MachineLearningPipelineState
    let backbonePipeline: any MTL4MachineLearningPipelineState
    let offsetArguments: any MTL4ArgumentTable
    let backboneArguments: any MTL4ArgumentTable
    let transformArguments: any MTL4ArgumentTable
    let deformArguments: any MTL4ArgumentTable
    var assemblyArguments: (any MTL4ArgumentTable)?
    var fusedAssemblyArguments: (any MTL4ArgumentTable)?
    let residualArguments: any MTL4ArgumentTable
    let offsetHeap: MTLHeap
    let backboneHeap: MTLHeap
    let tensors: [any MTLTensor]
    let conditionBuffer: MTLBuffer
    let rawBuffer: MTLBuffer
    let flow1Buffer: MTLBuffer
    let flow2Buffer: MTLBuffer
    let deformInputBuffer: MTLBuffer
    let offsetBuffer: MTLBuffer
    let maskBuffer: MTLBuffer
    let alignedBuffer: MTLBuffer
    let backboneInputBuffer: MTLBuffer
    let backboneOutputBuffer: MTLBuffer
    let outputBuffer: MTLBuffer
    let branchIndexBuffer: MTLBuffer
    let weightBuffer: MTLBuffer
    let biasBuffer: MTLBuffer
}

@available(macOS 27.0, *)
func verifyZeroCopyFrameGraph(
    device: MTLDevice,
    modelsURL: URL,
    weightsURL: URL,
    groupOffsetPredictions: Bool = false,
    groupAlignments: Bool = false,
    fusePropagationResiduals: Bool = false,
    firstOrderFlows: [String: [Float16]] = [:]
) throws -> ZeroCopyFrameGraphResult {
    guard !groupAlignments || groupOffsetPredictions else {
        throw DeformConvError.commandFailed("grouped alignments require grouped offset predictions")
    }
    guard !fusePropagationResiduals || groupAlignments else {
        throw DeformConvError.commandFailed("fused residuals require grouped alignments")
    }
    for (name, values) in firstOrderFlows where values.count != 2 * 64 * 64 {
        throw DeformConvError.commandFailed("invalid \(name) first-order flow size")
    }
    let featurePlane = 64 * 64
    let featureCount = 64 * featurePlane
    let frameCount = 3 * 256 * 256
    let branchSpecifications = [
        ("backward_1", 128), ("forward_1", 192),
        ("backward_2", 256), ("forward_2", 320),
    ]

    func makeSharedBuffer(elements: Int) throws -> MTLBuffer {
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
        let buffer = try makeSharedBuffer(elements: elements)
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
    func makeConstantBuffer<T>(_ value: inout T) throws -> MTLBuffer {
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
    func fill(_ buffer: MTLBuffer, elements: Int, seed: Int, scale: Float, offset: Float = 0) {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: elements)
        for index in 0..<elements {
            let centered = Float((index * 37 + seed) % 257) / 256 - 0.5
            pointer[index] = Float16(centered * scale + offset)
        }
    }

    let featurePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("feature_extract.mtlpackage")
    )
    let upsamplePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("upsample.mtlpackage")
    )
    let (frameTensor, frameBuffer) = try makeTensor(dimensions: [256, 256, 3, 1])
    let (spatialTensor, spatialBuffer) = try makeTensor(dimensions: [64, 64, 64, 1])
    let (reconstructionTensor, reconstructionBuffer) = try makeTensor(dimensions: [64, 64, 320, 1])
    let (predictedTensor, predictedBuffer) = try makeTensor(dimensions: [256, 256, 3, 1])
    let restoredBuffer = try makeSharedBuffer(elements: frameCount)
    let zeroFeatureBuffer = try makeSharedBuffer(elements: featureCount)

    let featureArguments = try makeMLArguments(
        pipeline: featurePipeline,
        resources: ["frames": frameTensor.gpuResourceID, "output": spatialTensor.gpuResourceID]
    )
    let upsampleArguments = try makeMLArguments(
        pipeline: upsamplePipeline,
        resources: ["features": reconstructionTensor.gpuResourceID, "output": predictedTensor.gpuResourceID]
    )
    let featureHeap = try makeHeap(size: featurePipeline.intermediatesHeapSize)
    let upsampleHeap = try makeHeap(size: upsamplePipeline.intermediatesHeapSize)

    var planeValue = UInt32(featurePlane)
    var frameCountValue = UInt32(frameCount)
    var featureCountValue = UInt32(featureCount)
    var deformShape = PropagationDeformConvShape()
    let planeBuffer = try makeConstantBuffer(&planeValue)
    let frameCountBuffer = try makeConstantBuffer(&frameCountValue)
    let featureCountBuffer = try makeConstantBuffer(&featureCountValue)
    let deformShapeBuffer = try makeConstantBuffer(&deformShape)

    let shaderOptions = MTLCompileOptions()
    shaderOptions.mathMode = .safe
    let library = try device.makeLibrary(source: MetalShader.source, options: shaderOptions)
    guard let transformFunction = library.makeFunction(name: "prepare_dcn_offsets_fp16"),
          let deformFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_tiled"),
          let temporalAssemblyFunction = library.makeFunction(name: "assemble_temporal_backbone_fp16"),
          let fusedTemporalAssemblyFunction = library.makeFunction(name: "assemble_temporal_backbone_fused_fp16"),
          let propagationResidualFunction = library.makeFunction(name: "add_propagation_residual_fp16"),
          let reconstructionAssemblyFunction = library.makeFunction(name: "assemble_reconstruction_fp16"),
          let frameResidualFunction = library.makeFunction(name: "add_frame_residual_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let transformCompute = try device.makeComputePipelineState(function: transformFunction)
    let deformCompute = try device.makeComputePipelineState(function: deformFunction)
    let temporalAssemblyCompute = try device.makeComputePipelineState(function: temporalAssemblyFunction)
    let fusedTemporalAssemblyCompute = try device.makeComputePipelineState(function: fusedTemporalAssemblyFunction)
    let propagationResidualCompute = try device.makeComputePipelineState(function: propagationResidualFunction)
    let reconstructionAssemblyCompute = try device.makeComputePipelineState(function: reconstructionAssemblyFunction)
    let frameResidualCompute = try device.makeComputePipelineState(function: frameResidualFunction)

    var branches = [ZeroCopyBranchRuntime]()
    for (branchIndex, specification) in branchSpecifications.enumerated() {
        let (direction, backboneInputChannels) = specification
        let offsetPipeline = try makeMetalMLPipeline(
            device: device,
            packageURL: modelsURL.appendingPathComponent("offset_\(direction).mtlpackage")
        )
        let backbonePipeline = try makeMetalMLPipeline(
            device: device,
            packageURL: modelsURL.appendingPathComponent("backbone_\(direction).mtlpackage")
        )
        let (conditionTensor, conditionBuffer) = try makeTensor(dimensions: [64, 64, 196, 1])
        let (rawTensor, rawBuffer) = try makeTensor(dimensions: [64, 64, 432, 1])
        let (backboneInputTensor, backboneInputBuffer) = try makeTensor(
            dimensions: [64, 64, backboneInputChannels, 1]
        )
        let (backboneOutputTensor, backboneOutputBuffer) = try makeTensor(dimensions: [64, 64, 64, 1])
        let flow1Buffer = try makeSharedBuffer(elements: 2 * featurePlane)
        let flow2Buffer = try makeSharedBuffer(elements: 2 * featurePlane)
        let deformInputBuffer = try makeSharedBuffer(elements: 128 * featurePlane)
        let offsetBuffer = try makeSharedBuffer(elements: 288 * featurePlane)
        let maskBuffer = try makeSharedBuffer(elements: 144 * featurePlane)
        let alignedBuffer = try makeSharedBuffer(elements: featureCount)
        let outputBuffer = try makeSharedBuffer(elements: featureCount)
        var branchIndexValue = UInt32(branchIndex)
        let branchIndexBuffer = try makeConstantBuffer(&branchIndexValue)
        let checkpoint = try DeformConvWeightSet(
            direction: direction,
            url: weightsURL.appendingPathComponent("\(direction).dcnfp16")
        ).makeBuffers(device: device)
        let offsetArguments = try makeMLArguments(
            pipeline: offsetPipeline,
            resources: ["conditions": conditionTensor.gpuResourceID, "output": rawTensor.gpuResourceID]
        )
        let backboneArguments = try makeMLArguments(
            pipeline: backbonePipeline,
            resources: [
                "features": backboneInputTensor.gpuResourceID,
                "output": backboneOutputTensor.gpuResourceID,
            ]
        )
        let transformArguments = try makeComputeArguments([
            rawBuffer, flow1Buffer, flow2Buffer, offsetBuffer, maskBuffer, planeBuffer,
        ])
        let deformArguments = try makeComputeArguments([
            deformInputBuffer, offsetBuffer, maskBuffer, checkpoint.weight, checkpoint.bias,
            alignedBuffer, deformShapeBuffer,
        ])
        let residualArguments = try makeComputeArguments([
            alignedBuffer, backboneOutputBuffer, outputBuffer, featureCountBuffer,
        ])
        branches.append(ZeroCopyBranchRuntime(
            direction: direction,
            backboneInputChannels: backboneInputChannels,
            offsetPipeline: offsetPipeline,
            backbonePipeline: backbonePipeline,
            offsetArguments: offsetArguments,
            backboneArguments: backboneArguments,
            transformArguments: transformArguments,
            deformArguments: deformArguments,
            assemblyArguments: nil,
            fusedAssemblyArguments: nil,
            residualArguments: residualArguments,
            offsetHeap: try makeHeap(size: offsetPipeline.intermediatesHeapSize),
            backboneHeap: try makeHeap(size: backbonePipeline.intermediatesHeapSize),
            tensors: [conditionTensor, rawTensor, backboneInputTensor, backboneOutputTensor],
            conditionBuffer: conditionBuffer,
            rawBuffer: rawBuffer,
            flow1Buffer: flow1Buffer,
            flow2Buffer: flow2Buffer,
            deformInputBuffer: deformInputBuffer,
            offsetBuffer: offsetBuffer,
            maskBuffer: maskBuffer,
            alignedBuffer: alignedBuffer,
            backboneInputBuffer: backboneInputBuffer,
            backboneOutputBuffer: backboneOutputBuffer,
            outputBuffer: outputBuffer,
            branchIndexBuffer: branchIndexBuffer,
            weightBuffer: checkpoint.weight,
            biasBuffer: checkpoint.bias
        ))
    }

    for index in branches.indices {
        let prior0 = branches.indices.contains(0) ? branches[0].outputBuffer : zeroFeatureBuffer
        let prior1 = index >= 2 ? branches[1].outputBuffer : zeroFeatureBuffer
        let prior2 = index >= 3 ? branches[2].outputBuffer : zeroFeatureBuffer
        branches[index].assemblyArguments = try makeComputeArguments([
            spatialBuffer, index >= 1 ? prior0 : zeroFeatureBuffer, prior1, prior2,
            branches[index].alignedBuffer, branches[index].backboneInputBuffer,
            planeBuffer, branches[index].branchIndexBuffer,
        ])
        if index > 0 {
            let previous = branches[index - 1]
            branches[index].fusedAssemblyArguments = try makeComputeArguments([
                spatialBuffer,
                index >= 2 ? branches[0].outputBuffer : zeroFeatureBuffer,
                index >= 3 ? branches[1].outputBuffer : zeroFeatureBuffer,
                zeroFeatureBuffer,
                previous.alignedBuffer, previous.backboneOutputBuffer, previous.outputBuffer,
                branches[index].alignedBuffer, branches[index].backboneInputBuffer,
                planeBuffer, branches[index].branchIndexBuffer,
            ])
        }
    }
    let reconstructionAssemblyArguments = try makeComputeArguments(
        [spatialBuffer] + branches.map(\.outputBuffer) + [reconstructionBuffer, planeBuffer]
    )
    let frameResidualArguments = try makeComputeArguments([
        predictedBuffer, frameBuffer, restoredBuffer, frameCountBuffer,
    ])

    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "zero-copy complete-frame graph"
    residencyDescriptor.initialCapacity = 80
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    var residentBuffers = [
        frameBuffer, spatialBuffer, reconstructionBuffer, predictedBuffer,
        restoredBuffer, zeroFeatureBuffer, planeBuffer, frameCountBuffer,
        featureCountBuffer, deformShapeBuffer,
    ]
    for branch in branches {
        residentBuffers += [
            branch.conditionBuffer, branch.rawBuffer, branch.flow1Buffer, branch.flow2Buffer,
            branch.deformInputBuffer, branch.offsetBuffer, branch.maskBuffer,
            branch.alignedBuffer, branch.backboneInputBuffer, branch.backboneOutputBuffer,
            branch.outputBuffer, branch.branchIndexBuffer, branch.weightBuffer, branch.biasBuffer,
        ]
        residencySet.addAllocation(branch.offsetHeap)
        residencySet.addAllocation(branch.backboneHeap)
    }
    for buffer in residentBuffers { residencySet.addAllocation(buffer) }
    residencySet.addAllocation(featureHeap)
    residencySet.addAllocation(upsampleHeap)
    residencySet.commit()
    guard let queue = device.makeMTL4CommandQueue() else {
        throw DeformConvError.metalUnavailable
    }

    let transientBuffers = [spatialBuffer, reconstructionBuffer, predictedBuffer, restoredBuffer]
        + branches.flatMap {
            [$0.rawBuffer, $0.offsetBuffer, $0.maskBuffer, $0.alignedBuffer,
             $0.backboneInputBuffer, $0.backboneOutputBuffer, $0.outputBuffer]
        }
    func initializeBuffers() {
        for buffer in transientBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        fill(frameBuffer, elements: frameCount, seed: 17, scale: 1.0, offset: 0.5)
        for (index, branch) in branches.enumerated() {
            fill(branch.conditionBuffer, elements: 196 * featurePlane, seed: 31 + index * 13, scale: 0.25)
            let flowName = branch.direction.hasPrefix("backward") ? "backward" : "forward"
            if let flow = firstOrderFlows[flowName] {
                let destination = branch.flow1Buffer.contents().bindMemory(
                    to: Float16.self, capacity: flow.count
                )
                flow.withUnsafeBufferPointer { source in
                    destination.update(from: source.baseAddress!, count: source.count)
                }
                // A two-frame pair has no second-order predecessor.
                branch.flow2Buffer.contents().initializeMemory(
                    as: UInt8.self, repeating: 0, count: branch.flow2Buffer.length
                )
            } else {
                fill(branch.flow1Buffer, elements: 2 * featurePlane, seed: 53 + index * 7, scale: 0.20)
                fill(branch.flow2Buffer, elements: 2 * featurePlane, seed: 71 + index * 11, scale: 0.16)
            }
            fill(branch.deformInputBuffer, elements: 128 * featurePlane, seed: 97 + index * 17, scale: 0.50)
        }
    }

    func execute() throws -> (Double, [Float16]) {
        initializeBuffers()
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)

        guard let featureEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        featureEncoder.setPipelineState(featurePipeline)
        featureEncoder.setArgumentTable(featureArguments)
        featureEncoder.dispatchNetwork(intermediatesHeap: featureHeap)
        featureEncoder.endEncoding()

        if groupOffsetPredictions {
            for branch in branches {
                guard let offsetEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                offsetEncoder.setPipelineState(branch.offsetPipeline)
                offsetEncoder.setArgumentTable(branch.offsetArguments)
                offsetEncoder.dispatchNetwork(intermediatesHeap: branch.offsetHeap)
                offsetEncoder.endEncoding()
            }
        }

        if groupAlignments {
            for branch in branches {
                guard let alignmentEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                alignmentEncoder.barrier(
                    afterQueueStages: .machineLearning,
                    beforeStages: .dispatch,
                    visibilityOptions: .device
                )
                alignmentEncoder.setComputePipelineState(transformCompute)
                alignmentEncoder.setArgumentTable(branch.transformArguments)
                alignmentEncoder.dispatchThreads(
                    threadsPerGrid: MTLSize(width: 432 * featurePlane, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: transformCompute.threadExecutionWidth, height: 1, depth: 1)
                )
                alignmentEncoder.barrier(
                    afterEncoderStages: .dispatch,
                    beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                alignmentEncoder.setComputePipelineState(deformCompute)
                alignmentEncoder.setArgumentTable(branch.deformArguments)
                alignmentEncoder.dispatchThreadgroups(
                    threadgroupsPerGrid: MTLSize(width: featurePlane, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
                alignmentEncoder.endEncoding()
            }
        }

        for (branchIndex, branch) in branches.enumerated() {
            if !groupOffsetPredictions {
                guard let offsetEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                offsetEncoder.setPipelineState(branch.offsetPipeline)
                offsetEncoder.setArgumentTable(branch.offsetArguments)
                offsetEncoder.dispatchNetwork(intermediatesHeap: branch.offsetHeap)
                offsetEncoder.endEncoding()
            }

            guard let propagationEncoder = commandBuffer.makeComputeCommandEncoder(),
                  let assemblyArguments = branch.assemblyArguments
            else { throw DeformConvError.metalUnavailable }
            if groupAlignments {
                propagationEncoder.barrier(
                    afterQueueStages: .dispatch,
                    beforeStages: .dispatch,
                    visibilityOptions: .device
                )
            } else {
                propagationEncoder.barrier(
                    afterQueueStages: [.machineLearning, .dispatch],
                    beforeStages: .dispatch,
                    visibilityOptions: .device
                )
                propagationEncoder.setComputePipelineState(transformCompute)
                propagationEncoder.setArgumentTable(branch.transformArguments)
                propagationEncoder.dispatchThreads(
                    threadsPerGrid: MTLSize(width: 432 * featurePlane, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: transformCompute.threadExecutionWidth, height: 1, depth: 1)
                )
                propagationEncoder.barrier(
                    afterEncoderStages: .dispatch,
                    beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                propagationEncoder.setComputePipelineState(deformCompute)
                propagationEncoder.setArgumentTable(branch.deformArguments)
                propagationEncoder.dispatchThreadgroups(
                    threadgroupsPerGrid: MTLSize(width: featurePlane, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
                propagationEncoder.barrier(
                    afterEncoderStages: .dispatch,
                    beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
            }
            propagationEncoder.setComputePipelineState(temporalAssemblyCompute)
            if fusePropagationResiduals, branchIndex > 0,
               let fusedArguments = branch.fusedAssemblyArguments {
                propagationEncoder.setComputePipelineState(fusedTemporalAssemblyCompute)
                propagationEncoder.setArgumentTable(fusedArguments)
            } else {
                propagationEncoder.setComputePipelineState(temporalAssemblyCompute)
                propagationEncoder.setArgumentTable(assemblyArguments)
            }
            propagationEncoder.dispatchThreads(
                threadsPerGrid: MTLSize(width: branch.backboneInputChannels * featurePlane, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: temporalAssemblyCompute.threadExecutionWidth, height: 1, depth: 1)
            )
            propagationEncoder.barrier(
                afterStages: .dispatch, beforeQueueStages: .machineLearning, visibilityOptions: .device
            )
            propagationEncoder.endEncoding()

            guard let backboneEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            backboneEncoder.setPipelineState(branch.backbonePipeline)
            backboneEncoder.setArgumentTable(branch.backboneArguments)
            backboneEncoder.dispatchNetwork(intermediatesHeap: branch.backboneHeap)
            backboneEncoder.barrier(
                afterStages: .machineLearning, beforeQueueStages: .dispatch, visibilityOptions: .device
            )
            backboneEncoder.endEncoding()

            if !fusePropagationResiduals || branchIndex == branches.count - 1 {
                guard let residualEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                residualEncoder.setComputePipelineState(propagationResidualCompute)
                residualEncoder.setArgumentTable(branch.residualArguments)
                residualEncoder.dispatchThreads(
                    threadsPerGrid: MTLSize(width: featureCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: propagationResidualCompute.threadExecutionWidth, height: 1, depth: 1)
                )
                residualEncoder.barrier(
                    afterStages: .dispatch,
                    beforeQueueStages: [.machineLearning, .dispatch],
                    visibilityOptions: .device
                )
                residualEncoder.endEncoding()
            }
        }

        guard let reconstructionEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        reconstructionEncoder.barrier(
            afterQueueStages: [.machineLearning, .dispatch],
            beforeStages: .dispatch,
            visibilityOptions: .device
        )
        reconstructionEncoder.setComputePipelineState(reconstructionAssemblyCompute)
        reconstructionEncoder.setArgumentTable(reconstructionAssemblyArguments)
        reconstructionEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: 320 * featurePlane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: reconstructionAssemblyCompute.threadExecutionWidth, height: 1, depth: 1)
        )
        reconstructionEncoder.barrier(
            afterStages: .dispatch, beforeQueueStages: .machineLearning, visibilityOptions: .device
        )
        reconstructionEncoder.endEncoding()

        guard let upsampleEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        upsampleEncoder.setPipelineState(upsamplePipeline)
        upsampleEncoder.setArgumentTable(upsampleArguments)
        upsampleEncoder.dispatchNetwork(intermediatesHeap: upsampleHeap)
        upsampleEncoder.barrier(
            afterStages: .machineLearning, beforeQueueStages: .dispatch, visibilityOptions: .device
        )
        upsampleEncoder.endEncoding()

        guard let frameResidualEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        frameResidualEncoder.setComputePipelineState(frameResidualCompute)
        frameResidualEncoder.setArgumentTable(frameResidualArguments)
        frameResidualEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: frameCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: frameResidualCompute.threadExecutionWidth, height: 1, depth: 1)
        )
        frameResidualEncoder.endEncoding()
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
        let pointer = restoredBuffer.contents().bindMemory(to: Float16.self, capacity: frameCount)
        return (milliseconds, Array(UnsafeBufferPointer(start: pointer, count: frameCount)))
    }

    _ = try execute()
    _ = try execute()
    let (firstMilliseconds, first) = try execute()
    var samples = [firstMilliseconds]
    var second = first
    for _ in 1..<7 {
        let (milliseconds, output) = try execute()
        samples.append(milliseconds)
        second = output
    }
    samples.sort()
    let milliseconds = samples[samples.count / 2]
    let frame = frameBuffer.contents().bindMemory(to: Float16.self, capacity: frameCount)
    let predicted = predictedBuffer.contents().bindMemory(to: Float16.self, capacity: frameCount)
    var maximumMagnitude: Float = 0
    var residualMaximumError: Float = 0
    var repeatMaximumError: Float = 0
    var checksum = 0.0
    for index in 0..<frameCount {
        let value = Float(second[index])
        guard value.isFinite else {
            throw DeformConvError.commandFailed("zero-copy frame graph produced a non-finite value")
        }
        maximumMagnitude = max(maximumMagnitude, abs(value))
        residualMaximumError = max(
            residualMaximumError,
            abs((Float(predicted[index]) + Float(frame[index])) - value)
        )
        repeatMaximumError = max(repeatMaximumError, abs(Float(first[index]) - value))
        if index.isMultiple(of: 257) { checksum += Double(value) }
    }
    guard maximumMagnitude > 0.001,
          residualMaximumError <= 0.001,
          repeatMaximumError <= 0.001
    else {
        throw DeformConvError.commandFailed(
            "zero-copy validation failed (max=\(maximumMagnitude), "
                + "residual=\(residualMaximumError), repeat=\(repeatMaximumError))"
        )
    }
    return ZeroCopyFrameGraphResult(
        gpuMilliseconds: milliseconds,
        minimumMilliseconds: samples[0],
        maximumMilliseconds: samples[samples.count - 1],
        iterations: samples.count,
        elementCount: frameCount,
        allocatedBytes: residentBuffers.reduce(0) { $0 + $1.length },
        maximumMagnitude: maximumMagnitude,
        residualMaximumError: residualMaximumError,
        repeatMaximumError: repeatMaximumError,
        checksum: checksum
    )
}
