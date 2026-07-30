import Foundation
import Metal

struct FusedFourPassRecurrenceResult {
    let statistics: BenchmarkStatistics
    let propagationRepeatMaximumError: Float
    let propagationStagedMaximumError: Float
    let restoredRepeatMaximumError: Float
    let restoredStagedMaximumError: Float
    let residualMaximumError: Float
    let flowRepeatMaximumError: Float
    let flowOracleMaximumError: Float
    let flowChecksums: [Double]
    let propagationChecksums: [Double]
    let restoredChecksums: [Double]
    let propagatedFrames: [[[Float16]]]
    let restoredFrames: [[Float16]]
}

@available(macOS 27.0, *)
private struct FusedBranchRuntime {
    let name: String
    let direction: PropagationDirection
    let backboneInputChannels: Int
    let offsetPipeline: any MTL4MachineLearningPipelineState
    let backbonePipeline: any MTL4MachineLearningPipelineState
    let offsetArguments: any MTL4ArgumentTable
    let backboneArguments: any MTL4ArgumentTable
    let initialAssemblyArguments: any MTL4ArgumentTable
    let initialResidualArguments: any MTL4ArgumentTable
    let accumulateArguments: [any MTL4ArgumentTable]
    let prepareArguments: [any MTL4ArgumentTable]
    let transformArguments: [any MTL4ArgumentTable]
    let gemmArguments: any MTL4ArgumentTable
    let gemmOutputArguments: any MTL4ArgumentTable
    let assemblyArguments: [any MTL4ArgumentTable]
    let residualArguments: [any MTL4ArgumentTable]
    let offsetHeap: MTLHeap
    let backboneHeap: MTLHeap
    let backboneInputTensor: any MTLTensor
    let backboneInputBuffer: MTLBuffer
    let weightBuffer: MTLBuffer
    let biasBuffer: MTLBuffer
}

@available(macOS 27.0, *)
func verifyFusedFourPassRecurrence(
    device: MTLDevice,
    modelsURL: URL,
    weightsURL: URL,
    backwardFlows: [[Float16]],
    forwardFlows: [[Float16]],
    inputFrames: [[Float16]],
    stagedBranchFrames: [[[Float16]]],
    stagedRestoredFrames: [[Float16]]
) throws -> FusedFourPassRecurrenceResult {
    typealias Support = Metal4GraphSupport
    let plane = 64 * 64
    let featureCount = 64 * plane
    let frameElements = 3 * 256 * 256
    let branchSpecs: [(String, PropagationDirection)] = [
        ("backward_1", .backward), ("forward_1", .forward),
        ("backward_2", .backward), ("forward_2", .forward),
    ]
    guard backwardFlows.count == 2, forwardFlows.count == 2,
          (backwardFlows + forwardFlows).allSatisfy({ $0.count == 2 * plane }),
          inputFrames.count == 3,
          inputFrames.allSatisfy({ $0.count == frameElements }),
          stagedBranchFrames.count == 4,
          stagedBranchFrames.allSatisfy({ branch in
              branch.count == 3 && branch.allSatisfy({ $0.count == featureCount })
          }),
          stagedRestoredFrames.count == 3,
          stagedRestoredFrames.allSatisfy({ $0.count == frameElements })
    else { throw DeformConvError.invalidShape }

    let featurePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("feature_extract.mtlpackage")
    )
    let upsamplePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("upsample.mtlpackage")
    )
    let featureHeaps = try (0..<3).map { _ in
        try Support.makeHeap(device: device, size: featurePipeline.intermediatesHeapSize)
    }
    let upsampleHeaps = try (0..<3).map { _ in
        try Support.makeHeap(device: device, size: upsamplePipeline.intermediatesHeapSize)
    }
    var heldTensors = [any MTLTensor]()
    var frameBuffers = [MTLBuffer]()
    var spatialBuffers = [MTLBuffer]()
    var featureArguments = [any MTL4ArgumentTable]()
    for _ in 0..<3 {
        let (frameTensor, frameBuffer) = try Support.makeTensor(
            device: device, dimensions: [256, 256, 3, 1]
        )
        let (spatialTensor, spatialBuffer) = try Support.makeTensor(
            device: device, dimensions: [64, 64, 64, 1]
        )
        heldTensors += [frameTensor, spatialTensor]
        frameBuffers.append(frameBuffer)
        spatialBuffers.append(spatialBuffer)
        featureArguments.append(try Support.makeMLArguments(
            device: device,
            pipeline: featurePipeline,
            resources: ["frames": frameTensor.gpuResourceID, "output": spatialTensor.gpuResourceID]
        ))
    }

    let (conditionTensor, conditionBuffer) = try Support.makeTensor(
        device: device, dimensions: [64, 64, 196, 1]
    )
    let (rawTensor, rawBuffer) = try Support.makeTensor(
        device: device, dimensions: [64, 64, 432, 1]
    )
    let (backboneOutputTensor, backboneOutputBuffer) = try Support.makeTensor(
        device: device, dimensions: [64, 64, 64, 1]
    )
    heldTensors += [conditionTensor, rawTensor, backboneOutputTensor]

    let zeroFeatureBuffer = try Support.makeSharedFP16Buffer(device: device, elements: featureCount)
    let flow2Buffer = try Support.makeSharedFP16Buffer(device: device, elements: 2 * plane)
    let deformInputBuffer = try Support.makeSharedFP16Buffer(device: device, elements: 128 * plane)
    let offsetBuffer = try Support.makeSharedFP16Buffer(device: device, elements: 288 * plane)
    let maskBuffer = try Support.makeSharedFP16Buffer(device: device, elements: 144 * plane)
    let alignedBuffer = try Support.makeSharedFP16Buffer(device: device, elements: featureCount)
    let gatheredBuffer = try Support.makePrivateBuffer(device: device, bytes: plane * 128 * 9 * 2)
    let matrixOutputBuffer = try Support.makePrivateBuffer(device: device, bytes: plane * 64 * 4)
    let propagationBuffers = try (0..<4).map { _ in
        try (0..<3).map { _ in
            try Support.makeSharedFP16Buffer(device: device, elements: featureCount)
        }
    }
    var firstShape = ThreeFramePrepareShape(hasSecondOrder: 0)
    var secondShape = ThreeFramePrepareShape(hasSecondOrder: 1)
    var planeValue = UInt32(plane)
    var prefixChannelsValue = UInt32(64)
    var featureCountValue = UInt32(featureCount)
    var frameElementsValue = UInt32(frameElements)
    var deformShape = PropagationDeformConvShape()
    let firstShapeBuffer = try Support.makeConstant(device: device, value: &firstShape)
    let secondShapeBuffer = try Support.makeConstant(device: device, value: &secondShape)
    let planeBuffer = try Support.makeConstant(device: device, value: &planeValue)
    let prefixChannelsBuffer = try Support.makeConstant(device: device, value: &prefixChannelsValue)
    let featureCountBuffer = try Support.makeConstant(device: device, value: &featureCountValue)
    let frameElementsBuffer = try Support.makeConstant(device: device, value: &frameElementsValue)
    let deformShapeBuffer = try Support.makeConstant(device: device, value: &deformShape)

    let library = try device.makeLibrary(source: MetalShader.source, options: nil)
    guard let accumulateFunction = library.makeFunction(name: "accumulate_second_order_flow_fp16"),
          let prepareFunction = library.makeFunction(name: "assemble_temporal_alignment_fp16"),
          let transformFunction = library.makeFunction(name: "prepare_dcn_offsets_fp16"),
          let gatherFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_gather"),
          let gemmFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_simdgroup_gemm"),
          let gemmOutputFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_float_gemm_output"),
          let simpleAssemblyFunction = library.makeFunction(name: "assemble_propagation_backbone_fp16"),
          let temporalAssemblyFunction = library.makeFunction(name: "assemble_temporal_backbone_fp16"),
          let residualFunction = library.makeFunction(name: "add_propagation_residual_fp16"),
          let reconstructionAssemblyFunction = library.makeFunction(name: "assemble_reconstruction_fp16"),
          let frameResidualFunction = library.makeFunction(name: "add_frame_residual_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let accumulatePipeline = try device.makeComputePipelineState(function: accumulateFunction)
    let preparePipeline = try device.makeComputePipelineState(function: prepareFunction)
    let transformPipeline = try device.makeComputePipelineState(function: transformFunction)
    let gatherPipeline = try device.makeComputePipelineState(function: gatherFunction)
    let gemmPipeline = try device.makeComputePipelineState(function: gemmFunction)
    let gemmOutputPipeline = try device.makeComputePipelineState(function: gemmOutputFunction)
    let simpleAssemblyPipeline = try device.makeComputePipelineState(function: simpleAssemblyFunction)
    let temporalAssemblyPipeline = try device.makeComputePipelineState(function: temporalAssemblyFunction)
    let residualPipeline = try device.makeComputePipelineState(function: residualFunction)
    let reconstructionAssemblyPipeline = try device.makeComputePipelineState(
        function: reconstructionAssemblyFunction
    )
    let frameResidualPipeline = try device.makeComputePipelineState(function: frameResidualFunction)
    let spynetGraph = try FusedSPyNetClipGraph(
        device: device, modelsURL: modelsURL, library: library, sourceFrames: frameBuffers
    )
    let backwardFlowBuffers = spynetGraph.backwardFlowBuffers
    let forwardFlowBuffers = spynetGraph.forwardFlowBuffers
    let gatherArguments = try Support.makeComputeArguments(
        device: device,
        buffers: [deformInputBuffer, offsetBuffer, maskBuffer, gatheredBuffer, deformShapeBuffer]
    )

    var branches = [FusedBranchRuntime]()
    var branchIndexBuffers = [MTLBuffer]()
    for (branchIndex, spec) in branchSpecs.enumerated() {
        let (name, direction) = spec
        let backboneInputChannels = (branchIndex + 2) * 64
        let offsetPipeline = try makeMetalMLPipeline(
            device: device,
            packageURL: modelsURL.appendingPathComponent("offset_\(name).mtlpackage")
        )
        let backbonePipeline = try makeMetalMLPipeline(
            device: device,
            packageURL: modelsURL.appendingPathComponent("backbone_\(name).mtlpackage")
        )
        let (backboneInputTensor, backboneInputBuffer) = try Support.makeTensor(
            device: device, dimensions: [64, 64, backboneInputChannels, 1]
        )
        heldTensors.append(backboneInputTensor)
        let checkpoint = try DeformConvWeightSet(
            direction: name,
            url: weightsURL.appendingPathComponent("\(name).dcnfp16")
        ).makeBuffers(device: device)
        var branchIndexValue = UInt32(branchIndex)
        let branchIndexBuffer = try Support.makeConstant(device: device, value: &branchIndexValue)
        branchIndexBuffers.append(branchIndexBuffer)
        let offsetArguments = try Support.makeMLArguments(
            device: device,
            pipeline: offsetPipeline,
            resources: ["conditions": conditionTensor.gpuResourceID, "output": rawTensor.gpuResourceID]
        )
        let backboneArguments = try Support.makeMLArguments(
            device: device,
            pipeline: backbonePipeline,
            resources: [
                "features": backboneInputTensor.gpuResourceID,
                "output": backboneOutputTensor.gpuResourceID,
            ]
        )
        let traversal = direction == .backward ? [2, 1, 0] : [0, 1, 2]
        func prior(_ index: Int, _ frame: Int) -> MTLBuffer {
            index < branchIndex ? propagationBuffers[index][frame] : zeroFeatureBuffer
        }
        let firstFrame = traversal[0]
        let initialAssemblyArguments = try Support.makeComputeArguments(
            device: device,
            buffers: branchIndex == 0
                ? [spatialBuffers[firstFrame], zeroFeatureBuffer, backboneInputBuffer,
                   planeBuffer, prefixChannelsBuffer]
                : [spatialBuffers[firstFrame], prior(0, firstFrame), prior(1, firstFrame),
                   prior(2, firstFrame), zeroFeatureBuffer, backboneInputBuffer,
                   planeBuffer, branchIndexBuffer]
        )
        let initialResidualArguments = try Support.makeComputeArguments(
            device: device,
            buffers: [zeroFeatureBuffer, backboneOutputBuffer,
                      propagationBuffers[branchIndex][firstFrame], featureCountBuffer]
        )
        var accumulateArguments = [any MTL4ArgumentTable]()
        var prepareArguments = [any MTL4ArgumentTable]()
        var transformArguments = [any MTL4ArgumentTable]()
        var assemblyArguments = [any MTL4ArgumentTable]()
        var residualArguments = [any MTL4ArgumentTable]()
        let flowBuffers = direction == .backward ? backwardFlowBuffers : forwardFlowBuffers
        for step in 1...2 {
            let frame = traversal[step]
            let previousFrame = traversal[step - 1]
            let flowIndex = direction == .backward ? frame : frame - 1
            let previousFlowIndex = step == 2
                ? (direction == .backward ? previousFrame : previousFrame - 1)
                : flowIndex
            let shapeBuffer = step == 2 ? secondShapeBuffer : firstShapeBuffer
            let featN2 = step == 2
                ? propagationBuffers[branchIndex][traversal[0]] : zeroFeatureBuffer
            accumulateArguments.append(try Support.makeComputeArguments(
                device: device,
                buffers: [flowBuffers[flowIndex], flowBuffers[previousFlowIndex],
                          flow2Buffer, shapeBuffer]
            ))
            prepareArguments.append(try Support.makeComputeArguments(
                device: device,
                buffers: [propagationBuffers[branchIndex][previousFrame], spatialBuffers[frame],
                          featN2, flowBuffers[flowIndex], flow2Buffer, conditionBuffer,
                          deformInputBuffer, shapeBuffer]
            ))
            transformArguments.append(try Support.makeComputeArguments(
                device: device,
                buffers: [rawBuffer, flowBuffers[flowIndex], flow2Buffer,
                          offsetBuffer, maskBuffer, planeBuffer]
            ))
            assemblyArguments.append(try Support.makeComputeArguments(
                device: device,
                buffers: branchIndex == 0
                    ? [spatialBuffers[frame], alignedBuffer, backboneInputBuffer,
                       planeBuffer, prefixChannelsBuffer]
                    : [spatialBuffers[frame], prior(0, frame), prior(1, frame), prior(2, frame),
                       alignedBuffer, backboneInputBuffer, planeBuffer, branchIndexBuffer]
            ))
            residualArguments.append(try Support.makeComputeArguments(
                device: device,
                buffers: [alignedBuffer, backboneOutputBuffer,
                          propagationBuffers[branchIndex][frame], featureCountBuffer]
            ))
        }
        let gemmArguments = try Support.makeComputeArguments(
            device: device,
            buffers: [gatheredBuffer, checkpoint.weight, matrixOutputBuffer]
        )
        let biasedGEMMOutputArguments = try Support.makeComputeArguments(
            device: device,
            buffers: [matrixOutputBuffer, checkpoint.bias, alignedBuffer, deformShapeBuffer]
        )
        branches.append(FusedBranchRuntime(
            name: name, direction: direction, backboneInputChannels: backboneInputChannels,
            offsetPipeline: offsetPipeline, backbonePipeline: backbonePipeline,
            offsetArguments: offsetArguments, backboneArguments: backboneArguments,
            initialAssemblyArguments: initialAssemblyArguments,
            initialResidualArguments: initialResidualArguments,
            accumulateArguments: accumulateArguments, prepareArguments: prepareArguments,
            transformArguments: transformArguments,
            gemmArguments: gemmArguments,
            gemmOutputArguments: biasedGEMMOutputArguments,
            assemblyArguments: assemblyArguments,
            residualArguments: residualArguments,
            offsetHeap: try Support.makeHeap(device: device, size: offsetPipeline.intermediatesHeapSize),
            backboneHeap: try Support.makeHeap(device: device, size: backbonePipeline.intermediatesHeapSize),
            backboneInputTensor: backboneInputTensor,
            backboneInputBuffer: backboneInputBuffer,
            weightBuffer: checkpoint.weight, biasBuffer: checkpoint.bias
        ))
    }

    var reconstructionBuffers = [MTLBuffer]()
    var predictedBuffers = [MTLBuffer]()
    var restoredBuffers = [MTLBuffer]()
    var reconstructionAssemblyArguments = [any MTL4ArgumentTable]()
    var upsampleArguments = [any MTL4ArgumentTable]()
    var frameResidualArguments = [any MTL4ArgumentTable]()
    for frame in 0..<3 {
        let (reconstructionTensor, reconstructionBuffer) = try Support.makeTensor(
            device: device, dimensions: [64, 64, 320, 1]
        )
        let (predictedTensor, predictedBuffer) = try Support.makeTensor(
            device: device, dimensions: [256, 256, 3, 1]
        )
        let restoredBuffer = try Support.makeSharedFP16Buffer(
            device: device, elements: frameElements
        )
        heldTensors += [reconstructionTensor, predictedTensor]
        reconstructionBuffers.append(reconstructionBuffer)
        predictedBuffers.append(predictedBuffer)
        restoredBuffers.append(restoredBuffer)
        reconstructionAssemblyArguments.append(try Support.makeComputeArguments(
            device: device,
            buffers: [spatialBuffers[frame]]
                + propagationBuffers.map { $0[frame] }
                + [reconstructionBuffer, planeBuffer]
        ))
        upsampleArguments.append(try Support.makeMLArguments(
            device: device,
            pipeline: upsamplePipeline,
            resources: [
                "features": reconstructionTensor.gpuResourceID,
                "output": predictedTensor.gpuResourceID,
            ]
        ))
        frameResidualArguments.append(try Support.makeComputeArguments(
            device: device,
            buffers: [predictedBuffer, frameBuffers[frame], restoredBuffer, frameElementsBuffer]
        ))
    }
    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "fused three-frame feature-to-restored graph"
    residencyDescriptor.initialCapacity = 144
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    let sharedBuffers = frameBuffers + spatialBuffers + propagationBuffers.flatMap { $0 } +
        backwardFlowBuffers + forwardFlowBuffers + branchIndexBuffers + [
            conditionBuffer, rawBuffer, backboneOutputBuffer, zeroFeatureBuffer, flow2Buffer,
            deformInputBuffer, offsetBuffer, maskBuffer, alignedBuffer, gatheredBuffer,
            matrixOutputBuffer, firstShapeBuffer, secondShapeBuffer, planeBuffer,
            prefixChannelsBuffer, featureCountBuffer, frameElementsBuffer, deformShapeBuffer,
        ]
        + reconstructionBuffers + predictedBuffers + restoredBuffers
    for buffer in sharedBuffers { residencySet.addAllocation(buffer) }
    for heap in featureHeaps { residencySet.addAllocation(heap) }
    for heap in upsampleHeaps { residencySet.addAllocation(heap) }
    spynetGraph.addAllocations(to: residencySet)
    for branch in branches {
        residencySet.addAllocation(branch.offsetHeap)
        residencySet.addAllocation(branch.backboneHeap)
        residencySet.addAllocation(branch.backboneInputBuffer)
        residencySet.addAllocation(branch.weightBuffer)
        residencySet.addAllocation(branch.biasBuffer)
    }
    residencySet.commit()
    guard let queue = device.makeMTL4CommandQueue() else {
        throw DeformConvError.metalUnavailable
    }

    let transientBuffers = spatialBuffers + propagationBuffers.flatMap { $0 } + [
        conditionBuffer, rawBuffer, backboneOutputBuffer, zeroFeatureBuffer, flow2Buffer,
        deformInputBuffer, offsetBuffer, maskBuffer, alignedBuffer,
    ] + reconstructionBuffers + predictedBuffers + restoredBuffers
    func initializeBuffers() {
        spynetGraph.initializeBuffers()
        for buffer in transientBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        for frame in 0..<3 {
            inputFrames[frame].withUnsafeBufferPointer { source in
                frameBuffers[frame].contents().bindMemory(to: Float16.self, capacity: frameElements)
                    .update(from: source.baseAddress!, count: frameElements)
            }
        }
    }

    func execute() throws -> (Double, [[[Float16]]], [[Float16]], [[Float16]]) {
        initializeBuffers()
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        for index in 0..<3 {
            guard let encoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            encoder.setPipelineState(featurePipeline)
            encoder.setArgumentTable(featureArguments[index])
            encoder.dispatchNetwork(intermediatesHeap: featureHeaps[index])
            encoder.endEncoding()
        }
        try spynetGraph.encode(into: commandBuffer)
        for (branchIndex, branch) in branches.enumerated() {
            let assemblyPipeline = branchIndex == 0 ? simpleAssemblyPipeline : temporalAssemblyPipeline
            guard let initialAssembly = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            initialAssembly.barrier(
                afterQueueStages: branchIndex == 0 ? .machineLearning : .dispatch,
                beforeStages: .dispatch, visibilityOptions: .device
            )
            Support.dispatch1D(
                initialAssembly, pipeline: assemblyPipeline,
                arguments: branch.initialAssemblyArguments,
                count: branch.backboneInputChannels * plane
            )
            initialAssembly.barrier(
                afterStages: .dispatch, beforeQueueStages: .machineLearning,
                visibilityOptions: .device
            )
            initialAssembly.endEncoding()
            guard let initialBackbone = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            initialBackbone.setPipelineState(branch.backbonePipeline)
            initialBackbone.setArgumentTable(branch.backboneArguments)
            initialBackbone.dispatchNetwork(intermediatesHeap: branch.backboneHeap)
            initialBackbone.barrier(
                afterStages: .machineLearning, beforeQueueStages: .dispatch,
                visibilityOptions: .device
            )
            initialBackbone.endEncoding()
            guard let initialResidual = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            Support.dispatch1D(
                initialResidual, pipeline: residualPipeline,
                arguments: branch.initialResidualArguments, count: featureCount
            )
            initialResidual.endEncoding()

            for step in 0..<2 {
                guard let preparation = commandBuffer.makeComputeCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                preparation.barrier(
                    afterQueueStages: .dispatch, beforeStages: .dispatch,
                    visibilityOptions: .device
                )
                Support.dispatch1D(
                    preparation, pipeline: accumulatePipeline,
                    arguments: branch.accumulateArguments[step], count: 2 * plane
                )
                preparation.barrier(
                    afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                Support.dispatch1D(
                    preparation, pipeline: preparePipeline,
                    arguments: branch.prepareArguments[step], count: 196 * plane
                )
                preparation.barrier(
                    afterStages: .dispatch, beforeQueueStages: .machineLearning,
                    visibilityOptions: .device
                )
                preparation.endEncoding()
                guard let offsetEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                offsetEncoder.setPipelineState(branch.offsetPipeline)
                offsetEncoder.setArgumentTable(branch.offsetArguments)
                offsetEncoder.dispatchNetwork(intermediatesHeap: branch.offsetHeap)
                offsetEncoder.barrier(
                    afterStages: .machineLearning, beforeQueueStages: .dispatch,
                    visibilityOptions: .device
                )
                offsetEncoder.endEncoding()
                guard let alignment = commandBuffer.makeComputeCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                Support.dispatch1D(
                    alignment, pipeline: transformPipeline,
                    arguments: branch.transformArguments[step], count: 432 * plane
                )
                alignment.barrier(
                    afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                Support.dispatch1D(
                    alignment, pipeline: gatherPipeline, arguments: gatherArguments,
                    count: plane, threads: 128, threadgroups: true
                )
                alignment.barrier(
                    afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                Support.dispatch1D(
                    alignment, pipeline: gemmPipeline,
                    arguments: branch.gemmArguments, count: plane / 8,
                    threads: 8 * gemmPipeline.threadExecutionWidth, threadgroups: true
                )
                alignment.barrier(
                    afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                Support.dispatch1D(
                    alignment, pipeline: gemmOutputPipeline,
                    arguments: branch.gemmOutputArguments, count: featureCount
                )
                alignment.barrier(
                    afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                Support.dispatch1D(
                    alignment, pipeline: assemblyPipeline,
                    arguments: branch.assemblyArguments[step],
                    count: branch.backboneInputChannels * plane
                )
                alignment.barrier(
                    afterStages: .dispatch, beforeQueueStages: .machineLearning,
                    visibilityOptions: .device
                )
                alignment.endEncoding()
                guard let backbone = commandBuffer.makeMachineLearningCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                backbone.setPipelineState(branch.backbonePipeline)
                backbone.setArgumentTable(branch.backboneArguments)
                backbone.dispatchNetwork(intermediatesHeap: branch.backboneHeap)
                backbone.barrier(
                    afterStages: .machineLearning, beforeQueueStages: .dispatch,
                    visibilityOptions: .device
                )
                backbone.endEncoding()
                guard let residual = commandBuffer.makeComputeCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                Support.dispatch1D(
                    residual, pipeline: residualPipeline,
                    arguments: branch.residualArguments[step], count: featureCount
                )
                residual.endEncoding()
            }
        }
        guard let reconstructionAssembly = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        reconstructionAssembly.barrier(
            afterQueueStages: .dispatch, beforeStages: .dispatch,
            visibilityOptions: .device
        )
        for arguments in reconstructionAssemblyArguments {
            Support.dispatch1D(
                reconstructionAssembly,
                pipeline: reconstructionAssemblyPipeline,
                arguments: arguments,
                count: 320 * plane
            )
        }
        reconstructionAssembly.barrier(
            afterStages: .dispatch, beforeQueueStages: .machineLearning,
            visibilityOptions: .device
        )
        reconstructionAssembly.endEncoding()
        for frame in 0..<3 {
            guard let upsample = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            upsample.setPipelineState(upsamplePipeline)
            upsample.setArgumentTable(upsampleArguments[frame])
            upsample.dispatchNetwork(intermediatesHeap: upsampleHeaps[frame])
            upsample.endEncoding()
        }
        guard let frameResidual = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        frameResidual.barrier(
            afterQueueStages: .machineLearning, beforeStages: .dispatch,
            visibilityOptions: .device
        )
        for arguments in frameResidualArguments {
            Support.dispatch1D(
                frameResidual,
                pipeline: frameResidualPipeline,
                arguments: arguments,
                count: frameElements
            )
        }
        frameResidual.endEncoding()
        commandBuffer.endCommandBuffer()
        let semaphore = DispatchSemaphore(value: 0)
        let commitResult = CommitResult()
        let commitOptions = MTL4CommitOptions()
        commitOptions.addFeedbackHandler { feedback in
            commitResult.store(
                milliseconds: (feedback.gpuEndTime - feedback.gpuStartTime) * 1_000,
                error: feedback.error
            )
            semaphore.signal()
        }
        queue.commit([commandBuffer], options: commitOptions)
        semaphore.wait()
        let (milliseconds, error) = commitResult.load()
        if let error { throw error }
        let outputs = propagationBuffers.map { branch in
            branch.map { buffer -> [Float16] in
                let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: featureCount)
                return Array(UnsafeBufferPointer(start: pointer, count: featureCount))
            }
        }
        let restored = restoredBuffers.map { buffer -> [Float16] in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: frameElements)
            return Array(UnsafeBufferPointer(start: pointer, count: frameElements))
        }
        let flows = (backwardFlowBuffers + forwardFlowBuffers).map { buffer -> [Float16] in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: 2 * plane)
            return Array(UnsafeBufferPointer(start: pointer, count: 2 * plane))
        }
        return (milliseconds, outputs, restored, flows)
    }

    let measurementCount = 20
    _ = try execute()
    _ = try execute()
    _ = try execute()
    let (firstMilliseconds, firstOutputs, firstRestored, firstFlows) = try execute()
    var samples = [firstMilliseconds]
    var lastOutputs = firstOutputs
    var lastRestored = firstRestored
    var lastFlows = firstFlows
    for _ in 1..<measurementCount {
        let (milliseconds, outputs, restored, flows) = try execute()
        samples.append(milliseconds)
        lastOutputs = outputs
        lastRestored = restored
        lastFlows = flows
    }
    guard let statistics = BenchmarkStatistics(samples) else {
        throw DeformConvError.commandFailed("invalid fused four-pass benchmark samples")
    }
    var propagationRepeatMaximumError: Float = 0
    var propagationStagedMaximumError: Float = 0
    var stagedBranchErrors = [Float](repeating: 0, count: 4)
    var checksums = [Double](repeating: 0, count: 4)
    for branch in 0..<4 {
        let finalFrame = branchSpecs[branch].1 == .backward ? 0 : 2
        for frame in 0..<3 {
            for index in 0..<featureCount {
                let value = Float(lastOutputs[branch][frame][index])
                guard value.isFinite else {
                    throw DeformConvError.commandFailed("fused recurrence produced a non-finite feature")
                }
                propagationRepeatMaximumError = max(
                    propagationRepeatMaximumError,
                    abs(Float(firstOutputs[branch][frame][index]) - value)
                )
                propagationStagedMaximumError = max(
                    propagationStagedMaximumError,
                    abs(Float(stagedBranchFrames[branch][frame][index]) - value)
                )
                stagedBranchErrors[branch] = max(
                    stagedBranchErrors[branch],
                    abs(Float(stagedBranchFrames[branch][frame][index]) - value)
                )
                if frame == finalFrame, index.isMultiple(of: 257) {
                    checksums[branch] += Double(value)
                }
            }
        }
    }
    var restoredRepeatMaximumError: Float = 0
    var restoredStagedMaximumError: Float = 0
    var residualMaximumError: Float = 0
    var restoredChecksums = [Double](repeating: 0, count: 3)
    for frame in 0..<3 {
        let predicted = predictedBuffers[frame].contents().bindMemory(
            to: Float16.self, capacity: frameElements
        )
        for index in 0..<frameElements {
            let value = Float(lastRestored[frame][index])
            guard value.isFinite else {
                throw DeformConvError.commandFailed("fused graph produced a non-finite frame")
            }
            restoredRepeatMaximumError = max(
                restoredRepeatMaximumError,
                abs(Float(firstRestored[frame][index]) - value)
            )
            restoredStagedMaximumError = max(
                restoredStagedMaximumError,
                abs(Float(stagedRestoredFrames[frame][index]) - value)
            )
            residualMaximumError = max(
                residualMaximumError,
                abs(Float(predicted[index]) + Float(inputFrames[frame][index]) - value)
            )
            if index.isMultiple(of: 257) { restoredChecksums[frame] += Double(value) }
        }
    }
    let flowOracles = backwardFlows + forwardFlows
    var flowRepeatMaximumError: Float = 0
    var flowOracleMaximumError: Float = 0
    var flowChecksums = [Double](repeating: 0, count: 4)
    for flow in 0..<4 {
        for index in 0..<(2 * plane) {
            let value = Float(lastFlows[flow][index])
            guard value.isFinite else {
                throw DeformConvError.commandFailed("fused SPyNet produced a non-finite flow")
            }
            flowRepeatMaximumError = max(
                flowRepeatMaximumError, abs(Float(firstFlows[flow][index]) - value)
            )
            flowOracleMaximumError = max(
                flowOracleMaximumError, abs(Float(flowOracles[flow][index]) - value)
            )
            if index.isMultiple(of: 257) { flowChecksums[flow] += Double(value) }
        }
    }
    guard propagationRepeatMaximumError <= 0.001,
          propagationStagedMaximumError <= 0.002,
          restoredRepeatMaximumError <= 0.001,
          restoredStagedMaximumError <= 0.002,
          residualMaximumError <= 0.001,
          flowRepeatMaximumError <= 0.001,
          flowOracleMaximumError <= 0.002
    else {
        throw DeformConvError.commandFailed(
            "fused graph mismatch (propagation repeat=\(propagationRepeatMaximumError), "
                + "propagation staged=\(propagationStagedMaximumError), "
                + "restored repeat=\(restoredRepeatMaximumError), "
                + "restored staged=\(restoredStagedMaximumError), "
                + "residual=\(residualMaximumError), flow repeat=\(flowRepeatMaximumError), "
                + "flow oracle=\(flowOracleMaximumError), branches=\(stagedBranchErrors))"
        )
    }
    _ = heldTensors
    _ = branches.map(\.backboneInputTensor)
    return FusedFourPassRecurrenceResult(
        statistics: statistics,
        propagationRepeatMaximumError: propagationRepeatMaximumError,
        propagationStagedMaximumError: propagationStagedMaximumError,
        restoredRepeatMaximumError: restoredRepeatMaximumError,
        restoredStagedMaximumError: restoredStagedMaximumError,
        residualMaximumError: residualMaximumError,
        flowRepeatMaximumError: flowRepeatMaximumError,
        flowOracleMaximumError: flowOracleMaximumError,
        flowChecksums: flowChecksums,
        propagationChecksums: checksums,
        restoredChecksums: restoredChecksums,
        propagatedFrames: lastOutputs,
        restoredFrames: lastRestored
    )
}
