import Foundation
import Metal

struct FusedFourPassRecurrenceResult {
    let statistics: BenchmarkStatistics
    let propagationRepeatMaximumError: Float
    let propagationStagedMaximumError: Float
    let restoredRepeatMaximumError: Float
    let restoredStagedMaximumError: Float
    let residualMaximumError: Float
    let flowOracleCompared: Bool
    let flowRepeatMaximumError: Float
    let flowOracleMaximumError: Float
    let flowChecksums: [Double]
    let propagationChecksums: [Double]
    let restoredChecksums: [Double]
    let propagatedFrames: [[[Float16]]]
    let restoredFrames: [[Float16]]
}

@available(macOS 27.0, *)
private final class ProductionFusedGraphRunner: @unchecked Sendable {
    private let lock = NSLock()
    private let execute: ([[Float16]]) throws -> FusedFourPassRecurrenceResult

    init(execute: @escaping ([[Float16]]) throws -> FusedFourPassRecurrenceResult) {
        self.execute = execute
    }

    func restore(_ frames: [[Float16]]) throws -> FusedFourPassRecurrenceResult {
        lock.lock()
        defer { lock.unlock() }
        return try execute(frames)
    }
}

@available(macOS 27.0, *)
private final class ProductionFusedGraphCache: @unchecked Sendable {
    static let shared = ProductionFusedGraphCache()

    private let lock = NSLock()
    private var runners = [String: ProductionFusedGraphRunner]()

    private init() {}

    func runner(for key: String) -> ProductionFusedGraphRunner? {
        lock.lock()
        defer { lock.unlock() }
        return runners[key]
    }

    func retain(_ runner: ProductionFusedGraphRunner, for key: String) {
        lock.lock()
        defer { lock.unlock() }
        if runners[key] == nil { runners[key] = runner }
    }
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
    stagedRestoredFrames: [[Float16]],
    warmupCount: Int = 3,
    measurementCount: Int = 20,
    collectDiagnostics: Bool = true
) throws -> FusedFourPassRecurrenceResult {
    typealias Support = Metal4GraphSupport
    let plane = 64 * 64
    let featureCount = 64 * plane
    let frameElements = 3 * 256 * 256
    let frameCount = inputFrames.count
    let flowCount = frameCount - 1
    let hasFlowOracle = !backwardFlows.isEmpty || !forwardFlows.isEmpty
    let hasStagedPropagation = !stagedBranchFrames.isEmpty
    let hasStagedRestoration = !stagedRestoredFrames.isEmpty
    let reusableProductionGraph = frameCount == 30
        && warmupCount == 0
        && measurementCount == 1
        && !collectDiagnostics
        && !hasFlowOracle
        && !hasStagedPropagation
        && !hasStagedRestoration
    let productionGraphKey = "\(device.registryID):\(modelsURL.standardizedFileURL.path):"
        + "\(weightsURL.standardizedFileURL.path):\(frameCount)"
    let branchSpecs: [(String, PropagationDirection)] = [
        ("backward_1", .backward), ("forward_1", .forward),
        ("backward_2", .backward), ("forward_2", .forward),
    ]
    guard frameCount >= 3,
          warmupCount >= 0,
          measurementCount > 0,
          (!hasFlowOracle || (
              backwardFlows.count == flowCount
                  && forwardFlows.count == flowCount
                  && (backwardFlows + forwardFlows).allSatisfy({ $0.count == 2 * plane })
          )),
          inputFrames.allSatisfy({ $0.count == frameElements }),
          collectDiagnostics || (!hasFlowOracle && !hasStagedPropagation),
          (!hasStagedPropagation || (
              stagedBranchFrames.count == 4
                  && stagedBranchFrames.allSatisfy({ branch in
                      branch.count == frameCount
                          && branch.allSatisfy({ $0.count == featureCount })
                  })
          )),
          (!hasStagedRestoration || (
              stagedRestoredFrames.count == frameCount
                  && stagedRestoredFrames.allSatisfy({ $0.count == frameElements })
          ))
    else { throw DeformConvError.invalidShape }

    if reusableProductionGraph,
       let runner = ProductionFusedGraphCache.shared.runner(for: productionGraphKey)
    {
        return try runner.restore(inputFrames)
    }

    var activeInputFrames = inputFrames

    let featurePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("feature_extract.mtlpackage")
    )
    let upsamplePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("upsample.mtlpackage")
    )
    // Metal ML's intermediates heap is scratch storage for one dispatch. These
    // networks execute serially in one command buffer, so a pipeline can reuse
    // the same heap for every frame instead of allocating 2 * frameCount heaps
    // for every restored crop.
    let featureHeap = try Support.makeHeap(
        device: device, size: featurePipeline.intermediatesHeapSize
    )
    let upsampleHeap = try Support.makeHeap(
        device: device, size: upsamplePipeline.intermediatesHeapSize
    )
    let featureHeaps = [MTLHeap](repeating: featureHeap, count: frameCount)
    let upsampleHeaps = [MTLHeap](repeating: upsampleHeap, count: frameCount)
    var heldTensors = [any MTLTensor]()
    var frameBuffers = [MTLBuffer]()
    var spatialBuffers = [MTLBuffer]()
    var featureArguments = [any MTL4ArgumentTable]()
    for _ in 0..<frameCount {
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
    let propagationBuffers = try (0..<4).map { _ in
        try (0..<frameCount).map { _ in
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

    let library = try MetalResourceCache.shared.shaderLibrary(device: device) {
        try device.makeLibrary(source: MetalShader.source, options: nil)
    }
    guard let accumulateFunction = library.makeFunction(name: "accumulate_second_order_flow_fp16"),
          let prepareFunction = library.makeFunction(name: "assemble_temporal_alignment_fp16"),
          let transformFunction = library.makeFunction(name: "prepare_dcn_offsets_fp16"),
          let gatherFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_gather"),
          let gemmFunction = library.makeFunction(
              name: "deform_conv2d_fp16_jasna_simdgroup_gemm_fused"
          ),
          let simpleAssemblyFunction = library.makeFunction(name: "assemble_propagation_backbone_fp16"),
          let temporalAssemblyFunction = library.makeFunction(name: "assemble_temporal_backbone_fp16"),
          let residualFunction = library.makeFunction(name: "add_propagation_residual_fp16"),
          let reconstructionAssemblyFunction = library.makeFunction(name: "assemble_reconstruction_fp16"),
          let frameResidualFunction = library.makeFunction(name: "add_frame_residual_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let cache = MetalResourceCache.shared
    let accumulatePipeline = try cache.computePipeline(device: device, function: accumulateFunction)
    let preparePipeline = try cache.computePipeline(device: device, function: prepareFunction)
    let transformPipeline = try cache.computePipeline(device: device, function: transformFunction)
    let gatherPipeline = try cache.computePipeline(device: device, function: gatherFunction)
    let gemmPipeline = try cache.computePipeline(device: device, function: gemmFunction)
    let simpleAssemblyPipeline = try cache.computePipeline(
        device: device, function: simpleAssemblyFunction
    )
    let temporalAssemblyPipeline = try cache.computePipeline(
        device: device, function: temporalAssemblyFunction
    )
    let residualPipeline = try cache.computePipeline(device: device, function: residualFunction)
    let reconstructionAssemblyPipeline = try cache.computePipeline(
        device: device, function: reconstructionAssemblyFunction
    )
    let frameResidualPipeline = try cache.computePipeline(
        device: device, function: frameResidualFunction
    )
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
        let checkpoint = try MetalResourceCache.shared.deformConvWeightBuffers(
            device: device,
            direction: name,
            url: weightsURL.appendingPathComponent("\(name).dcnfp16")
        )
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
        let traversal = direction == .backward
            ? Array((0..<frameCount).reversed()) : Array(0..<frameCount)
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
        for step in 1..<frameCount {
            let frame = traversal[step]
            let previousFrame = traversal[step - 1]
            let flowIndex = direction == .backward ? frame : frame - 1
            let previousFlowIndex = step >= 2
                ? (direction == .backward ? previousFrame : previousFrame - 1)
                : flowIndex
            let shapeBuffer = step >= 2 ? secondShapeBuffer : firstShapeBuffer
            let featN2 = step >= 2
                ? propagationBuffers[branchIndex][traversal[step - 2]] : zeroFeatureBuffer
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
            buffers: [
                gatheredBuffer, checkpoint.weight, checkpoint.bias,
                alignedBuffer, deformShapeBuffer,
            ]
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
    for frame in 0..<frameCount {
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
            firstShapeBuffer, secondShapeBuffer, planeBuffer,
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
        for frame in 0..<frameCount {
            activeInputFrames[frame].withUnsafeBufferPointer { source in
                frameBuffers[frame].contents().bindMemory(to: Float16.self, capacity: frameElements)
                    .update(from: source.baseAddress!, count: frameElements)
            }
        }
    }

    func execute() throws -> (Double, [[[Float16]]], [[Float16]], [[Float16]]) {
        MetalResourceCache.shared.beginMachineLearningExecution()
        defer { MetalResourceCache.shared.endMachineLearningExecution() }
        initializeBuffers()
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        for index in 0..<frameCount {
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

            for step in 0..<flowCount {
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
        for frame in 0..<frameCount {
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
        let outputs = collectDiagnostics ? propagationBuffers.map { branch in
            branch.map { buffer -> [Float16] in
                let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: featureCount)
                return Array(UnsafeBufferPointer(start: pointer, count: featureCount))
            }
        } : []
        let restored = restoredBuffers.map { buffer -> [Float16] in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: frameElements)
            return Array(UnsafeBufferPointer(start: pointer, count: frameElements))
        }
        let flows = collectDiagnostics ? (backwardFlowBuffers + forwardFlowBuffers).map {
            buffer -> [Float16] in
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: 2 * plane)
            return Array(UnsafeBufferPointer(start: pointer, count: 2 * plane))
        } : []
        return (milliseconds, outputs, restored, flows)
    }

    for _ in 0..<warmupCount { _ = try execute() }
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
    if collectDiagnostics {
        for branch in 0..<4 {
            let finalFrame = branchSpecs[branch].1 == .backward ? 0 : frameCount - 1
            for frame in 0..<frameCount {
                for index in 0..<featureCount {
                    let value = Float(lastOutputs[branch][frame][index])
                    guard value.isFinite else {
                        throw DeformConvError.nonFiniteOutput(
                            "fused recurrence produced a non-finite feature in "
                                + "\(branchSpecs[branch].0), frame \(frame), element \(index)"
                        )
                    }
                    propagationRepeatMaximumError = max(
                        propagationRepeatMaximumError,
                        abs(Float(firstOutputs[branch][frame][index]) - value)
                    )
                    if hasStagedPropagation {
                        propagationStagedMaximumError = max(
                            propagationStagedMaximumError,
                            abs(Float(stagedBranchFrames[branch][frame][index]) - value)
                        )
                        stagedBranchErrors[branch] = max(
                            stagedBranchErrors[branch],
                            abs(Float(stagedBranchFrames[branch][frame][index]) - value)
                        )
                    }
                    if frame == finalFrame, index.isMultiple(of: 257) {
                        checksums[branch] += Double(value)
                    }
                }
            }
        }
    }
    var restoredRepeatMaximumError: Float = 0
    var restoredStagedMaximumError: Float = 0
    var residualMaximumError: Float = 0
    var restoredChecksums = [Double](repeating: 0, count: frameCount)
    for frame in 0..<frameCount {
        let predicted = predictedBuffers[frame].contents().bindMemory(
            to: Float16.self, capacity: frameElements
        )
        for index in 0..<frameElements {
            let value = Float(lastRestored[frame][index])
            guard value.isFinite else {
                throw DeformConvError.nonFiniteOutput(
                    "fused graph produced a non-finite output at frame \(frame), element \(index)"
                )
            }
            restoredRepeatMaximumError = max(
                restoredRepeatMaximumError,
                abs(Float(firstRestored[frame][index]) - value)
            )
            if hasStagedRestoration {
                restoredStagedMaximumError = max(
                    restoredStagedMaximumError,
                    abs(Float(stagedRestoredFrames[frame][index]) - value)
                )
            }
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
    var flowChecksums = [Double](repeating: 0, count: 2 * flowCount)
    if collectDiagnostics {
        for flow in 0..<(2 * flowCount) {
            for index in 0..<(2 * plane) {
                let value = Float(lastFlows[flow][index])
                guard value.isFinite else {
                    throw DeformConvError.nonFiniteOutput(
                        "fused SPyNet produced a non-finite flow \(flow), element \(index)"
                    )
                }
                flowRepeatMaximumError = max(
                    flowRepeatMaximumError, abs(Float(firstFlows[flow][index]) - value)
                )
                if hasFlowOracle {
                    flowOracleMaximumError = max(
                        flowOracleMaximumError, abs(Float(flowOracles[flow][index]) - value)
                    )
                }
                if index.isMultiple(of: 257) { flowChecksums[flow] += Double(value) }
            }
        }
    }
    guard propagationRepeatMaximumError <= 0.001,
          propagationStagedMaximumError <= 0.002,
          restoredRepeatMaximumError <= 0.001,
          restoredStagedMaximumError <= 0.002,
          residualMaximumError <= 0.001,
          flowRepeatMaximumError <= 0.001,
          (!hasFlowOracle || flowOracleMaximumError <= 0.002)
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
    let result = FusedFourPassRecurrenceResult(
        statistics: statistics,
        propagationRepeatMaximumError: propagationRepeatMaximumError,
        propagationStagedMaximumError: propagationStagedMaximumError,
        restoredRepeatMaximumError: restoredRepeatMaximumError,
        restoredStagedMaximumError: restoredStagedMaximumError,
        residualMaximumError: residualMaximumError,
        flowOracleCompared: hasFlowOracle,
        flowRepeatMaximumError: flowRepeatMaximumError,
        flowOracleMaximumError: flowOracleMaximumError,
        flowChecksums: flowChecksums,
        propagationChecksums: checksums,
        restoredChecksums: restoredChecksums,
        propagatedFrames: lastOutputs,
        restoredFrames: lastRestored
    )
    if reusableProductionGraph {
        let runner = ProductionFusedGraphRunner { frames in
            guard frames.count == frameCount,
                  frames.allSatisfy({ $0.count == frameElements })
            else { throw DeformConvError.invalidShape }
            // Argument tables store resource IDs, not strong ownership of the
            // tensor wrappers that created those IDs. A retained production
            // graph must therefore keep every tensor alive across executions.
            _ = heldTensors
            _ = branches.map(\.backboneInputTensor)
            activeInputFrames = frames
            let (milliseconds, _, restored, _) = try execute()
            for (frame, values) in restored.enumerated() {
                for (index, value) in values.enumerated() where !Float(value).isFinite {
                    throw DeformConvError.nonFiniteOutput(
                        "fused graph produced a non-finite output at frame \(frame), "
                            + "element \(index)"
                    )
                }
            }
            guard let statistics = BenchmarkStatistics([milliseconds]) else {
                throw DeformConvError.commandFailed("invalid fused four-pass timing")
            }
            return FusedFourPassRecurrenceResult(
                statistics: statistics,
                propagationRepeatMaximumError: 0,
                propagationStagedMaximumError: 0,
                restoredRepeatMaximumError: 0,
                restoredStagedMaximumError: 0,
                residualMaximumError: 0,
                flowOracleCompared: false,
                flowRepeatMaximumError: 0,
                flowOracleMaximumError: 0,
                flowChecksums: [],
                propagationChecksums: [],
                restoredChecksums: [Double](repeating: 0, count: frameCount),
                propagatedFrames: [],
                restoredFrames: restored
            )
        }
        ProductionFusedGraphCache.shared.retain(runner, for: productionGraphKey)
    }
    return result
}
