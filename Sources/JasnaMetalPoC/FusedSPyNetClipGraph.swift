import Metal

private struct FusedSPyNetShape {
    var width: UInt32
    var height: UInt32
    var rowStride: UInt32
    var sourceFlowWidth: UInt32
    var sourceFlowHeight: UInt32
    var sourceFlowRowStride: UInt32
    var firstLevel: UInt32
}

@available(macOS 27.0, *)
private struct FusedSPyNetLevel {
    let size: Int
    let pipeline: any MTL4MachineLearningPipelineState
    let heap: MTLHeap
    let featureBuffer: MTLBuffer
    let residualBuffer: MTLBuffer
    let baseFlowBuffer: MTLBuffer
    let outputFlowBuffer: MTLBuffer
    let shapeBuffer: MTLBuffer
    let prepareArguments: any MTL4ArgumentTable
    let mlArguments: any MTL4ArgumentTable
    let addArguments: any MTL4ArgumentTable
}

@available(macOS 27.0, *)
final class FusedSPyNetClipGraph {
    typealias Support = Metal4GraphSupport

    let backwardFlowBuffers: [MTLBuffer]
    let forwardFlowBuffers: [MTLBuffer]

    private let downsamplePipeline: MTLComputePipelineState
    private let pyramidPipeline: MTLComputePipelineState
    private let preparePipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    private let downsampleArguments: [any MTL4ArgumentTable]
    private let pyramidArguments: [any MTL4ArgumentTable]
    private let pairs: [[[FusedSPyNetLevel]]]
    private let residentBuffers: [MTLBuffer]
    private let transientBuffers: [MTLBuffer]
    private let heaps: [MTLHeap]
    private let heldTensors: [any MTLTensor]

    init(
        device: MTLDevice,
        modelsURL: URL,
        library: MTLLibrary,
        sourceFrames: [MTLBuffer]
    ) throws {
        let sizes = [2, 4, 8, 16, 32, 64]
        guard sourceFrames.count >= 2 else { throw DeformConvError.invalidShape }
        let frameCount = sourceFrames.count
        guard let downsampleFunction = library.makeFunction(
                  name: "jasna_bicubic_downsample_quarter_fp16"
              ),
              let pyramidFunction = library.makeFunction(name: "spynet_build_pyramid_pair_fp16"),
              let prepareFunction = library.makeFunction(name: "spynet_prepare_padded_fp16"),
              let addFunction = library.makeFunction(name: "spynet_add_flow_padded_fp16")
        else { throw DeformConvError.shaderResourceMissing }
        let cache = MetalResourceCache.shared
        downsamplePipeline = try cache.computePipeline(device: device, function: downsampleFunction)
        pyramidPipeline = try cache.computePipeline(device: device, function: pyramidFunction)
        preparePipeline = try cache.computePipeline(device: device, function: prepareFunction)
        addPipeline = try cache.computePipeline(device: device, function: addFunction)

        func makePaddedTensor(
            dimensions: [Int], rowStride: Int
        ) throws -> (any MTLTensor, MTLBuffer) {
            guard dimensions.count == 4 else { throw DeformConvError.invalidShape }
            let strides = [
                1, rowStride, rowStride * dimensions[1],
                rowStride * dimensions[1] * dimensions[2],
            ]
            let buffer = try Support.makeSharedFP16Buffer(
                device: device, elements: strides[3] * dimensions[3]
            )
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

        let downsampledFrames = try (0..<frameCount).map { _ in
            try Support.makeSharedFP16Buffer(device: device, elements: 3 * 64 * 64)
        }
        downsampleArguments = try (0..<frameCount).map { frame in
            try Support.makeComputeArguments(
                device: device, buffers: [sourceFrames[frame], downsampledFrames[frame]]
            )
        }
        let zeroFlow = try Support.makeSharedFP16Buffer(device: device, elements: 2 * 32 * 2)
        let levelPipelines = try sizes.indices.map { level in
            try makeMetalMLPipeline(
                device: device,
                packageURL: modelsURL.appendingPathComponent("spynet_level_\(level).mtlpackage")
            )
        }
        // All pair/direction dispatches at a level are encoded serially. Keep
        // one scratch heap per level rather than one per pair and direction.
        // For a 30-frame clip this removes 342 identical MTLHeap allocations.
        let levelHeaps = try levelPipelines.map {
            try Support.makeHeap(device: device, size: $0.intermediatesHeapSize)
        }

        var builtPairs = [[[FusedSPyNetLevel]]]()
        var builtPyramidArguments = [any MTL4ArgumentTable]()
        var buffers = sourceFrames + downsampledFrames + [zeroFlow]
        var mutableTransient = downsampledFrames + [zeroFlow]
        let mutableHeaps = levelHeaps
        var tensors = [any MTLTensor]()
        for pairIndex in 0..<(frameCount - 1) {
            let referencePyramid = try sizes.map {
                try Support.makeSharedFP16Buffer(device: device, elements: 3 * $0 * $0)
            }
            let supportPyramid = try sizes.map {
                try Support.makeSharedFP16Buffer(device: device, elements: 3 * $0 * $0)
            }
            var pyramidBindings = [downsampledFrames[pairIndex], downsampledFrames[pairIndex + 1]]
            for level in sizes.indices {
                pyramidBindings += [referencePyramid[level], supportPyramid[level]]
            }
            builtPyramidArguments.append(try Support.makeComputeArguments(
                device: device, buffers: pyramidBindings
            ))
            buffers += referencePyramid + supportPyramid
            mutableTransient += referencePyramid + supportPyramid

            var directions = [[FusedSPyNetLevel](), [FusedSPyNetLevel]()]
            for direction in 0..<2 {
                for (level, size) in sizes.enumerated() {
                    let rowStride = max(size, 32)
                    let storagePlane = rowStride * size
                    let (featureTensor, featureBuffer) = try makePaddedTensor(
                        dimensions: [size, size, 8, 1], rowStride: rowStride
                    )
                    let (residualTensor, residualBuffer) = try makePaddedTensor(
                        dimensions: [size, size, 2, 1], rowStride: rowStride
                    )
                    let baseFlow = try Support.makeSharedFP16Buffer(
                        device: device, elements: 2 * storagePlane
                    )
                    let outputFlow = try Support.makeSharedFP16Buffer(
                        device: device, elements: 2 * storagePlane
                    )
                    let previousFlow = level == 0
                        ? zeroFlow : directions[direction][level - 1].outputFlowBuffer
                    let reference = direction == 0
                        ? referencePyramid[level] : supportPyramid[level]
                    let support = direction == 0
                        ? supportPyramid[level] : referencePyramid[level]
                    var shape = FusedSPyNetShape(
                        width: UInt32(size), height: UInt32(size), rowStride: UInt32(rowStride),
                        sourceFlowWidth: UInt32(level == 0 ? 2 : sizes[level - 1]),
                        sourceFlowHeight: UInt32(level == 0 ? 2 : sizes[level - 1]),
                        sourceFlowRowStride: UInt32(level == 0 ? 32 : max(sizes[level - 1], 32)),
                        firstLevel: level == 0 ? 1 : 0
                    )
                    let shapeBuffer = try Support.makeConstant(device: device, value: &shape)
                    let pipeline = levelPipelines[level]
                    let heap = levelHeaps[level]
                    directions[direction].append(FusedSPyNetLevel(
                        size: size, pipeline: pipeline, heap: heap,
                        featureBuffer: featureBuffer, residualBuffer: residualBuffer,
                        baseFlowBuffer: baseFlow, outputFlowBuffer: outputFlow,
                        shapeBuffer: shapeBuffer,
                        prepareArguments: try Support.makeComputeArguments(
                            device: device,
                            buffers: [reference, support, previousFlow, featureBuffer, baseFlow,
                                      shapeBuffer]
                        ),
                        mlArguments: try Support.makeMLArguments(
                            device: device, pipeline: pipeline,
                            resources: [
                                "features": featureTensor.gpuResourceID,
                                "output": residualTensor.gpuResourceID,
                            ]
                        ),
                        addArguments: try Support.makeComputeArguments(
                            device: device,
                            buffers: [baseFlow, residualBuffer, outputFlow, shapeBuffer]
                        )
                    ))
                    tensors += [featureTensor, residualTensor]
                    buffers += [featureBuffer, residualBuffer, baseFlow, outputFlow, shapeBuffer]
                    mutableTransient += [featureBuffer, residualBuffer, baseFlow, outputFlow]
                }
            }
            builtPairs.append(directions)
        }
        pairs = builtPairs
        pyramidArguments = builtPyramidArguments
        residentBuffers = buffers
        transientBuffers = mutableTransient
        heaps = mutableHeaps
        heldTensors = tensors
        backwardFlowBuffers = builtPairs.map { $0[0][5].outputFlowBuffer }
        forwardFlowBuffers = builtPairs.map { $0[1][5].outputFlowBuffer }
    }

    func addAllocations(to residencySet: any MTLResidencySet) {
        for buffer in residentBuffers { residencySet.addAllocation(buffer) }
        for heap in heaps { residencySet.addAllocation(heap) }
    }

    func initializeBuffers() {
        for buffer in transientBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
    }

    func encode(into commandBuffer: any MTL4CommandBuffer) throws {
        guard let downsample = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        for arguments in downsampleArguments {
            Support.dispatch1D(
                downsample, pipeline: downsamplePipeline, arguments: arguments,
                count: 3 * 64 * 64
            )
        }
        downsample.barrier(
            afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device
        )
        downsample.endEncoding()

        guard let pyramid = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        pyramid.barrier(
            afterQueueStages: .dispatch, beforeStages: .dispatch, visibilityOptions: .device
        )
        for arguments in pyramidArguments {
            pyramid.setComputePipelineState(pyramidPipeline)
            pyramid.setArgumentTable(arguments)
            pyramid.dispatchThreads(
                threadsPerGrid: MTLSize(width: 3 * 64 * 64, height: 6, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: min(pyramidPipeline.threadExecutionWidth * 4, 256), height: 1, depth: 1
                )
            )
        }
        pyramid.barrier(
            afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device
        )
        pyramid.endEncoding()

        for level in 0..<6 {
            guard let prepare = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            prepare.barrier(
                afterQueueStages: .dispatch, beforeStages: .dispatch, visibilityOptions: .device
            )
            for pair in pairs {
                for direction in pair {
                    let runtime = direction[level]
                    Support.dispatch1D(
                        prepare, pipeline: preparePipeline,
                        arguments: runtime.prepareArguments,
                        count: runtime.size * runtime.size
                    )
                }
            }
            prepare.barrier(
                afterStages: .dispatch, beforeQueueStages: .machineLearning,
                visibilityOptions: .device
            )
            prepare.endEncoding()

            for pair in pairs {
                for direction in pair {
                    let runtime = direction[level]
                    guard let network = commandBuffer.makeMachineLearningCommandEncoder() else {
                        throw DeformConvError.metalUnavailable
                    }
                    network.setPipelineState(runtime.pipeline)
                    network.setArgumentTable(runtime.mlArguments)
                    network.dispatchNetwork(intermediatesHeap: runtime.heap)
                    network.endEncoding()
                }
            }

            guard let add = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            add.barrier(
                afterQueueStages: .machineLearning, beforeStages: .dispatch,
                visibilityOptions: .device
            )
            for pair in pairs {
                for direction in pair {
                    let runtime = direction[level]
                    Support.dispatch1D(
                        add, pipeline: addPipeline, arguments: runtime.addArguments,
                        count: 2 * runtime.size * runtime.size
                    )
                }
            }
            add.barrier(
                afterStages: .dispatch, beforeQueueStages: .dispatch,
                visibilityOptions: .device
            )
            add.endEncoding()
        }
        _ = heldTensors
    }
}
