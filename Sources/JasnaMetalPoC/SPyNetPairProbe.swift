import Foundation
import Metal

struct SPyNetPairResult {
    let medianMilliseconds: Double
    let minimumMilliseconds: Double
    let maximumMilliseconds: Double
    let iterations: Int
    let elementCount: Int
    let backwardMaximum: Float
    let forwardMaximum: Float
    let repeatMaximumError: Float
    let oracleMaximumError: Float
    let backwardChecksum: Double
    let forwardChecksum: Double
    let backwardFlow: [Float16]
    let forwardFlow: [Float16]
}

private struct SPyNetGraphPrepareShape {
    var width: UInt32
    var height: UInt32
    var rowStride: UInt32
    var sourceFlowWidth: UInt32
    var sourceFlowHeight: UInt32
    var sourceFlowRowStride: UInt32
    var firstLevel: UInt32
}

@available(macOS 27.0, *)
private struct SPyNetLevelRuntime {
    let size: Int
    let rowStride: Int
    let pipeline: any MTL4MachineLearningPipelineState
    let heap: MTLHeap
    let tensors: [any MTLTensor]
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
func verifySPyNetPair(
    device: MTLDevice,
    modelsURL: URL,
    oracleURL: URL,
    inputVariant: Int = 0,
    validateOracle: Bool = true,
    referenceInput: [Float16]? = nil,
    supportInput: [Float16]? = nil
) throws -> SPyNetPairResult {
    let sizes = [2, 4, 8, 16, 32, 64]
    guard (referenceInput == nil) == (supportInput == nil),
          referenceInput?.count == nil || referenceInput?.count == 3 * 64 * 64,
          supportInput?.count == nil || supportInput?.count == 3 * 64 * 64
    else { throw DeformConvError.invalidShape }

    func makeBuffer(elements: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: elements * 2, options: .storageModeShared) else {
            throw DeformConvError.metalUnavailable
        }
        return buffer
    }
    func makeTensor(dimensions: [Int], rowStride: Int) throws -> (any MTLTensor, MTLBuffer) {
        guard dimensions.count == 4 else { throw DeformConvError.invalidShape }
        let strides = [
            1,
            rowStride,
            rowStride * dimensions[1],
            rowStride * dimensions[1] * dimensions[2],
        ]
        let elements = strides[3] * dimensions[3]
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
            throw DeformConvError.commandFailed("SPyNet package has no reflected bindings")
        }
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = (bindings.map(\.index).max() ?? 0) + 1
        descriptor.initializeBindings = true
        let table = try device.makeArgumentTable(descriptor: descriptor)
        for (name, resource) in resources {
            guard let binding = bindings.first(where: { $0.name == name }) else {
                throw DeformConvError.commandFailed("missing SPyNet binding: \(name)")
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

    let referenceFrame = try makeBuffer(elements: 3 * 64 * 64)
    let supportFrame = try makeBuffer(elements: 3 * 64 * 64)
    let referencePyramid = try sizes.map { try makeBuffer(elements: 3 * $0 * $0) }
    let supportPyramid = try sizes.map { try makeBuffer(elements: 3 * $0 * $0) }
    let zeroFlow = try makeBuffer(elements: 2 * 32 * 2)

    let library = try device.makeLibrary(source: MetalShader.source, options: nil)
    guard let pyramidFunction = library.makeFunction(name: "spynet_build_pyramid_pair_fp16"),
          let prepareFunction = library.makeFunction(name: "spynet_prepare_padded_fp16"),
          let addFunction = library.makeFunction(name: "spynet_add_flow_padded_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let pyramidPipeline = try device.makeComputePipelineState(function: pyramidFunction)
    let preparePipeline = try device.makeComputePipelineState(function: prepareFunction)
    let addPipeline = try device.makeComputePipelineState(function: addFunction)
    var pyramidBindings = [referenceFrame, supportFrame]
    for index in sizes.indices {
        pyramidBindings.append(referencePyramid[index])
        pyramidBindings.append(supportPyramid[index])
    }
    let pyramidArguments = try makeComputeArguments(pyramidBindings)

    let levelPipelines = try sizes.indices.map { level in
        try makeMetalMLPipeline(
            device: device,
            packageURL: modelsURL.appendingPathComponent("spynet_level_\(level).mtlpackage")
        )
    }
    var directions = [[SPyNetLevelRuntime](), [SPyNetLevelRuntime]()]
    for direction in 0..<2 {
        for (level, size) in sizes.enumerated() {
            let rowStride = max(size, 32)
            let storagePlane = rowStride * size
            let (featureTensor, featureBuffer) = try makeTensor(
                dimensions: [size, size, 8, 1], rowStride: rowStride
            )
            let (residualTensor, residualBuffer) = try makeTensor(
                dimensions: [size, size, 2, 1], rowStride: rowStride
            )
            let baseFlowBuffer = try makeBuffer(elements: 2 * storagePlane)
            let outputFlowBuffer = try makeBuffer(elements: 2 * storagePlane)
            let previousFlow = level == 0 ? zeroFlow : directions[direction][level - 1].outputFlowBuffer
            let reference = direction == 0 ? referencePyramid[level] : supportPyramid[level]
            let support = direction == 0 ? supportPyramid[level] : referencePyramid[level]
            var shape = SPyNetGraphPrepareShape(
                width: UInt32(size),
                height: UInt32(size),
                rowStride: UInt32(rowStride),
                sourceFlowWidth: UInt32(level == 0 ? 2 : sizes[level - 1]),
                sourceFlowHeight: UInt32(level == 0 ? 2 : sizes[level - 1]),
                sourceFlowRowStride: UInt32(level == 0 ? 32 : max(sizes[level - 1], 32)),
                firstLevel: level == 0 ? 1 : 0
            )
            let shapeBuffer = try makeConstant(&shape)
            let pipeline = levelPipelines[level]
            directions[direction].append(SPyNetLevelRuntime(
                size: size,
                rowStride: rowStride,
                pipeline: pipeline,
                heap: try makeHeap(size: pipeline.intermediatesHeapSize),
                tensors: [featureTensor, residualTensor],
                featureBuffer: featureBuffer,
                residualBuffer: residualBuffer,
                baseFlowBuffer: baseFlowBuffer,
                outputFlowBuffer: outputFlowBuffer,
                shapeBuffer: shapeBuffer,
                prepareArguments: try makeComputeArguments([
                    reference, support, previousFlow, featureBuffer, baseFlowBuffer, shapeBuffer,
                ]),
                mlArguments: try makeMLArguments(
                    pipeline: pipeline,
                    resources: ["features": featureTensor.gpuResourceID, "output": residualTensor.gpuResourceID]
                ),
                addArguments: try makeComputeArguments([
                    baseFlowBuffer, residualBuffer, outputFlowBuffer, shapeBuffer,
                ])
            ))
        }
    }

    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "bidirectional SPyNet graph"
    residencyDescriptor.initialCapacity = 80
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    var residentBuffers = [referenceFrame, supportFrame, zeroFlow]
        + referencePyramid + supportPyramid
    for direction in directions {
        for level in direction {
            residentBuffers += [
                level.featureBuffer, level.residualBuffer, level.baseFlowBuffer,
                level.outputFlowBuffer, level.shapeBuffer,
            ]
            residencySet.addAllocation(level.heap)
        }
    }
    for buffer in residentBuffers { residencySet.addAllocation(buffer) }
    residencySet.commit()
    guard let queue = device.makeMTL4CommandQueue() else {
        throw DeformConvError.metalUnavailable
    }

    let transientBuffers = [referenceFrame, supportFrame, zeroFlow]
        + referencePyramid + supportPyramid
        + directions.flatMap { direction in
            direction.flatMap {
                [$0.featureBuffer, $0.residualBuffer, $0.baseFlowBuffer, $0.outputFlowBuffer]
            }
        }
    func initializeBuffers() {
        for buffer in transientBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        let reference = referenceFrame.contents().bindMemory(to: Float16.self, capacity: 3 * 4096)
        let support = supportFrame.contents().bindMemory(to: Float16.self, capacity: 3 * 4096)
        if let referenceInput, let supportInput {
            referenceInput.withUnsafeBufferPointer { source in
                reference.update(from: source.baseAddress!, count: source.count)
            }
            supportInput.withUnsafeBufferPointer { source in
                support.update(from: source.baseAddress!, count: source.count)
            }
            return
        }
        for index in 0..<(3 * 4096) {
            if inputVariant == 0 {
                reference[index] = Float16(Float((index * 29 + 17) % 1021) / 1020)
                support[index] = Float16(Float((index * 43 + 31) % 1019) / 1018)
            } else {
                // The second adjacent pair starts with the first pair's support
                // frame so a three-frame clip has a consistent middle frame.
                reference[index] = Float16(Float((index * 43 + 31) % 1019) / 1018)
                support[index] = Float16(Float((index * 53 + 47) % 1013) / 1012)
            }
        }
    }

    func execute() throws -> (Double, [[Float16]]) {
        initializeBuffers()
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)

        guard let pyramidEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        pyramidEncoder.setComputePipelineState(pyramidPipeline)
        pyramidEncoder.setArgumentTable(pyramidArguments)
        pyramidEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: 3 * 64 * 64, height: 6, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(pyramidPipeline.threadExecutionWidth * 4, 256), height: 1, depth: 1)
        )
        pyramidEncoder.barrier(
            afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device
        )
        pyramidEncoder.endEncoding()

        for level in sizes.indices {
            guard let prepareEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            prepareEncoder.barrier(
                afterQueueStages: .dispatch, beforeStages: .dispatch, visibilityOptions: .device
            )
            for direction in 0..<2 {
                let runtime = directions[direction][level]
                prepareEncoder.setComputePipelineState(preparePipeline)
                prepareEncoder.setArgumentTable(runtime.prepareArguments)
                prepareEncoder.dispatchThreads(
                    threadsPerGrid: MTLSize(width: runtime.size * runtime.size, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: preparePipeline.threadExecutionWidth, height: 1, depth: 1)
                )
            }
            prepareEncoder.barrier(
                afterStages: .dispatch, beforeQueueStages: .machineLearning, visibilityOptions: .device
            )
            prepareEncoder.endEncoding()

            for direction in 0..<2 {
                let runtime = directions[direction][level]
                guard let mlEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                mlEncoder.setPipelineState(runtime.pipeline)
                mlEncoder.setArgumentTable(runtime.mlArguments)
                mlEncoder.dispatchNetwork(intermediatesHeap: runtime.heap)
                mlEncoder.endEncoding()
            }

            guard let addEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            addEncoder.barrier(
                afterQueueStages: .machineLearning, beforeStages: .dispatch, visibilityOptions: .device
            )
            for direction in 0..<2 {
                let runtime = directions[direction][level]
                addEncoder.setComputePipelineState(addPipeline)
                addEncoder.setArgumentTable(runtime.addArguments)
                addEncoder.dispatchThreads(
                    threadsPerGrid: MTLSize(width: 2 * runtime.size * runtime.size, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: addPipeline.threadExecutionWidth, height: 1, depth: 1)
                )
            }
            addEncoder.barrier(
                afterStages: .dispatch, beforeQueueStages: .dispatch, visibilityOptions: .device
            )
            addEncoder.endEncoding()
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
        let outputs = directions.map { direction -> [Float16] in
            let runtime = direction[5]
            let pointer = runtime.outputFlowBuffer.contents().bindMemory(
                to: Float16.self, capacity: 2 * runtime.rowStride * runtime.size
            )
            var packed = [Float16](repeating: 0, count: 2 * runtime.size * runtime.size)
            let plane = runtime.size * runtime.size
            let storagePlane = runtime.rowStride * runtime.size
            for channel in 0..<2 {
                for y in 0..<runtime.size {
                    for x in 0..<runtime.size {
                        packed[channel * plane + y * runtime.size + x]
                            = pointer[channel * storagePlane + y * runtime.rowStride + x]
                    }
                }
            }
            return packed
        }
        return (milliseconds, outputs)
    }

    _ = try execute()
    _ = try execute()
    let (firstMilliseconds, first) = try execute()
    var samples = [firstMilliseconds]
    var last = first
    for _ in 1..<7 {
        let (milliseconds, outputs) = try execute()
        samples.append(milliseconds)
        last = outputs
    }
    samples.sort()
    var maxima = [Float](repeating: 0, count: 2)
    var checksums = [Double](repeating: 0, count: 2)
    var repeatError: Float = 0
    var oracleError: Float = 0
    let oracleNames = ["backward", "forward"]
    for direction in 0..<2 {
        let oracle: [Float16]
        if validateOracle {
            let oracleData = try Data(
                contentsOf: oracleURL.appendingPathComponent("\(oracleNames[direction]).f16")
            )
            guard oracleData.count == last[direction].count * 2 else {
                throw DeformConvError.commandFailed("invalid SPyNet oracle size")
            }
            oracle = oracleData.withUnsafeBytes {
                Array($0.bindMemory(to: UInt16.self)).map(Float16.init(bitPattern:))
            }
        } else {
            oracle = []
        }
        for index in last[direction].indices {
            let value = Float(last[direction][index])
            guard value.isFinite else {
                throw DeformConvError.commandFailed("SPyNet produced a non-finite flow")
            }
            maxima[direction] = max(maxima[direction], abs(value))
            repeatError = max(repeatError, abs(Float(first[direction][index]) - value))
            if validateOracle {
                oracleError = max(oracleError, abs(Float(oracle[index]) - value))
            }
            if index.isMultiple(of: 257) { checksums[direction] += Double(value) }
        }
    }
    guard maxima.allSatisfy({ $0 > 0.001 }),
          repeatError <= 0.001,
          oracleError <= 0.05
    else {
        throw DeformConvError.commandFailed(
            "SPyNet validation failed (maxima=\(maxima), repeat=\(repeatError), oracle=\(oracleError))"
        )
    }
    return SPyNetPairResult(
        medianMilliseconds: samples[samples.count / 2],
        minimumMilliseconds: samples[0],
        maximumMilliseconds: samples[samples.count - 1],
        iterations: samples.count,
        elementCount: 2 * 2 * 4096,
        backwardMaximum: maxima[0],
        forwardMaximum: maxima[1],
        repeatMaximumError: repeatError,
        oracleMaximumError: oracleError,
        backwardChecksum: checksums[0],
        forwardChecksum: checksums[1],
        backwardFlow: last[0],
        forwardFlow: last[1]
    )
}
