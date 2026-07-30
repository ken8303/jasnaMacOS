import Metal

@available(macOS 27.0, *)
enum Metal4GraphSupport {
    static func makeSharedFP16Buffer(device: MTLDevice, elements: Int) throws -> MTLBuffer {
        guard elements > 0,
              let buffer = device.makeBuffer(length: elements * 2, options: .storageModeShared)
        else { throw DeformConvError.metalUnavailable }
        return buffer
    }

    static func makePrivateBuffer(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
        guard bytes > 0,
              let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate)
        else { throw DeformConvError.metalUnavailable }
        return buffer
    }

    static func makeTensor(
        device: MTLDevice,
        dimensions: [Int]
    ) throws -> (any MTLTensor, MTLBuffer) {
        var strides = [Int]()
        var elements = 1
        for dimension in dimensions {
            strides.append(elements)
            elements *= dimension
        }
        let buffer = try makeSharedFP16Buffer(device: device, elements: elements)
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

    static func makeMLArguments(
        device: MTLDevice,
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

    static func makeComputeArguments(
        device: MTLDevice,
        buffers: [MTLBuffer]
    ) throws -> any MTL4ArgumentTable {
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = buffers.count
        let table = try device.makeArgumentTable(descriptor: descriptor)
        for (index, buffer) in buffers.enumerated() {
            table.setAddress(buffer.gpuAddress, index: index)
        }
        return table
    }

    static func makeConstant<T>(device: MTLDevice, value: inout T) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: 256, options: .storageModeShared) else {
            throw DeformConvError.metalUnavailable
        }
        withUnsafeBytes(of: &value) { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        return buffer
    }

    static func makeHeap(device: MTLDevice, size: Int) throws -> MTLHeap {
        let descriptor = MTLHeapDescriptor()
        descriptor.storageMode = .private
        descriptor.size = max(size, 4_096)
        guard let heap = device.makeHeap(descriptor: descriptor) else {
            throw DeformConvError.metalUnavailable
        }
        return heap
    }

    static func dispatch1D(
        _ encoder: any MTL4ComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        arguments: any MTL4ArgumentTable,
        count: Int,
        threads: Int? = nil,
        threadgroups: Bool = false
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setArgumentTable(arguments)
        let threadCount = threads ?? pipeline.threadExecutionWidth
        if threadgroups {
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
            )
        } else {
            encoder.dispatchThreads(
                threadsPerGrid: MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
            )
        }
    }
}
