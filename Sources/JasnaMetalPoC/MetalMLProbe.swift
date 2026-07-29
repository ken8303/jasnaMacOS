import Foundation
import Metal

struct MetalMLPipelineInfo {
    let bindingDescriptions: [String]
    let intermediatesHeapSize: Int
}

struct MetalMLBenchmarkResult {
    let medianMilliseconds: Double
    let minimumMilliseconds: Double
    let iterations: Int
}

struct MetalMLInteropResult {
    let maximumError: Float
    let gpuMilliseconds: Double
    let elementCount: Int
    let baselineMaximum: Float
    let baselineChecksum: Double
}

struct MetalMLGraphValidationResult {
    let packageCount: Int
    let bindingCount: Int
}

struct PropagationSmokeResult {
    let gpuMilliseconds: Double
    let elementCount: Int
    let maximumMagnitude: Float
    let repeatMaximumError: Float
    let checksum: Double
    let outputValues: [Float16]
}

struct ReconstructionSmokeResult {
    let reconstructionMilliseconds: Double
    let estimatedPipelineMilliseconds: Double
    let elementCount: Int
    let maximumMagnitude: Float
    let residualMaximumError: Float
    let repeatMaximumError: Float
    let checksum: Double
}

struct PropagationDeformConvShape {
    var batch: UInt32 = 1
    var inputChannels: UInt32 = 128
    var inputHeight: UInt32 = 64
    var inputWidth: UInt32 = 64
    var outputChannels: UInt32 = 64
    var outputHeight: UInt32 = 64
    var outputWidth: UInt32 = 64
    var kernelHeight: UInt32 = 3
    var kernelWidth: UInt32 = 3
    var padHeight: UInt32 = 1
    var padWidth: UInt32 = 1
    var strideHeight: UInt32 = 1
    var strideWidth: UInt32 = 1
    var dilationHeight: UInt32 = 1
    var dilationWidth: UInt32 = 1
    var groups: UInt32 = 1
    var offsetGroups: UInt32 = 16
    var hasMask: UInt32 = 1
}

final class CommitResult: @unchecked Sendable {
    private let lock = NSLock()
    private var _milliseconds: Double = 0
    private var _error: Error?

    func store(milliseconds: Double, error: Error?) {
        lock.lock()
        _milliseconds = milliseconds
        _error = error
        lock.unlock()
    }

    func load() -> (Double, Error?) {
        lock.lock()
        defer { lock.unlock() }
        return (_milliseconds, _error)
    }
}

func tensorExtents(_ values: [Int]) throws -> MTLTensorExtents {
    var mutableValues = values
    guard let result = mutableValues.withUnsafeMutableBufferPointer({
        MTLTensorExtents(__rank: $0.count, values: $0.baseAddress)
    }) else {
        throw DeformConvError.invalidShape
    }
    return result
}

private func extentValues(_ extents: MTLTensorExtents) -> [Int] {
    // The beta 4 Swift importer omits extentAtDimensionIndex(_:), although the
    // Objective-C method is present. Invoke its typed implementation directly.
    let selector = NSSelectorFromString("extentAtDimensionIndex:")
    typealias ExtentFunction = @convention(c) (AnyObject, Selector, UInt) -> Int
    let implementation = extents.method(for: selector)
    let function = unsafeBitCast(implementation, to: ExtentFunction.self)
    return (0..<extents.rank).map { function(extents, selector, UInt($0)) }
}

@available(macOS 26.0, *)
func makeMetalMLPipeline(
    device: MTLDevice,
    packageURL: URL
) throws -> any MTL4MachineLearningPipelineState {
    let library = try device.makeLibrary(URL: packageURL)

    let function = MTL4LibraryFunctionDescriptor()
    function.name = "main"
    function.library = library

    let options = MTL4PipelineOptions()
    options.shaderReflection = [.bindingInfo, .bufferTypeInfo]

    let descriptor = MTL4MachineLearningPipelineDescriptor()
    descriptor.label = packageURL.deletingPathExtension().lastPathComponent
    descriptor.machineLearningFunctionDescriptor = function
    descriptor.options = options

    let compilerDescriptor = MTL4CompilerDescriptor()
    compilerDescriptor.label = "Jasna Metal ML probe"
    let compiler = try device.makeCompiler(descriptor: compilerDescriptor)
    return try compiler.makeMachineLearningPipelineState(descriptor: descriptor)
}

@available(macOS 26.0, *)
func compileMetalMLPackage(device: MTLDevice, packageURL: URL) throws -> MetalMLPipelineInfo {
    let pipeline = try makeMetalMLPipeline(device: device, packageURL: packageURL)

    // Avoid NSObject.description here: Xcode 27 beta 4's private
    // MTLTensorBindingInternal implementation sends an invalid selector while
    // formatting an empty auxiliary-planes array.
    let descriptions: [String] = pipeline.reflection?.bindings.map { binding -> String in
        var detail = "index=\(binding.index), name=\(binding.name), type=\(binding.type.rawValue), access=\(binding.access.rawValue)"
        if let tensor = binding as? any MTLTensorBinding {
            let dimensions = tensor.dimensions.map(extentValues) ?? []
            detail += ", dimensions=\(dimensions), dataType=\(tensor.tensorDataType.rawValue)"
        }
        return detail
    } ?? []
    return MetalMLPipelineInfo(
        bindingDescriptions: descriptions,
        intermediatesHeapSize: pipeline.intermediatesHeapSize
    )
}

@available(macOS 26.0, *)
func validateMetalMLPackageGraph(
    device: MTLDevice,
    directoryURL: URL
) throws -> MetalMLGraphValidationResult {
    struct Specification {
        let package: String
        let inputName: String
        let input: [Int]
        let output: [Int]
    }
    var specifications = [
        Specification(package: "feature_extract", inputName: "frames", input: [256, 256, 3, 1], output: [64, 64, 64, 1]),
        Specification(package: "offset_backward_1", inputName: "conditions", input: [64, 64, 196, 1], output: [64, 64, 432, 1]),
        Specification(package: "offset_forward_1", inputName: "conditions", input: [64, 64, 196, 1], output: [64, 64, 432, 1]),
        Specification(package: "offset_backward_2", inputName: "conditions", input: [64, 64, 196, 1], output: [64, 64, 432, 1]),
        Specification(package: "offset_forward_2", inputName: "conditions", input: [64, 64, 196, 1], output: [64, 64, 432, 1]),
        Specification(package: "backbone_backward_1", inputName: "features", input: [64, 64, 128, 1], output: [64, 64, 64, 1]),
        Specification(package: "backbone_forward_1", inputName: "features", input: [64, 64, 192, 1], output: [64, 64, 64, 1]),
        Specification(package: "backbone_backward_2", inputName: "features", input: [64, 64, 256, 1], output: [64, 64, 64, 1]),
        Specification(package: "backbone_forward_2", inputName: "features", input: [64, 64, 320, 1], output: [64, 64, 64, 1]),
        Specification(package: "upsample", inputName: "features", input: [64, 64, 320, 1], output: [256, 256, 3, 1]),
    ]
    for (level, size) in [2, 4, 8, 16, 32, 64].enumerated() {
        specifications.append(Specification(
            package: "spynet_level_\(level)", inputName: "features",
            input: [size, size, 8, 1], output: [size, size, 2, 1]
        ))
    }

    var bindingCount = 0
    for specification in specifications {
        let packageURL = directoryURL.appendingPathComponent("\(specification.package).mtlpackage")
        let pipeline = try makeMetalMLPipeline(device: device, packageURL: packageURL)
        guard let bindings = pipeline.reflection?.bindings,
              let input = bindings.first(where: { $0.name == specification.inputName }) as? any MTLTensorBinding,
              let output = bindings.first(where: { $0.name == "output" }) as? any MTLTensorBinding,
              let inputDimensions = input.dimensions,
              let outputDimensions = output.dimensions,
              extentValues(inputDimensions) == specification.input,
              extentValues(outputDimensions) == specification.output,
              input.tensorDataType == .float16,
              output.tensorDataType == .float16
        else {
            throw DeformConvError.commandFailed("Metal ML package boundary mismatch: \(specification.package)")
        }
        bindingCount += bindings.count
    }
    return MetalMLGraphValidationResult(
        packageCount: specifications.count,
        bindingCount: bindingCount
    )
}

@available(macOS 27.0, *)
func benchmarkMetalMLPackage(
    device: MTLDevice,
    packageURL: URL,
    iterations: Int = 8
) throws -> MetalMLBenchmarkResult {
    let pipeline = try makeMetalMLPipeline(device: device, packageURL: packageURL)
    guard let bindings = pipeline.reflection?.bindings,
          !bindings.isEmpty
    else { throw DeformConvError.commandFailed("Metal ML package has no reflected bindings") }

    func makeTensor(binding: any MTLTensorBinding) throws -> (any MTLTensor, MTLBuffer) {
        guard let dimensions = binding.dimensions else {
            throw DeformConvError.commandFailed("dynamic tensor shapes are not supported by this benchmark")
        }
        let sizes = extentValues(dimensions)
        var strideValues = [Int]()
        var runningStride = 1
        for size in sizes {
            strideValues.append(runningStride)
            runningStride *= size
        }
        guard binding.tensorDataType == .float16,
              let buffer = device.makeBuffer(length: runningStride * 2, options: .storageModeShared)
        else {
            throw DeformConvError.commandFailed("buffer-backed benchmark currently requires FP16 tensors")
        }
        let descriptor = MTLTensorDescriptor()
        descriptor.dimensions = dimensions
        descriptor.strides = try tensorExtents(strideValues)
        descriptor.dataType = binding.tensorDataType
        descriptor.usage = [.compute, .machineLearning]
        descriptor.storageMode = .shared
        let attachments = MTLTensorBufferAttachments()
        attachments.setBuffer(buffer, offset: 0, for: .data)
        return (try device.makeTensor(descriptor: descriptor, attachments: attachments), buffer)
    }

    let argumentDescriptor = MTL4ArgumentTableDescriptor()
    argumentDescriptor.maxBufferBindCount = (bindings.map(\.index).max() ?? 0) + 1
    argumentDescriptor.initializeBindings = true
    let arguments = try device.makeArgumentTable(descriptor: argumentDescriptor)
    var tensors = [any MTLTensor]()
    var backingBuffers = [MTLBuffer]()
    for binding in bindings {
        guard let tensorBinding = binding as? any MTLTensorBinding else { continue }
        let (tensor, buffer) = try makeTensor(binding: tensorBinding)
        tensors.append(tensor)
        backingBuffers.append(buffer)
        arguments.setResource(tensor.gpuResourceID, bufferIndex: binding.index)
    }
    guard tensors.count == bindings.count else {
        throw DeformConvError.commandFailed("benchmark currently supports tensor bindings only")
    }

    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.storageMode = .private
    heapDescriptor.size = max(pipeline.intermediatesHeapSize, 4_096)
    guard let heap = device.makeHeap(descriptor: heapDescriptor),
          let queue = device.makeMTL4CommandQueue()
    else { throw DeformConvError.metalUnavailable }

    func execute() throws -> Double {
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }

        commandBuffer.beginCommandBuffer(allocator: allocator)
        guard let encoder = commandBuffer.makeMachineLearningCommandEncoder()
        else { throw DeformConvError.metalUnavailable }
        encoder.setPipelineState(pipeline)
        encoder.setArgumentTable(arguments)
        encoder.dispatchNetwork(intermediatesHeap: heap)
        encoder.endEncoding()
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
        return milliseconds
    }

    _ = try execute()
    _ = try execute()
    var samples = [Double]()
    for _ in 0..<max(iterations, 1) {
        samples.append(try execute())
    }
    samples.sort()
    return MetalMLBenchmarkResult(
        medianMilliseconds: samples[samples.count / 2],
        minimumMilliseconds: samples[0],
        iterations: samples.count
    )
}

@available(macOS 27.0, *)
func verifyMetalMLComputeInterop(
    device: MTLDevice,
    packageURL: URL
) throws -> MetalMLInteropResult {
    let pipeline = try makeMetalMLPipeline(device: device, packageURL: packageURL)
    guard let bindings = pipeline.reflection?.bindings,
          let outputBinding = bindings.first(where: { $0.name == "output" }) as? any MTLTensorBinding,
          let outputDimensions = outputBinding.dimensions
    else { throw DeformConvError.commandFailed("expected reflected output tensor") }

    let argumentDescriptor = MTL4ArgumentTableDescriptor()
    argumentDescriptor.maxBufferBindCount = (bindings.map(\.index).max() ?? 0) + 1
    argumentDescriptor.initializeBindings = true
    let mlArguments = try device.makeArgumentTable(descriptor: argumentDescriptor)
    var tensors = [any MTLTensor]()
    var buffersByIndex = [Int: MTLBuffer]()

    for binding in bindings {
        guard let tensorBinding = binding as? any MTLTensorBinding,
              let dimensions = tensorBinding.dimensions,
              tensorBinding.tensorDataType == .float16
        else { throw DeformConvError.commandFailed("interop requires static FP16 tensor bindings") }
        let sizes = extentValues(dimensions)
        var strides = [Int]()
        var elementCount = 1
        for size in sizes {
            strides.append(elementCount)
            elementCount *= size
        }
        guard let buffer = device.makeBuffer(length: elementCount * 2, options: .storageModeShared)
        else { throw DeformConvError.metalUnavailable }
        let descriptor = MTLTensorDescriptor()
        descriptor.dimensions = dimensions
        descriptor.strides = try tensorExtents(strides)
        descriptor.dataType = .float16
        descriptor.usage = [.compute, .machineLearning]
        descriptor.storageMode = .shared
        let attachments = MTLTensorBufferAttachments()
        attachments.setBuffer(buffer, offset: 0, for: .data)
        let tensor = try device.makeTensor(descriptor: descriptor, attachments: attachments)
        tensors.append(tensor)
        buffersByIndex[binding.index] = buffer
        mlArguments.setResource(tensor.gpuResourceID, bufferIndex: binding.index)
    }

    guard let outputBuffer = buffersByIndex[outputBinding.index] else {
        throw DeformConvError.commandFailed("output buffer was not allocated")
    }
    let outputCount = extentValues(outputDimensions).reduce(1, *)

    let addLibrary = try device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_one_fp16(device half *values [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
            values[gid] += half(1.0h);
        }
        """, options: nil)
    guard let addFunction = addLibrary.makeFunction(name: "add_one_fp16") else {
        throw DeformConvError.shaderResourceMissing
    }
    let addPipeline = try device.makeComputePipelineState(function: addFunction)
    let computeDescriptor = MTL4ArgumentTableDescriptor()
    computeDescriptor.maxBufferBindCount = 1
    let computeArguments = try device.makeArgumentTable(descriptor: computeDescriptor)
    computeArguments.setAddress(outputBuffer.gpuAddress, index: 0)

    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.storageMode = .private
    heapDescriptor.size = max(pipeline.intermediatesHeapSize, 4_096)
    guard let heap = device.makeHeap(descriptor: heapDescriptor),
          let queue = device.makeMTL4CommandQueue()
    else { throw DeformConvError.metalUnavailable }

    func execute(addOne: Bool) throws -> (Double, [Float16]) {
        for buffer in buffersByIndex.values {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        for binding in bindings where binding.name != "output" {
            guard let buffer = buffersByIndex[binding.index] else { continue }
            let count = buffer.length / MemoryLayout<Float16>.stride
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            for index in 0..<count {
                pointer[index] = Float16(Float((index * 17 + 11) % 251) / 250 - 0.5)
            }
        }
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        guard let mlEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        mlEncoder.setPipelineState(pipeline)
        mlEncoder.setArgumentTable(mlArguments)
        mlEncoder.dispatchNetwork(intermediatesHeap: heap)
        mlEncoder.endEncoding()
        if addOne {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            computeEncoder.barrier(
                afterQueueStages: .machineLearning,
                beforeStages: .dispatch,
                visibilityOptions: .device
            )
            computeEncoder.setComputePipelineState(addPipeline)
            computeEncoder.setArgumentTable(computeArguments)
            computeEncoder.dispatchThreads(
                threadsPerGrid: MTLSize(width: outputCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: addPipeline.threadExecutionWidth, height: 1, depth: 1)
            )
            computeEncoder.endEncoding()
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
        let pointer = outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputCount)
        return (milliseconds, Array(UnsafeBufferPointer(start: pointer, count: outputCount)))
    }

    let (_, baseline) = try execute(addOne: false)
    let (milliseconds, integrated) = try execute(addOne: true)
    var maximumError: Float = 0
    for (before, after) in zip(baseline, integrated) {
        maximumError = max(maximumError, abs((Float(before) + 1) - Float(after)))
    }
    guard maximumError <= 0.001 else {
        throw DeformConvError.commandFailed("Metal ML/compute interop error \(maximumError)")
    }
    let baselineMaximum = baseline.map { abs(Float($0)) }.max() ?? 0
    var baselineChecksum = 0.0
    for index in stride(from: 0, to: baseline.count, by: 257) {
        baselineChecksum += Double(Float(baseline[index]))
    }
    guard baselineMaximum > 0.001 else {
        throw DeformConvError.commandFailed("Metal ML produced an unexpectedly empty output")
    }
    return MetalMLInteropResult(
        maximumError: maximumError,
        gpuMilliseconds: milliseconds,
        elementCount: outputCount,
        baselineMaximum: baselineMaximum,
        baselineChecksum: baselineChecksum
    )
}

@available(macOS 27.0, *)
func verifyPropagationDirection(
    device: MTLDevice,
    modelsURL: URL,
    weightsURL: URL,
    direction: String,
    backboneInputChannels: Int
) throws -> PropagationSmokeResult {
    guard [128, 192, 256, 320].contains(backboneInputChannels) else {
        throw DeformConvError.commandFailed("unsupported backbone width \(backboneInputChannels)")
    }
    let plane = 64 * 64
    let outputCount = 64 * plane
    let prefixChannels = backboneInputChannels - 64
    let offsetPipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("offset_\(direction).mtlpackage")
    )
    let backbonePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("backbone_\(direction).mtlpackage")
    )

    func makeSharedBuffer(elements: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: elements * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else { throw DeformConvError.metalUnavailable }
        return buffer
    }

    func makeTensor(dimensions: [Int]) throws -> (any MTLTensor, MTLBuffer) {
        var strides = [Int]()
        var elementCount = 1
        for dimension in dimensions {
            strides.append(elementCount)
            elementCount *= dimension
        }
        let buffer = try makeSharedBuffer(elements: elementCount)
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

    let (conditionTensor, conditionBuffer) = try makeTensor(dimensions: [64, 64, 196, 1])
    let (rawTensor, rawBuffer) = try makeTensor(dimensions: [64, 64, 432, 1])
    let (backboneInputTensor, backboneInputBuffer) = try makeTensor(
        dimensions: [64, 64, backboneInputChannels, 1]
    )
    let (backboneOutputTensor, backboneOutputBuffer) = try makeTensor(dimensions: [64, 64, 64, 1])
    let flow1Buffer = try makeSharedBuffer(elements: 2 * plane)
    let flow2Buffer = try makeSharedBuffer(elements: 2 * plane)
    let deformInputBuffer = try makeSharedBuffer(elements: 128 * plane)
    let offsetBuffer = try makeSharedBuffer(elements: 288 * plane)
    let maskBuffer = try makeSharedBuffer(elements: 144 * plane)
    let alignedBuffer = try makeSharedBuffer(elements: outputCount)
    let prefixBuffer = try makeSharedBuffer(elements: prefixChannels * plane)
    let propagationOutputBuffer = try makeSharedBuffer(elements: outputCount)

    var planeValue = UInt32(plane)
    var prefixChannelsValue = UInt32(prefixChannels)
    var outputCountValue = UInt32(outputCount)
    var deformShape = PropagationDeformConvShape()
    func makeConstantBuffer<T>(_ value: inout T) throws -> MTLBuffer {
        // Keep address-bound constants in their own aligned MTL4 allocation.
        guard let buffer = device.makeBuffer(length: 256, options: .storageModeShared) else {
            throw DeformConvError.metalUnavailable
        }
        withUnsafeBytes(of: &value) { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        return buffer
    }
    let planeBuffer = try makeConstantBuffer(&planeValue)
    let prefixChannelsBuffer = try makeConstantBuffer(&prefixChannelsValue)
    let outputCountBuffer = try makeConstantBuffer(&outputCountValue)
    let deformShapeBuffer = try makeConstantBuffer(&deformShape)

    let weights = try DeformConvWeightSet(
        direction: direction,
        url: weightsURL.appendingPathComponent("\(direction).dcnfp16")
    ).makeBuffers(device: device)

    let mlOffsetArguments = try makeMLArguments(
        pipeline: offsetPipeline,
        resources: ["conditions": conditionTensor.gpuResourceID, "output": rawTensor.gpuResourceID]
    )
    let mlBackboneArguments = try makeMLArguments(
        pipeline: backbonePipeline,
        resources: [
            "features": backboneInputTensor.gpuResourceID,
            "output": backboneOutputTensor.gpuResourceID,
        ]
    )

    let options = MTLCompileOptions()
    options.mathMode = .safe
    let library = try device.makeLibrary(source: MetalShader.source, options: options)
    guard let transformFunction = library.makeFunction(name: "prepare_dcn_offsets_fp16"),
          let deformFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_tiled"),
          let assembleFunction = library.makeFunction(name: "assemble_propagation_backbone_fp16"),
          let residualFunction = library.makeFunction(name: "add_propagation_residual_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let transformPipeline = try device.makeComputePipelineState(function: transformFunction)
    let deformPipeline = try device.makeComputePipelineState(function: deformFunction)
    let assemblePipeline = try device.makeComputePipelineState(function: assembleFunction)
    let residualPipeline = try device.makeComputePipelineState(function: residualFunction)

    let transformArguments = try makeComputeArguments([
        rawBuffer, flow1Buffer, flow2Buffer, offsetBuffer, maskBuffer, planeBuffer,
    ])
    let deformArguments = try makeComputeArguments([
        deformInputBuffer, offsetBuffer, maskBuffer, weights.weight, weights.bias,
        alignedBuffer, deformShapeBuffer,
    ])
    let assembleArguments = try makeComputeArguments([
        prefixBuffer, alignedBuffer, backboneInputBuffer, planeBuffer, prefixChannelsBuffer,
    ])
    let residualArguments = try makeComputeArguments([
        alignedBuffer, backboneOutputBuffer, propagationOutputBuffer, outputCountBuffer,
    ])

    func makeHeap(size: Int) throws -> MTLHeap {
        let descriptor = MTLHeapDescriptor()
        descriptor.storageMode = .private
        descriptor.size = max(size, 4_096)
        guard let heap = device.makeHeap(descriptor: descriptor) else {
            throw DeformConvError.metalUnavailable
        }
        return heap
    }
    let offsetHeap = try makeHeap(size: offsetPipeline.intermediatesHeapSize)
    let backboneHeap = try makeHeap(size: backbonePipeline.intermediatesHeapSize)
    guard let queue = device.makeMTL4CommandQueue() else {
        throw DeformConvError.metalUnavailable
    }
    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "\(direction) propagation resources"
    residencyDescriptor.initialCapacity = 20
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    let residentBuffers = [
        conditionBuffer, rawBuffer, backboneInputBuffer, backboneOutputBuffer,
        flow1Buffer, flow2Buffer, deformInputBuffer, offsetBuffer, maskBuffer,
        alignedBuffer, prefixBuffer, propagationOutputBuffer, planeBuffer,
        prefixChannelsBuffer, outputCountBuffer, deformShapeBuffer, weights.weight, weights.bias,
    ]
    for buffer in residentBuffers { residencySet.addAllocation(buffer) }
    residencySet.addAllocation(offsetHeap)
    residencySet.addAllocation(backboneHeap)
    residencySet.commit()

    func fill(_ buffer: MTLBuffer, elements: Int, seed: Int, scale: Float) {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: elements)
        for index in 0..<elements {
            let centered = Float((index * 37 + seed) % 257) / 256 - 0.5
            pointer[index] = Float16(centered * scale)
        }
    }

    let transientBuffers = [
        rawBuffer, offsetBuffer, maskBuffer, alignedBuffer, backboneInputBuffer,
        backboneOutputBuffer, propagationOutputBuffer,
    ]
    func initializeBuffers() {
        for buffer in transientBuffers {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        fill(conditionBuffer, elements: 196 * plane, seed: 11, scale: 0.25)
        fill(flow1Buffer, elements: 2 * plane, seed: 23, scale: 0.20)
        fill(flow2Buffer, elements: 2 * plane, seed: 47, scale: 0.16)
        fill(deformInputBuffer, elements: 128 * plane, seed: 71, scale: 0.50)
        fill(prefixBuffer, elements: prefixChannels * plane, seed: 101, scale: 0.50)
    }

    func execute() throws -> (Double, [Float16]) {
        initializeBuffers()
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)

        guard let offsetEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        offsetEncoder.setPipelineState(offsetPipeline)
        offsetEncoder.setArgumentTable(mlOffsetArguments)
        offsetEncoder.dispatchNetwork(intermediatesHeap: offsetHeap)
        offsetEncoder.endEncoding()

        guard let propagationEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        propagationEncoder.barrier(
            afterQueueStages: .machineLearning,
            beforeStages: .dispatch,
            visibilityOptions: .device
        )
        propagationEncoder.setComputePipelineState(transformPipeline)
        propagationEncoder.setArgumentTable(transformArguments)
        propagationEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: 432 * plane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: transformPipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        propagationEncoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
        propagationEncoder.setComputePipelineState(deformPipeline)
        propagationEncoder.setArgumentTable(deformArguments)
        propagationEncoder.dispatchThreadgroups(
            threadgroupsPerGrid: MTLSize(width: plane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        propagationEncoder.barrier(
            afterEncoderStages: .dispatch,
            beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
        propagationEncoder.setComputePipelineState(assemblePipeline)
        propagationEncoder.setArgumentTable(assembleArguments)
        propagationEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: backboneInputChannels * plane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: assemblePipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        propagationEncoder.barrier(
            afterStages: .dispatch,
            beforeQueueStages: .machineLearning,
            visibilityOptions: .device
        )
        propagationEncoder.endEncoding()

        guard let backboneEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        backboneEncoder.setPipelineState(backbonePipeline)
        backboneEncoder.setArgumentTable(mlBackboneArguments)
        backboneEncoder.dispatchNetwork(intermediatesHeap: backboneHeap)
        backboneEncoder.barrier(
            afterStages: .machineLearning,
            beforeQueueStages: .dispatch,
            visibilityOptions: .device
        )
        backboneEncoder.endEncoding()

        guard let residualEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        residualEncoder.setComputePipelineState(residualPipeline)
        residualEncoder.setArgumentTable(residualArguments)
        residualEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: outputCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: residualPipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        residualEncoder.endEncoding()
        commandBuffer.endCommandBuffer()

        let semaphore = DispatchSemaphore(value: 0)
        let result = CommitResult()
        let commitOptions = MTL4CommitOptions()
        commitOptions.addFeedbackHandler { feedback in
            result.store(
                milliseconds: (feedback.gpuEndTime - feedback.gpuStartTime) * 1_000,
                error: feedback.error
            )
            semaphore.signal()
        }
        queue.commit([commandBuffer], options: commitOptions)
        semaphore.wait()
        let (milliseconds, error) = result.load()
        if let error { throw error }
        let pointer = propagationOutputBuffer.contents().bindMemory(
            to: Float16.self, capacity: outputCount
        )
        return (milliseconds, Array(UnsafeBufferPointer(start: pointer, count: outputCount)))
    }

    _ = try execute()
    let (_, first) = try execute()
    let (milliseconds, second) = try execute()
    var maximumMagnitude: Float = 0
    var repeatMaximumError: Float = 0
    var checksum = 0.0
    for index in 0..<outputCount {
        let value = Float(second[index])
        guard value.isFinite else {
            throw DeformConvError.commandFailed("propagation produced a non-finite value")
        }
        maximumMagnitude = max(maximumMagnitude, abs(value))
        repeatMaximumError = max(repeatMaximumError, abs(Float(first[index]) - value))
        if index.isMultiple(of: 257) { checksum += Double(value) }
    }
    guard maximumMagnitude > 0.001 else {
        func maximum(in buffer: MTLBuffer, count: Int) -> Float {
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            var result: Float = 0
            for index in 0..<count { result = max(result, abs(Float(pointer[index]))) }
            return result
        }
        throw DeformConvError.commandFailed(
            "propagation produced an unexpectedly empty output "
                + "(raw=\(maximum(in: rawBuffer, count: 432 * plane)), "
                + "mask=\(maximum(in: maskBuffer, count: 144 * plane)), "
                + "aligned=\(maximum(in: alignedBuffer, count: outputCount)), "
                + "backbone=\(maximum(in: backboneOutputBuffer, count: outputCount)))"
        )
    }
    guard repeatMaximumError <= 0.001 else {
        throw DeformConvError.commandFailed("propagation repeat error \(repeatMaximumError)")
    }
    return PropagationSmokeResult(
        gpuMilliseconds: milliseconds,
        elementCount: outputCount,
        maximumMagnitude: maximumMagnitude,
        repeatMaximumError: repeatMaximumError,
        checksum: checksum,
        outputValues: second
    )
}

@available(macOS 27.0, *)
func verifyReconstructedFrame(
    device: MTLDevice,
    modelsURL: URL,
    propagationResults: [String: PropagationSmokeResult],
    spatialValues: [Float16]? = nil,
    frameValues: [Float16]? = nil
) throws -> ReconstructionSmokeResult {
    let featurePlane = 64 * 64
    let framePlane = 256 * 256
    let frameElementCount = 3 * framePlane
    let directions = ["backward_1", "forward_1", "backward_2", "forward_2"]
    guard directions.allSatisfy({ propagationResults[$0]?.outputValues.count == 64 * featurePlane }) else {
        throw DeformConvError.commandFailed("reconstruction requires all four propagation outputs")
    }
    guard spatialValues == nil || spatialValues?.count == 64 * featurePlane,
          frameValues == nil || frameValues?.count == frameElementCount
    else { throw DeformConvError.invalidShape }
    let propagationMilliseconds = directions.reduce(0.0) {
        $0 + (propagationResults[$1]?.gpuMilliseconds ?? 0)
    }
    let featurePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("feature_extract.mtlpackage")
    )
    let upsamplePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("upsample.mtlpackage")
    )

    func makeSharedBuffer(elements: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: elements * 2, options: .storageModeShared) else {
            throw DeformConvError.metalUnavailable
        }
        return buffer
    }
    func makeTensor(dimensions: [Int]) throws -> (any MTLTensor, MTLBuffer) {
        var strides = [Int]()
        var elementCount = 1
        for dimension in dimensions {
            strides.append(elementCount)
            elementCount *= dimension
        }
        let buffer = try makeSharedBuffer(elements: elementCount)
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
    func makeBuffer(values: [Float16]) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let buffer = device.makeBuffer(
                    bytes: base, length: bytes.count, options: .storageModeShared
                  )
            else { throw DeformConvError.metalUnavailable }
            return buffer
        }
    }

    let (frameTensor, frameBuffer) = try makeTensor(dimensions: [256, 256, 3, 1])
    let (spatialTensor, spatialBuffer) = try makeTensor(dimensions: [64, 64, 64, 1])
    let (reconstructionTensor, reconstructionBuffer) = try makeTensor(dimensions: [64, 64, 320, 1])
    let (predictedTensor, predictedBuffer) = try makeTensor(dimensions: [256, 256, 3, 1])
    let restoredBuffer = try makeSharedBuffer(elements: frameElementCount)
    let branchBuffers = try directions.map { direction in
        try makeBuffer(values: propagationResults[direction]!.outputValues)
    }
    var featurePlaneValue = UInt32(featurePlane)
    var frameElementCountValue = UInt32(frameElementCount)
    let featurePlaneBuffer = try makeConstantBuffer(&featurePlaneValue)
    let frameElementCountBuffer = try makeConstantBuffer(&frameElementCountValue)

    let featureArguments = try makeMLArguments(
        pipeline: featurePipeline,
        resources: ["frames": frameTensor.gpuResourceID, "output": spatialTensor.gpuResourceID]
    )
    let upsampleArguments = try makeMLArguments(
        pipeline: upsamplePipeline,
        resources: ["features": reconstructionTensor.gpuResourceID, "output": predictedTensor.gpuResourceID]
    )
    let library = try device.makeLibrary(source: MetalShader.source, options: nil)
    guard let assembleFunction = library.makeFunction(name: "assemble_reconstruction_fp16"),
          let residualFunction = library.makeFunction(name: "add_frame_residual_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let assemblePipeline = try device.makeComputePipelineState(function: assembleFunction)
    let residualPipeline = try device.makeComputePipelineState(function: residualFunction)
    let assembleArguments = try makeComputeArguments(
        [spatialBuffer] + branchBuffers + [reconstructionBuffer, featurePlaneBuffer]
    )
    let residualArguments = try makeComputeArguments([
        predictedBuffer, frameBuffer, restoredBuffer, frameElementCountBuffer,
    ])

    func makeHeap(size: Int) throws -> MTLHeap {
        let descriptor = MTLHeapDescriptor()
        descriptor.storageMode = .private
        descriptor.size = max(size, 4_096)
        guard let heap = device.makeHeap(descriptor: descriptor) else {
            throw DeformConvError.metalUnavailable
        }
        return heap
    }
    let featureHeap = try makeHeap(size: featurePipeline.intermediatesHeapSize)
    let upsampleHeap = try makeHeap(size: upsamplePipeline.intermediatesHeapSize)
    guard let queue = device.makeMTL4CommandQueue() else {
        throw DeformConvError.metalUnavailable
    }
    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "complete-frame reconstruction resources"
    residencyDescriptor.initialCapacity = 16
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    for buffer in [
        frameBuffer, spatialBuffer, reconstructionBuffer, predictedBuffer,
        restoredBuffer, featurePlaneBuffer, frameElementCountBuffer,
    ] + branchBuffers {
        residencySet.addAllocation(buffer)
    }
    residencySet.addAllocation(featureHeap)
    residencySet.addAllocation(upsampleHeap)
    residencySet.commit()

    func initializeBuffers() {
        for buffer in [spatialBuffer, reconstructionBuffer, predictedBuffer, restoredBuffer] {
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        }
        let frame = frameBuffer.contents().bindMemory(to: Float16.self, capacity: frameElementCount)
        if let frameValues {
            frameValues.withUnsafeBufferPointer { source in
                frame.update(from: source.baseAddress!, count: source.count)
            }
        } else {
            for index in 0..<frameElementCount {
                frame[index] = Float16(Float((index * 29 + 17) % 1021) / 1020)
            }
        }
        if let spatialValues {
            let spatial = spatialBuffer.contents().bindMemory(
                to: Float16.self, capacity: 64 * featurePlane
            )
            spatialValues.withUnsafeBufferPointer { source in
                spatial.update(from: source.baseAddress!, count: source.count)
            }
        }
    }
    func execute() throws -> (Double, [Float16]) {
        initializeBuffers()
        guard let allocator = device.makeCommandAllocator(),
              let commandBuffer = device.makeCommandBuffer()
        else { throw DeformConvError.metalUnavailable }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)

        if spatialValues == nil {
            guard let featureEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            featureEncoder.setPipelineState(featurePipeline)
            featureEncoder.setArgumentTable(featureArguments)
            featureEncoder.dispatchNetwork(intermediatesHeap: featureHeap)
            featureEncoder.endEncoding()
        }

        guard let assembleEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        if spatialValues == nil {
            assembleEncoder.barrier(
                afterQueueStages: .machineLearning,
                beforeStages: .dispatch,
                visibilityOptions: .device
            )
        }
        assembleEncoder.setComputePipelineState(assemblePipeline)
        assembleEncoder.setArgumentTable(assembleArguments)
        assembleEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: 320 * featurePlane, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: assemblePipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        assembleEncoder.barrier(
            afterStages: .dispatch,
            beforeQueueStages: .machineLearning,
            visibilityOptions: .device
        )
        assembleEncoder.endEncoding()

        guard let upsampleEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        upsampleEncoder.setPipelineState(upsamplePipeline)
        upsampleEncoder.setArgumentTable(upsampleArguments)
        upsampleEncoder.dispatchNetwork(intermediatesHeap: upsampleHeap)
        upsampleEncoder.barrier(
            afterStages: .machineLearning,
            beforeQueueStages: .dispatch,
            visibilityOptions: .device
        )
        upsampleEncoder.endEncoding()

        guard let residualEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeformConvError.metalUnavailable
        }
        residualEncoder.setComputePipelineState(residualPipeline)
        residualEncoder.setArgumentTable(residualArguments)
        residualEncoder.dispatchThreads(
            threadsPerGrid: MTLSize(width: frameElementCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: residualPipeline.threadExecutionWidth, height: 1, depth: 1)
        )
        residualEncoder.endEncoding()
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
        let pointer = restoredBuffer.contents().bindMemory(to: Float16.self, capacity: frameElementCount)
        return (milliseconds, Array(UnsafeBufferPointer(start: pointer, count: frameElementCount)))
    }

    _ = try execute()
    let (_, first) = try execute()
    let (milliseconds, second) = try execute()
    let input = frameBuffer.contents().bindMemory(to: Float16.self, capacity: frameElementCount)
    let predicted = predictedBuffer.contents().bindMemory(to: Float16.self, capacity: frameElementCount)
    var maximumMagnitude: Float = 0
    var residualMaximumError: Float = 0
    var repeatMaximumError: Float = 0
    var checksum = 0.0
    for index in 0..<frameElementCount {
        let value = Float(second[index])
        guard value.isFinite else {
            throw DeformConvError.commandFailed("reconstruction produced a non-finite value")
        }
        maximumMagnitude = max(maximumMagnitude, abs(value))
        residualMaximumError = max(
            residualMaximumError,
            abs((Float(predicted[index]) + Float(input[index])) - value)
        )
        repeatMaximumError = max(repeatMaximumError, abs(Float(first[index]) - value))
        if index.isMultiple(of: 257) { checksum += Double(value) }
    }
    guard maximumMagnitude > 0.001,
          residualMaximumError <= 0.001,
          repeatMaximumError <= 0.001
    else {
        throw DeformConvError.commandFailed(
            "reconstruction validation failed (max=\(maximumMagnitude), "
                + "residual=\(residualMaximumError), repeat=\(repeatMaximumError))"
        )
    }
    return ReconstructionSmokeResult(
        reconstructionMilliseconds: milliseconds,
        estimatedPipelineMilliseconds: propagationMilliseconds + milliseconds,
        elementCount: frameElementCount,
        maximumMagnitude: maximumMagnitude,
        residualMaximumError: residualMaximumError,
        repeatMaximumError: repeatMaximumError,
        checksum: checksum
    )
}
