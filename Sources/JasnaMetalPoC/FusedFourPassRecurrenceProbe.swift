import Foundation
import Metal

struct FusedGraphComponentTimings: Codable, Equatable, Sendable {
    let featureExtraction: Double
    let spynet: Double
    let backward1: Double
    let forward1: Double
    let backward2: Double
    let forward2: Double
    let reconstruction: Double

    var total: Double {
        featureExtraction + spynet + backward1 + forward1 + backward2 + forward2
            + reconstruction
    }

    func isPlausible(totalGPUMilliseconds: Double) -> Bool {
        let values = [
            featureExtraction, spynet, backward1, forward1, backward2, forward2,
            reconstruction,
        ]
        return totalGPUMilliseconds > 0
            && values.allSatisfy { $0.isFinite && $0 >= 0 }
            && total > 0
            && total <= totalGPUMilliseconds * 1.10
    }
}

struct FusedGraphPropagationTimings: Codable, Equatable, Sendable {
    let offsetNetwork: Double
    let tensorPreparation: Double
    let dcnTransform: Double
    let dcnGather: Double
    let dcnGEMM: Double
    let backboneNetwork: Double
    let residual: Double
    var branches: [FusedGraphBranchPropagationTimings] = []
    var offsetLocality: [FusedGraphBranchOffsetLocality] = []

    var total: Double {
        offsetNetwork + tensorPreparation + dcnTransform + dcnGather + dcnGEMM
            + backboneNetwork + residual
    }

    func isPlausible(totalPropagationMilliseconds: Double) -> Bool {
        let values = [
            offsetNetwork, tensorPreparation, dcnTransform, dcnGather, dcnGEMM,
            backboneNetwork, residual,
        ]
        return totalPropagationMilliseconds > 0
            && values.allSatisfy { $0.isFinite && $0 >= 0 }
            && offsetNetwork > 0
            && dcnTransform > 0
            && dcnGather > 0
            && dcnGEMM > 0
            && backboneNetwork > 0
            && total > 0
            && total <= totalPropagationMilliseconds * 1.10
    }
}

struct FusedGraphBranchOffsetLocality: Codable, Equatable, Sendable {
    let name: String
    let sampleCount: UInt64
    let meanMagnitude: Double
    let maximumMagnitude: Double
    let fractionAbove2: Double
    let fractionAbove4: Double
    let fractionAbove8: Double
    let outOfBoundsFraction: Double
    let meanNeighborDelta: Double

    var isPlausible: Bool {
        let finite = [
            meanMagnitude, maximumMagnitude, fractionAbove2, fractionAbove4,
            fractionAbove8, outOfBoundsFraction, meanNeighborDelta,
        ].allSatisfy(\.isFinite)
        return !name.isEmpty && sampleCount > 0 && finite
            && meanMagnitude >= 0 && maximumMagnitude >= meanMagnitude
            && meanNeighborDelta >= 0
            && (0...1).contains(fractionAbove2)
            && (0...1).contains(fractionAbove4)
            && (0...1).contains(fractionAbove8)
            && (0...1).contains(outOfBoundsFraction)
            && fractionAbove8 <= fractionAbove4
            && fractionAbove4 <= fractionAbove2
    }
}

struct FusedGraphBranchPropagationTimings: Codable, Equatable, Sendable {
    let name: String
    let offsetNetwork: Double
    let tensorPreparation: Double
    let dcnTransform: Double
    let dcnGather: Double
    let dcnGEMM: Double
    let backboneNetwork: Double
    let residual: Double

    var total: Double {
        offsetNetwork + tensorPreparation + dcnTransform + dcnGather + dcnGEMM
            + backboneNetwork + residual
    }

    func isPlausible(totalBranchMilliseconds: Double) -> Bool {
        let values = [
            offsetNetwork, tensorPreparation, dcnTransform, dcnGather, dcnGEMM,
            backboneNetwork, residual,
        ]
        return !name.isEmpty
            && totalBranchMilliseconds > 0
            && values.allSatisfy { $0.isFinite && $0 >= 0 }
            && offsetNetwork > 0
            && dcnTransform > 0
            && dcnGather > 0
            && dcnGEMM > 0
            && backboneNetwork > 0
            && total > 0
            && total <= totalBranchMilliseconds * 1.10
    }
}

private enum PropagationTimingCategory: Int, CaseIterable {
    case offsetNetwork
    case tensorPreparation
    case dcnTransform
    case dcnGather
    case dcnGEMM
    case backboneNetwork
    case residual
}

func dcnOffsetLocalityEnabled(
    detailedTelemetry: Bool,
    environment: [String: String] = ProcessInfo.processInfo.environment
) -> Bool {
    detailedTelemetry && environment["JASNA_DCN_OFFSET_LOCALITY"] == "1"
}

func dcnChannelLastGatherEnabled(
    environment: [String: String] = ProcessInfo.processInfo.environment
) -> Bool {
    environment["JASNA_DCN_CHANNEL_LAST_GATHER"] == "1"
}

struct FusedFourPassRecurrenceResult {
    var graphCacheHit: Bool
    var graphLookupMilliseconds: Double
    var graphSetupMilliseconds: Double
    let statistics: BenchmarkStatistics
    let componentTimings: FusedGraphComponentTimings?
    let propagationTimings: FusedGraphPropagationTimings?
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

    func restore(
        _ frames: [[Float16]],
        reportLockWait: ((Double) -> Void)? = nil
    ) throws -> FusedFourPassRecurrenceResult {
        let lockStarted = ContinuousClock.now
        lock.lock()
        defer { lock.unlock() }
        reportLockWait?(SideBySideRestoration.elapsedMilliseconds(since: lockStarted))
        return try execute(frames)
    }
}

@available(macOS 27.0, *)
private final class ProductionFusedGraphCache: @unchecked Sendable {
    static let shared = ProductionFusedGraphCache()

    private let lock = NSLock()
    private struct Entry {
        let graphKey: String
        let runner: ProductionFusedGraphRunner
    }
    private var runners = [String: Entry]()
    private var initiallyWarmedFamilies = Set<String>()

    private init() {}

    func runner(for key: String, family: String) -> ProductionFusedGraphRunner? {
        lock.lock()
        defer { lock.unlock() }
        guard let entry = runners[family] else { return nil }
        guard entry.graphKey == key else {
            // Only one temporal shape remains resident for each device/model/
            // batch family. Evict the old shape before constructing the
            // replacement to keep graph memory bounded.
            runners.removeValue(forKey: family)
            return nil
        }
        return entry.runner
    }

    func retain(
        _ runner: ProductionFusedGraphRunner,
        for key: String,
        family: String
    ) {
        lock.lock()
        defer { lock.unlock() }
        if runners[family]?.graphKey != key {
            runners[family] = Entry(graphKey: key, runner: runner)
        }
    }

    func needsInitialWarmup(for family: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !initiallyWarmedFamilies.contains(family)
    }

    func markInitialWarmupComplete(for family: String) {
        lock.lock()
        defer { lock.unlock() }
        initiallyWarmedFamilies.insert(family)
    }
}

func productionGraphReuseEligible(
    frameCount: Int,
    warmupCount: Int,
    measurementCount: Int,
    collectDiagnostics: Bool,
    hasFlowOracle: Bool,
    hasStagedPropagation: Bool,
    hasStagedRestoration: Bool
) -> Bool {
    // Full windows normally contain 30 output frames, or 35 inputs after the
    // quality-preserving temporal warm-up. Sparse regions that enter or leave
    // during a window legitimately use any shorter length down to three. The
    // scheduler groups equal lengths, so retaining the current length avoids
    // rebuilding identical partial graphs while the family cache still holds
    // only one temporal shape at a time.
    frameCount >= 3
        && warmupCount == 0
        && measurementCount == 1
        && !collectDiagnostics
        && !hasFlowOracle
        && !hasStagedPropagation
        && !hasStagedRestoration
}

func coldGraphWarmupRequired(
    setupMilliseconds: Double,
    reusableProductionGraph: Bool,
    isFirstGraphForFamily: Bool = false,
    environment: [String: String] = ProcessInfo.processInfo.environment
) -> Bool {
    guard reusableProductionGraph else { return false }
    if isFirstGraphForFamily,
       environment["JASNA_INITIAL_GRAPH_WARMUP"] != "0"
    {
        return true
    }
    let configured = Double(environment["JASNA_COLD_GRAPH_WARMUP_THRESHOLD_MS"] ?? "")
    let threshold = configured ?? 5_000
    return threshold > 0 && setupMilliseconds >= threshold
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
    let localityArguments: [any MTL4ArgumentTable]
    let gemmArguments: any MTL4ArgumentTable
    let assemblyArguments: [any MTL4ArgumentTable]
    let residualArguments: [any MTL4ArgumentTable]
    let offsetHeap: MTLHeap
    let backboneHeap: MTLHeap
    let backboneInputTensor: any MTLTensor
    let backboneInputBuffer: MTLBuffer
    let weightBuffer: MTLBuffer
    let biasBuffer: MTLBuffer
    let localityBuffer: MTLBuffer?
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
    collectDiagnostics: Bool = true,
    batch: Int = 1
) throws -> FusedFourPassRecurrenceResult {
    typealias Support = Metal4GraphSupport
    let traceGraph = ProcessInfo.processInfo.environment["JASNA_GRAPH_TRACE"] == "1"
    let traceGraphPhases = ProcessInfo.processInfo.environment[
        "JASNA_GRAPH_PHASE_TELEMETRY"
    ] == "1"
    let traceGraphPropagation = ProcessInfo.processInfo.environment[
        "JASNA_GRAPH_PROPAGATION_TELEMETRY"
    ] == "1"
    let traceGraphComponents = traceGraphPropagation || ProcessInfo.processInfo.environment[
        "JASNA_GRAPH_COMPONENT_TELEMETRY"
    ] == "1"
    let collectOffsetLocality = dcnOffsetLocalityEnabled(
        detailedTelemetry: traceGraphPropagation
    )
    let useChannelLastGather = dcnChannelLastGatherEnabled()
    let frameCount = inputFrames.count
    func trace(_ message: String) {
        if traceGraph {
            // Write directly instead of using buffered print, so a hung build
            // leaves its last phase visible when stdout is redirected through tee.
            SideBySideRestoration.report(
                "Fused graph [batch \(batch), frames \(frameCount)]: \(message)"
            )
        }
    }
    func elapsedMilliseconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = start.duration(to: .now).components
        return Double(elapsed.seconds) * 1_000
            + Double(elapsed.attoseconds) / 1_000_000_000_000_000
    }
    let plane = 64 * 64
    let featureCount = batch * 64 * plane
    let frameElements = batch * 3 * 256 * 256
    let flowCount = frameCount - 1
    let hasFlowOracle = !backwardFlows.isEmpty || !forwardFlows.isEmpty
    let hasStagedPropagation = !stagedBranchFrames.isEmpty
    let hasStagedRestoration = !stagedRestoredFrames.isEmpty
    let reusableProductionGraph = ProcessInfo.processInfo.environment[
        "JASNA_RETAINED_GRAPH"
    ] != "0" && productionGraphReuseEligible(
        frameCount: frameCount,
        warmupCount: warmupCount,
        measurementCount: measurementCount,
        collectDiagnostics: collectDiagnostics,
        hasFlowOracle: hasFlowOracle,
        hasStagedPropagation: hasStagedPropagation,
        hasStagedRestoration: hasStagedRestoration
    )
    let diagnosticKey = (collectOffsetLocality ? ":offset-locality" : "")
        + (useChannelLastGather ? ":channel-last-gather" : "")
    let productionGraphFamily = "\(device.registryID):"
        + "\(modelsURL.standardizedFileURL.path):"
        + "\(weightsURL.standardizedFileURL.path):batch\(batch)\(diagnosticKey)"
    let productionGraphKey = "\(device.registryID):\(modelsURL.standardizedFileURL.path):"
        + "\(weightsURL.standardizedFileURL.path):\(frameCount):batch\(batch)"
        + diagnosticKey
    let branchSpecs: [(String, PropagationDirection)] = [
        ("backward_1", .backward), ("forward_1", .forward),
        ("backward_2", .backward), ("forward_2", .forward),
    ]
    guard batch > 0,
          frameCount >= 3,
          warmupCount >= 0,
          measurementCount > 0,
          batch == 1 || (!collectDiagnostics && !hasFlowOracle
              && !hasStagedPropagation && !hasStagedRestoration),
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

    trace("looking up retained graph")
    let lookupStarted = ContinuousClock.now
    let retainedRunner = reusableProductionGraph
        ? ProductionFusedGraphCache.shared.runner(
            for: productionGraphKey,
            family: productionGraphFamily
        ) : nil
    let lookupMilliseconds = elapsedMilliseconds(since: lookupStarted)
    if let retainedRunner {
        trace("retained graph hit; waiting for runner lock")
        var reused = try retainedRunner.restore(inputFrames) { lockMilliseconds in
            if traceGraphPhases {
                SideBySideRestoration.report(
                    "Fused graph reuse: batch \(batch), frames \(frameCount), lookup "
                        + "\(String(format: "%.3f", lookupMilliseconds)) ms, runner lock "
                        + "\(String(format: "%.3f", lockMilliseconds)) ms"
                )
            }
        }
        reused.graphCacheHit = true
        reused.graphLookupMilliseconds = lookupMilliseconds
        reused.graphSetupMilliseconds = 0
        return reused
    }

    let setupStarted = ContinuousClock.now
    trace("loading batch-\(batch) Metal ML pipelines")
    var activeInputFrames = inputFrames

    let featurePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("feature_extract.mtlpackage")
    )
    trace("feature pipeline ready")
    let upsamplePipeline = try makeMetalMLPipeline(
        device: device,
        packageURL: modelsURL.appendingPathComponent("upsample.mtlpackage")
    )
    trace("reconstruction pipeline ready")
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
    trace("feature and reconstruction heaps ready")
    let featureHeaps = [MTLHeap](repeating: featureHeap, count: frameCount)
    let upsampleHeaps = [MTLHeap](repeating: upsampleHeap, count: frameCount)
    trace("feature and reconstruction pipelines ready")
    var heldTensors = [any MTLTensor]()
    var frameBuffers = [MTLBuffer]()
    var spatialBuffers = [MTLBuffer]()
    var featureArguments = [any MTL4ArgumentTable]()
    for _ in 0..<frameCount {
        let (frameTensor, frameBuffer) = try Support.makeTensor(
            device: device, dimensions: [256, 256, 3, batch]
        )
        let (spatialTensor, spatialBuffer) = try Support.makeTensor(
            device: device, dimensions: [64, 64, 64, batch]
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
        device: device, dimensions: [64, 64, 196, batch]
    )
    let (rawTensor, rawBuffer) = try Support.makeTensor(
        device: device, dimensions: [64, 64, 432, batch]
    )
    let (backboneOutputTensor, backboneOutputBuffer) = try Support.makeTensor(
        device: device, dimensions: [64, 64, 64, batch]
    )
    heldTensors += [conditionTensor, rawTensor, backboneOutputTensor]

    let zeroFeatureBuffer = try Support.makeSharedFP16Buffer(device: device, elements: featureCount)
    let flow2Buffer = try Support.makeSharedFP16Buffer(
        device: device, elements: batch * 2 * plane
    )
    let deformInputBuffer = try Support.makeSharedFP16Buffer(
        device: device, elements: batch * 128 * plane
    )
    let offsetBuffer = try Support.makeSharedFP16Buffer(
        device: device, elements: batch * 288 * plane
    )
    let maskBuffer = try Support.makeSharedFP16Buffer(
        device: device, elements: batch * 144 * plane
    )
    let alignedBuffer = try Support.makeSharedFP16Buffer(device: device, elements: featureCount)
    let gatheredBuffer = try Support.makePrivateBuffer(
        device: device, bytes: batch * plane * 128 * 9 * 2
    )
    let channelLastInputBuffer: MTLBuffer? = useChannelLastGather
        ? try Support.makePrivateBuffer(device: device, bytes: deformInputBuffer.length)
        : nil
    let propagationBuffers = try (0..<4).map { _ in
        try (0..<frameCount).map { _ in
            try Support.makeSharedFP16Buffer(device: device, elements: featureCount)
        }
    }
    var firstShape = ThreeFramePrepareShape(hasSecondOrder: 0, batch: UInt32(batch))
    var secondShape = ThreeFramePrepareShape(hasSecondOrder: 1, batch: UInt32(batch))
    var planeValue = UInt32(plane)
    var prefixChannelsValue = UInt32(64)
    var featureCountValue = UInt32(featureCount)
    var frameElementsValue = UInt32(frameElements)
    var deformShape = PropagationDeformConvShape(batch: UInt32(batch))
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
          let localityFunction = library.makeFunction(
              name: "summarize_dcn_offset_locality_fp16"
          ),
          let transposeFunction = library.makeFunction(
              name: "transpose_dcn_input_channel_last_fp16"
          ),
          let gatherFunction = library.makeFunction(name: "deform_conv2d_fp16_jasna_gather"),
          let channelLastGatherFunction = library.makeFunction(
              name: "deform_conv2d_fp16_jasna_gather_channel_last"
          ),
          let gemmFunction = library.makeFunction(
              name: "deform_conv2d_fp16_jasna_simdgroup_gemm_fused"
          ),
          let simpleAssemblyFunction = library.makeFunction(name: "assemble_propagation_backbone_fp16"),
          let temporalAssemblyFunction = library.makeFunction(name: "assemble_temporal_backbone_fp16"),
          let residualFunction = library.makeFunction(name: "add_propagation_residual_fp16"),
          let reconstructionAssemblyFunction = library.makeFunction(name: "assemble_reconstruction_fp16"),
          let frameResidualFunction = library.makeFunction(name: "add_frame_residual_fp16"),
          let nonFiniteFunction = library.makeFunction(name: "flag_non_finite_fp16")
    else { throw DeformConvError.shaderResourceMissing }
    let cache = MetalResourceCache.shared
    let accumulatePipeline = try cache.computePipeline(device: device, function: accumulateFunction)
    let preparePipeline = try cache.computePipeline(device: device, function: prepareFunction)
    let transformPipeline = try cache.computePipeline(device: device, function: transformFunction)
    let localityPipeline: MTLComputePipelineState? = collectOffsetLocality
        ? try cache.computePipeline(device: device, function: localityFunction) : nil
    let transposePipeline: MTLComputePipelineState? = useChannelLastGather
        ? try cache.computePipeline(device: device, function: transposeFunction) : nil
    let gatherPipeline = try cache.computePipeline(device: device, function: gatherFunction)
    let channelLastGatherPipeline: MTLComputePipelineState? = useChannelLastGather
        ? try cache.computePipeline(device: device, function: channelLastGatherFunction) : nil
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
    let nonFinitePipeline = try cache.computePipeline(device: device, function: nonFiniteFunction)
    let spynetGraph = try FusedSPyNetClipGraph(
        device: device, modelsURL: modelsURL, library: library,
        sourceFrames: frameBuffers, batch: batch
    )
    trace("SPyNet graph ready")
    let backwardFlowBuffers = spynetGraph.backwardFlowBuffers
    let forwardFlowBuffers = spynetGraph.forwardFlowBuffers
    let gatherArguments = try Support.makeComputeArguments(
        device: device,
        buffers: [deformInputBuffer, offsetBuffer, maskBuffer, gatheredBuffer, deformShapeBuffer]
    )
    let transposeArguments: (any MTL4ArgumentTable)? = if let channelLastInputBuffer {
        try Support.makeComputeArguments(
            device: device,
            buffers: [deformInputBuffer, channelLastInputBuffer, deformShapeBuffer]
        )
    } else {
        nil
    }
    let channelLastGatherArguments: (any MTL4ArgumentTable)? = if let channelLastInputBuffer {
        try Support.makeComputeArguments(
            device: device,
            buffers: [
                channelLastInputBuffer, offsetBuffer, maskBuffer, gatheredBuffer,
                deformShapeBuffer,
            ]
        )
    } else {
        nil
    }
    let localityStepBuffers: [MTLBuffer]
    if collectOffsetLocality {
        localityStepBuffers = try (0..<flowCount).map { step in
            var value = UInt32(step)
            return try Support.makeConstant(device: device, value: &value)
        }
    } else {
        localityStepBuffers = []
    }

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
            device: device, dimensions: [64, 64, backboneInputChannels, batch]
        )
        heldTensors.append(backboneInputTensor)
        let checkpoint = try MetalResourceCache.shared.deformConvWeightBuffers(
            device: device,
            direction: name,
            url: weightsURL.appendingPathComponent("\(name).dcnfp16")
        )
        let localityBuffer: MTLBuffer? = if collectOffsetLocality {
            device.makeBuffer(length: flowCount * 8 * 4, options: .storageModeShared)
        } else {
            nil
        }
        if collectOffsetLocality, localityBuffer == nil {
            throw DeformConvError.metalUnavailable
        }
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
        var localityArguments = [any MTL4ArgumentTable]()
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
            if let localityBuffer {
                localityArguments.append(try Support.makeComputeArguments(
                    device: device,
                    buffers: [
                        offsetBuffer, localityBuffer, deformShapeBuffer,
                        localityStepBuffers[step - 1],
                    ]
                ))
            }
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
            transformArguments: transformArguments, localityArguments: localityArguments,
            gemmArguments: gemmArguments,
            assemblyArguments: assemblyArguments,
            residualArguments: residualArguments,
            offsetHeap: try Support.makeHeap(device: device, size: offsetPipeline.intermediatesHeapSize),
            backboneHeap: try Support.makeHeap(device: device, size: backbonePipeline.intermediatesHeapSize),
            backboneInputTensor: backboneInputTensor,
            backboneInputBuffer: backboneInputBuffer,
            weightBuffer: checkpoint.weight, biasBuffer: checkpoint.bias,
            localityBuffer: localityBuffer
        ))
        trace("branch \(name) ready")
    }

    var reconstructionBuffers = [MTLBuffer]()
    var predictedBuffers = [MTLBuffer]()
    var restoredBuffers = [MTLBuffer]()
    var reconstructionAssemblyArguments = [any MTL4ArgumentTable]()
    var upsampleArguments = [any MTL4ArgumentTable]()
    var frameResidualArguments = [any MTL4ArgumentTable]()
    var nonFiniteArguments = [any MTL4ArgumentTable]()
    guard let nonFiniteFlagBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
    else { throw DeformConvError.metalUnavailable }
    for frame in 0..<frameCount {
        let (reconstructionTensor, reconstructionBuffer) = try Support.makeTensor(
            device: device, dimensions: [64, 64, 320, batch]
        )
        let (predictedTensor, predictedBuffer) = try Support.makeTensor(
            device: device, dimensions: [256, 256, 3, batch]
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
        nonFiniteArguments.append(try Support.makeComputeArguments(
            device: device,
            buffers: [restoredBuffer, nonFiniteFlagBuffer, frameElementsBuffer]
        ))
    }
    let residencyDescriptor = MTLResidencySetDescriptor()
    residencyDescriptor.label = "fused three-frame feature-to-restored graph"
    residencyDescriptor.initialCapacity = 144
    let residencySet = try device.makeResidencySet(descriptor: residencyDescriptor)
    var sharedBuffers = frameBuffers
    sharedBuffers += spatialBuffers
    sharedBuffers += propagationBuffers.flatMap { $0 }
    sharedBuffers += backwardFlowBuffers
    sharedBuffers += forwardFlowBuffers
    sharedBuffers += branchIndexBuffers
    sharedBuffers += [
            conditionBuffer, rawBuffer, backboneOutputBuffer, zeroFeatureBuffer, flow2Buffer,
            deformInputBuffer, offsetBuffer, maskBuffer, alignedBuffer, gatheredBuffer,
            firstShapeBuffer, secondShapeBuffer, planeBuffer,
            prefixChannelsBuffer, featureCountBuffer, frameElementsBuffer, deformShapeBuffer,
        ]
    sharedBuffers += localityStepBuffers
    if let channelLastInputBuffer { sharedBuffers.append(channelLastInputBuffer) }
    sharedBuffers += reconstructionBuffers
    sharedBuffers += predictedBuffers
    sharedBuffers += restoredBuffers
    sharedBuffers.append(nonFiniteFlagBuffer)
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
        if let localityBuffer = branch.localityBuffer {
            residencySet.addAllocation(localityBuffer)
        }
    }
    residencySet.commit()
    trace("residency set committed")
    guard let queue = device.makeMTL4CommandQueue(),
          let commandAllocator = device.makeCommandAllocator(),
          let commandBuffer = device.makeCommandBuffer()
    else {
        throw DeformConvError.metalUnavailable
    }
    var commandAllocatorHasCompletedSubmission = false
    let componentCounterHeap: (any MTL4CounterHeap)? = if traceGraphComponents {
        try {
            let descriptor = MTL4CounterHeapDescriptor()
            descriptor.type = .timestamp
            descriptor.count = 8
            return try device.makeCounterHeap(descriptor: descriptor)
        }()
    } else {
        nil
    }
    let propagationCounterCountPerBranch = 1 + 3 + flowCount * 8
    let propagationCounterHeaps: [any MTL4CounterHeap]? = if traceGraphPropagation {
        try (0..<4).map { _ in
            let descriptor = MTL4CounterHeapDescriptor()
            descriptor.type = .timestamp
            descriptor.count = propagationCounterCountPerBranch
            return try device.makeCounterHeap(descriptor: descriptor)
        }
    } else {
        nil
    }

    let transientBuffers = spatialBuffers + propagationBuffers.flatMap { $0 } + [
        conditionBuffer, rawBuffer, backboneOutputBuffer, zeroFeatureBuffer, flow2Buffer,
        deformInputBuffer, offsetBuffer, maskBuffer, alignedBuffer,
    ] + reconstructionBuffers + predictedBuffers + restoredBuffers
    let setupMilliseconds = elapsedMilliseconds(since: setupStarted)
    if traceGraphPhases {
        SideBySideRestoration.report(
            "Fused graph setup: batch \(batch), frames \(frameCount), cache "
                + "\(reusableProductionGraph ? "miss" : "ineligible/disabled"), lookup "
                + "\(String(format: "%.3f", lookupMilliseconds)) ms, setup "
                + "\(String(format: "%.3f", setupMilliseconds)) ms"
        )
    }
    var buffersNeedInitialClear = true
    func initializeBuffers() {
        // SPyNet's small tensors use padded row strides; their padding must
        // remain zero because Metal ML can read the complete physical rows.
        spynetGraph.initializeBuffers()
        nonFiniteFlagBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        for buffer in branches.compactMap(\.localityBuffer) {
            buffer.contents().initializeMemory(
                as: UInt8.self, repeating: 0, count: buffer.length
            )
        }
        // The main graph fully overwrites every mutable destination, while
        // zeroFeatureBuffer is immutable. Clear these shared buffers once
        // instead of rewriting roughly 190 MB before every retained-graph run.
        if buffersNeedInitialClear {
            for buffer in transientBuffers {
                buffer.contents().initializeMemory(
                    as: UInt8.self, repeating: 0, count: buffer.length
                )
            }
            buffersNeedInitialClear = false
        }
        for frame in 0..<frameCount {
            activeInputFrames[frame].withUnsafeBufferPointer { source in
                frameBuffers[frame].contents().bindMemory(to: Float16.self, capacity: frameElements)
                    .update(from: source.baseAddress!, count: frameElements)
            }
        }
    }

    func execute() throws -> (
        Double, [[[Float16]]], [[Float16]], [[Float16]], FusedGraphComponentTimings?,
        FusedGraphPropagationTimings?
    ) {
        trace("waiting for Metal ML execution lock")
        let lockStarted = ContinuousClock.now
        MetalResourceCache.shared.beginMachineLearningExecution()
        defer { MetalResourceCache.shared.endMachineLearningExecution() }
        let lockMilliseconds = elapsedMilliseconds(since: lockStarted)
        trace("execution lock acquired; uploading inputs")
        let uploadStarted = ContinuousClock.now
        initializeBuffers()
        let uploadMilliseconds = elapsedMilliseconds(since: uploadStarted)
        trace("encoding command buffer")
        let encodingStarted = ContinuousClock.now
        if commandAllocatorHasCompletedSubmission { commandAllocator.reset() }
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        commandBuffer.useResidencySet(residencySet)
        var propagationTimestampIndices = [Int](repeating: 0, count: 4)
        var propagationCategories = [[PropagationTimingCategory]](
            repeating: [], count: 4
        )
        func beginPropagationTiming(branchIndex: Int) {
            guard let propagationCounterHeaps else { return }
            commandBuffer.writeTimestamp(
                counterHeap: propagationCounterHeaps[branchIndex], index: 0
            )
            propagationTimestampIndices[branchIndex] = 1
        }
        func markPropagation(_ category: PropagationTimingCategory, branchIndex: Int) {
            guard let propagationCounterHeaps else { return }
            let timestampIndex = propagationTimestampIndices[branchIndex]
            commandBuffer.writeTimestamp(
                counterHeap: propagationCounterHeaps[branchIndex], index: timestampIndex
            )
            propagationTimestampIndices[branchIndex] += 1
            propagationCategories[branchIndex].append(category)
        }
        if let componentCounterHeap {
            commandBuffer.writeTimestamp(counterHeap: componentCounterHeap, index: 0)
        }
        for index in 0..<frameCount {
            guard let encoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            encoder.setPipelineState(featurePipeline)
            encoder.setArgumentTable(featureArguments[index])
            encoder.dispatchNetwork(intermediatesHeap: featureHeaps[index])
            encoder.endEncoding()
        }
        if let componentCounterHeap {
            commandBuffer.writeTimestamp(counterHeap: componentCounterHeap, index: 1)
        }
        try spynetGraph.encode(into: commandBuffer)
        if let componentCounterHeap {
            commandBuffer.writeTimestamp(counterHeap: componentCounterHeap, index: 2)
        }
        for (branchIndex, branch) in branches.enumerated() {
            beginPropagationTiming(branchIndex: branchIndex)
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
                count: batch * branch.backboneInputChannels * plane
            )
            initialAssembly.barrier(
                afterStages: .dispatch, beforeQueueStages: .machineLearning,
                visibilityOptions: .device
            )
            initialAssembly.endEncoding()
            markPropagation(.tensorPreparation, branchIndex: branchIndex)
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
            markPropagation(.backboneNetwork, branchIndex: branchIndex)
            guard let initialResidual = commandBuffer.makeComputeCommandEncoder() else {
                throw DeformConvError.metalUnavailable
            }
            Support.dispatch1D(
                initialResidual, pipeline: residualPipeline,
                arguments: branch.initialResidualArguments, count: featureCount
            )
            initialResidual.endEncoding()
            markPropagation(.residual, branchIndex: branchIndex)

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
                    arguments: branch.accumulateArguments[step], count: batch * 2 * plane
                )
                preparation.barrier(
                    afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                    visibilityOptions: .device
                )
                Support.dispatch1D(
                    preparation, pipeline: preparePipeline,
                    arguments: branch.prepareArguments[step], count: batch * 196 * plane
                )
                preparation.barrier(
                    afterStages: .dispatch, beforeQueueStages: .machineLearning,
                    visibilityOptions: .device
                )
                preparation.endEncoding()
                markPropagation(.tensorPreparation, branchIndex: branchIndex)
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
                markPropagation(.offsetNetwork, branchIndex: branchIndex)
                if traceGraphPropagation {
                    guard let transform = commandBuffer.makeComputeCommandEncoder() else {
                        throw DeformConvError.metalUnavailable
                    }
                    Support.dispatch1D(
                        transform, pipeline: transformPipeline,
                        arguments: branch.transformArguments[step],
                        count: batch * 432 * plane
                    )
                    transform.barrier(
                        afterStages: .dispatch, beforeQueueStages: .dispatch,
                        visibilityOptions: .device
                    )
                    transform.endEncoding()
                    markPropagation(.dcnTransform, branchIndex: branchIndex)

                    if let localityPipeline {
                        guard branch.localityArguments.indices.contains(step),
                              let locality = commandBuffer.makeComputeCommandEncoder()
                        else { throw DeformConvError.metalUnavailable }
                        Support.dispatch1D(
                            locality, pipeline: localityPipeline,
                            arguments: branch.localityArguments[step],
                            count: batch * 144 * plane
                        )
                        locality.barrier(
                            afterStages: .dispatch, beforeQueueStages: .dispatch,
                            visibilityOptions: .device
                        )
                        locality.endEncoding()
                    }

                    guard let gather = commandBuffer.makeComputeCommandEncoder() else {
                        throw DeformConvError.metalUnavailable
                    }
                    gather.barrier(
                        afterQueueStages: .dispatch, beforeStages: .dispatch,
                        visibilityOptions: .device
                    )
                    if let transposePipeline, let transposeArguments,
                       let channelLastGatherPipeline, let channelLastGatherArguments {
                        gather.setComputePipelineState(transposePipeline)
                        gather.setArgumentTable(transposeArguments)
                        gather.dispatchThreads(
                            threadsPerGrid: MTLSize(
                                width: plane, height: batch * 128, depth: 1
                            ),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
                        )
                        gather.barrier(
                            afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                            visibilityOptions: .device
                        )
                        Support.dispatch1D(
                            gather, pipeline: channelLastGatherPipeline,
                            arguments: channelLastGatherArguments,
                            count: batch * plane, threads: 128, threadgroups: true
                        )
                    } else {
                        Support.dispatch1D(
                            gather, pipeline: gatherPipeline,
                            arguments: gatherArguments,
                            count: batch * plane, threads: 128, threadgroups: true
                        )
                    }
                    gather.barrier(
                        afterStages: .dispatch, beforeQueueStages: .dispatch,
                        visibilityOptions: .device
                    )
                    gather.endEncoding()
                    markPropagation(.dcnGather, branchIndex: branchIndex)

                    guard let gemm = commandBuffer.makeComputeCommandEncoder() else {
                        throw DeformConvError.metalUnavailable
                    }
                    gemm.barrier(
                        afterQueueStages: .dispatch, beforeStages: .dispatch,
                        visibilityOptions: .device
                    )
                    Support.dispatch1D(
                        gemm, pipeline: gemmPipeline, arguments: branch.gemmArguments,
                        count: batch * plane / MetalShader.fusedGEMMRowsPerTile,
                        threads: 8 * gemmPipeline.threadExecutionWidth, threadgroups: true
                    )
                    gemm.barrier(
                        afterStages: .dispatch, beforeQueueStages: .dispatch,
                        visibilityOptions: .device
                    )
                    gemm.endEncoding()
                    markPropagation(.dcnGEMM, branchIndex: branchIndex)

                    guard let assembly = commandBuffer.makeComputeCommandEncoder() else {
                        throw DeformConvError.metalUnavailable
                    }
                    assembly.barrier(
                        afterQueueStages: .dispatch, beforeStages: .dispatch,
                        visibilityOptions: .device
                    )
                    Support.dispatch1D(
                        assembly, pipeline: assemblyPipeline,
                        arguments: branch.assemblyArguments[step],
                        count: batch * branch.backboneInputChannels * plane
                    )
                    assembly.barrier(
                        afterStages: .dispatch, beforeQueueStages: .machineLearning,
                        visibilityOptions: .device
                    )
                    assembly.endEncoding()
                    markPropagation(.tensorPreparation, branchIndex: branchIndex)
                } else {
                    guard let alignment = commandBuffer.makeComputeCommandEncoder() else {
                        throw DeformConvError.metalUnavailable
                    }
                    Support.dispatch1D(
                        alignment, pipeline: transformPipeline,
                        arguments: branch.transformArguments[step],
                        count: batch * 432 * plane
                    )
                    alignment.barrier(
                        afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                        visibilityOptions: .device
                    )
                    if let transposePipeline, let transposeArguments,
                       let channelLastGatherPipeline, let channelLastGatherArguments {
                        alignment.setComputePipelineState(transposePipeline)
                        alignment.setArgumentTable(transposeArguments)
                        alignment.dispatchThreads(
                            threadsPerGrid: MTLSize(
                                width: plane, height: batch * 128, depth: 1
                            ),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
                        )
                        alignment.barrier(
                            afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                            visibilityOptions: .device
                        )
                        Support.dispatch1D(
                            alignment, pipeline: channelLastGatherPipeline,
                            arguments: channelLastGatherArguments,
                            count: batch * plane, threads: 128, threadgroups: true
                        )
                    } else {
                        Support.dispatch1D(
                            alignment, pipeline: gatherPipeline,
                            arguments: gatherArguments,
                            count: batch * plane, threads: 128, threadgroups: true
                        )
                    }
                    alignment.barrier(
                        afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                        visibilityOptions: .device
                    )
                    Support.dispatch1D(
                        alignment, pipeline: gemmPipeline,
                        arguments: branch.gemmArguments,
                        count: batch * plane / MetalShader.fusedGEMMRowsPerTile,
                        threads: 8 * gemmPipeline.threadExecutionWidth, threadgroups: true
                    )
                    alignment.barrier(
                        afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                        visibilityOptions: .device
                    )
                    Support.dispatch1D(
                        alignment, pipeline: assemblyPipeline,
                        arguments: branch.assemblyArguments[step],
                        count: batch * branch.backboneInputChannels * plane
                    )
                    alignment.barrier(
                        afterStages: .dispatch, beforeQueueStages: .machineLearning,
                        visibilityOptions: .device
                    )
                    alignment.endEncoding()
                }
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
                markPropagation(.backboneNetwork, branchIndex: branchIndex)
                guard let residual = commandBuffer.makeComputeCommandEncoder() else {
                    throw DeformConvError.metalUnavailable
                }
                Support.dispatch1D(
                    residual, pipeline: residualPipeline,
                    arguments: branch.residualArguments[step], count: featureCount
                )
                residual.endEncoding()
                markPropagation(.residual, branchIndex: branchIndex)
            }
            if let componentCounterHeap {
                commandBuffer.writeTimestamp(
                    counterHeap: componentCounterHeap, index: 3 + branchIndex
                )
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
                count: batch * 320 * plane
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
        frameResidual.barrier(
            afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
            visibilityOptions: .device
        )
        for arguments in nonFiniteArguments {
            Support.dispatch1D(
                frameResidual,
                pipeline: nonFinitePipeline,
                arguments: arguments,
                count: frameElements
            )
        }
        frameResidual.endEncoding()
        if let componentCounterHeap {
            commandBuffer.writeTimestamp(counterHeap: componentCounterHeap, index: 7)
        }
        commandBuffer.endCommandBuffer()
        let encodingMilliseconds = elapsedMilliseconds(since: encodingStarted)
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
        trace("submitting command buffer")
        let waitStarted = ContinuousClock.now
        queue.commit([commandBuffer], options: commitOptions)
        let submitMilliseconds = elapsedMilliseconds(since: waitStarted)
        trace("command buffer committed")
        let feedbackStarted = ContinuousClock.now
        semaphore.wait()
        let feedbackMilliseconds = elapsedMilliseconds(since: feedbackStarted)
        let waitMilliseconds = elapsedMilliseconds(since: waitStarted)
        trace("command buffer completed")
        commandAllocatorHasCompletedSubmission = true
        let (milliseconds, error) = commitResult.load()
        if let error { throw error }
        let readbackStarted = ContinuousClock.now
        if nonFiniteFlagBuffer.contents().load(as: UInt32.self) != 0 {
            throw DeformConvError.nonFiniteOutput(
                "fused graph produced a non-finite restored-frame value"
            )
        }
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
        let readbackMilliseconds = elapsedMilliseconds(since: readbackStarted)
        var componentTimings: FusedGraphComponentTimings?
        if let componentCounterHeap,
           let data = try componentCounterHeap.resolveCounterRange(0..<8)
        {
            let timestamps = data.withUnsafeBytes { rawBuffer in
                Array(rawBuffer.bindMemory(to: UInt64.self).prefix(8))
            }
            let frequency = Double(device.queryTimestampFrequency())
            func milliseconds(_ start: Int, _ end: Int) -> Double {
                guard timestamps.count == 8, timestamps[end] >= timestamps[start], frequency > 0
                else { return 0 }
                return Double(timestamps[end] - timestamps[start]) * 1_000 / frequency
            }
            componentTimings = FusedGraphComponentTimings(
                featureExtraction: milliseconds(0, 1),
                spynet: milliseconds(1, 2),
                backward1: milliseconds(2, 3),
                forward1: milliseconds(3, 4),
                backward2: milliseconds(4, 5),
                forward2: milliseconds(5, 6),
                reconstruction: milliseconds(6, 7)
            )
        }
        if traceGraphComponents,
           componentTimings?.isPlausible(totalGPUMilliseconds: milliseconds) != true
        {
            throw DeformConvError.commandFailed(
                "Metal counter timestamps returned invalid fused-graph component timings"
            )
        }
        var propagationTimings: FusedGraphPropagationTimings?
        if let propagationCounterHeaps {
            let frequency = Double(device.queryTimestampFrequency())
            guard propagationCounterHeaps.count == 4, frequency > 0 else {
                throw DeformConvError.commandFailed(
                    "Metal propagation counter timestamps could not be resolved"
                )
            }
            var totals = [Double](
                repeating: 0, count: PropagationTimingCategory.allCases.count
            )
            var branchTimings = [FusedGraphBranchPropagationTimings]()
            var offsetLocality = [FusedGraphBranchOffsetLocality]()
            for branchIndex in propagationCounterHeaps.indices {
                guard propagationTimestampIndices[branchIndex]
                        == propagationCounterCountPerBranch,
                      propagationCategories[branchIndex].count
                        == propagationCounterCountPerBranch - 1,
                      let data = try propagationCounterHeaps[branchIndex]
                        .resolveCounterRange(0..<propagationCounterCountPerBranch)
                else {
                    throw DeformConvError.commandFailed(
                        "Metal propagation counter timestamps were incomplete"
                    )
                }
                let timestamps = data.withUnsafeBytes { rawBuffer in
                    Array(rawBuffer.bindMemory(to: UInt64.self)
                        .prefix(propagationCounterCountPerBranch))
                }
                guard timestamps.count == propagationCounterCountPerBranch else {
                    throw DeformConvError.commandFailed(
                        "Metal propagation counter timestamps could not be resolved"
                    )
                }
                var branchTotals = [Double](
                    repeating: 0, count: PropagationTimingCategory.allCases.count
                )
                for (index, category) in propagationCategories[branchIndex].enumerated() {
                    guard timestamps[index + 1] >= timestamps[index] else {
                        throw DeformConvError.commandFailed(
                            "Metal propagation counter timestamps were not monotonic"
                        )
                    }
                    let duration = Double(
                        timestamps[index + 1] - timestamps[index]
                    ) * 1_000 / frequency
                    totals[category.rawValue] += duration
                    branchTotals[category.rawValue] += duration
                }
                let branchTiming = FusedGraphBranchPropagationTimings(
                    name: branchSpecs[branchIndex].0,
                    offsetNetwork: branchTotals[PropagationTimingCategory.offsetNetwork.rawValue],
                    tensorPreparation: branchTotals[PropagationTimingCategory.tensorPreparation.rawValue],
                    dcnTransform: branchTotals[PropagationTimingCategory.dcnTransform.rawValue],
                    dcnGather: branchTotals[PropagationTimingCategory.dcnGather.rawValue],
                    dcnGEMM: branchTotals[PropagationTimingCategory.dcnGEMM.rawValue],
                    backboneNetwork: branchTotals[PropagationTimingCategory.backboneNetwork.rawValue],
                    residual: branchTotals[PropagationTimingCategory.residual.rawValue]
                )
                let branchComponentMilliseconds: Double
                switch branchIndex {
                case 0: branchComponentMilliseconds = componentTimings?.backward1 ?? 0
                case 1: branchComponentMilliseconds = componentTimings?.forward1 ?? 0
                case 2: branchComponentMilliseconds = componentTimings?.backward2 ?? 0
                default: branchComponentMilliseconds = componentTimings?.forward2 ?? 0
                }
                guard branchTiming.isPlausible(
                    totalBranchMilliseconds: branchComponentMilliseconds
                ) else {
                    throw DeformConvError.commandFailed(
                        "Metal counter timestamps returned invalid \(branchTiming.name) timings"
                    )
                }
                branchTimings.append(branchTiming)
                if let localityBuffer = branches[branchIndex].localityBuffer {
                    let values = localityBuffer.contents().bindMemory(
                        to: UInt32.self, capacity: flowCount * 8
                    )
                    var sampleCount: UInt64 = 0
                    var magnitudeSum: UInt64 = 0
                    var maximumBits: UInt32 = 0
                    var above2: UInt64 = 0
                    var above4: UInt64 = 0
                    var above8: UInt64 = 0
                    var outOfBounds: UInt64 = 0
                    var neighborDeltaSum: UInt64 = 0
                    for step in 0..<flowCount {
                        let base = step * 8
                        sampleCount += UInt64(values[base])
                        magnitudeSum += UInt64(values[base + 1])
                        maximumBits = max(maximumBits, values[base + 2])
                        above2 += UInt64(values[base + 3])
                        above4 += UInt64(values[base + 4])
                        above8 += UInt64(values[base + 5])
                        outOfBounds += UInt64(values[base + 6])
                        neighborDeltaSum += UInt64(values[base + 7])
                    }
                    guard sampleCount > 0 else {
                        throw DeformConvError.commandFailed(
                            "DCNv2 offset-locality counters were empty"
                        )
                    }
                    let denominator = Double(sampleCount)
                    let locality = FusedGraphBranchOffsetLocality(
                        name: branchTiming.name,
                        sampleCount: sampleCount,
                        meanMagnitude: Double(magnitudeSum) / (64 * denominator),
                        maximumMagnitude: Double(Float(bitPattern: maximumBits)),
                        fractionAbove2: Double(above2) / denominator,
                        fractionAbove4: Double(above4) / denominator,
                        fractionAbove8: Double(above8) / denominator,
                        outOfBoundsFraction: Double(outOfBounds) / denominator,
                        meanNeighborDelta: Double(neighborDeltaSum) / (64 * denominator)
                    )
                    guard locality.isPlausible else {
                        throw DeformConvError.commandFailed(
                            "DCNv2 offset-locality counters were invalid"
                        )
                    }
                    offsetLocality.append(locality)
                }
            }
            propagationTimings = FusedGraphPropagationTimings(
                offsetNetwork: totals[PropagationTimingCategory.offsetNetwork.rawValue],
                tensorPreparation: totals[PropagationTimingCategory.tensorPreparation.rawValue],
                dcnTransform: totals[PropagationTimingCategory.dcnTransform.rawValue],
                dcnGather: totals[PropagationTimingCategory.dcnGather.rawValue],
                dcnGEMM: totals[PropagationTimingCategory.dcnGEMM.rawValue],
                backboneNetwork: totals[PropagationTimingCategory.backboneNetwork.rawValue],
                residual: totals[PropagationTimingCategory.residual.rawValue],
                branches: branchTimings,
                offsetLocality: offsetLocality
            )
            let totalPropagationMilliseconds = componentTimings.map {
                $0.backward1 + $0.forward1 + $0.backward2 + $0.forward2
            } ?? 0
            guard propagationTimings?.isPlausible(
                totalPropagationMilliseconds: totalPropagationMilliseconds
            ) == true else {
                throw DeformConvError.commandFailed(
                    "Metal counter timestamps returned invalid propagation timings"
                )
            }
        }
        if traceGraphPhases {
            SideBySideRestoration.report(
                "Fused graph host phases: upload "
                    + "\(String(format: "%.3f", uploadMilliseconds)) ms, encode "
                    + "\(String(format: "%.3f", encodingMilliseconds)) ms, wait "
                    + "\(String(format: "%.3f", waitMilliseconds)) ms, readback "
                    + "\(String(format: "%.3f", readbackMilliseconds)) ms, GPU "
                    + "\(String(format: "%.3f", milliseconds)) ms, execution lock "
                    + "\(String(format: "%.3f", lockMilliseconds)) ms, submit "
                    + "\(String(format: "%.3f", submitMilliseconds)) ms, feedback wait "
                    + "\(String(format: "%.3f", feedbackMilliseconds)) ms"
            )
        }
        if let componentTimings {
            SideBySideRestoration.report(
                "Fused graph GPU components: feature "
                    + "\(String(format: "%.3f", componentTimings.featureExtraction)) ms, SPyNet "
                    + "\(String(format: "%.3f", componentTimings.spynet)) ms, backward_1 "
                    + "\(String(format: "%.3f", componentTimings.backward1)) ms, forward_1 "
                    + "\(String(format: "%.3f", componentTimings.forward1)) ms, backward_2 "
                    + "\(String(format: "%.3f", componentTimings.backward2)) ms, forward_2 "
                    + "\(String(format: "%.3f", componentTimings.forward2)) ms, reconstruction "
                    + "\(String(format: "%.3f", componentTimings.reconstruction)) ms"
            )
        }
        if let propagationTimings {
            SideBySideRestoration.report(
                "Fused graph propagation GPU components: offset "
                    + "\(String(format: "%.3f", propagationTimings.offsetNetwork)) ms, prepare/assemble "
                    + "\(String(format: "%.3f", propagationTimings.tensorPreparation)) ms, DCN transform "
                    + "\(String(format: "%.3f", propagationTimings.dcnTransform)) ms, DCN gather "
                    + "\(String(format: "%.3f", propagationTimings.dcnGather)) ms, DCN GEMM "
                    + "\(String(format: "%.3f", propagationTimings.dcnGEMM)) ms, backbone "
                    + "\(String(format: "%.3f", propagationTimings.backboneNetwork)) ms, residual "
                    + "\(String(format: "%.3f", propagationTimings.residual)) ms"
            )
            for branch in propagationTimings.branches {
                SideBySideRestoration.report(
                    "Fused graph \(branch.name) GPU packages: offset "
                        + "\(String(format: "%.3f", branch.offsetNetwork)) ms, prepare/assemble "
                        + "\(String(format: "%.3f", branch.tensorPreparation)) ms, DCNv2 "
                        + "\(String(format: "%.3f", branch.dcnTransform + branch.dcnGather + branch.dcnGEMM)) ms, backbone "
                        + "\(String(format: "%.3f", branch.backboneNetwork)) ms, residual "
                        + "\(String(format: "%.3f", branch.residual)) ms"
                )
            }
            for locality in propagationTimings.offsetLocality {
                SideBySideRestoration.report(
                    "Fused graph \(locality.name) offset locality: mean/max "
                        + "\(String(format: "%.3f", locality.meanMagnitude))/"
                        + "\(String(format: "%.3f", locality.maximumMagnitude)) px, >2/4/8 px "
                        + "\(String(format: "%.2f", 100 * locality.fractionAbove2))/"
                        + "\(String(format: "%.2f", 100 * locality.fractionAbove4))/"
                        + "\(String(format: "%.2f", 100 * locality.fractionAbove8))%, outside "
                        + "\(String(format: "%.2f", 100 * locality.outOfBoundsFraction))%, neighbor delta "
                        + "\(String(format: "%.3f", locality.meanNeighborDelta)) px"
                )
            }
        }
        return (
            milliseconds, outputs, restored, flows, componentTimings, propagationTimings
        )
    }

    let isFirstGraphForFamily = reusableProductionGraph
        && ProductionFusedGraphCache.shared.needsInitialWarmup(
            for: productionGraphFamily
        )
    if coldGraphWarmupRequired(
        setupMilliseconds: setupMilliseconds,
        reusableProductionGraph: reusableProductionGraph,
        isFirstGraphForFamily: isFirstGraphForFamily
    ) {
        trace(
            "initial/cold graph safety warm-up; discarding first graph execution"
        )
        let safetyReason = isFirstGraphForFamily
            ? "first-family correctness"
            : "slow graph setup"
        SideBySideRestoration.report(
            "Fused graph discarded safety execution started: setup "
                + "\(String(format: "%.3f", setupMilliseconds)) ms; "
                + "reason \(safetyReason)"
        )
        let safetyStarted = ContinuousClock.now
        let safetyResult = try execute()
        let safetyWallMilliseconds = elapsedMilliseconds(since: safetyStarted)
        SideBySideRestoration.report(
            "Fused graph discarded safety execution completed: wall "
                + "\(String(format: "%.3f", safetyWallMilliseconds)) ms, GPU "
                + "\(String(format: "%.3f", safetyResult.0)) ms; "
                + "reason \(safetyReason)"
        )
    }
    if isFirstGraphForFamily {
        ProductionFusedGraphCache.shared.markInitialWarmupComplete(
            for: productionGraphFamily
        )
    }
    for _ in 0..<warmupCount { _ = try execute() }
    let (
        firstMilliseconds, firstOutputs, firstRestored, firstFlows, firstComponents,
        firstPropagation
    ) = try execute()
    var samples = [firstMilliseconds]
    var lastOutputs = firstOutputs
    var lastRestored = firstRestored
    var lastFlows = firstFlows
    var lastComponents = firstComponents
    var lastPropagation = firstPropagation
    for _ in 1..<measurementCount {
        let (milliseconds, outputs, restored, flows, components, propagation) = try execute()
        samples.append(milliseconds)
        lastOutputs = outputs
        lastRestored = restored
        lastFlows = flows
        lastComponents = components
        lastPropagation = propagation
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
    if collectDiagnostics || hasStagedRestoration {
        for frame in 0..<frameCount {
            let predicted = predictedBuffers[frame].contents().bindMemory(
                to: Float16.self, capacity: frameElements
            )
            for index in 0..<frameElements {
                let value = Float(lastRestored[frame][index])
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
    _ = channelLastInputBuffer
    let result = FusedFourPassRecurrenceResult(
        graphCacheHit: false,
        graphLookupMilliseconds: lookupMilliseconds,
        graphSetupMilliseconds: setupMilliseconds,
        statistics: statistics,
        componentTimings: lastComponents,
        propagationTimings: lastPropagation,
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
            _ = channelLastInputBuffer
            activeInputFrames = frames
            let (milliseconds, _, restored, _, components, propagation) = try execute()
            guard let statistics = BenchmarkStatistics([milliseconds]) else {
                throw DeformConvError.commandFailed("invalid fused four-pass timing")
            }
            return FusedFourPassRecurrenceResult(
                graphCacheHit: true,
                graphLookupMilliseconds: 0,
                graphSetupMilliseconds: 0,
                statistics: statistics,
                componentTimings: components,
                propagationTimings: propagation,
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
        ProductionFusedGraphCache.shared.retain(
            runner,
            for: productionGraphKey,
            family: productionGraphFamily
        )
    }
    return result
}
