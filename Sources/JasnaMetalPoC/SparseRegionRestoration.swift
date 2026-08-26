import CoreVideo
import Foundation
import Metal

@available(macOS 27.0, *)
final class RestorationBatchCircuitBreaker: @unchecked Sendable {
    private let lock = NSLock()
    private var disabled = false

    var isDisabled: Bool {
        lock.withLock { disabled }
    }

    @discardableResult
    func disable() -> Bool {
        lock.withLock {
            guard !disabled else { return false }
            disabled = true
            return true
        }
    }
}

@available(macOS 27.0, *)
private let sharedBatch2CircuitBreaker = RestorationBatchCircuitBreaker()

@available(macOS 27.0, *)
extension SideBySideRestoration {
    struct ModelCropReuseSummary: Equatable, Sendable {
        let cropCount: Int
        let uniqueExactCropCount: Int
        let exactDuplicateCount: Int
        let highOverlapPairCount: Int
        let containedPairCount: Int
    }

    private struct ModelCropReuseKey: Hashable {
        let x: Int
        let y: Int
        let width: Int
        let height: Int
        let modelStartFrame: Int
        let modelEndFrame: Int
    }

    static func modelCropReuseSummary(
        regions: [MosaicRegion],
        windowStartFrame: Int,
        outputCount: Int,
        temporalWarmupFrames: Int
    ) -> ModelCropReuseSummary {
        let windowEndFrame = windowStartFrame + outputCount
        let keys = regions.map { region in
            let targetStart = max(windowStartFrame, region.startFrame)
            let targetEnd = min(windowEndFrame, region.endFrame)
            return ModelCropReuseKey(
                x: region.x,
                y: region.y,
                width: region.width,
                height: region.height,
                modelStartFrame: max(0, targetStart - max(0, temporalWarmupFrames)),
                modelEndFrame: targetEnd
            )
        }
        var highOverlapPairCount = 0
        var containedPairCount = 0
        if regions.count > 1 {
            for leftIndex in 0..<(regions.count - 1) {
                for rightIndex in (leftIndex + 1)..<regions.count {
                    let leftKey = keys[leftIndex]
                    let rightKey = keys[rightIndex]
                    guard leftKey.modelStartFrame == rightKey.modelStartFrame,
                          leftKey.modelEndFrame == rightKey.modelEndFrame,
                          leftKey != rightKey
                    else { continue }
                    let left = regions[leftIndex]
                    let right = regions[rightIndex]
                    let intersectionWidth = max(
                        0, min(left.x + left.width, right.x + right.width)
                            - max(left.x, right.x)
                    )
                    let intersectionHeight = max(
                        0, min(left.y + left.height, right.y + right.height)
                            - max(left.y, right.y)
                    )
                    let intersection = intersectionWidth * intersectionHeight
                    guard intersection > 0 else { continue }
                    let leftArea = left.width * left.height
                    let rightArea = right.width * right.height
                    let union = leftArea + rightArea - intersection
                    if union > 0, Double(intersection) / Double(union) >= 0.8 {
                        highOverlapPairCount += 1
                    }
                    if intersection == min(leftArea, rightArea) {
                        containedPairCount += 1
                    }
                }
            }
        }
        let uniqueCount = Set(keys).count
        return ModelCropReuseSummary(
            cropCount: regions.count,
            uniqueExactCropCount: uniqueCount,
            exactDuplicateCount: regions.count - uniqueCount,
            highOverlapPairCount: highOverlapPairCount,
            containedPairCount: containedPairCount
        )
    }

    static func reportModelCropReuseSummary(
        _ summary: ModelCropReuseSummary,
        windowIndex: Int,
        eye: String? = nil
    ) {
        let eyeText = eye.map { " \($0)" } ?? ""
        report(
            "Window \(windowIndex + 1)\(eyeText) crop reuse: "
                + "\(summary.uniqueExactCropCount)/\(summary.cropCount) unique exact crops, "
                + "\(summary.exactDuplicateCount) reusable duplicate(s), "
                + "\(summary.highOverlapPairCount) high-overlap pair(s), "
                + "\(summary.containedPairCount) contained pair(s)"
        )
    }

    static var fullDetectedRegionBlendEnabled: Bool {
        ProcessInfo.processInfo.environment["JASNA_DIAGNOSTIC_FULL_REGION_BLEND"] == "1"
    }

    static func elapsedMilliseconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = start.duration(to: .now).components
        return Double(elapsed.seconds) * 1_000
            + Double(elapsed.attoseconds) / 1_000_000_000_000_000
    }

    struct PreparedRegionRestoration: Sendable {
        let regionIndex: Int
        let localStart: Int
        let activeFrameCount: Int
        let restoredFrameOffset: Int
        let inputFrames: [[Float16]]
        let context: String
    }

    struct CompletedRegionRestoration: Sendable {
        let prepared: PreparedRegionRestoration
        let frames: [[Float16]]
        let gpuMilliseconds: Double
        let wallMilliseconds: Double
        let inputPackingMilliseconds: Double
        let graphExecutionMilliseconds: Double
        let outputSplittingMilliseconds: Double

        init(
            prepared: PreparedRegionRestoration,
            frames: [[Float16]],
            gpuMilliseconds: Double,
            wallMilliseconds: Double,
            inputPackingMilliseconds: Double = 0,
            graphExecutionMilliseconds: Double? = nil,
            outputSplittingMilliseconds: Double = 0
        ) {
            self.prepared = prepared
            self.frames = frames
            self.gpuMilliseconds = gpuMilliseconds
            self.wallMilliseconds = wallMilliseconds
            self.inputPackingMilliseconds = inputPackingMilliseconds
            self.graphExecutionMilliseconds = graphExecutionMilliseconds ?? wallMilliseconds
            self.outputSplittingMilliseconds = outputSplittingMilliseconds
        }
    }

    static func batchOptimizedRegions(
        _ regions: [MosaicRegion],
        windowStartFrame: Int,
        outputCount: Int,
        batch2Enabled: Bool,
        temporalWarmupFrames: Int = 0
    ) -> [MosaicRegion] {
        guard batch2Enabled, regions.count > 2 else { return regions }
        func modelFrameCount(_ region: MosaicRegion) -> Int {
            let targetStart = max(windowStartFrame, region.startFrame)
            let targetEnd = min(windowStartFrame + outputCount, region.endFrame)
            let modelStart = max(
                0, targetStart - max(0, temporalWarmupFrames)
            )
            return max(3, targetEnd - modelStart)
        }
        let grouped = Dictionary(grouping: regions.enumerated()) {
            modelFrameCount($0.element)
        }
        var paired = [MosaicRegion]()
        var leftovers = [MosaicRegion]()
        for frameCount in grouped.keys.sorted() {
            let stableGroup = grouped[frameCount, default: []].sorted {
                $0.offset < $1.offset
            }
            let pairedCount = stableGroup.count - stableGroup.count % 2
            paired.append(contentsOf: stableGroup[..<pairedCount].map(\.element))
            if pairedCount < stableGroup.count {
                leftovers.append(stableGroup[pairedCount].element)
            }
        }
        // The restoration loop consumes two adjacent entries at a time. Put all
        // compatible pairs first so an odd group cannot shift every later group
        // off its pair boundary. At most one stable leftover remains per length.
        return paired + leftovers
    }

    final class RegionRestorationBatch: @unchecked Sendable {
        private let device: MTLDevice
        private let modelsURL: URL
        private let weightsURL: URL
        private let work: [PreparedRegionRestoration]
        private let lock = NSLock()
        private var completed: [CompletedRegionRestoration?]
        private var failures: [(any Error)?]

        init(
            device: MTLDevice,
            modelsURL: URL,
            weightsURL: URL,
            work: [PreparedRegionRestoration]
        ) {
            self.device = device
            self.modelsURL = modelsURL
            self.weightsURL = weightsURL
            self.work = work
            completed = [CompletedRegionRestoration?](repeating: nil, count: work.count)
            failures = [(any Error)?](repeating: nil, count: work.count)
        }

        func execute(_ index: Int) {
            do {
                let item = work[index]
                let started = ContinuousClock.now
                let restored = try SideBySideRestoration.restoreTileWithFallback(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    inputFrames: item.inputFrames,
                    context: item.context
                )
                let wallMilliseconds = SideBySideRestoration.elapsedMilliseconds(
                    since: started
                )
                lock.lock()
                completed[index] = CompletedRegionRestoration(
                    prepared: item,
                    frames: restored.frames,
                    gpuMilliseconds: restored.gpuMilliseconds,
                    wallMilliseconds: wallMilliseconds
                )
                lock.unlock()
            } catch {
                lock.lock()
                failures[index] = error
                lock.unlock()
            }
        }

        func results() throws -> [CompletedRegionRestoration] {
            lock.lock()
            defer { lock.unlock() }
            if let failure = failures.compactMap({ $0 }).first { throw failure }
            guard completed.allSatisfy({ $0 != nil }) else {
                throw DeformConvError.commandFailed("parallel mosaic restoration was incomplete")
            }
            return completed.compactMap { $0 }
        }
    }

    static func processRegionWindow(
        device: MTLDevice,
        plan: SideBySideVideoPlan,
        regions: [MosaicRegion],
        decodedFrames: [CVPixelBuffer],
        decodedStartFrame: Int,
        outputCount: Int,
        windowIndex: Int,
        windowCount: Int,
        modelsURL: URL,
        weightsURL: URL,
        cacheVariant: String,
        projection: VRMosaicProjection,
        samplingMaps: [MosaicCropSamplingMap],
        workDirectoryURL: URL? = nil
    ) throws -> WindowResult {
        guard samplingMaps.count == regions.count else {
            throw DeformConvError.invalidShape
        }
        let cacheBytes = regions.count * outputCount * tileBytes
        let configuredWorkPath = workDirectoryURL?.path
            ?? ProcessInfo.processInfo.environment["JASNA_WORK_DIR"]
        let workURL = configuredWorkPath.map {
            URL(fileURLWithPath: $0, isDirectory: true)
        } ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(at: workURL, withIntermediateDirectories: true)
        let resumed = configuredWorkPath == nil ? nil : try resumableWindowCache(
            in: workURL,
            windowIndex: windowIndex,
            outputCount: outputCount,
            tileCount: regions.count,
            cacheVariant: cacheVariant
        )
        let prefix = cacheDirectoryPrefix(windowIndex: windowIndex, cacheVariant: cacheVariant)
        let directory = resumed?.directory ?? workURL.appendingPathComponent(
            "\(prefix)\(UUID().uuidString)", isDirectory: true
        )
        if resumed == nil {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        }
        do {
            let configuredMemoryLimitMiB = Int(
                ProcessInfo.processInfo.environment["JASNA_IN_MEMORY_CACHE_LIMIT_MB"] ?? ""
            ) ?? 512
            let memoryLimitBytes = max(0, configuredMemoryLimitMiB) * 1_048_576
            let useInMemoryCache = resumed == nil
                && ProcessInfo.processInfo.environment["JASNA_IN_MEMORY_CROP_CACHE"] != "0"
                && cacheBytes <= memoryLimitBytes
            let inMemoryCache = useInMemoryCache
                ? try InMemoryRegionFrameCache(
                    frameCount: outputCount, regionCount: regions.count
                ) : nil
            let urls = resumed?.urls ?? (0..<outputCount).map {
                directory.appendingPathComponent("frame-\($0).fp16")
            }
            var handles = try useInMemoryCache ? [] : urls.map { url -> FileHandle in
                if resumed == nil {
                    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                        throw DeformConvError.commandFailed("failed creating crop cache: \(url.path)")
                    }
                }
                let handle = try FileHandle(forUpdating: url)
                if let resumed {
                    let safeOffset = UInt64(resumed.completedTiles * tileBytes)
                    try handle.truncate(atOffset: safeOffset)
                    try handle.seek(toOffset: safeOffset)
                }
                return handle
            }
            defer { for handle in handles { try? handle.close() } }
            let completedRegions = resumed?.completedTiles ?? 0
            var gpuMilliseconds: Double = 0
            var extractionMilliseconds: Double = 0
            var graphWallMilliseconds: Double = 0
            var inputPackingMilliseconds: Double = 0
            var graphExecutionMilliseconds: Double = 0
            var outputSplittingMilliseconds: Double = 0
            var cacheWriteMilliseconds: Double = 0
            var restoredModelFrames = 0
            let configuredCheckpointInterval = Int(
                ProcessInfo.processInfo.environment["JASNA_REGION_CHECKPOINT_INTERVAL"] ?? ""
            )
            let checkpointInterval = configuredWorkPath == nil
                ? max(1, regions.count)
                : max(1, configuredCheckpointInterval ?? 5)
            let batchModelsURL = sharedBatch2CircuitBreaker.isDisabled ? nil
                : ProcessInfo.processInfo.environment[
                    "JASNA_BATCH2_MODELS_DIR"
                ].map { URL(fileURLWithPath: $0, isDirectory: true) }.flatMap { candidate in
                    FileManager.default.fileExists(
                        atPath: candidate.appendingPathComponent(
                            "feature_extract.mtlpackage"
                        ).path
                    ) ? candidate : nil
                }
            // Keep construction serial: two simultaneous first-use Metal ML
            // graphs are unstable in the macOS 27 beta. A fixed-batch graph can
            // still restore two compatible crops in one command buffer.
            let regionBatchSize = batchModelsURL == nil ? 1 : 2
            report(
                "Window \(windowIndex + 1)/\(windowCount): restoring "
                    + "\(regions.count) tight mosaic crops; cache "
                    + "\(String(format: "%.2f", Double(cacheBytes) / 1_073_741_824)) GiB; "
                    + "model batch \(regionBatchSize); handoff "
                    + (useInMemoryCache ? "bounded memory" : "restartable disk")
            )
            var nextRegion = completedRegions
            while nextRegion < regions.count {
                let batchEnd = min(regions.count, nextRegion + regionBatchSize)
                let extractionStarted = ContinuousClock.now
                let work = try (nextRegion..<batchEnd).map { regionIndex in
                    let windowStartFrame = windowIndex * SideBySideVideoPlan.temporalWindowFrames
                    let region = regions[regionIndex]
                    let windowSchedule = TemporalWindowSchedule(
                        outputStartFrame: windowStartFrame,
                        outputFrameCount: outputCount,
                        requestedWarmupFrames: windowStartFrame - decodedStartFrame
                    )
                    guard let regionSchedule = TemporalRegionSchedule(
                        regionStartFrame: region.startFrame,
                        regionEndFrame: region.endFrame,
                        window: windowSchedule
                    ) else {
                        throw DeformConvError.commandFailed(
                            "mosaic crop does not intersect its assigned window"
                        )
                    }
                    guard regionSchedule.decodedLocalRange.lowerBound >= 0,
                          regionSchedule.decodedLocalRange.upperBound <= decodedFrames.count,
                          !regionSchedule.decodedLocalRange.isEmpty
                    else { throw DeformConvError.invalidShape }
                    let samplingMap = samplingMaps[regionIndex]
                    let modelFrames = decodedFrames[regionSchedule.decodedLocalRange]
                    var cropFrames = try modelFrames.map {
                        try samplingMap.extractPlanarRGB(from: $0)
                    }
                    while cropFrames.count < 3 {
                        guard let last = cropFrames.last else {
                            throw DeformConvError.invalidShape
                        }
                        cropFrames.append(last)
                    }
                    let context = "Window \(windowIndex + 1)/\(windowCount): mosaic crop "
                        + "\(regionIndex + 1)/\(regions.count), x \(region.x), y \(region.y), "
                        + "size \(region.width)×\(region.height), frames "
                        + "\(region.startFrame)..<\(region.endFrame)"
                    return PreparedRegionRestoration(
                        regionIndex: regionIndex,
                        localStart: regionSchedule.outputLocalStart,
                        activeFrameCount: regionSchedule.activeFrameCount,
                        restoredFrameOffset: regionSchedule.restoredFrameOffset,
                        inputFrames: cropFrames,
                        context: context
                    )
                }
                extractionMilliseconds += elapsedMilliseconds(since: extractionStarted)
                let completedWork: [CompletedRegionRestoration]
                if let batchModelsURL, !sharedBatch2CircuitBreaker.isDisabled {
                    completedWork = try restorePreparedRegionsWithBatchFallback(
                        work: work,
                        batchRestore: { batchWork in
                            try restorePreparedRegionBatch(
                                device: device,
                                modelsURL: batchModelsURL,
                                weightsURL: weightsURL,
                                work: batchWork
                            )
                        },
                        individualRestore: { individualWork in
                            try restorePreparedRegionsIndividually(
                                device: device,
                                modelsURL: modelsURL,
                                weightsURL: weightsURL,
                                work: individualWork
                            )
                        },
                        onBatchFailure: { error in
                            let firstFailure = sharedBatch2CircuitBreaker.disable()
                            report(
                                "Window \(windowIndex + 1)/\(windowCount): batch-2 graph failed "
                                    + "(\(error)); retrying both crops independently"
                                    + (firstFailure
                                        ? "; batch 2 disabled for the rest of this process"
                                        : "")
                            )
                        }
                    )
                } else {
                    completedWork = try restorePreparedRegionsIndividually(
                        device: device,
                        modelsURL: modelsURL,
                        weightsURL: weightsURL,
                        work: work
                    )
                }
                for restored in completedWork {
                    let prepared = restored.prepared
                    try writeRecoveryDiagnosticIfRequested(
                        restored: restored,
                        region: regions[prepared.regionIndex],
                        samplingMap: samplingMaps[prepared.regionIndex],
                        windowIndex: windowIndex,
                        workDirectoryURL: workDirectoryURL
                    )
                    let cacheWriteStarted = ContinuousClock.now
                    for frame in 0..<outputCount {
                        if frame >= prepared.localStart
                            && frame < prepared.localStart + prepared.activeFrameCount
                        {
                            let values = restored.frames[
                                prepared.restoredFrameOffset + frame - prepared.localStart
                            ]
                            if let inMemoryCache {
                                try inMemoryCache.store(
                                    values, frame: frame, region: prepared.regionIndex
                                )
                            } else {
                                try values.withUnsafeBytes { bytes in
                                    try handles[frame].write(contentsOf: Data(bytes))
                                }
                            }
                        } else if inMemoryCache == nil {
                            try handles[frame].seek(
                                toOffset: UInt64((prepared.regionIndex + 1) * tileBytes)
                            )
                        }
                    }
                    gpuMilliseconds += restored.gpuMilliseconds
                    graphWallMilliseconds += restored.wallMilliseconds
                    inputPackingMilliseconds += restored.inputPackingMilliseconds
                    graphExecutionMilliseconds += restored.graphExecutionMilliseconds
                    outputSplittingMilliseconds += restored.outputSplittingMilliseconds
                    restoredModelFrames += prepared.inputFrames.count
                    let completedCount = prepared.regionIndex + 1
                    let shouldCheckpoint = completedCount == regions.count
                        || completedCount.isMultiple(of: checkpointInterval)
                    if shouldCheckpoint, inMemoryCache == nil {
                        let completedBytes = UInt64(completedCount * tileBytes)
                        for handle in handles {
                            try handle.truncate(atOffset: completedBytes)
                            try handle.synchronize()
                        }
                        try Data("\(completedCount)\n".utf8).write(
                            to: directory.appendingPathComponent("completed-tiles.txt"),
                            options: .atomic
                        )
                    }
                    cacheWriteMilliseconds += elapsedMilliseconds(since: cacheWriteStarted)
                    report(
                        "Window \(windowIndex + 1)/\(windowCount): mosaic crop "
                            + "\(completedCount)/\(regions.count), GPU "
                            + "\(String(format: "%.3f", gpuMilliseconds)) ms cumulative, "
                            + "crop wall \(String(format: "%.3f", restored.wallMilliseconds)) ms"
                    )
                }
                nextRegion = batchEnd
            }
            report(
                "Window \(windowIndex + 1)/\(windowCount): sparse hot-path phases: "
                    + "\(restoredModelFrames) model frames, extraction "
                    + "\(String(format: "%.3f", extractionMilliseconds)) ms, graph wall "
                    + "\(String(format: "%.3f", graphWallMilliseconds)) ms, GPU "
                    + "\(String(format: "%.3f", gpuMilliseconds)) ms, cache writes "
                    + "\(String(format: "%.3f", cacheWriteMilliseconds)) ms"
            )
            report(
                "Window \(windowIndex + 1)/\(windowCount): graph host phases: input packing "
                    + "\(String(format: "%.3f", inputPackingMilliseconds)) ms, execution "
                    + "\(String(format: "%.3f", graphExecutionMilliseconds)) ms, output split "
                    + "\(String(format: "%.3f", outputSplittingMilliseconds)) ms"
            )
            for handle in handles { try handle.close() }
            handles.removeAll()
            return WindowResult(
                cacheDirectory: directory,
                cacheURLs: urls,
                gpuMilliseconds: gpuMilliseconds,
                cacheBytes: cacheBytes,
                inMemoryRegionCache: inMemoryCache
            )
        } catch {
            if configuredWorkPath == nil {
                try? FileManager.default.removeItem(at: directory)
            } else {
                report("Preserving failed crop cache at \(directory.path)")
            }
            throw error
        }
    }

    static func reportSubdivisionConfiguration(
        _ configuration: MosaicRegionSubdivisionConfiguration
    ) {
        if fullDetectedRegionBlendEnabled {
            report(
                "WARNING: diagnostic full detected-region blend enabled; "
                    + "segmentation masks are bypassed"
            )
        }
        guard configuration.maximumBlendDimension > 0,
              configuration.splitLimit > 0
        else {
            report("Large-region subdivision: disabled")
            return
        }
        report(
            "Large-region subdivision: max blend "
                + "\(configuration.maximumBlendDimension)px, overlap "
                + "\(configuration.overlap)px, up to "
                + "\(configuration.splitLimit) region(s)/window and "
                + "\(configuration.maximumAxisCrops) crops/axis; "
                + "normalized Metal overlap; all-region mask growth "
                + "\(String(format: "%.3f", configuration.maskGrowthFraction)), feather "
                + "\(String(format: "%.3f", configuration.maskFeatherFraction)), block halo "
                + "\(String(format: "%.3f", configuration.blockResidualGrowthFraction)), "
                + "temporal radius \(configuration.maskTemporalRadius); lower detail "
                + "\(configuration.detailCropCount)x"
                + "\(configuration.detailCropDimension)px"
        )
    }

    static func reportTemporalCropConfiguration(
        _ configuration: MosaicTemporalCropConfiguration
    ) {
        guard configuration.chunkFrames >= 3 else {
            report("Motion-tight temporal crops: disabled")
            return
        }
        report(
            "Motion-tight temporal crops: \(configuration.chunkFrames) frames, padding "
                + "\(configuration.padding)px, minimum region "
                + "\(configuration.minimumDimension)px, motion threshold "
                + "\(String(format: "%.3f", configuration.minimumMotionFraction))"
        )
    }

    static func reportTemporalWarmupConfiguration(
        _ configuration: TemporalWarmupConfiguration
    ) {
        guard configuration.frames > 0 else {
            report("Temporal crop warm-up: disabled")
            return
        }
        report(
            "Temporal crop warm-up: \(configuration.frames) preceding frame(s); "
                + "warm-up outputs are discarded"
        )
    }

    static func restorePreparedRegionsWithBatchFallback(
        work: [PreparedRegionRestoration],
        batchRestore: ([PreparedRegionRestoration]) throws -> [CompletedRegionRestoration],
        individualRestore: ([PreparedRegionRestoration]) throws -> [CompletedRegionRestoration],
        onBatchFailure: (any Error) -> Void
    ) throws -> [CompletedRegionRestoration] {
        guard work.count == 2,
              work[0].inputFrames.count == work[1].inputFrames.count
        else {
            return try individualRestore(work)
        }
        do {
            return try batchRestore(work)
        } catch {
            onBatchFailure(error)
            return try individualRestore(work)
        }
    }

    private static func restorePreparedRegionsIndividually(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        work: [PreparedRegionRestoration]
    ) throws -> [CompletedRegionRestoration] {
        let batch = RegionRestorationBatch(
            device: device, modelsURL: modelsURL, weightsURL: weightsURL, work: work
        )
        for index in work.indices { batch.execute(index) }
        let completed = try batch.results()
        guard completed.count == work.count,
              zip(completed, work).allSatisfy({ restored, expected in
                  restored.prepared.regionIndex == expected.regionIndex
                      && restored.frames.count
                          >= expected.restoredFrameOffset + expected.activeFrameCount
                      && restored.frames.allSatisfy({ $0.count == tileElements })
              })
        else {
            throw DeformConvError.commandFailed(
                "individual restoration returned malformed crop tensors"
            )
        }
        return completed
    }

    private static func restorePreparedRegionBatch(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        work: [PreparedRegionRestoration]
    ) throws -> [CompletedRegionRestoration] {
        guard work.count == 2,
              work[0].inputFrames.count == work[1].inputFrames.count,
              work.allSatisfy({ item in
                  item.inputFrames.allSatisfy({ $0.count == tileElements })
              })
        else { throw DeformConvError.invalidShape }
        let packingStarted = ContinuousClock.now
        let batchedFrames = try work[0].inputFrames.indices.map { frame in
            let values = work.flatMap { $0.inputFrames[frame] }
            guard values.count == 2 * tileElements else {
                throw DeformConvError.invalidShape
            }
            return values
        }
        let packingMilliseconds = elapsedMilliseconds(since: packingStarted)
        let graphStarted = ContinuousClock.now
        let restored = try restoreTileFrames(
            device: device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            inputFrames: batchedFrames,
            maximumFramesPerChunk: batchedFrames.count,
            batch: 2
        )
        let graphMilliseconds = elapsedMilliseconds(since: graphStarted)
        let splittingStarted = ContinuousClock.now
        let split = try splitBatchedRegionFrames(
            restored.frames,
            work: work,
            gpuMilliseconds: restored.gpuMilliseconds,
            wallMilliseconds: packingMilliseconds + graphMilliseconds,
            inputPackingMilliseconds: packingMilliseconds,
            graphExecutionMilliseconds: graphMilliseconds
        )
        let splittingMilliseconds = elapsedMilliseconds(since: splittingStarted)
        let perItemSplittingMilliseconds = splittingMilliseconds / Double(split.count)
        return split.map { item in
            CompletedRegionRestoration(
                prepared: item.prepared,
                frames: item.frames,
                gpuMilliseconds: item.gpuMilliseconds,
                wallMilliseconds: item.wallMilliseconds + perItemSplittingMilliseconds,
                inputPackingMilliseconds: item.inputPackingMilliseconds,
                graphExecutionMilliseconds: item.graphExecutionMilliseconds,
                outputSplittingMilliseconds: perItemSplittingMilliseconds
            )
        }
    }

    static func splitBatchedRegionFrames(
        _ frames: [[Float16]],
        work: [PreparedRegionRestoration],
        gpuMilliseconds: Double,
        wallMilliseconds: Double,
        inputPackingMilliseconds: Double = 0,
        graphExecutionMilliseconds: Double? = nil
    ) throws -> [CompletedRegionRestoration] {
        guard work.count == 2,
              frames.count == work[0].inputFrames.count,
              frames.allSatisfy({ $0.count == 2 * tileElements })
        else {
            throw DeformConvError.commandFailed(
                "batch-2 restoration returned malformed crop tensors"
            )
        }
        return work.indices.map { sample in
            let start = sample * tileElements
            let end = start + tileElements
            return CompletedRegionRestoration(
                prepared: work[sample],
                frames: frames.map { Array($0[start..<end]) },
                gpuMilliseconds: gpuMilliseconds / Double(work.count),
                wallMilliseconds: wallMilliseconds / Double(work.count),
                inputPackingMilliseconds: inputPackingMilliseconds / Double(work.count),
                graphExecutionMilliseconds: (graphExecutionMilliseconds ?? wallMilliseconds)
                    / Double(work.count)
            )
        }
    }

    static func restoreTileWithFallback(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        inputFrames: [[Float16]],
        context: String
    ) throws -> (frames: [[Float16]], gpuMilliseconds: Double) {
        do {
            return try restoreTileFrames(
                device: device,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                inputFrames: inputFrames,
                maximumFramesPerChunk: inputFrames.count
            )
        } catch {
            guard let modelError = error as? DeformConvError,
                  modelError.isRecoverableNumericalFailure
            else { throw error }
            report(
                "\(context) failed its full temporal window (\(error)); "
                    + "retrying shorter recurrence chunks"
            )
            var lastError: Error = error
            for chunkSize in [10, 5, 3] where chunkSize < inputFrames.count {
                do {
                    let recovered = try restoreTileFrames(
                        device: device,
                        modelsURL: modelsURL,
                        weightsURL: weightsURL,
                        inputFrames: inputFrames,
                        maximumFramesPerChunk: chunkSize
                    )
                    report(
                        "\(context) recovered with at most \(chunkSize) frames "
                            + "per recurrence chunk"
                    )
                    return recovered
                } catch {
                    guard let modelError = error as? DeformConvError,
                          modelError.isRecoverableNumericalFailure
                    else { throw error }
                    lastError = error
                    report("\(context) also failed with \(chunkSize)-frame chunks (\(error))")
                }
            }
            do {
                let recovered = try restoreTileFramesIndependently(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    inputFrames: inputFrames
                )
                report("\(context) recovered with independent zero-motion frame triplets")
                return recovered
            } catch {
                guard let modelError = error as? DeformConvError,
                      modelError.isRecoverableNumericalFailure
                else { throw error }
                lastError = error
                report("\(context) also failed independent frame recovery (\(error))")
            }
            guard inputFrames.joined().allSatisfy({ Float($0).isFinite }) else {
                throw DeformConvError.commandFailed(
                    "\(context) has non-finite input pixels after model failure: \(lastError)"
                )
            }
            guard ProcessInfo.processInfo.environment["JASNA_ALLOW_PASSTHROUGH"] == "1" else {
                throw DeformConvError.commandFailed(
                    "\(context) exhausted every Metal recurrence recovery mode; refusing "
                        + "to preserve the original mosaic via silent passthrough. Last error: "
                        + "\(lastError). Set JASNA_ALLOW_PASSTHROUGH=1 only to produce a "
                        + "known-degraded diagnostic output."
                )
            }
            report(
                "WARNING: \(context) is using explicitly enabled finite input-pixel "
                    + "passthrough because all Metal recurrence recovery modes failed "
                    + "(\(lastError)); output is degraded"
            )
            return (inputFrames, 0)
        }
    }

    private static func restoreTileFramesIndependently(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        inputFrames: [[Float16]]
    ) throws -> (frames: [[Float16]], gpuMilliseconds: Double) {
        var frames = [[Float16]]()
        frames.reserveCapacity(inputFrames.count)
        var gpuMilliseconds: Double = 0
        for frame in inputFrames {
            let recovered = try autoreleasepool {
                try verifyFusedFourPassRecurrence(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    backwardFlows: [],
                    forwardFlows: [],
                    inputFrames: [frame, frame, frame],
                    stagedBranchFrames: [],
                    stagedRestoredFrames: [],
                    warmupCount: 0,
                    measurementCount: 1,
                    collectDiagnostics: false
                )
            }
            guard recovered.restoredFrames.count == 3 else {
                throw DeformConvError.commandFailed("Metal graph returned the wrong frame count")
            }
            frames.append(recovered.restoredFrames[1])
            gpuMilliseconds += recovered.statistics.median
        }
        return (frames, gpuMilliseconds)
    }

    private static func restoreTileFrames(
        device: MTLDevice,
        modelsURL: URL,
        weightsURL: URL,
        inputFrames: [[Float16]],
        maximumFramesPerChunk: Int,
        batch: Int = 1
    ) throws -> (frames: [[Float16]], gpuMilliseconds: Double) {
        let ranges = try temporalChunkRanges(
            frameCount: inputFrames.count,
            maximumFramesPerChunk: maximumFramesPerChunk
        )
        var frames = [[Float16]]()
        frames.reserveCapacity(inputFrames.count)
        var gpuMilliseconds: Double = 0
        for range in ranges {
            let result = try autoreleasepool {
                try verifyFusedFourPassRecurrence(
                    device: device,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    backwardFlows: [],
                    forwardFlows: [],
                    inputFrames: Array(inputFrames[range]),
                    stagedBranchFrames: [],
                    stagedRestoredFrames: [],
                    warmupCount: 0,
                    measurementCount: 1,
                    collectDiagnostics: false,
                    batch: batch
                )
            }
            guard result.restoredFrames.count == range.count else {
                throw DeformConvError.commandFailed("Metal graph returned the wrong frame count")
            }
            frames.append(contentsOf: result.restoredFrames)
            gpuMilliseconds += result.statistics.median
        }
        return (frames, gpuMilliseconds)
    }
}
