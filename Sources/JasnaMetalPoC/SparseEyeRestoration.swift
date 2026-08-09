import AVFoundation
import CoreVideo
import Foundation
import Metal

@available(macOS 27.0, *)
extension SideBySideRestoration {
    static let defaultEncoderWindowsPerSegment = 120

    static func restorationWindowRange(
        windowCount: Int,
        environment: [String: String]
    ) throws -> Range<Int> {
        guard windowCount > 0 else { throw DeformConvError.invalidShape }
        let start = Int(environment["JASNA_WINDOW_START"] ?? "") ?? 0
        let requestedCount = Int(environment["JASNA_WINDOW_COUNT"] ?? "") ?? windowCount
        guard start >= 0, start < windowCount, requestedCount > 0 else {
            throw DeformConvError.commandFailed(
                "invalid restoration window range: start \(start), count \(requestedCount), "
                    + "available \(windowCount)"
            )
        }
        return start..<min(windowCount, start + requestedCount)
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
        let inputFrames: [[Float16]]
        let context: String
    }

    struct CompletedRegionRestoration: Sendable {
        let prepared: PreparedRegionRestoration
        let frames: [[Float16]]
        let gpuMilliseconds: Double
        let wallMilliseconds: Double
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

    static func restoreSparseSingleEyeVideoWindows(
        device: MTLDevice,
        inputURL: URL,
        windowsDirectoryURL: URL,
        manifestURL: URL,
        modelsURL: URL,
        weightsURL: URL,
        projection: VRMosaicProjection = .raw,
        workDirectoryURL: URL? = nil
    ) async throws -> Int {
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds,
            eyeLayout: .singleEye
        )
        let manifest = try MosaicRegionManifest.load(from: manifestURL)
        try manifest.validate(for: plan)
        let subdivisionConfiguration = MosaicRegionSubdivisionConfiguration.fromEnvironment(
            ProcessInfo.processInfo.environment
        )
        let windowFrameCounts = stride(
            from: 0,
            to: manifest.frameCount,
            by: SideBySideVideoPlan.temporalWindowFrames
        ).map {
            min(SideBySideVideoPlan.temporalWindowFrames, manifest.frameCount - $0)
        }
        try FileManager.default.createDirectory(
            at: windowsDirectoryURL, withIntermediateDirectories: true
        )
        report(
            "Sparse single-eye plan: \(plan.dimensions.width)×\(plan.dimensions.height), "
                + "\(manifest.frameCount) frames, \(manifest.regions.count) regions, "
                + "VR projection \(projection.rawValue)"
        )
        reportSubdivisionConfiguration(subdivisionConfiguration)
        let decoder = try await FrameDecoder(
            inputURL: inputURL,
            plan: plan,
            sourceDimensions: inputInfo.dimensions,
            cropX: 0
        )
        let encoderWindowsPerSegment = min(
            windowFrameCounts.count,
            max(
                1,
                Int(
                    ProcessInfo.processInfo.environment[
                        "JASNA_ENCODER_WINDOWS_PER_SEGMENT"
                    ] ?? ""
                ) ?? defaultEncoderWindowsPerSegment
            )
        )
        report(
            "HEVC encoder segment: up to \(encoderWindowsPerSegment) recurrence window(s)"
        )
        var completedWindows = 0
        var restoredRegionWindows = 0
        var skippedTileWindows = 0
        var windowIndex = 0
        while windowIndex < windowFrameCounts.count {
            let outputURL = windowsDirectoryURL.appendingPathComponent(
                String(format: "window-%05d.mov", windowIndex + 1)
            )
            let segmentEnd = encoderSegmentEnd(
                windowIndex: windowIndex,
                windowCount: windowFrameCounts.count,
                maximumWindows: encoderWindowsPerSegment
            ) { candidate in
                let candidateURL = windowsDirectoryURL.appendingPathComponent(
                    String(format: "window-%05d.mov", candidate + 1)
                )
                return FileManager.default.fileExists(atPath: candidateURL.path)
            }
            let segmentFrameCount = windowFrameCounts[windowIndex..<segmentEnd].reduce(0, +)
            if await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: segmentFrameCount
            ) {
                report(
                    "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                        + "encoded segment already complete"
                )
                completedWindows += segmentEnd - windowIndex
                windowIndex = segmentEnd
                continue
            }
            if segmentEnd > windowIndex + 1,
               await validWindowOutput(
                   outputURL,
                   dimensions: plan.dimensions,
                   frameCount: windowFrameCounts[windowIndex]
               )
            {
                report(
                    "Window \(windowIndex + 1)/\(windowFrameCounts.count): "
                        + "legacy one-window output already complete"
                )
                completedWindows += 1
                windowIndex += 1
                continue
            }
            if FileManager.default.fileExists(atPath: outputURL.path) {
                let archived = try archiveInterruptedOutput(outputURL)
                report(
                    "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                        + "archived incomplete output at \(archived.path)"
                )
            }

            let writer = try RestoredFrameWriter(device: device, outputURL: outputURL, plan: plan)
            var segmentCacheDirectories = [URL]()
            var segmentOutputFrame = 0
            for currentWindowIndex in windowIndex..<segmentEnd {
                let outputCount = windowFrameCounts[currentWindowIndex]
                let windowStart = currentWindowIndex * SideBySideVideoPlan.temporalWindowFrames
                let frameRange = windowStart..<(windowStart + outputCount)
                let detectedRegions = manifest.regions(intersecting: frameRange)
                let subdivision = MosaicRegionSubdivision.expand(
                    detectedRegions, configuration: subdivisionConfiguration
                )
                let activeRegions = subdivision.regions
                report(
                    "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): decoding "
                        + "\(outputCount) frames; mosaic regions \(detectedRegions.count), "
                        + "model crops \(activeRegions.count)"
                )
                if subdivision.splitRegionCount > 0 {
                    report(
                        "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): "
                            + "split \(subdivision.splitRegionCount) oversized region(s), "
                            + "added \(subdivision.addedModelCropCount) overlapping crop(s)"
                    )
                }
                let baseFrames = try (0..<outputCount).map {
                    try decoder.copyFrame(outputIndex: windowStart + $0)
                }
                let attachments = baseFrames.map {
                    CVBufferCopyAttachments($0, .shouldPropagate)
                }
                var modelFrames = baseFrames
                while modelFrames.count < 3 {
                    guard let last = modelFrames.last else {
                        throw DeformConvError.invalidShape
                    }
                    modelFrames.append(last)
                }
                let cacheVariant = sparseRegionCacheVariant(
                    regions: activeRegions, projection: projection
                )
                let samplingMaps = activeRegions.map {
                    MosaicCropSamplingMap(
                        region: $0,
                        eyeWidth: plan.dimensions.width,
                        eyeHeight: plan.dimensions.height,
                        projection: projection
                    )
                }
                let window = try processRegionWindow(
                    device: device,
                    plan: plan,
                    regions: activeRegions,
                    decodedFrames: modelFrames,
                    outputCount: outputCount,
                    windowIndex: currentWindowIndex,
                    windowCount: windowFrameCounts.count,
                    modelsURL: modelsURL,
                    weightsURL: weightsURL,
                    cacheVariant: cacheVariant,
                    projection: projection,
                    samplingMaps: samplingMaps,
                    workDirectoryURL: workDirectoryURL
                )
                segmentCacheDirectories.append(window.cacheDirectory)
                let compositeStarted = ContinuousClock.now
                try await writer.appendRegionCachedFrames(
                    cacheURLs: window.cacheURLs,
                    attachments: attachments,
                    startFrame: segmentOutputFrame,
                    progressStartFrame: windowStart,
                    progressFrameCount: manifest.frameCount,
                    plan: plan,
                    baseFrames: baseFrames,
                    regions: activeRegions,
                    projection: projection,
                    samplingMaps: samplingMaps
                )
                let compositeMilliseconds = elapsedMilliseconds(since: compositeStarted)
                report(
                    "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): "
                        + "compositor/writer "
                        + "\(String(format: "%.3f", compositeMilliseconds)) ms"
                )
                segmentOutputFrame += outputCount
                if activeRegions.isEmpty {
                    skippedTileWindows += 1
                } else {
                    restoredRegionWindows += activeRegions.count
                }
            }
            let finishStarted = ContinuousClock.now
            try await writer.finish()
            let finishMilliseconds = elapsedMilliseconds(since: finishStarted)
            report(
                "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                    + "encoder finish "
                    + "\(String(format: "%.3f", finishMilliseconds)) ms"
            )
            guard await validWindowOutput(
                outputURL,
                dimensions: plan.dimensions,
                frameCount: segmentFrameCount
            ) else {
                throw DeformConvError.commandFailed(
                    "encoded sparse segment \(windowIndex + 1)-\(segmentEnd) failed validation"
                )
            }
            for cacheDirectory in segmentCacheDirectories {
                try FileManager.default.removeItem(at: cacheDirectory)
            }
            completedWindows += segmentEnd - windowIndex
            report(
                "Windows \(windowIndex + 1)-\(segmentEnd)/\(windowFrameCounts.count): "
                    + "sparse segment encoded, validated, and caches removed"
            )
            windowIndex = segmentEnd
        }
        report(
            "Sparse restoration completed: \(completedWindows) windows, "
                + "\(restoredRegionWindows) mosaic crops restored, "
                + "\(skippedTileWindows) clean windows bypassed"
        )
        return completedWindows
    }

    static func restoreSparseStereoEyeSegment(
        device: MTLDevice,
        leftInputURL: URL,
        rightInputURL: URL,
        outputURL: URL,
        leftManifestURL: URL,
        rightManifestURL: URL,
        modelsURL: URL,
        weightsURL: URL,
        projection: VRMosaicProjection,
        leftWorkDirectoryURL: URL,
        rightWorkDirectoryURL: URL
    ) async throws -> Int {
        let leftInfo = try await SideBySideVideoIO.inspect(url: leftInputURL)
        let rightInfo = try await SideBySideVideoIO.inspect(url: rightInputURL)
        guard leftInfo.dimensions == rightInfo.dimensions,
              abs(leftInfo.nominalFramesPerSecond - rightInfo.nominalFramesPerSecond) < 0.01,
              abs(leftInfo.durationSeconds - rightInfo.durationSeconds) < 0.01
        else {
            throw DeformConvError.commandFailed(
                "direct SBS left/right source segments do not match"
            )
        }
        let eyePlan = try SideBySideVideoPlan(
            width: leftInfo.dimensions.width,
            height: leftInfo.dimensions.height,
            sourceFramesPerSecond: leftInfo.nominalFramesPerSecond,
            durationSeconds: leftInfo.durationSeconds,
            eyeLayout: .singleEye
        )
        let leftManifest = try MosaicRegionManifest.load(from: leftManifestURL)
        let rightManifest = try MosaicRegionManifest.load(from: rightManifestURL)
        try leftManifest.validate(for: eyePlan)
        try rightManifest.validate(for: eyePlan)
        let subdivisionConfiguration = MosaicRegionSubdivisionConfiguration.fromEnvironment(
            ProcessInfo.processInfo.environment
        )
        guard leftManifest.frameCount == rightManifest.frameCount else {
            throw DeformConvError.commandFailed(
                "direct SBS left/right manifests have different frame counts"
            )
        }
        let windowCount = (leftManifest.frameCount
            + SideBySideVideoPlan.temporalWindowFrames - 1)
            / SideBySideVideoPlan.temporalWindowFrames
        let windowRange = try restorationWindowRange(
            windowCount: windowCount,
            environment: ProcessInfo.processInfo.environment
        )
        let rangeStartFrame = windowRange.lowerBound * SideBySideVideoPlan.temporalWindowFrames
        let rangeEndFrame = min(
            leftManifest.frameCount,
            windowRange.upperBound * SideBySideVideoPlan.temporalWindowFrames
        )
        let rangeFrameCount = rangeEndFrame - rangeStartFrame
        let stereoPlan = try SideBySideVideoPlan(
            width: leftInfo.dimensions.width * 2,
            height: leftInfo.dimensions.height,
            sourceFramesPerSecond: leftInfo.nominalFramesPerSecond,
            durationSeconds: Double(rangeFrameCount)
                / SideBySideVideoPlan.outputFramesPerSecond
        )
        if await validWindowOutput(
            outputURL,
            dimensions: stereoPlan.dimensions,
            frameCount: rangeFrameCount
        ) {
            report("Direct SBS segment already complete: \(outputURL.path)")
            return windowRange.count
        }
        if FileManager.default.fileExists(atPath: outputURL.path) {
            let archived = try archiveInterruptedOutput(outputURL)
            report("Archived interrupted direct SBS segment at \(archived.path)")
        }
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let leftDecoder = try await FrameDecoder(
            inputURL: leftInputURL,
            plan: eyePlan,
            sourceDimensions: leftInfo.dimensions
        )
        let rightDecoder = try await FrameDecoder(
            inputURL: rightInputURL,
            plan: eyePlan,
            sourceDimensions: rightInfo.dimensions
        )
        let writer = try RestoredFrameWriter(
            device: device, outputURL: outputURL, plan: stereoPlan
        )
        var completedCacheDirectories = [URL]()
        report(
            "Direct SBS segment: \(stereoPlan.dimensions.width)×"
                + "\(stereoPlan.dimensions.height), \(rangeFrameCount) frames, windows "
                + "\(windowRange.lowerBound + 1)-\(windowRange.upperBound)/\(windowCount)"
        )
        reportSubdivisionConfiguration(subdivisionConfiguration)
        for windowIndex in windowRange {
            let windowStart = windowIndex * SideBySideVideoPlan.temporalWindowFrames
            let outputCount = min(
                SideBySideVideoPlan.temporalWindowFrames,
                leftManifest.frameCount - windowStart
            )
            let frameRange = windowStart..<(windowStart + outputCount)
            let leftDetectedRegions = leftManifest.regions(intersecting: frameRange)
            let rightDetectedRegions = rightManifest.regions(intersecting: frameRange)
            let leftSubdivision = MosaicRegionSubdivision.expand(
                leftDetectedRegions, configuration: subdivisionConfiguration
            )
            let rightSubdivision = MosaicRegionSubdivision.expand(
                rightDetectedRegions, configuration: subdivisionConfiguration
            )
            let leftRegions = leftSubdivision.regions
            let rightRegions = rightSubdivision.regions
            report(
                "Direct SBS window \(windowIndex + 1)/\(windowCount): "
                    + "decoding \(outputCount) frames; left/right regions "
                    + "\(leftDetectedRegions.count)/\(rightDetectedRegions.count), "
                    + "model crops \(leftRegions.count)/\(rightRegions.count)"
            )
            if leftSubdivision.splitRegionCount + rightSubdivision.splitRegionCount > 0 {
                report(
                    "Direct SBS window \(windowIndex + 1)/\(windowCount): split "
                        + "left/right oversized regions "
                        + "\(leftSubdivision.splitRegionCount)/"
                        + "\(rightSubdivision.splitRegionCount), added crops "
                        + "\(leftSubdivision.addedModelCropCount)/"
                        + "\(rightSubdivision.addedModelCropCount)"
                )
            }
            var leftFrames = try (0..<outputCount).map {
                try leftDecoder.copyFrame(outputIndex: windowStart + $0)
            }
            var rightFrames = try (0..<outputCount).map {
                try rightDecoder.copyFrame(outputIndex: windowStart + $0)
            }
            let leftOutputFrames = leftFrames
            let rightOutputFrames = rightFrames
            while leftFrames.count < 3 {
                guard let last = leftFrames.last else { throw DeformConvError.invalidShape }
                leftFrames.append(last)
            }
            while rightFrames.count < 3 {
                guard let last = rightFrames.last else { throw DeformConvError.invalidShape }
                rightFrames.append(last)
            }
            let leftSamplingMaps = leftRegions.map {
                MosaicCropSamplingMap(
                    region: $0,
                    eyeWidth: eyePlan.dimensions.width,
                    eyeHeight: eyePlan.dimensions.height,
                    projection: projection
                )
            }
            let rightSamplingMaps = rightRegions.map {
                MosaicCropSamplingMap(
                    region: $0,
                    eyeWidth: eyePlan.dimensions.width,
                    eyeHeight: eyePlan.dimensions.height,
                    projection: projection
                )
            }
            let leftWindow = try processRegionWindow(
                device: device,
                plan: eyePlan,
                regions: leftRegions,
                decodedFrames: leftFrames,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: windowCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                cacheVariant: sparseRegionCacheVariant(
                    regions: leftRegions, projection: projection
                ),
                projection: projection,
                samplingMaps: leftSamplingMaps,
                workDirectoryURL: leftWorkDirectoryURL
            )
            let rightWindow = try processRegionWindow(
                device: device,
                plan: eyePlan,
                regions: rightRegions,
                decodedFrames: rightFrames,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: windowCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                cacheVariant: sparseRegionCacheVariant(
                    regions: rightRegions, projection: projection
                ),
                projection: projection,
                samplingMaps: rightSamplingMaps,
                workDirectoryURL: rightWorkDirectoryURL
            )
            completedCacheDirectories += [
                leftWindow.cacheDirectory, rightWindow.cacheDirectory,
            ]
            try await writer.appendStereoRegionCachedFrames(
                leftCacheURLs: leftWindow.cacheURLs,
                rightCacheURLs: rightWindow.cacheURLs,
                leftBaseFrames: leftOutputFrames,
                rightBaseFrames: rightOutputFrames,
                leftRegions: leftRegions,
                rightRegions: rightRegions,
                leftSamplingMaps: leftSamplingMaps,
                rightSamplingMaps: rightSamplingMaps,
                presentationStartFrame: windowStart - rangeStartFrame,
                absoluteStartFrame: windowStart,
                progressFrameCount: leftManifest.frameCount,
                plan: stereoPlan,
                projection: projection
            )
        }
        try await writer.finish()
        guard await validWindowOutput(
            outputURL,
            dimensions: stereoPlan.dimensions,
            frameCount: rangeFrameCount
        ) else {
            throw DeformConvError.commandFailed("direct SBS segment failed validation")
        }
        for directory in completedCacheDirectories {
            try FileManager.default.removeItem(at: directory)
        }
        report(
            "Direct SBS segment encoded and validated: \(outputURL.path)"
        )
        return windowRange.count
    }

    private static func processRegionWindow(
        device: MTLDevice,
        plan: SideBySideVideoPlan,
        regions: [MosaicRegion],
        decodedFrames: [CVPixelBuffer],
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
            let urls = resumed?.urls ?? (0..<outputCount).map {
                directory.appendingPathComponent("frame-\($0).fp16")
            }
            var handles = try urls.map { url -> FileHandle in
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
            var cacheWriteMilliseconds: Double = 0
            var restoredModelFrames = 0
            let configuredCheckpointInterval = Int(
                ProcessInfo.processInfo.environment["JASNA_REGION_CHECKPOINT_INTERVAL"] ?? ""
            )
            let checkpointInterval = configuredWorkPath == nil
                ? max(1, regions.count)
                : max(1, configuredCheckpointInterval ?? 5)
            let batchModelsURL = ProcessInfo.processInfo.environment[
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
                    + "model batch \(regionBatchSize)"
            )
            var nextRegion = completedRegions
            while nextRegion < regions.count {
                let batchEnd = min(regions.count, nextRegion + regionBatchSize)
                let extractionStarted = ContinuousClock.now
                let work = try (nextRegion..<batchEnd).map { regionIndex in
                    let windowStartFrame = windowIndex * SideBySideVideoPlan.temporalWindowFrames
                    let region = regions[regionIndex]
                    let localStart = max(0, region.startFrame - windowStartFrame)
                    let localEnd = min(outputCount, region.endFrame - windowStartFrame)
                    guard localStart < localEnd else {
                        throw DeformConvError.commandFailed(
                            "mosaic crop does not intersect its assigned window"
                        )
                    }
                    let samplingMap = samplingMaps[regionIndex]
                    let activeFrames = decodedFrames[localStart..<localEnd]
                    var cropFrames = try activeFrames.map {
                        try samplingMap.extractPlanarRGB(from: $0)
                    }
                    let activeFrameCount = cropFrames.count
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
                        localStart: localStart,
                        activeFrameCount: activeFrameCount,
                        inputFrames: cropFrames,
                        context: context
                    )
                }
                extractionMilliseconds += elapsedMilliseconds(since: extractionStarted)
                let completedWork: [CompletedRegionRestoration]
                if let batchModelsURL,
                   work.count == 2,
                   work[0].inputFrames.count == work[1].inputFrames.count
                {
                    do {
                        completedWork = try restorePreparedRegionBatch(
                            device: device,
                            modelsURL: batchModelsURL,
                            weightsURL: weightsURL,
                            work: work
                        )
                    } catch {
                        report(
                            "Window \(windowIndex + 1)/\(windowCount): batch-2 graph failed "
                                + "(\(error)); retrying both crops independently"
                        )
                        completedWork = try restorePreparedRegionsIndividually(
                            device: device,
                            modelsURL: modelsURL,
                            weightsURL: weightsURL,
                            work: work
                        )
                    }
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
                    let cacheWriteStarted = ContinuousClock.now
                    for frame in 0..<outputCount {
                        if frame >= prepared.localStart
                            && frame < prepared.localStart + prepared.activeFrameCount
                        {
                            let values = restored.frames[
                                min(
                                    frame - prepared.localStart,
                                    prepared.activeFrameCount - 1
                                )
                            ]
                            try values.withUnsafeBytes { bytes in
                                try handles[frame].write(contentsOf: Data(bytes))
                            }
                        } else {
                            try handles[frame].seek(
                                toOffset: UInt64((prepared.regionIndex + 1) * tileBytes)
                            )
                        }
                    }
                    gpuMilliseconds += restored.gpuMilliseconds
                    graphWallMilliseconds += restored.wallMilliseconds
                    restoredModelFrames += prepared.activeFrameCount
                    let completedCount = prepared.regionIndex + 1
                    let shouldCheckpoint = completedCount == regions.count
                        || completedCount.isMultiple(of: checkpointInterval)
                    if shouldCheckpoint {
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
            for handle in handles { try handle.close() }
            handles.removeAll()
            return WindowResult(
                cacheDirectory: directory,
                cacheURLs: urls,
                gpuMilliseconds: gpuMilliseconds,
                cacheBytes: cacheBytes
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

    private static func reportSubdivisionConfiguration(
        _ configuration: MosaicRegionSubdivisionConfiguration
    ) {
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
                + "normalized Metal overlap; adaptive parent mask growth "
                + "\(String(format: "%.3f", configuration.maskGrowthFraction)), feather "
                + "\(String(format: "%.3f", configuration.maskFeatherFraction)), block halo "
                + "\(String(format: "%.3f", configuration.blockResidualGrowthFraction)), "
                + "temporal radius \(configuration.maskTemporalRadius); lower detail "
                + "\(configuration.detailCropCount)x"
                + "\(configuration.detailCropDimension)px"
        )
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
        return try batch.results()
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
        let started = ContinuousClock.now
        let batchedFrames = try work[0].inputFrames.indices.map { frame in
            let values = work.flatMap { $0.inputFrames[frame] }
            guard values.count == 2 * tileElements else {
                throw DeformConvError.invalidShape
            }
            return values
        }
        let restored = try restoreTileFrames(
            device: device,
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            inputFrames: batchedFrames,
            maximumFramesPerChunk: batchedFrames.count,
            batch: 2
        )
        let wallMilliseconds = elapsedMilliseconds(since: started)
        return work.indices.map { sample in
            let start = sample * tileElements
            let end = start + tileElements
            return CompletedRegionRestoration(
                prepared: work[sample],
                frames: restored.frames.map { Array($0[start..<end]) },
                gpuMilliseconds: restored.gpuMilliseconds / Double(work.count),
                wallMilliseconds: wallMilliseconds / Double(work.count)
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
            report(
                "WARNING: \(context) is using finite input-pixel passthrough because all "
                    + "Metal recurrence recovery modes failed (\(lastError))"
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
