import AVFoundation
import CoreVideo
import Foundation
import Metal

@available(macOS 27.0, *)
extension SideBySideRestoration {
    // Keep encoded segments small enough that their per-window FP16 caches stay bounded.
    // Four windows measured 5.75 GiB peak resident memory on an 8K eye-by-eye run,
    // while the former 120-window default retained more than 7 GiB of crop caches.
    static let defaultEncoderWindowsPerSegment = 4

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
        let temporalCropConfiguration = MosaicTemporalCropConfiguration.fromEnvironment(
            ProcessInfo.processInfo.environment
        )
        let temporalWarmupConfiguration = TemporalWarmupConfiguration.fromEnvironment(
            ProcessInfo.processInfo.environment
        )
        let restorationIdentity = restorationCacheIdentity(
            sourceURLs: [inputURL],
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            additionalModelURLs: configuredAdditionalModelURLs(
                environment: ProcessInfo.processInfo.environment
            )
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
        reportTemporalCropConfiguration(temporalCropConfiguration)
        reportTemporalWarmupConfiguration(temporalWarmupConfiguration)
        reportCropExtractionConfiguration()
        var decoder: FrameDecoder?
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
        var subdivisionModelCrops = 0
        var classifiedSubdivisionMasks = 0
        var zeroSubdivisionMasks = 0
        var previousFrames = [CVPixelBuffer]()
        var previousFramesEnd = 0
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

            if decoder == nil {
                let outputStartFrame = windowIndex * SideBySideVideoPlan.temporalWindowFrames
                let decoderStartFrame = max(
                    0, outputStartFrame - temporalWarmupConfiguration.frames
                )
                decoder = try await FrameDecoder(
                    inputURL: inputURL,
                    plan: plan,
                    sourceDimensions: inputInfo.dimensions,
                    cropX: 0,
                    startOutputIndex: decoderStartFrame
                )
                report(
                    "Single-eye decoder seek: frame \(decoderStartFrame) for first "
                        + "unfinished window \(windowIndex + 1)"
                )
            }
            guard let activeDecoder = decoder else {
                throw DeformConvError.commandFailed("single-eye decoder failed to initialize")
            }

            let writer = try RestoredFrameWriter(device: device, outputURL: outputURL, plan: plan)
            var segmentCacheDirectories = [URL]()
            var segmentOutputFrame = 0
            for currentWindowIndex in windowIndex..<segmentEnd {
                let outputCount = windowFrameCounts[currentWindowIndex]
                let windowStart = currentWindowIndex * SideBySideVideoPlan.temporalWindowFrames
                let schedule = TemporalWindowSchedule(
                    outputStartFrame: windowStart,
                    outputFrameCount: outputCount,
                    requestedWarmupFrames: temporalWarmupConfiguration.frames
                )
                let frameRange = windowStart..<(windowStart + outputCount)
                let detectedRegions = manifest.regions(intersecting: frameRange)
                let temporalCrops = MosaicRegionSubdivision.tightenMovingRegions(
                    detectedRegions, configuration: temporalCropConfiguration
                )
                let subdivision = MosaicRegionSubdivision.expand(
                    temporalCrops.regions, configuration: subdivisionConfiguration
                )
                subdivisionModelCrops += subdivision.subdivisionModelCropCount
                classifiedSubdivisionMasks += subdivision.classifiedMaskCropCount
                zeroSubdivisionMasks += subdivision.zeroMaskCropCount
                let unsortedActiveRegions = fullDetectedRegionBlendEnabled
                    ? subdivision.regions.map { $0.usingFullDetectedRegionBlend() }
                    : subdivision.regions
                let activeRegions = batchOptimizedRegions(
                    unsortedActiveRegions,
                    windowStartFrame: windowStart,
                    outputCount: outputCount,
                    batch2Enabled: ProcessInfo.processInfo.environment[
                        "JASNA_BATCH2_MODELS_DIR"
                    ] != nil,
                    temporalWarmupFrames: schedule.warmupFrameCount
                )
                report(
                    "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): decoding "
                        + "\(outputCount) frames; mosaic regions \(detectedRegions.count), "
                        + "model crops \(activeRegions.count)"
                )
                reportModelCropReuseSummary(
                    modelCropReuseSummary(
                        regions: activeRegions,
                        windowStartFrame: windowStart,
                        outputCount: outputCount,
                        temporalWarmupFrames: schedule.warmupFrameCount
                    ),
                    windowIndex: currentWindowIndex
                )
                if subdivision.splitRegionCount > 0 {
                    report(
                        "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): "
                            + "split \(subdivision.splitRegionCount) oversized region(s), "
                            + "added \(subdivision.addedModelCropCount) overlapping crop(s)"
                    )
                    report(
                        "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): "
                            + "subdivision mask occupancy "
                            + "\(subdivision.classifiedMaskCropCount)/"
                            + "\(subdivision.subdivisionModelCropCount) classified, "
                            + "\(subdivision.zeroMaskCropCount) all-zero candidate(s); "
                            + "telemetry only, scheduling unchanged"
                    )
                }
                if temporalCrops.movingRegionCount > 0 {
                    report(
                        "Window \(currentWindowIndex + 1)/\(windowFrameCounts.count): "
                            + "motion-tightened \(temporalCrops.movingRegionCount) region(s), "
                            + "added \(temporalCrops.addedTemporalCropCount) temporal crop(s)"
                    )
                }
                var warmupFrames = [CVPixelBuffer]()
                warmupFrames.reserveCapacity(schedule.warmupFrameCount)
                if schedule.warmupFrameCount > 0,
                   previousFramesEnd == windowStart,
                   previousFrames.count >= schedule.warmupFrameCount
                {
                    warmupFrames.append(
                        contentsOf: previousFrames.suffix(schedule.warmupFrameCount)
                    )
                } else if schedule.warmupFrameCount > 0 {
                    for absoluteFrame in schedule.decodedStartFrame..<windowStart {
                        warmupFrames.append(
                            try await activeDecoder.copyFrame(outputIndex: absoluteFrame)
                        )
                    }
                }
                var baseFrames = [CVPixelBuffer]()
                baseFrames.reserveCapacity(outputCount)
                for localFrame in 0..<outputCount {
                    baseFrames.append(
                        try await activeDecoder.copyFrame(outputIndex: windowStart + localFrame)
                    )
                }
                let attachments = baseFrames.map {
                    CVBufferCopyAttachments($0, .shouldPropagate)
                }
                previousFrames = Array(baseFrames.suffix(temporalWarmupConfiguration.frames))
                previousFramesEnd = windowStart + outputCount
                var modelFrames = warmupFrames + baseFrames
                while modelFrames.count < 3 {
                    guard let last = modelFrames.last else {
                        throw DeformConvError.invalidShape
                    }
                    modelFrames.append(last)
                }
                let cacheVariant = sparseRegionCacheVariant(
                    regions: activeRegions,
                    projection: projection,
                    restorationIdentity: restorationIdentity,
                    temporalWarmupFrames: schedule.warmupFrameCount,
                    allowPassthrough: ProcessInfo.processInfo.environment[
                        "JASNA_ALLOW_PASSTHROUGH"
                    ] == "1"
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
                    decodedStartFrame: schedule.decodedStartFrame,
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
                    inMemoryCache: window.inMemoryRegionCache,
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
        if subdivisionModelCrops > 0 {
            let zeroPercent = classifiedSubdivisionMasks > 0
                ? 100 * Double(zeroSubdivisionMasks) / Double(classifiedSubdivisionMasks)
                : 0
            report(
                "Subdivision mask occupancy for processed windows: "
                    + "\(classifiedSubdivisionMasks)/\(subdivisionModelCrops) classified, "
                    + "\(zeroSubdivisionMasks) all-zero candidate(s) "
                    + "(\(String(format: "%.1f", zeroPercent))% of classified); "
                    + "telemetry only, scheduling unchanged"
            )
        }
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
        let leftManifest = try MosaicRegionManifest.load(from: leftManifestURL)
        let rightManifest = try MosaicRegionManifest.load(from: rightManifestURL)
        let sharedSBSInput = leftInputURL.resolvingSymlinksInPath().standardizedFileURL
            == rightInputURL.resolvingSymlinksInPath().standardizedFileURL
            && leftInfo.dimensions.width == leftManifest.width * 2
            && leftInfo.dimensions.height == leftManifest.height
            && rightManifest.width == leftManifest.width
            && rightManifest.height == leftManifest.height
        let eyeWidth = sharedSBSInput
            ? leftInfo.dimensions.width / 2 : leftInfo.dimensions.width
        let eyePlan = try SideBySideVideoPlan(
            width: eyeWidth,
            height: leftInfo.dimensions.height,
            sourceFramesPerSecond: leftInfo.nominalFramesPerSecond,
            durationSeconds: leftInfo.durationSeconds,
            eyeLayout: .singleEye
        )
        try leftManifest.validate(for: eyePlan)
        try rightManifest.validate(for: eyePlan)
        let subdivisionConfiguration = MosaicRegionSubdivisionConfiguration.fromEnvironment(
            ProcessInfo.processInfo.environment
        )
        let temporalCropConfiguration = MosaicTemporalCropConfiguration.fromEnvironment(
            ProcessInfo.processInfo.environment
        )
        let temporalWarmupConfiguration = TemporalWarmupConfiguration.fromEnvironment(
            ProcessInfo.processInfo.environment
        )
        let restorationIdentity = restorationCacheIdentity(
            sourceURLs: [leftInputURL, rightInputURL],
            modelsURL: modelsURL,
            weightsURL: weightsURL,
            additionalModelURLs: configuredAdditionalModelURLs(
                environment: ProcessInfo.processInfo.environment
            )
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
            width: eyeWidth * 2,
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
        let sharedDecoder = sharedSBSInput ? try await FrameDecoder(
            inputURL: leftInputURL,
            plan: eyePlan,
            sourceDimensions: leftInfo.dimensions,
            startOutputIndex: max(0, rangeStartFrame - temporalWarmupConfiguration.frames)
        ) : nil
        let leftDecoder = sharedSBSInput ? nil : try await FrameDecoder(
            inputURL: leftInputURL,
            plan: eyePlan,
            sourceDimensions: leftInfo.dimensions,
            startOutputIndex: max(0, rangeStartFrame - temporalWarmupConfiguration.frames)
        )
        let rightDecoder = sharedSBSInput ? nil : try await FrameDecoder(
            inputURL: rightInputURL,
            plan: eyePlan,
            sourceDimensions: rightInfo.dimensions,
            startOutputIndex: max(0, rangeStartFrame - temporalWarmupConfiguration.frames)
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
        if sharedSBSInput {
            report("Direct SBS source: one shared 8K decode with in-memory eye crops")
        }
        reportSubdivisionConfiguration(subdivisionConfiguration)
        reportTemporalCropConfiguration(temporalCropConfiguration)
        reportTemporalWarmupConfiguration(temporalWarmupConfiguration)
        reportCropExtractionConfiguration()
        var previousLeftFrames = [CVPixelBuffer]()
        var previousRightFrames = [CVPixelBuffer]()
        var previousFramesEnd = rangeStartFrame
        var leftSubdivisionModelCrops = 0
        var rightSubdivisionModelCrops = 0
        var leftClassifiedSubdivisionMasks = 0
        var rightClassifiedSubdivisionMasks = 0
        var leftZeroSubdivisionMasks = 0
        var rightZeroSubdivisionMasks = 0
        for windowIndex in windowRange {
            let windowStart = windowIndex * SideBySideVideoPlan.temporalWindowFrames
            let outputCount = min(
                SideBySideVideoPlan.temporalWindowFrames,
                leftManifest.frameCount - windowStart
            )
            let schedule = TemporalWindowSchedule(
                outputStartFrame: windowStart,
                outputFrameCount: outputCount,
                requestedWarmupFrames: temporalWarmupConfiguration.frames
            )
            let frameRange = windowStart..<(windowStart + outputCount)
            let leftDetectedRegions = leftManifest.regions(intersecting: frameRange)
            let rightDetectedRegions = rightManifest.regions(intersecting: frameRange)
            let leftTemporalCrops = MosaicRegionSubdivision.tightenMovingRegions(
                leftDetectedRegions, configuration: temporalCropConfiguration
            )
            let rightTemporalCrops = MosaicRegionSubdivision.tightenMovingRegions(
                rightDetectedRegions, configuration: temporalCropConfiguration
            )
            let leftSubdivision = MosaicRegionSubdivision.expand(
                leftTemporalCrops.regions, configuration: subdivisionConfiguration
            )
            let rightSubdivision = MosaicRegionSubdivision.expand(
                rightTemporalCrops.regions, configuration: subdivisionConfiguration
            )
            leftSubdivisionModelCrops += leftSubdivision.subdivisionModelCropCount
            rightSubdivisionModelCrops += rightSubdivision.subdivisionModelCropCount
            leftClassifiedSubdivisionMasks += leftSubdivision.classifiedMaskCropCount
            rightClassifiedSubdivisionMasks += rightSubdivision.classifiedMaskCropCount
            leftZeroSubdivisionMasks += leftSubdivision.zeroMaskCropCount
            rightZeroSubdivisionMasks += rightSubdivision.zeroMaskCropCount
            let unsortedLeftRegions = fullDetectedRegionBlendEnabled
                ? leftSubdivision.regions.map { $0.usingFullDetectedRegionBlend() }
                : leftSubdivision.regions
            let unsortedRightRegions = fullDetectedRegionBlendEnabled
                ? rightSubdivision.regions.map { $0.usingFullDetectedRegionBlend() }
                : rightSubdivision.regions
            let batch2Enabled = ProcessInfo.processInfo.environment[
                "JASNA_BATCH2_MODELS_DIR"
            ] != nil
            let leftRegions = batchOptimizedRegions(
                unsortedLeftRegions,
                windowStartFrame: windowStart,
                outputCount: outputCount,
                batch2Enabled: batch2Enabled,
                temporalWarmupFrames: schedule.warmupFrameCount
            )
            let rightRegions = batchOptimizedRegions(
                unsortedRightRegions,
                windowStartFrame: windowStart,
                outputCount: outputCount,
                batch2Enabled: batch2Enabled,
                temporalWarmupFrames: schedule.warmupFrameCount
            )
            report(
                "Direct SBS window \(windowIndex + 1)/\(windowCount): "
                    + "decoding \(outputCount) frames; left/right regions "
                    + "\(leftDetectedRegions.count)/\(rightDetectedRegions.count), "
                    + "model crops \(leftRegions.count)/\(rightRegions.count)"
            )
            reportModelCropReuseSummary(
                modelCropReuseSummary(
                    regions: leftRegions,
                    windowStartFrame: windowStart,
                    outputCount: outputCount,
                    temporalWarmupFrames: schedule.warmupFrameCount
                ),
                windowIndex: windowIndex,
                eye: "left"
            )
            reportModelCropReuseSummary(
                modelCropReuseSummary(
                    regions: rightRegions,
                    windowStartFrame: windowStart,
                    outputCount: outputCount,
                    temporalWarmupFrames: schedule.warmupFrameCount
                ),
                windowIndex: windowIndex,
                eye: "right"
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
                report(
                    "Direct SBS window \(windowIndex + 1)/\(windowCount): subdivision "
                        + "mask occupancy left/right classified "
                        + "\(leftSubdivision.classifiedMaskCropCount)/"
                        + "\(leftSubdivision.subdivisionModelCropCount), "
                        + "\(rightSubdivision.classifiedMaskCropCount)/"
                        + "\(rightSubdivision.subdivisionModelCropCount); all-zero "
                        + "candidates \(leftSubdivision.zeroMaskCropCount)/"
                        + "\(rightSubdivision.zeroMaskCropCount); telemetry only, "
                        + "scheduling unchanged"
                )
            }
            if leftTemporalCrops.movingRegionCount + rightTemporalCrops.movingRegionCount > 0 {
                report(
                    "Direct SBS window \(windowIndex + 1)/\(windowCount): motion-tightened "
                        + "left/right regions \(leftTemporalCrops.movingRegionCount)/"
                        + "\(rightTemporalCrops.movingRegionCount), added temporal crops "
                        + "\(leftTemporalCrops.addedTemporalCropCount)/"
                        + "\(rightTemporalCrops.addedTemporalCropCount)"
                )
            }
            var leftWarmupFrames = [CVPixelBuffer]()
            var rightWarmupFrames = [CVPixelBuffer]()
            leftWarmupFrames.reserveCapacity(schedule.warmupFrameCount)
            rightWarmupFrames.reserveCapacity(schedule.warmupFrameCount)
            let decodeStarted = ContinuousClock.now
            if schedule.warmupFrameCount > 0,
               previousFramesEnd == windowStart,
               previousLeftFrames.count >= schedule.warmupFrameCount,
               previousRightFrames.count >= schedule.warmupFrameCount
            {
                leftWarmupFrames.append(
                    contentsOf: previousLeftFrames.suffix(schedule.warmupFrameCount)
                )
                rightWarmupFrames.append(
                    contentsOf: previousRightFrames.suffix(schedule.warmupFrameCount)
                )
            } else if schedule.warmupFrameCount > 0 {
                for absoluteFrame in schedule.decodedStartFrame..<windowStart {
                    if let sharedDecoder {
                        let pair = try await sharedDecoder.copyStereoFrames(
                            outputIndex: absoluteFrame
                        )
                        leftWarmupFrames.append(pair.left)
                        rightWarmupFrames.append(pair.right)
                    } else {
                        guard let leftDecoder, let rightDecoder else {
                            throw DeformConvError.commandFailed(
                                "direct SBS eye decoders were not initialized"
                            )
                        }
                        leftWarmupFrames.append(
                            try await leftDecoder.copyFrame(outputIndex: absoluteFrame)
                        )
                        rightWarmupFrames.append(
                            try await rightDecoder.copyFrame(outputIndex: absoluteFrame)
                        )
                    }
                }
            }
            var leftOutputFrames = [CVPixelBuffer]()
            var rightOutputFrames = [CVPixelBuffer]()
            leftOutputFrames.reserveCapacity(outputCount)
            rightOutputFrames.reserveCapacity(outputCount)
            for localFrame in 0..<outputCount {
                if let sharedDecoder {
                    let pair = try await sharedDecoder.copyStereoFrames(
                        outputIndex: windowStart + localFrame
                    )
                    leftOutputFrames.append(pair.left)
                    rightOutputFrames.append(pair.right)
                } else {
                    guard let leftDecoder, let rightDecoder else {
                        throw DeformConvError.commandFailed(
                            "direct SBS eye decoders were not initialized"
                        )
                    }
                    leftOutputFrames.append(
                        try await leftDecoder.copyFrame(outputIndex: windowStart + localFrame)
                    )
                    rightOutputFrames.append(
                        try await rightDecoder.copyFrame(outputIndex: windowStart + localFrame)
                    )
                }
            }
            report(
                "Direct SBS window \(windowIndex + 1)/\(windowCount): decode/split "
                    + "\(String(format: "%.3f", elapsedMilliseconds(since: decodeStarted))) ms"
            )
            previousLeftFrames = Array(
                leftOutputFrames.suffix(temporalWarmupConfiguration.frames)
            )
            previousRightFrames = Array(
                rightOutputFrames.suffix(temporalWarmupConfiguration.frames)
            )
            previousFramesEnd = windowStart + outputCount
            var leftFrames = leftWarmupFrames + leftOutputFrames
            var rightFrames = rightWarmupFrames + rightOutputFrames
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
                decodedStartFrame: schedule.decodedStartFrame,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: windowCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                cacheVariant: sparseRegionCacheVariant(
                    regions: leftRegions,
                    projection: projection,
                    restorationIdentity: restorationIdentity,
                    temporalWarmupFrames: schedule.warmupFrameCount,
                    allowPassthrough: ProcessInfo.processInfo.environment[
                        "JASNA_ALLOW_PASSTHROUGH"
                    ] == "1"
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
                decodedStartFrame: schedule.decodedStartFrame,
                outputCount: outputCount,
                windowIndex: windowIndex,
                windowCount: windowCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                cacheVariant: sparseRegionCacheVariant(
                    regions: rightRegions,
                    projection: projection,
                    restorationIdentity: restorationIdentity,
                    temporalWarmupFrames: schedule.warmupFrameCount,
                    allowPassthrough: ProcessInfo.processInfo.environment[
                        "JASNA_ALLOW_PASSTHROUGH"
                    ] == "1"
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
                leftInMemoryCache: leftWindow.inMemoryRegionCache,
                rightInMemoryCache: rightWindow.inMemoryRegionCache,
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
        if leftSubdivisionModelCrops + rightSubdivisionModelCrops > 0 {
            let classified = leftClassifiedSubdivisionMasks
                + rightClassifiedSubdivisionMasks
            let zero = leftZeroSubdivisionMasks + rightZeroSubdivisionMasks
            let zeroPercent = classified > 0 ? 100 * Double(zero) / Double(classified) : 0
            report(
                "Direct SBS subdivision mask occupancy: left/right classified "
                    + "\(leftClassifiedSubdivisionMasks)/\(leftSubdivisionModelCrops), "
                    + "\(rightClassifiedSubdivisionMasks)/\(rightSubdivisionModelCrops); "
                    + "all-zero candidates \(leftZeroSubdivisionMasks)/"
                    + "\(rightZeroSubdivisionMasks) "
                    + "(\(String(format: "%.1f", zeroPercent))% combined); "
                    + "telemetry only, scheduling unchanged"
            )
        }
        report(
            "Direct SBS segment encoded and validated: \(outputURL.path)"
        )
        return windowRange.count
    }

}
