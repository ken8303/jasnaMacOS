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
        let restorationIdentity = restorationCacheIdentity(
            sourceURLs: [inputURL], modelsURL: modelsURL, weightsURL: weightsURL
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
                let unsortedActiveRegions = fullDetectedRegionBlendEnabled
                    ? subdivision.regions.map { $0.usingFullDetectedRegionBlend() }
                    : subdivision.regions
                let activeRegions = batchOptimizedRegions(
                    unsortedActiveRegions,
                    windowStartFrame: windowStart,
                    outputCount: outputCount,
                    batch2Enabled: ProcessInfo.processInfo.environment[
                        "JASNA_BATCH2_MODELS_DIR"
                    ] != nil
                )
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
                var baseFrames = [CVPixelBuffer]()
                baseFrames.reserveCapacity(outputCount)
                for localFrame in 0..<outputCount {
                    baseFrames.append(
                        try await decoder.copyFrame(outputIndex: windowStart + localFrame)
                    )
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
                    regions: activeRegions,
                    projection: projection,
                    restorationIdentity: restorationIdentity
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
        let restorationIdentity = restorationCacheIdentity(
            sourceURLs: [leftInputURL, rightInputURL],
            modelsURL: modelsURL,
            weightsURL: weightsURL
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
            startOutputIndex: rangeStartFrame
        ) : nil
        let leftDecoder = sharedSBSInput ? nil : try await FrameDecoder(
            inputURL: leftInputURL,
            plan: eyePlan,
            sourceDimensions: leftInfo.dimensions,
            startOutputIndex: rangeStartFrame
        )
        let rightDecoder = sharedSBSInput ? nil : try await FrameDecoder(
            inputURL: rightInputURL,
            plan: eyePlan,
            sourceDimensions: rightInfo.dimensions,
            startOutputIndex: rangeStartFrame
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
                batch2Enabled: batch2Enabled
            )
            let rightRegions = batchOptimizedRegions(
                unsortedRightRegions,
                windowStartFrame: windowStart,
                outputCount: outputCount,
                batch2Enabled: batch2Enabled
            )
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
            var leftFrames = [CVPixelBuffer]()
            var rightFrames = [CVPixelBuffer]()
            leftFrames.reserveCapacity(outputCount)
            rightFrames.reserveCapacity(outputCount)
            for localFrame in 0..<outputCount {
                if let sharedDecoder {
                    let pair = try await sharedDecoder.copyStereoFrames(
                        outputIndex: windowStart + localFrame
                    )
                    leftFrames.append(pair.left)
                    rightFrames.append(pair.right)
                } else {
                    guard let leftDecoder, let rightDecoder else {
                        throw DeformConvError.commandFailed(
                            "direct SBS eye decoders were not initialized"
                        )
                    }
                    leftFrames.append(
                        try await leftDecoder.copyFrame(outputIndex: windowStart + localFrame)
                    )
                    rightFrames.append(
                        try await rightDecoder.copyFrame(outputIndex: windowStart + localFrame)
                    )
                }
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
                    regions: leftRegions,
                    projection: projection,
                    restorationIdentity: restorationIdentity
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
                    regions: rightRegions,
                    projection: projection,
                    restorationIdentity: restorationIdentity
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
        report(
            "Direct SBS segment encoded and validated: \(outputURL.path)"
        )
        return windowRange.count
    }

}
