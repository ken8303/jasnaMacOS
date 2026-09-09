import AVFoundation
import CoreVideo
import Foundation
import Metal

@available(macOS 27.0, *)
extension SideBySideRestoration {
    final class RestoredFrameWriter: @unchecked Sendable {
        // Created only after all CPU/Metal writes finish; the receiver gets read-only ownership.
        private struct FinishedPixelBuffer: @unchecked Sendable {
            let value: CVPixelBuffer
        }

        private struct CompositedRegionFrame {
            let localFrame: Int
            let pixelBuffer: CVPixelBuffer
            let wallSeconds: Double
        }

        private struct CompositeInputReadResult {
            let inputs: [MetalMosaicCompositeInput]
            let totalMilliseconds: Double
            let cacheMilliseconds: Double
            let extractionMilliseconds: Double
        }

        /// Owns everything needed to prepare one frame away from the caller's executor.
        /// Core Video pixel buffers are reference-counted and each preparation uses distinct
        /// source/output surfaces; the finished output is handed read-only to AVFoundation.
        private struct StereoFramePreparationInput: @unchecked Sendable {
            let localFrame: Int
            let absoluteFrame: Int
            let leftCacheURL: URL
            let rightCacheURL: URL
            let leftInMemoryCache: InMemoryRegionFrameCache?
            let rightInMemoryCache: InMemoryRegionFrameCache?
            let leftBaseFrame: CVPixelBuffer
            let rightBaseFrame: CVPixelBuffer
            let leftRegions: [MosaicRegion]
            let rightRegions: [MosaicRegion]
            let leftSamplingMaps: [MosaicCropSamplingMap]
            let rightSamplingMaps: [MosaicCropSamplingMap]
            let plan: SideBySideVideoPlan
            let projection: VRMosaicProjection
        }

        private struct PreparedStereoFrame: @unchecked Sendable {
            let localFrame: Int
            let absoluteFrame: Int
            let outputBuffer: CVPixelBuffer
            let preparationMilliseconds: Double
            let inputMilliseconds: Double
            let cacheMilliseconds: Double
            let extractionMilliseconds: Double
            let compositeMilliseconds: Double
            let fusedFallbackMessage: String?
            let cpuFallbackMessage: String?
        }

        private final class RegionFrameCompositeBatch: @unchecked Sendable {
            private let localFrames: [Int]
            private let cacheURLs: [URL]
            private let inMemoryCache: InMemoryRegionFrameCache?
            private let baseFrames: [CVPixelBuffer]
            private let outputBuffers: [CVPixelBuffer]
            private let dimensions: VideoDimensions
            private let regions: [MosaicRegion]
            private let projection: VRMosaicProjection
            private let samplingMaps: [MosaicCropSamplingMap]
            private let progressStartFrame: Int
            private let metalCompositor: MetalMosaicCompositor?
            private let lock = NSLock()
            private var completed: [CompositedRegionFrame?]
            private var failures: [(any Error)?]

            init(
                localFrames: [Int],
                cacheURLs: [URL],
                inMemoryCache: InMemoryRegionFrameCache?,
                baseFrames: [CVPixelBuffer],
                outputBuffers: [CVPixelBuffer],
                dimensions: VideoDimensions,
                regions: [MosaicRegion],
                projection: VRMosaicProjection,
                samplingMaps: [MosaicCropSamplingMap],
                progressStartFrame: Int,
                metalCompositor: MetalMosaicCompositor?
            ) {
                self.localFrames = localFrames
                self.cacheURLs = cacheURLs
                self.inMemoryCache = inMemoryCache
                self.baseFrames = baseFrames
                self.outputBuffers = outputBuffers
                self.dimensions = dimensions
                self.regions = regions
                self.projection = projection
                self.samplingMaps = samplingMaps
                self.progressStartFrame = progressStartFrame
                self.metalCompositor = metalCompositor
                completed = [CompositedRegionFrame?](repeating: nil, count: localFrames.count)
                failures = [(any Error)?](repeating: nil, count: localFrames.count)
            }

            func execute(_ index: Int) {
                do {
                    let started = Date()
                    let localFrame = localFrames[index]
                    let cache = try inMemoryCache == nil
                        ? FileHandle(forReadingFrom: cacheURLs[index]) : nil
                    defer { try? cache?.close() }
                    var compositeInputs = [MetalMosaicCompositeInput]()
                    compositeInputs.reserveCapacity(regions.count)
                    for (regionIndex, region) in regions.enumerated() {
                        let absoluteFrame = progressStartFrame + localFrame
                        guard region.frameRange.contains(absoluteFrame) else {
                            try cache?.seek(toOffset: UInt64((regionIndex + 1) * tileBytes))
                            continue
                        }
                        let values: [Float16]
                        if let inMemoryCache {
                            values = try inMemoryCache.values(
                                frame: localFrame, region: regionIndex
                            )
                        } else {
                            guard let data = try cache?.read(upToCount: tileBytes),
                                  data.count == tileBytes
                            else {
                                throw DeformConvError.commandFailed(
                                    "restored mosaic-crop cache is truncated"
                                )
                            }
                            var decoded = [Float16](repeating: 0, count: tileElements)
                            _ = decoded.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                            values = decoded
                        }
                        let original = projection == .fisheye
                            ? try samplingMaps[regionIndex].extractPlanarRGB(
                                from: baseFrames[index]
                            )
                            : nil
                        compositeInputs.append(
                            MetalMosaicCompositeInput(
                                region: region.resolvingSegmentationMask(at: absoluteFrame),
                                restored: values,
                                original: original ?? [],
                                samples: samplingMaps[regionIndex].compositeSamples
                            )
                        )
                    }
                    if projection == .fisheye, let metalCompositor {
                        var usedMetal = false
                        do {
                            try metalCompositor.composite(
                                basePixelBuffer: baseFrames[index],
                                outputPixelBuffer: outputBuffers[index],
                                dimensions: dimensions,
                                inputs: compositeInputs
                            )
                            usedMetal = true
                        } catch {
                            report(
                                "WARNING: Metal mosaic compositing failed; using CPU for frame "
                                    + "\(progressStartFrame + localFrame + 1) (\(error))"
                            )
                            try Self.compositeOnCPU(
                                basePixelBuffer: baseFrames[index],
                                outputPixelBuffer: outputBuffers[index],
                                dimensions: dimensions,
                                inputs: compositeInputs,
                                projection: projection
                            )
                        }
                        if usedMetal,
                           localFrame == 0,
                           ProcessInfo.processInfo.environment[
                            "JASNA_VERIFY_METAL_COMPOSITOR"
                           ] == "1"
                        {
                            let cpuReference = try Self.makePixelBuffer(dimensions: dimensions)
                            try Self.compositeOnCPU(
                                basePixelBuffer: baseFrames[index],
                                outputPixelBuffer: cpuReference,
                                dimensions: dimensions,
                                inputs: compositeInputs,
                                projection: projection
                            )
                            let difference = try Self.pixelDifference(
                                outputBuffers[index], cpuReference, dimensions: dimensions
                            )
                            report(
                                "Metal compositor raw check: max byte error "
                                    + "\(difference.maximum), differing bytes "
                                    + "\(difference.differing)/\(difference.total)"
                            )
                        }
                    } else {
                        try Self.compositeOnCPU(
                            basePixelBuffer: baseFrames[index],
                            outputPixelBuffer: outputBuffers[index],
                            dimensions: dimensions,
                            inputs: compositeInputs,
                            projection: projection
                        )
                    }
                    let result = CompositedRegionFrame(
                        localFrame: localFrame,
                        pixelBuffer: outputBuffers[index],
                        wallSeconds: Date().timeIntervalSince(started)
                    )
                    lock.lock()
                    completed[index] = result
                    lock.unlock()
                } catch {
                    lock.lock()
                    failures[index] = error
                    lock.unlock()
                }
            }

            private static func compositeOnCPU(
                basePixelBuffer: CVPixelBuffer,
                outputPixelBuffer: CVPixelBuffer,
                dimensions: VideoDimensions,
                inputs: [MetalMosaicCompositeInput],
                projection: VRMosaicProjection
            ) throws {
                var accumulator = try MosaicRegionFrameAccumulator(
                    basePixelBuffer: basePixelBuffer, dimensions: dimensions
                )
                for input in inputs {
                    try accumulator.composite(
                        region: input.region,
                        planarRGB: input.restored,
                        originalPlanarRGB: projection == .fisheye ? input.original : nil,
                        projection: projection
                    )
                }
                try accumulator.writeBGRA(to: outputPixelBuffer)
            }

            private static func makePixelBuffer(
                dimensions: VideoDimensions
            ) throws -> CVPixelBuffer {
                var optionalBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(
                    nil,
                    dimensions.width,
                    dimensions.height,
                    kCVPixelFormatType_32BGRA,
                    [
                        kCVPixelBufferMetalCompatibilityKey as String: true,
                        kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                    ] as CFDictionary,
                    &optionalBuffer
                )
                guard status == kCVReturnSuccess, let buffer = optionalBuffer else {
                    throw DeformConvError.commandFailed(
                        "failed allocating Metal compositor verification frame"
                    )
                }
                return buffer
            }

            private static func pixelDifference(
                _ lhs: CVPixelBuffer,
                _ rhs: CVPixelBuffer,
                dimensions: VideoDimensions
            ) throws -> (maximum: Int, differing: Int, total: Int) {
                CVPixelBufferLockBaseAddress(lhs, .readOnly)
                CVPixelBufferLockBaseAddress(rhs, .readOnly)
                defer {
                    CVPixelBufferUnlockBaseAddress(rhs, .readOnly)
                    CVPixelBufferUnlockBaseAddress(lhs, .readOnly)
                }
                guard let lhsBase = CVPixelBufferGetBaseAddress(lhs),
                      let rhsBase = CVPixelBufferGetBaseAddress(rhs)
                else {
                    throw DeformConvError.commandFailed(
                        "Metal compositor verification frame is not CPU accessible"
                    )
                }
                let packedRowBytes = dimensions.width * 4
                var maximum = 0
                var differing = 0
                for row in 0..<dimensions.height {
                    let lhsRow = lhsBase.advanced(
                        by: row * CVPixelBufferGetBytesPerRow(lhs)
                    ).assumingMemoryBound(to: UInt8.self)
                    let rhsRow = rhsBase.advanced(
                        by: row * CVPixelBufferGetBytesPerRow(rhs)
                    ).assumingMemoryBound(to: UInt8.self)
                    for column in 0..<packedRowBytes {
                        let difference = abs(Int(lhsRow[column]) - Int(rhsRow[column]))
                        maximum = max(maximum, difference)
                        if difference != 0 { differing += 1 }
                    }
                }
                return (maximum, differing, packedRowBytes * dimensions.height)
            }

            func results() throws -> [CompositedRegionFrame] {
                lock.lock()
                defer { lock.unlock() }
                if let failure = failures.compactMap({ $0 }).first { throw failure }
                guard completed.allSatisfy({ $0 != nil }) else {
                    throw DeformConvError.commandFailed(
                        "parallel mosaic compositing was incomplete"
                    )
                }
                return completed.compactMap { $0 }.sorted { $0.localFrame < $1.localFrame }
            }
        }

        private let writer: AVAssetWriter
        private let receiver: AVAssetWriterInput.PixelBufferReceiver
        private let pixelBufferPool: CVPixelBufferPool
        private let metalCompositor: MetalMosaicCompositor?
        private var directStereoFrameCount = 0
        private var fusedStereoFallbackCount = 0
        private var cpuStereoFallbackCount = 0

        init(device: MTLDevice, outputURL: URL, plan: SideBySideVideoPlan) throws {
            metalCompositor = ProcessInfo.processInfo.environment["JASNA_METAL_COMPOSITOR"] == "0"
                ? nil : try? MetalMosaicCompositor(device: device)
            writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
            let automaticBitRate = plan.dimensions.pixelCount * 5 / 2
            let configuredBitRate = Int(
                ProcessInfo.processInfo.environment["JASNA_VIDEO_BITRATE"] ?? ""
            )
            let bitRate = min(
                160_000_000, max(8_000_000, configuredBitRate ?? automaticBitRate)
            )
            let input = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.hevc,
                    AVVideoWidthKey: plan.dimensions.width,
                    AVVideoHeightKey: plan.dimensions.height,
                    AVVideoCompressionPropertiesKey: [
                        AVVideoAverageBitRateKey: bitRate,
                        AVVideoExpectedSourceFrameRateKey: 30,
                        AVVideoMaxKeyFrameIntervalKey: 60,
                        AVVideoAllowFrameReorderingKey: false,
                    ],
                ]
            )
            // Match the packet time base used by clean bypass segments so the
            // grouped output can be concatenated without a second file rewrite.
            input.mediaTimeScale = 600
            guard writer.canAdd(input) else {
                throw DeformConvError.commandFailed("video writer rejected restored frames")
            }
            var creationAttributes = CVPixelBufferCreationAttributes(
                pixelFormatType: CVPixelFormatType(
                    rawValue: kCVPixelFormatType_32BGRA
                ),
                size: CVImageSize(
                    width: plan.dimensions.width, height: plan.dimensions.height
                ),
                compatibility: [.metalTexture]
            )
            creationAttributes.backing = .ioSurface
            receiver = writer.inputPixelBufferReceiver(
                for: input, pixelBufferAttributes: creationAttributes
            )
            let legacyAttributes: CFDictionary = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: plan.dimensions.width,
                kCVPixelBufferHeightKey as String: plan.dimensions.height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ] as CFDictionary
            var optionalPool: CVPixelBufferPool?
            let poolStatus = CVPixelBufferPoolCreate(
                nil, nil, legacyAttributes, &optionalPool
            )
            guard poolStatus == kCVReturnSuccess, let optionalPool else {
                throw DeformConvError.commandFailed("failed creating restored-frame pool")
            }
            pixelBufferPool = optionalPool
            try writer.start()
            writer.startSession(atSourceTime: .zero)
        }

        func appendCachedFrames(
            cacheURLs: [URL],
            attachments: [CFDictionary?],
            startFrame: Int,
            progressStartFrame: Int? = nil,
            plan: SideBySideVideoPlan,
            tiles: [VideoTile]? = nil,
            baseFrames: [CVPixelBuffer]? = nil,
            regions: [MosaicRegion] = []
        ) async throws {
            let compositeTiles = tiles ?? plan.tiles
            guard baseFrames == nil || baseFrames?.count == cacheURLs.count else {
                throw DeformConvError.invalidShape
            }
            report(
                "Compositing and encoding \(cacheURLs.count) cached frame(s) "
                    + "starting at output frame \(startFrame)"
            )
            for localFrame in cacheURLs.indices {
                let frameStarted = Date()
                let progressFrame = (progressStartFrame ?? startFrame) + localFrame
                report(
                    "Compositing output frame \(progressFrame + 1)/"
                        + "\(plan.frameRate.outputFrameCount)"
                )
                let cache = try FileHandle(forReadingFrom: cacheURLs[localFrame])
                var restoredTiles = [(VideoTile, [Float16])]()
                restoredTiles.reserveCapacity(compositeTiles.count)
                for tile in compositeTiles {
                    guard let data = try cache.read(upToCount: tileBytes), data.count == tileBytes else {
                        try? cache.close()
                        throw DeformConvError.commandFailed("restored tile cache is truncated")
                    }
                    var values = [Float16](repeating: 0, count: tileElements)
                    _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                    restoredTiles.append((tile, values))
                }
                try cache.close()
                var optionalBuffer: CVPixelBuffer?
                let status = CVPixelBufferPoolCreatePixelBuffer(
                    nil, pixelBufferPool, &optionalBuffer
                )
                guard status == kCVReturnSuccess, let pixelBuffer = optionalBuffer else {
                    throw DeformConvError.commandFailed("failed allocating restored output frame")
                }
                if let frameAttachments = attachments[localFrame] {
                    CVBufferSetAttachments(pixelBuffer, frameAttachments, .shouldPropagate)
                }
                if let baseFrames {
                    var accumulator = try SparseTileFrameAccumulator(
                        basePixelBuffer: baseFrames[localFrame],
                        dimensions: plan.dimensions
                    )
                    for (tile, values) in restoredTiles {
                        try accumulator.accumulate(tile: tile, planarRGB: values)
                    }
                    try accumulator.writeBGRA(to: pixelBuffer, regions: regions)
                } else {
                    var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
                    for (tile, values) in restoredTiles {
                        try accumulator.accumulate(tile: tile, planarRGB: values)
                    }
                    try accumulator.writeBGRA(to: pixelBuffer)
                }
                let outputFrame = startFrame + localFrame
                let time = CMTime(value: CMTimeValue(outputFrame), timescale: 30)
                try await append(FinishedPixelBuffer(value: pixelBuffer), at: time, frame: outputFrame)
                report(
                    "Queued output frame \(progressFrame + 1)/"
                        + "\(plan.frameRate.outputFrameCount) in "
                        + "\(String(format: "%.3f", Date().timeIntervalSince(frameStarted))) s"
                )
            }
        }

        func appendRegionCachedFrames(
            cacheURLs: [URL],
            inMemoryCache: InMemoryRegionFrameCache? = nil,
            attachments: [CFDictionary?],
            startFrame: Int,
            progressStartFrame: Int,
            progressFrameCount: Int? = nil,
            plan: SideBySideVideoPlan,
            baseFrames: [CVPixelBuffer],
            regions: [MosaicRegion],
            projection: VRMosaicProjection = .raw,
            samplingMaps: [MosaicCropSamplingMap]
        ) async throws {
            guard cacheURLs.count == baseFrames.count,
                  attachments.count == cacheURLs.count,
                  samplingMaps.count == regions.count
            else { throw DeformConvError.invalidShape }
            report(
                "Compositing \(regions.count) restored mosaic crops into "
                    + "\(cacheURLs.count) frame(s)"
            )
            let configuredConcurrency = Int(
                ProcessInfo.processInfo.environment["JASNA_COMPOSITE_CONCURRENCY"] ?? ""
            )
            let compositeConcurrency = min(
                2, max(1, configuredConcurrency ?? 1)
            )
            report("Frame compositing concurrency: \(compositeConcurrency)")
            report(
                "Fisheye compositor: "
                    + {
                        guard projection == .fisheye, let metalCompositor else {
                            return "CPU"
                        }
                        return metalCompositor.prefersTextureSurfaces
                            ? "Metal zero-copy texture" : "Metal buffer-copy fallback"
                    }()
            )
            if projection == .fisheye, let metalCompositor {
                report("Metal compositor cache limits: \(metalCompositor.memoryBudgetDescription)")
            }
            var nextFrame = 0
            while nextFrame < cacheURLs.count {
                let batchEnd = min(cacheURLs.count, nextFrame + compositeConcurrency)
                let localFrames = Array(nextFrame..<batchEnd)
                var outputBuffers = [CVPixelBuffer]()
                outputBuffers.reserveCapacity(localFrames.count)
                for localFrame in localFrames {
                    var optionalBuffer: CVPixelBuffer?
                    let status = CVPixelBufferPoolCreatePixelBuffer(
                        nil, pixelBufferPool, &optionalBuffer
                    )
                    guard status == kCVReturnSuccess, let pixelBuffer = optionalBuffer else {
                        throw DeformConvError.commandFailed(
                            "failed allocating restored output frame"
                        )
                    }
                    if let frameAttachments = attachments[localFrame] {
                        CVBufferSetAttachments(pixelBuffer, frameAttachments, .shouldPropagate)
                    }
                    outputBuffers.append(pixelBuffer)
                }
                let batch = RegionFrameCompositeBatch(
                    localFrames: localFrames,
                    cacheURLs: localFrames.map { cacheURLs[$0] },
                    inMemoryCache: inMemoryCache,
                    baseFrames: localFrames.map { baseFrames[$0] },
                    outputBuffers: outputBuffers,
                    dimensions: plan.dimensions,
                    regions: regions,
                    projection: projection,
                    samplingMaps: samplingMaps,
                    progressStartFrame: progressStartFrame,
                    metalCompositor: metalCompositor
                )
                if localFrames.count == 1 {
                    batch.execute(0)
                } else {
                    DispatchQueue.concurrentPerform(iterations: localFrames.count) {
                        batch.execute($0)
                    }
                }
                for composited in try batch.results() {
                    let outputFrame = startFrame + composited.localFrame
                    let time = CMTime(value: CMTimeValue(outputFrame), timescale: 30)
                    try await append(
                        FinishedPixelBuffer(value: composited.pixelBuffer),
                        at: time, frame: outputFrame
                    )
                    report(
                        "Queued crop-restored output frame "
                            + "\(progressStartFrame + composited.localFrame + 1)/"
                            + "\(progressFrameCount ?? plan.frameRate.outputFrameCount); "
                            + "composite \(String(format: "%.3f", composited.wallSeconds)) s"
                    )
                }
                nextFrame = batchEnd
            }
        }

        func appendStereoRegionCachedFrames(
            leftCacheURLs: [URL],
            rightCacheURLs: [URL],
            leftInMemoryCache: InMemoryRegionFrameCache? = nil,
            rightInMemoryCache: InMemoryRegionFrameCache? = nil,
            leftBaseFrames: [CVPixelBuffer],
            rightBaseFrames: [CVPixelBuffer],
            leftRegions: [MosaicRegion],
            rightRegions: [MosaicRegion],
            leftSamplingMaps: [MosaicCropSamplingMap],
            rightSamplingMaps: [MosaicCropSamplingMap],
            presentationStartFrame: Int,
            absoluteStartFrame: Int,
            progressFrameCount: Int,
            plan: SideBySideVideoPlan,
            projection: VRMosaicProjection
        ) async throws {
            let frameCount = leftCacheURLs.count
            guard frameCount == rightCacheURLs.count,
                  frameCount == leftBaseFrames.count,
                  frameCount == rightBaseFrames.count,
                  leftRegions.count == leftSamplingMaps.count,
                  rightRegions.count == rightSamplingMaps.count,
                  plan.eyeLayout == .sideBySide
            else { throw DeformConvError.invalidShape }
            report(
                "Direct SBS compositing \(frameCount) frame(s), left/right regions "
                    + "\(leftRegions.count)/\(rightRegions.count)"
            )
            guard frameCount > 0 else { return }
            let writerStarted = ContinuousClock.now
            var totalPreparationMilliseconds = 0.0
            var totalEncoderWaitMilliseconds = 0.0
            var totalInputMilliseconds = 0.0
            var totalCacheMilliseconds = 0.0
            var totalExtractionMilliseconds = 0.0
            var totalCompositeMilliseconds = 0.0
            let configuredDepth = Int(
                ProcessInfo.processInfo.environment["JASNA_STEREO_WRITER_DEPTH"] ?? ""
            ) ?? 2
            let pipelineDepth = min(2, max(1, configuredDepth))
            report("Direct SBS writer pipeline depth: \(pipelineDepth)")

            func preparationInput(_ localFrame: Int) -> StereoFramePreparationInput {
                StereoFramePreparationInput(
                    localFrame: localFrame,
                    absoluteFrame: absoluteStartFrame + localFrame,
                    leftCacheURL: leftCacheURLs[localFrame],
                    rightCacheURL: rightCacheURLs[localFrame],
                    leftInMemoryCache: leftInMemoryCache,
                    rightInMemoryCache: rightInMemoryCache,
                    leftBaseFrame: leftBaseFrames[localFrame],
                    rightBaseFrame: rightBaseFrames[localFrame],
                    leftRegions: leftRegions,
                    rightRegions: rightRegions,
                    leftSamplingMaps: leftSamplingMaps,
                    rightSamplingMaps: rightSamplingMaps,
                    plan: plan,
                    projection: projection
                )
            }

            var pendingPreparation: Task<PreparedStereoFrame, Error>?
            if pipelineDepth == 2 {
                let first = preparationInput(0)
                pendingPreparation = Task.detached { [self] in
                    try prepareStereoFrame(first)
                }
            }
            for localFrame in 0..<frameCount {
                directStereoFrameCount += 1
                let prepared: PreparedStereoFrame
                if let pendingPreparation {
                    prepared = try await pendingPreparation.value
                } else {
                    prepared = try prepareStereoFrame(preparationInput(localFrame))
                }
                let nextFrame = localFrame + 1
                if pipelineDepth == 2, nextFrame < frameCount {
                    let next = preparationInput(nextFrame)
                    pendingPreparation = Task.detached { [self] in
                        try prepareStereoFrame(next)
                    }
                } else {
                    pendingPreparation = nil
                }
                if let message = prepared.fusedFallbackMessage {
                    fusedStereoFallbackCount += 1
                    report(message)
                }
                if let message = prepared.cpuFallbackMessage {
                    cpuStereoFallbackCount += 1
                    report(message)
                }
                let presentationTime = CMTime(
                    value: CMTimeValue(presentationStartFrame + localFrame), timescale: 30
                )
                let appendStarted = ContinuousClock.now
                try await append(
                    FinishedPixelBuffer(value: prepared.outputBuffer), at: presentationTime,
                    frame: presentationStartFrame + localFrame
                )
                let appendMilliseconds = SideBySideRestoration.elapsedMilliseconds(
                    since: appendStarted
                )
                totalPreparationMilliseconds += prepared.preparationMilliseconds
                totalEncoderWaitMilliseconds += appendMilliseconds
                totalInputMilliseconds += prepared.inputMilliseconds
                totalCacheMilliseconds += prepared.cacheMilliseconds
                totalExtractionMilliseconds += prepared.extractionMilliseconds
                totalCompositeMilliseconds += prepared.compositeMilliseconds
                if localFrame == frameCount - 1
                    || (prepared.absoluteFrame + 1).isMultiple(of: 30)
                    || appendMilliseconds >= 250
                {
                    report(
                        "Queued direct SBS frame \(prepared.absoluteFrame + 1)/"
                            + "\(progressFrameCount); prepare/composite "
                            + "\(String(format: "%.3f", prepared.preparationMilliseconds)) ms, "
                            + "encoder wait \(String(format: "%.3f", appendMilliseconds)) ms"
                    )
                }
            }
            report(
                "Direct SBS writer phases: \(frameCount) frames, wall "
                    + "\(String(format: "%.3f", SideBySideRestoration.elapsedMilliseconds(since: writerStarted))) ms, "
                    + "preparation sum \(String(format: "%.3f", totalPreparationMilliseconds)) ms, "
                    + "encoder wait sum \(String(format: "%.3f", totalEncoderWaitMilliseconds)) ms"
            )
            let inputAssemblyMilliseconds = max(
                0,
                totalInputMilliseconds - totalCacheMilliseconds
                    - totalExtractionMilliseconds
            )
            let otherPreparationMilliseconds = max(
                0,
                totalPreparationMilliseconds - totalInputMilliseconds
                    - totalCompositeMilliseconds
            )
            report(
                "Direct SBS preparation detail: inputs "
                    + "\(String(format: "%.3f", totalInputMilliseconds)) ms "
                    + "(cache \(String(format: "%.3f", totalCacheMilliseconds)) ms, "
                    + "crop extraction \(String(format: "%.3f", totalExtractionMilliseconds)) ms, "
                    + "mask/assembly \(String(format: "%.3f", inputAssemblyMilliseconds)) ms); "
                    + "composite \(String(format: "%.3f", totalCompositeMilliseconds)) ms; "
                    + "buffers/attachments \(String(format: "%.3f", otherPreparationMilliseconds)) ms"
            )
        }

        private func prepareStereoFrame(
            _ input: StereoFramePreparationInput
        ) throws -> PreparedStereoFrame {
            let frameStarted = ContinuousClock.now
            var optionalOutput: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(
                nil, pixelBufferPool, &optionalOutput
            )
            guard status == kCVReturnSuccess, let outputBuffer = optionalOutput else {
                throw DeformConvError.commandFailed("failed allocating direct SBS frame")
            }
            let leftRead = try Self.readCompositeInputs(
                cacheURL: input.leftCacheURL,
                inMemoryCache: input.leftInMemoryCache,
                cacheFrame: input.localFrame,
                baseFrame: input.leftBaseFrame,
                regions: input.leftRegions,
                samplingMaps: input.leftSamplingMaps,
                absoluteFrame: input.absoluteFrame,
                xOffset: 0
            )
            let rightRead = try Self.readCompositeInputs(
                cacheURL: input.rightCacheURL,
                inMemoryCache: input.rightInMemoryCache,
                cacheFrame: input.localFrame,
                baseFrame: input.rightBaseFrame,
                regions: input.rightRegions,
                samplingMaps: input.rightSamplingMaps,
                absoluteFrame: input.absoluteFrame,
                xOffset: input.plan.eyeDimensions.width
            )
            let compositeInputs = leftRead.inputs + rightRead.inputs
            let compositeStarted = ContinuousClock.now
            var fusedFallbackMessage: String?
            var cpuFallbackMessage: String?
            if input.projection == .fisheye, let metalCompositor {
                do {
                    try metalCompositor.compositeStereo(
                        leftPixelBuffer: input.leftBaseFrame,
                        rightPixelBuffer: input.rightBaseFrame,
                        outputPixelBuffer: outputBuffer,
                        dimensions: input.plan.dimensions,
                        inputs: compositeInputs
                    )
                } catch {
                    fusedFallbackMessage = "WARNING: Fused Metal stereo composite failed; "
                        + "using split path for frame \(input.absoluteFrame + 1) (\(error))"
                    do {
                        try metalCompositor.copyStereo(
                            leftPixelBuffer: input.leftBaseFrame,
                            rightPixelBuffer: input.rightBaseFrame,
                            outputPixelBuffer: outputBuffer,
                            dimensions: input.plan.dimensions
                        )
                    } catch {
                        cpuFallbackMessage = "WARNING: Metal stereo copy failed; using CPU for frame "
                            + "\(input.absoluteFrame + 1) (\(error))"
                        try Self.copyStereoPixelBuffers(
                            left: input.leftBaseFrame,
                            right: input.rightBaseFrame,
                            destination: outputBuffer,
                            dimensions: input.plan.dimensions
                        )
                    }
                    try metalCompositor.compositeInPlace(
                        pixelBuffer: outputBuffer,
                        dimensions: input.plan.dimensions,
                        inputs: compositeInputs
                    )
                }
            } else {
                try Self.copyStereoPixelBuffers(
                    left: input.leftBaseFrame,
                    right: input.rightBaseFrame,
                    destination: outputBuffer,
                    dimensions: input.plan.dimensions
                )
                guard input.projection == .raw else {
                    throw DeformConvError.commandFailed(
                        "direct fisheye SBS output requires the Metal compositor"
                    )
                }
                var accumulator = try MosaicRegionFrameAccumulator(
                    basePixelBuffer: outputBuffer, dimensions: input.plan.dimensions
                )
                for composite in compositeInputs {
                    try accumulator.composite(
                        region: composite.region,
                        planarRGB: composite.restored,
                        originalPlanarRGB: nil,
                        projection: .raw
                    )
                }
                try accumulator.writeBGRA(to: outputBuffer)
            }
            let compositeMilliseconds = SideBySideRestoration.elapsedMilliseconds(
                since: compositeStarted
            )
            if let attachments = CVBufferCopyAttachments(
                input.leftBaseFrame, .shouldPropagate
            ) {
                CVBufferSetAttachments(outputBuffer, attachments, .shouldPropagate)
            }
            return PreparedStereoFrame(
                localFrame: input.localFrame,
                absoluteFrame: input.absoluteFrame,
                outputBuffer: outputBuffer,
                preparationMilliseconds: SideBySideRestoration.elapsedMilliseconds(
                    since: frameStarted
                ),
                inputMilliseconds: leftRead.totalMilliseconds + rightRead.totalMilliseconds,
                cacheMilliseconds: leftRead.cacheMilliseconds + rightRead.cacheMilliseconds,
                extractionMilliseconds: leftRead.extractionMilliseconds
                    + rightRead.extractionMilliseconds,
                compositeMilliseconds: compositeMilliseconds,
                fusedFallbackMessage: fusedFallbackMessage,
                cpuFallbackMessage: cpuFallbackMessage
            )
        }

        private static func readCompositeInputs(
            cacheURL: URL,
            inMemoryCache: InMemoryRegionFrameCache?,
            cacheFrame: Int,
            baseFrame: CVPixelBuffer,
            regions: [MosaicRegion],
            samplingMaps: [MosaicCropSamplingMap],
            absoluteFrame: Int,
            xOffset: Int
        ) throws -> CompositeInputReadResult {
            let totalStarted = ContinuousClock.now
            var cacheMilliseconds = 0.0
            var extractionMilliseconds = 0.0
            let cacheOpenStarted = ContinuousClock.now
            let cache = try inMemoryCache == nil ? FileHandle(forReadingFrom: cacheURL) : nil
            cacheMilliseconds += SideBySideRestoration.elapsedMilliseconds(
                since: cacheOpenStarted
            )
            defer { try? cache?.close() }
            var result = [MetalMosaicCompositeInput]()
            result.reserveCapacity(regions.count)
            for (index, region) in regions.enumerated() {
                guard region.frameRange.contains(absoluteFrame) else {
                    let seekStarted = ContinuousClock.now
                    try cache?.seek(toOffset: UInt64((index + 1) * tileBytes))
                    cacheMilliseconds += SideBySideRestoration.elapsedMilliseconds(
                        since: seekStarted
                    )
                    continue
                }
                let restored: [Float16]
                let cacheReadStarted = ContinuousClock.now
                if let inMemoryCache {
                    restored = try inMemoryCache.values(frame: cacheFrame, region: index)
                } else {
                    guard let data = try cache?.read(upToCount: tileBytes),
                          data.count == tileBytes
                    else {
                        throw DeformConvError.commandFailed(
                            "direct SBS mosaic-crop cache is truncated"
                        )
                    }
                    var decoded = [Float16](repeating: 0, count: tileElements)
                    _ = decoded.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                    restored = decoded
                }
                cacheMilliseconds += SideBySideRestoration.elapsedMilliseconds(
                    since: cacheReadStarted
                )
                let extractionStarted = ContinuousClock.now
                let original = try samplingMaps[index].extractPlanarRGB(from: baseFrame)
                extractionMilliseconds += SideBySideRestoration.elapsedMilliseconds(
                    since: extractionStarted
                )
                result.append(MetalMosaicCompositeInput(
                    region: translated(
                        region.resolvingSegmentationMask(at: absoluteFrame),
                        xOffset: xOffset
                    ),
                    restored: restored,
                    original: original,
                    samples: samplingMaps[index].compositeSamples
                ))
            }
            return CompositeInputReadResult(
                inputs: result,
                totalMilliseconds: SideBySideRestoration.elapsedMilliseconds(
                    since: totalStarted
                ),
                cacheMilliseconds: cacheMilliseconds,
                extractionMilliseconds: extractionMilliseconds
            )
        }

        static func translated(_ region: MosaicRegion, xOffset: Int) -> MosaicRegion {
            guard xOffset != 0 else { return region }
            return MosaicRegion(
                startFrame: region.startFrame,
                endFrame: region.endFrame,
                x: region.x + xOffset,
                y: region.y,
                width: region.width,
                height: region.height,
                confidence: region.confidence,
                blendX: region.blendX.map { $0 + xOffset },
                blendY: region.blendY,
                blendWidth: region.blendWidth,
                blendHeight: region.blendHeight,
                maskWidth: region.maskWidth,
                maskHeight: region.maskHeight,
                maskData: region.maskData,
                maskKeyframes: region.maskKeyframes,
                subdivisionGroup: region.subdivisionGroup.map {
                    xOffset == 0 ? $0 : $0 + 1_000_000
                },
                detailBlendFeather: region.detailBlendFeather
            )
        }

        private static func copyStereoPixelBuffers(
            left: CVPixelBuffer,
            right: CVPixelBuffer,
            destination: CVPixelBuffer,
            dimensions: VideoDimensions
        ) throws {
            let eyeWidth = dimensions.width / 2
            guard dimensions.width.isMultiple(of: 2),
                  CVPixelBufferGetWidth(left) == eyeWidth,
                  CVPixelBufferGetWidth(right) == eyeWidth,
                  CVPixelBufferGetHeight(left) == dimensions.height,
                  CVPixelBufferGetHeight(right) == dimensions.height,
                  CVPixelBufferGetWidth(destination) == dimensions.width,
                  CVPixelBufferGetHeight(destination) == dimensions.height
            else { throw DeformConvError.invalidShape }
            CVPixelBufferLockBaseAddress(left, .readOnly)
            CVPixelBufferLockBaseAddress(right, .readOnly)
            CVPixelBufferLockBaseAddress(destination, [])
            defer {
                CVPixelBufferUnlockBaseAddress(destination, [])
                CVPixelBufferUnlockBaseAddress(right, .readOnly)
                CVPixelBufferUnlockBaseAddress(left, .readOnly)
            }
            guard let leftBase = CVPixelBufferGetBaseAddress(left),
                  let rightBase = CVPixelBufferGetBaseAddress(right),
                  let destinationBase = CVPixelBufferGetBaseAddress(destination)
            else { throw DeformConvError.commandFailed("direct SBS frame is not CPU accessible") }
            let eyeBytes = eyeWidth * 4
            for row in 0..<dimensions.height {
                let destinationRow = destinationBase.advanced(
                    by: row * CVPixelBufferGetBytesPerRow(destination)
                )
                destinationRow.copyMemory(
                    from: leftBase.advanced(by: row * CVPixelBufferGetBytesPerRow(left)),
                    byteCount: eyeBytes
                )
                destinationRow.advanced(by: eyeBytes).copyMemory(
                    from: rightBase.advanced(by: row * CVPixelBufferGetBytesPerRow(right)),
                    byteCount: eyeBytes
                )
            }
        }

        func finish() async throws {
            receiver.finish()
            await withCheckedContinuation { continuation in
                writer.finishWriting { continuation.resume() }
            }
            guard writer.status == .completed else {
                throw writer.error ?? DeformConvError.commandFailed("video writer did not complete")
            }
            if directStereoFrameCount > 0 {
                let status = fusedStereoFallbackCount == 0 && cpuStereoFallbackCount == 0
                    ? "PASS" : "WARNING"
                report(
                    "Stereo compositor safety summary: \(status), "
                        + "\(directStereoFrameCount) frames, fused fallbacks "
                        + "\(fusedStereoFallbackCount), CPU fallbacks "
                        + "\(cpuStereoFallbackCount)"
                )
            }
        }

        private func append(
            _ pixelBuffer: consuming FinishedPixelBuffer,
            at presentationTime: CMTime,
            frame: Int
        ) async throws {
            let readOnly = CVReadOnlyPixelBuffer(unsafeBuffer: pixelBuffer.value)
            do {
                try await receiver.append(readOnly, with: presentationTime)
            } catch {
                throw writer.error
                    ?? DeformConvError.commandFailed(
                        "failed encoding frame \(frame): \(error)"
                    )
            }
        }
    }
}
