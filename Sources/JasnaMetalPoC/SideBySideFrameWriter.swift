import AVFoundation
import CoreVideo
import Foundation
import Metal

@available(macOS 27.0, *)
extension SideBySideRestoration {
    final class RestoredFrameWriter {
        // Created only after all CPU/Metal writes finish; the receiver gets read-only ownership.
        private struct FinishedPixelBuffer: @unchecked Sendable {
            let value: CVPixelBuffer
        }

        private struct CompositedRegionFrame {
            let localFrame: Int
            let pixelBuffer: CVPixelBuffer
            let wallSeconds: Double
        }

        private final class RegionFrameCompositeBatch: @unchecked Sendable {
            private let localFrames: [Int]
            private let cacheURLs: [URL]
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
                    let cache = try FileHandle(forReadingFrom: cacheURLs[index])
                    defer { try? cache.close() }
                    var compositeInputs = [MetalMosaicCompositeInput]()
                    compositeInputs.reserveCapacity(regions.count)
                    for (regionIndex, region) in regions.enumerated() {
                        let absoluteFrame = progressStartFrame + localFrame
                        guard region.frameRange.contains(absoluteFrame) else {
                            try cache.seek(toOffset: UInt64((regionIndex + 1) * tileBytes))
                            continue
                        }
                        guard let data = try cache.read(upToCount: tileBytes),
                              data.count == tileBytes
                        else {
                            throw DeformConvError.commandFailed(
                                "restored mosaic-crop cache is truncated"
                            )
                        }
                        var values = [Float16](repeating: 0, count: tileElements)
                        _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
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
            let automaticConcurrency = ProcessInfo.processInfo.physicalMemory
                >= UInt64(16 * 1_073_741_824) ? 2 : 1
            let compositeConcurrency = min(
                2, max(1, configuredConcurrency ?? automaticConcurrency)
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
            for localFrame in 0..<frameCount {
                directStereoFrameCount += 1
                let frameStarted = ContinuousClock.now
                var optionalOutput: CVPixelBuffer?
                let status = CVPixelBufferPoolCreatePixelBuffer(
                    nil, pixelBufferPool, &optionalOutput
                )
                guard status == kCVReturnSuccess, let outputBuffer = optionalOutput else {
                    throw DeformConvError.commandFailed("failed allocating direct SBS frame")
                }
                let absoluteFrame = absoluteStartFrame + localFrame
                let leftInputs = try Self.readCompositeInputs(
                    cacheURL: leftCacheURLs[localFrame],
                    baseFrame: leftBaseFrames[localFrame],
                    regions: leftRegions,
                    samplingMaps: leftSamplingMaps,
                    absoluteFrame: absoluteFrame,
                    xOffset: 0
                )
                let rightInputs = try Self.readCompositeInputs(
                    cacheURL: rightCacheURLs[localFrame],
                    baseFrame: rightBaseFrames[localFrame],
                    regions: rightRegions,
                    samplingMaps: rightSamplingMaps,
                    absoluteFrame: absoluteFrame,
                    xOffset: plan.eyeDimensions.width
                )
                if projection == .fisheye, let metalCompositor {
                    do {
                        try metalCompositor.compositeStereo(
                            leftPixelBuffer: leftBaseFrames[localFrame],
                            rightPixelBuffer: rightBaseFrames[localFrame],
                            outputPixelBuffer: outputBuffer,
                            dimensions: plan.dimensions,
                            inputs: leftInputs + rightInputs
                        )
                    } catch {
                        fusedStereoFallbackCount += 1
                        report(
                            "WARNING: Fused Metal stereo composite failed; "
                                + "using split path for frame \(absoluteFrame + 1) (\(error))"
                        )
                        do {
                            try metalCompositor.copyStereo(
                                leftPixelBuffer: leftBaseFrames[localFrame],
                                rightPixelBuffer: rightBaseFrames[localFrame],
                                outputPixelBuffer: outputBuffer,
                                dimensions: plan.dimensions
                            )
                        } catch {
                            cpuStereoFallbackCount += 1
                            report(
                                "WARNING: Metal stereo copy failed; using CPU for frame "
                                    + "\(absoluteFrame + 1) (\(error))"
                            )
                            try Self.copyStereoPixelBuffers(
                                left: leftBaseFrames[localFrame],
                                right: rightBaseFrames[localFrame],
                                destination: outputBuffer,
                                dimensions: plan.dimensions
                            )
                        }
                        try metalCompositor.compositeInPlace(
                            pixelBuffer: outputBuffer,
                            dimensions: plan.dimensions,
                            inputs: leftInputs + rightInputs
                        )
                    }
                } else {
                    try Self.copyStereoPixelBuffers(
                        left: leftBaseFrames[localFrame],
                        right: rightBaseFrames[localFrame],
                        destination: outputBuffer,
                        dimensions: plan.dimensions
                    )
                    guard projection == .raw else {
                        throw DeformConvError.commandFailed(
                            "direct fisheye SBS output requires the Metal compositor"
                        )
                    }
                    var accumulator = try MosaicRegionFrameAccumulator(
                        basePixelBuffer: outputBuffer, dimensions: plan.dimensions
                    )
                    for composite in leftInputs + rightInputs {
                        try accumulator.composite(
                            region: composite.region,
                            planarRGB: composite.restored,
                            originalPlanarRGB: nil,
                            projection: .raw
                        )
                    }
                    try accumulator.writeBGRA(to: outputBuffer)
                }
                if let attachments = CVBufferCopyAttachments(
                    leftBaseFrames[localFrame], .shouldPropagate
                ) {
                    CVBufferSetAttachments(outputBuffer, attachments, .shouldPropagate)
                }
                let preparationMilliseconds = SideBySideRestoration.elapsedMilliseconds(
                    since: frameStarted
                )
                let presentationTime = CMTime(
                    value: CMTimeValue(presentationStartFrame + localFrame), timescale: 30
                )
                let appendStarted = ContinuousClock.now
                try await append(
                    FinishedPixelBuffer(value: outputBuffer), at: presentationTime,
                    frame: presentationStartFrame + localFrame
                )
                let appendMilliseconds = SideBySideRestoration.elapsedMilliseconds(
                    since: appendStarted
                )
                totalPreparationMilliseconds += preparationMilliseconds
                totalEncoderWaitMilliseconds += appendMilliseconds
                if localFrame == frameCount - 1
                    || (absoluteFrame + 1).isMultiple(of: 30)
                    || appendMilliseconds >= 250
                {
                    report(
                        "Queued direct SBS frame \(absoluteFrame + 1)/"
                            + "\(progressFrameCount); prepare/composite "
                            + "\(String(format: "%.3f", preparationMilliseconds)) ms, "
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
        }

        private static func readCompositeInputs(
            cacheURL: URL,
            baseFrame: CVPixelBuffer,
            regions: [MosaicRegion],
            samplingMaps: [MosaicCropSamplingMap],
            absoluteFrame: Int,
            xOffset: Int
        ) throws -> [MetalMosaicCompositeInput] {
            let cache = try FileHandle(forReadingFrom: cacheURL)
            defer { try? cache.close() }
            var result = [MetalMosaicCompositeInput]()
            result.reserveCapacity(regions.count)
            for (index, region) in regions.enumerated() {
                guard region.frameRange.contains(absoluteFrame) else {
                    try cache.seek(toOffset: UInt64((index + 1) * tileBytes))
                    continue
                }
                guard let data = try cache.read(upToCount: tileBytes), data.count == tileBytes else {
                    throw DeformConvError.commandFailed(
                        "direct SBS mosaic-crop cache is truncated"
                    )
                }
                var restored = [Float16](repeating: 0, count: tileElements)
                _ = restored.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                let original = try samplingMaps[index].extractPlanarRGB(from: baseFrame)
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
            return result
        }

        private static func translated(_ region: MosaicRegion, xOffset: Int) -> MosaicRegion {
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
                }
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
