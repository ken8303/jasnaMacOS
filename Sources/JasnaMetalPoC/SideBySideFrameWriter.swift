import AVFoundation
import CoreVideo
import Foundation
import Metal

@available(macOS 27.0, *)
extension SideBySideRestoration {
    final class RestoredFrameWriter {
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
                                region: region,
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
        private let input: AVAssetWriterInput
        private let adaptor: AVAssetWriterInputPixelBufferAdaptor
        private let metalCompositor: MetalMosaicCompositor?

        init(device: MTLDevice, outputURL: URL, plan: SideBySideVideoPlan) throws {
            metalCompositor = ProcessInfo.processInfo.environment["JASNA_METAL_COMPOSITOR"] == "0"
                ? nil : try? MetalMosaicCompositor(device: device)
            writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
            let bitRate = min(160_000_000, max(8_000_000, plan.dimensions.pixelCount * 5 / 2))
            input = AVAssetWriterInput(
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
            adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: plan.dimensions.width,
                    kCVPixelBufferHeightKey as String: plan.dimensions.height,
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                    kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                ]
            )
            guard writer.canAdd(input) else {
                throw DeformConvError.commandFailed("video writer rejected restored frames")
            }
            writer.add(input)
            guard writer.startWriting() else {
                throw writer.error ?? DeformConvError.commandFailed("video writer failed to start")
            }
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
                while !input.isReadyForMoreMediaData {
                    if writer.status == .failed {
                        throw writer.error ?? DeformConvError.commandFailed("video writer failed")
                    }
                    try await Task.sleep(for: .milliseconds(1))
                }
                guard let pool = adaptor.pixelBufferPool else {
                    throw DeformConvError.commandFailed("video writer has no pixel-buffer pool")
                }
                var optionalBuffer: CVPixelBuffer?
                let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optionalBuffer)
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
                guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                    throw writer.error
                        ?? DeformConvError.commandFailed("failed encoding frame \(outputFrame)")
                }
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
            guard let pool = adaptor.pixelBufferPool else {
                throw DeformConvError.commandFailed("video writer has no pixel-buffer pool")
            }
            var nextFrame = 0
            while nextFrame < cacheURLs.count {
                let batchEnd = min(cacheURLs.count, nextFrame + compositeConcurrency)
                let localFrames = Array(nextFrame..<batchEnd)
                var outputBuffers = [CVPixelBuffer]()
                outputBuffers.reserveCapacity(localFrames.count)
                for localFrame in localFrames {
                    var optionalBuffer: CVPixelBuffer?
                    let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optionalBuffer)
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
                    while !input.isReadyForMoreMediaData {
                        if writer.status == .failed {
                            throw writer.error
                                ?? DeformConvError.commandFailed("video writer failed")
                        }
                        try await Task.sleep(for: .milliseconds(1))
                    }
                    let outputFrame = startFrame + composited.localFrame
                    let time = CMTime(value: CMTimeValue(outputFrame), timescale: 30)
                    guard adaptor.append(composited.pixelBuffer, withPresentationTime: time) else {
                        throw writer.error
                            ?? DeformConvError.commandFailed(
                                "failed encoding frame \(outputFrame)"
                            )
                    }
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

        func finish() async throws {
            input.markAsFinished()
            await withCheckedContinuation { continuation in
                writer.finishWriting { continuation.resume() }
            }
            guard writer.status == .completed else {
                throw writer.error ?? DeformConvError.commandFailed("video writer did not complete")
            }
        }
    }
}
