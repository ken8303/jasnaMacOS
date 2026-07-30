import AVFoundation
import CoreVideo
import Foundation
import Metal

struct SideBySideRestorationResult: Sendable {
    let input: VideoAssetInfo
    let output: VideoAssetInfo
    let tileCount: Int
    let frameCount: Int
    let gpuMilliseconds: Double
    let cacheBytes: Int
}

@available(macOS 27.0, *)
enum SideBySideRestoration {
    static func restoreSingleWindow(
        device: MTLDevice,
        inputURL: URL,
        outputURL: URL,
        modelsURL: URL,
        weightsURL: URL
    ) async throws -> SideBySideRestorationResult {
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw DeformConvError.commandFailed("output already exists: \(outputURL.path)")
        }
        let inputInfo = try await SideBySideVideoIO.inspect(url: inputURL)
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds
        )
        guard plan.frameRate.outputFrameCount == SideBySideVideoPlan.temporalWindowFrames else {
            throw DeformConvError.commandFailed(
                "first restored-video path requires exactly 30 output frames; got "
                    + "\(plan.frameRate.outputFrameCount)"
            )
        }

        var decodedFrames = try await decodeThirtyFrames(inputURL: inputURL, plan: plan)
        let propagatedAttachments = decodedFrames.map {
            CVBufferCopyAttachments($0, .shouldPropagate)
        }
        let tileBytes = 3 * SideBySideVideoPlan.modelTileSize
            * SideBySideVideoPlan.modelTileSize * MemoryLayout<Float16>.stride
        let cacheBytes = plan.tiles.count * plan.frameRate.outputFrameCount * tileBytes
        let cacheDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "JasnaMetalTileCache-\(UUID().uuidString)", isDirectory: true
        )
        let temporaryValues = try FileManager.default.temporaryDirectory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        )
        if let available = temporaryValues.volumeAvailableCapacityForImportantUsage,
           Int64(cacheBytes) + 1_073_741_824 > available {
            throw DeformConvError.commandFailed(
                "insufficient temporary storage for restored tiles: need at least "
                    + "\(cacheBytes + 1_073_741_824) bytes, available \(available)"
            )
        }
        try FileManager.default.createDirectory(
            at: cacheDirectory, withIntermediateDirectories: false
        )
        defer { try? FileManager.default.removeItem(at: cacheDirectory) }

        let frameCacheURLs = (0..<plan.frameRate.outputFrameCount).map {
            cacheDirectory.appendingPathComponent("frame-\($0).fp16")
        }
        var cacheWriters = try frameCacheURLs.map { url -> FileHandle in
            guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                throw DeformConvError.commandFailed("failed creating tile cache: \(url.path)")
            }
            return try FileHandle(forWritingTo: url)
        }
        defer { for writer in cacheWriters { try? writer.close() } }

        var totalGPUMilliseconds = 0.0
        for tile in plan.tiles {
            let tileFrames = try decodedFrames.map {
                try TilePixelPipeline.extractPlanarRGB(from: $0, tile: tile)
            }
            let restored = try verifyFusedFourPassRecurrence(
                device: device,
                modelsURL: modelsURL,
                weightsURL: weightsURL,
                backwardFlows: [],
                forwardFlows: [],
                inputFrames: tileFrames,
                stagedBranchFrames: [],
                stagedRestoredFrames: [],
                warmupCount: 0,
                measurementCount: 1
            )
            guard restored.restoredFrames.count == plan.frameRate.outputFrameCount else {
                throw DeformConvError.commandFailed("Metal graph returned the wrong frame count")
            }
            totalGPUMilliseconds += restored.statistics.median
            for frame in 0..<plan.frameRate.outputFrameCount {
                try restored.restoredFrames[frame].withUnsafeBytes { bytes in
                    try cacheWriters[frame].write(contentsOf: Data(bytes))
                }
            }
        }
        for writer in cacheWriters { try writer.close() }
        cacheWriters.removeAll()
        decodedFrames.removeAll(keepingCapacity: false)

        try await encodeCachedFrames(
            outputURL: outputURL,
            plan: plan,
            cacheURLs: frameCacheURLs,
            propagatedAttachments: propagatedAttachments,
            tileBytes: tileBytes
        )
        let outputInfo = try await SideBySideVideoIO.inspect(url: outputURL)
        guard outputInfo.dimensions == inputInfo.dimensions,
              abs(outputInfo.nominalFramesPerSecond - 30) < 0.01
        else {
            throw DeformConvError.commandFailed("restored video metadata validation failed")
        }
        return SideBySideRestorationResult(
            input: inputInfo,
            output: outputInfo,
            tileCount: plan.tiles.count,
            frameCount: plan.frameRate.outputFrameCount,
            gpuMilliseconds: totalGPUMilliseconds,
            cacheBytes: cacheBytes
        )
    }

    private static func decodeThirtyFrames(
        inputURL: URL,
        plan: SideBySideVideoPlan
    ) async throws -> [CVPixelBuffer] {
        let asset = AVURLAsset(url: inputURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw DeformConvError.commandFailed("video has no video track")
        }
        let naturalSize = try await track.load(.naturalSize)
        guard Int(abs(naturalSize.width).rounded()) == plan.dimensions.width,
              Int(abs(naturalSize.height).rounded()) == plan.dimensions.height
        else {
            throw DeformConvError.commandFailed("rotated video tracks are not supported yet")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
        )
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw DeformConvError.commandFailed("video reader rejected BGRA output")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? DeformConvError.commandFailed("video reader failed to start")
        }

        var previous: CMSampleBuffer?
        var next = output.copyNextSampleBuffer()
        var result = [CVPixelBuffer]()
        result.reserveCapacity(plan.frameRate.outputFrameCount)
        for outputIndex in 0..<plan.frameRate.outputFrameCount {
            let target = CMTime(value: CMTimeValue(outputIndex), timescale: 30)
            while let candidate = next,
                  CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(candidate), target) < 0 {
                previous = candidate
                next = output.copyNextSampleBuffer()
            }
            guard let sample = closestSample(previous: previous, next: next, to: target),
                  let source = CMSampleBufferGetImageBuffer(sample)
            else {
                throw reader.error
                    ?? DeformConvError.commandFailed("decoder ended before frame \(outputIndex)")
            }
            result.append(try copyBGRA(source, dimensions: plan.dimensions))
        }
        reader.cancelReading()
        return result
    }

    private static func copyBGRA(
        _ source: CVPixelBuffer,
        dimensions: VideoDimensions
    ) throws -> CVPixelBuffer {
        var optionalDestination: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            nil,
            dimensions.width,
            dimensions.height,
            kCVPixelFormatType_32BGRA,
            [
                kCVPixelBufferMetalCompatibilityKey as String: true,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ] as CFDictionary,
            &optionalDestination
        )
        guard status == kCVReturnSuccess, let destination = optionalDestination else {
            throw DeformConvError.commandFailed("failed allocating decoded frame copy")
        }
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination)
        else { throw DeformConvError.commandFailed("decoded frame is not CPU accessible") }
        let packedBytes = dimensions.width * 4
        for row in 0..<dimensions.height {
            memcpy(
                destinationBase.advanced(by: row * CVPixelBufferGetBytesPerRow(destination)),
                sourceBase.advanced(by: row * CVPixelBufferGetBytesPerRow(source)),
                packedBytes
            )
        }
        CVBufferPropagateAttachments(source, destination)
        return destination
    }

    private static func encodeCachedFrames(
        outputURL: URL,
        plan: SideBySideVideoPlan,
        cacheURLs: [URL],
        propagatedAttachments: [CFDictionary?],
        tileBytes: Int
    ) async throws {
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let bitRate = min(160_000_000, max(8_000_000, plan.dimensions.pixelCount * 5 / 2))
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
                ],
            ]
        )
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: plan.dimensions.width,
                kCVPixelBufferHeightKey as String: plan.dimensions.height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
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

        for frame in 0..<plan.frameRate.outputFrameCount {
            let cache = try FileHandle(forReadingFrom: cacheURLs[frame])
            defer { try? cache.close() }
            var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
            for tile in plan.tiles {
                guard let data = try cache.read(upToCount: tileBytes), data.count == tileBytes else {
                    throw DeformConvError.commandFailed("restored tile cache is truncated")
                }
                var values = [Float16](
                    repeating: 0, count: tileBytes / MemoryLayout<Float16>.stride
                )
                _ = values.withUnsafeMutableBytes { destination in
                    data.copyBytes(to: destination)
                }
                try accumulator.accumulate(tile: tile, planarRGB: values)
            }
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
            if let attachments = propagatedAttachments[frame] {
                CVBufferSetAttachments(pixelBuffer, attachments, .shouldPropagate)
            }
            try accumulator.writeBGRA(to: pixelBuffer)
            let time = CMTime(value: CMTimeValue(frame), timescale: 30)
            guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                throw writer.error ?? DeformConvError.commandFailed("failed encoding frame \(frame)")
            }
        }
        input.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw writer.error ?? DeformConvError.commandFailed("video writer did not complete")
        }
    }

    private static func closestSample(
        previous: CMSampleBuffer?, next: CMSampleBuffer?, to target: CMTime
    ) -> CMSampleBuffer? {
        guard let previous else { return next }
        guard let next else { return previous }
        let previousDistance = abs(CMTimeGetSeconds(
            CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(previous), target)
        ))
        let nextDistance = abs(CMTimeGetSeconds(
            CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(next), target)
        ))
        return previousDistance <= nextDistance ? previous : next
    }
}
