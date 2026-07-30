import AVFoundation
import CoreVideo
import Foundation
import Metal

struct SideBySideRestorationResult: Sendable {
    let input: VideoAssetInfo
    let output: VideoAssetInfo
    let tileCount: Int
    let frameCount: Int
    let windowCount: Int
    let gpuMilliseconds: Double
    let cacheBytes: Int
}

@available(macOS 27.0, *)
enum SideBySideRestoration {
    private static let tileElements = 3 * SideBySideVideoPlan.modelTileSize
        * SideBySideVideoPlan.modelTileSize
    private static let tileBytes = tileElements * MemoryLayout<Float16>.stride

    static func restoreVideo(
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
        let decoder = try await FrameDecoder(inputURL: inputURL, plan: plan)
        let writer = try RestoredFrameWriter(outputURL: outputURL, plan: plan)
        var totalGPU: Double = 0
        var peakCacheBytes = 0
        var windows = 0

        for (windowIndex, outputCount) in plan.temporalWindowFrameCounts.enumerated() {
            let windowStart = windowIndex * SideBySideVideoPlan.temporalWindowFrames
            var decoded = try (0..<outputCount).map {
                try decoder.copyFrame(outputIndex: windowStart + $0)
            }
            let attachments = decoded.map { CVBufferCopyAttachments($0, .shouldPropagate) }
            while decoded.count < 3 {
                guard let last = decoded.last else { throw DeformConvError.invalidShape }
                decoded.append(last)
            }
            let window = try processWindow(
                device: device,
                plan: plan,
                decodedFrames: decoded,
                outputCount: outputCount,
                modelsURL: modelsURL,
                weightsURL: weightsURL
            )
            defer { try? FileManager.default.removeItem(at: window.cacheDirectory) }
            try await writer.appendCachedFrames(
                cacheURLs: window.cacheURLs,
                attachments: attachments,
                startFrame: windowStart,
                plan: plan
            )
            try FileManager.default.removeItem(at: window.cacheDirectory)
            totalGPU += window.gpuMilliseconds
            peakCacheBytes = max(peakCacheBytes, window.cacheBytes)
            windows += 1
        }
        try await writer.finish()

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
            windowCount: windows,
            gpuMilliseconds: totalGPU,
            cacheBytes: peakCacheBytes
        )
    }

    private struct WindowResult {
        let cacheDirectory: URL
        let cacheURLs: [URL]
        let gpuMilliseconds: Double
        let cacheBytes: Int
    }

    private static func processWindow(
        device: MTLDevice,
        plan: SideBySideVideoPlan,
        decodedFrames: [CVPixelBuffer],
        outputCount: Int,
        modelsURL: URL,
        weightsURL: URL
    ) throws -> WindowResult {
        let cacheBytes = plan.tiles.count * outputCount * tileBytes
        let temporaryURL = FileManager.default.temporaryDirectory
        let available = try temporaryURL.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage
        if let available, Int64(cacheBytes) + 1_073_741_824 > available {
            throw DeformConvError.commandFailed(
                "insufficient temporary storage for restored tiles: need at least "
                    + "\(cacheBytes + 1_073_741_824) bytes, available \(available)"
            )
        }
        let directory = temporaryURL.appendingPathComponent(
            "JasnaMetalTileCache-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        do {
            let urls = (0..<outputCount).map {
                directory.appendingPathComponent("frame-\($0).fp16")
            }
            var handles = try urls.map { url -> FileHandle in
                guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
                    throw DeformConvError.commandFailed("failed creating tile cache: \(url.path)")
                }
                return try FileHandle(forWritingTo: url)
            }
            defer { for handle in handles { try? handle.close() } }
            var gpuMilliseconds: Double = 0
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
                guard restored.restoredFrames.count == decodedFrames.count else {
                    throw DeformConvError.commandFailed("Metal graph returned the wrong frame count")
                }
                gpuMilliseconds += restored.statistics.median
                for frame in 0..<outputCount {
                    try restored.restoredFrames[frame].withUnsafeBytes { bytes in
                        try handles[frame].write(contentsOf: Data(bytes))
                    }
                }
            }
            for handle in handles { try handle.close() }
            handles.removeAll()
            return WindowResult(
                cacheDirectory: directory,
                cacheURLs: urls,
                gpuMilliseconds: gpuMilliseconds,
                cacheBytes: cacheBytes
            )
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private final class FrameDecoder {
        private let reader: AVAssetReader
        private let output: AVAssetReaderTrackOutput
        private let dimensions: VideoDimensions
        private var previous: CMSampleBuffer?
        private var next: CMSampleBuffer?

        init(inputURL: URL, plan: SideBySideVideoPlan) async throws {
            let asset = AVURLAsset(url: inputURL)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw DeformConvError.commandFailed("video has no video track")
            }
            let naturalSize = try await track.load(.naturalSize)
            guard Int(abs(naturalSize.width).rounded()) == plan.dimensions.width,
                  Int(abs(naturalSize.height).rounded()) == plan.dimensions.height
            else { throw DeformConvError.commandFailed("rotated video tracks are not supported yet") }
            reader = try AVAssetReader(asset: asset)
            output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                ]
            )
            dimensions = plan.dimensions
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else {
                throw DeformConvError.commandFailed("video reader rejected BGRA output")
            }
            reader.add(output)
            guard reader.startReading() else {
                throw reader.error ?? DeformConvError.commandFailed("video reader failed to start")
            }
            next = output.copyNextSampleBuffer()
        }

        deinit { reader.cancelReading() }

        func copyFrame(outputIndex: Int) throws -> CVPixelBuffer {
            let target = CMTime(value: CMTimeValue(outputIndex), timescale: 30)
            while let candidate = next,
                  CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(candidate), target) < 0 {
                previous = candidate
                next = output.copyNextSampleBuffer()
            }
            guard let sample = Self.closest(previous: previous, next: next, to: target),
                  let source = CMSampleBufferGetImageBuffer(sample)
            else {
                if let error = reader.error as NSError?,
                   error.domain == AVFoundationErrorDomain,
                   error.code == AVError.Code.decoderNotFound.rawValue {
                    throw DeformConvError.commandFailed(
                        "Apple VideoToolbox cannot decode this source. For 8K Main 10 at "
                            + "59.94 fps, first run script/prepare_8k_30fps.sh, then restore "
                            + "the prepared 30 fps file. Underlying error: \(error)"
                    )
                }
                throw reader.error
                    ?? DeformConvError.commandFailed("decoder ended before frame \(outputIndex)")
            }
            return try Self.copyBGRA(source, dimensions: dimensions)
        }

        private static func closest(
            previous: CMSampleBuffer?, next: CMSampleBuffer?, to target: CMTime
        ) -> CMSampleBuffer? {
            guard let previous else { return next }
            guard let next else { return previous }
            let priorDistance = abs(CMTimeGetSeconds(
                CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(previous), target)
            ))
            let nextDistance = abs(CMTimeGetSeconds(
                CMTimeSubtract(CMSampleBufferGetPresentationTimeStamp(next), target)
            ))
            return priorDistance <= nextDistance ? previous : next
        }

        private static func copyBGRA(
            _ source: CVPixelBuffer, dimensions: VideoDimensions
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
            for row in 0..<dimensions.height {
                memcpy(
                    destinationBase.advanced(by: row * CVPixelBufferGetBytesPerRow(destination)),
                    sourceBase.advanced(by: row * CVPixelBufferGetBytesPerRow(source)),
                    dimensions.width * 4
                )
            }
            CVBufferPropagateAttachments(source, destination)
            return destination
        }
    }

    private final class RestoredFrameWriter {
        private let writer: AVAssetWriter
        private let input: AVAssetWriterInput
        private let adaptor: AVAssetWriterInputPixelBufferAdaptor

        init(outputURL: URL, plan: SideBySideVideoPlan) throws {
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
            plan: SideBySideVideoPlan
        ) async throws {
            for localFrame in cacheURLs.indices {
                let cache = try FileHandle(forReadingFrom: cacheURLs[localFrame])
                var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
                for tile in plan.tiles {
                    guard let data = try cache.read(upToCount: tileBytes), data.count == tileBytes else {
                        try? cache.close()
                        throw DeformConvError.commandFailed("restored tile cache is truncated")
                    }
                    var values = [Float16](repeating: 0, count: tileElements)
                    _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
                    try accumulator.accumulate(tile: tile, planarRGB: values)
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
                try accumulator.writeBGRA(to: pixelBuffer)
                let outputFrame = startFrame + localFrame
                let time = CMTime(value: CMTimeValue(outputFrame), timescale: 30)
                guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                    throw writer.error
                        ?? DeformConvError.commandFailed("failed encoding frame \(outputFrame)")
                }
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
