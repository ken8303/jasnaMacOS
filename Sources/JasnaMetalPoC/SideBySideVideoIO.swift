import AVFoundation
import CoreVideo
import Foundation

struct VideoAssetInfo: Sendable {
    let dimensions: VideoDimensions
    let nominalFramesPerSecond: Double
    let durationSeconds: Double
}

struct VideoTranscodeResult: Sendable {
    let input: VideoAssetInfo
    let output: VideoAssetInfo
    let writtenFrameCount: Int
}

enum SideBySideVideoIO {
    static func inspect(url: URL) async throws -> VideoAssetInfo {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw DeformConvError.commandFailed("video has no video track: \(url.path)")
        }
        let natural = try await track.load(.naturalSize)
        let preferredTransform = try await track.load(.preferredTransform)
        let transformed = natural.applying(preferredTransform)
        let width = Int(abs(transformed.width).rounded())
        let height = Int(abs(transformed.height).rounded())
        let duration = try await CMTimeGetSeconds(asset.load(.duration))
        let nominalFPS = try await Double(track.load(.nominalFrameRate))
        guard width > 0, height > 0, duration > 0, nominalFPS > 0 else {
            throw DeformConvError.commandFailed("video metadata is incomplete: \(url.path)")
        }
        return VideoAssetInfo(
            dimensions: VideoDimensions(width: width, height: height),
            nominalFramesPerSecond: nominalFPS,
            durationSeconds: duration
        )
    }

    static func transcodeTo30FPS(
        inputURL: URL,
        outputURL: URL,
        verifyTiledPixelPath: Bool = false
    ) async throws -> VideoTranscodeResult {
        guard inputURL.standardizedFileURL != outputURL.standardizedFileURL else {
            throw DeformConvError.commandFailed("input and output video paths must differ")
        }
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw DeformConvError.commandFailed("output already exists: \(outputURL.path)")
        }

        let asset = AVURLAsset(url: inputURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw DeformConvError.commandFailed("video has no video track: \(inputURL.path)")
        }
        let inputInfo = try await inspect(url: inputURL)
        let naturalSize = try await track.load(.naturalSize)
        guard Int(abs(naturalSize.width).rounded()) == inputInfo.dimensions.width,
              Int(abs(naturalSize.height).rounded()) == inputInfo.dimensions.height
        else {
            throw DeformConvError.commandFailed(
                "rotated video tracks are not supported by the SBS pixel-buffer path yet"
            )
        }
        let plan = try SideBySideVideoPlan(
            width: inputInfo.dimensions.width,
            height: inputInfo.dimensions.height,
            sourceFramesPerSecond: inputInfo.nominalFramesPerSecond,
            durationSeconds: inputInfo.durationSeconds
        )

        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
        )
        guard reader.canAdd(readerOutput) else {
            throw DeformConvError.commandFailed("AVAssetReader rejected the video output")
        }
        let provider = reader.outputProvider(for: readerOutput)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let configuredBitRate = Int(ProcessInfo.processInfo.environment["JASNA_VIDEO_BITRATE"] ?? "")
        let bitRate = configuredBitRate.flatMap { $0 > 0 ? $0 : nil }
            ?? min(160_000_000, max(8_000_000, inputInfo.dimensions.pixelCount * 5 / 2))
        let writerInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.hevc,
                AVVideoWidthKey: inputInfo.dimensions.width,
                AVVideoHeightKey: inputInfo.dimensions.height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: bitRate,
                    AVVideoExpectedSourceFrameRateKey: 30,
                    AVVideoMaxKeyFrameIntervalKey: 60,
                ],
            ]
        )
        guard writer.canAdd(writerInput) else {
            throw DeformConvError.commandFailed("AVAssetWriter rejected the HEVC video input")
        }
        var creationAttributes = CVPixelBufferCreationAttributes(
            pixelFormatType: CVPixelFormatType(rawValue: kCVPixelFormatType_32BGRA),
            size: CVImageSize(
                width: inputInfo.dimensions.width,
                height: inputInfo.dimensions.height
            ),
            compatibility: [.metalTexture]
        )
        creationAttributes.backing = .ioSurface
        let receiver = writer.inputPixelBufferReceiver(
            for: writerInput, pixelBufferAttributes: creationAttributes
        )
        try writer.start()
        try reader.start()
        writer.startSession(atSourceTime: .zero)

        var previous: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?
        var next = try await provider.next()
        var written = 0
        for outputIndex in 0..<plan.frameRate.outputFrameCount {
            let outputTime = CMTime(value: CMTimeValue(outputIndex), timescale: 30)
            while let candidate = next,
                  CMTimeCompare(candidate.presentationTimeStamp, outputTime) < 0 {
                previous = candidate
                next = try await provider.next()
            }
            guard let selected = closestSample(previous: previous, next: next, to: outputTime),
                  case .pixelBuffer(let decodedBuffer) = selected.content
            else {
                throw reader.error
                    ?? DeformConvError.commandFailed("decoder ended before output frame \(outputIndex)")
            }
            let outputBuffer: CVReadOnlyPixelBuffer
            if verifyTiledPixelPath {
                guard let pool = receiver.pixelBufferPool else {
                    throw DeformConvError.commandFailed("video writer has no pixel-buffer pool")
                }
                let created = try pool.makeMutablePixelBuffer()
                try decodedBuffer.withUnsafeBuffer { decodedUnsafe in
                    var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
                    for tile in plan.tiles {
                        let planar = try TilePixelPipeline.extractPlanarRGB(
                            from: decodedUnsafe, tile: tile
                        )
                        try accumulator.accumulate(tile: tile, planarRGB: planar)
                    }
                    try created.withUnsafeBuffer { createdUnsafe in
                        CVBufferPropagateAttachments(decodedUnsafe, createdUnsafe)
                        try accumulator.writeBGRA(to: createdUnsafe)
                    }
                }
                outputBuffer = CVReadOnlyPixelBuffer(created)
            } else {
                outputBuffer = decodedBuffer
            }
            try await receiver.append(outputBuffer, with: outputTime)
            written += 1
        }

        reader.cancelReading()
        receiver.finish()
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw writer.error ?? DeformConvError.commandFailed("video writer did not complete")
        }
        let outputInfo = try await inspect(url: outputURL)
        guard written == plan.frameRate.outputFrameCount,
              outputInfo.dimensions == inputInfo.dimensions,
              abs(outputInfo.nominalFramesPerSecond - 30) < 0.01
        else {
            throw DeformConvError.commandFailed(
                "30 FPS output validation failed (frames=\(written), "
                    + "size=\(outputInfo.dimensions.width)×\(outputInfo.dimensions.height), "
                    + "fps=\(outputInfo.nominalFramesPerSecond))"
            )
        }
        return VideoTranscodeResult(input: inputInfo, output: outputInfo, writtenFrameCount: written)
    }

    private static func closestSample(
        previous: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?,
        next: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?,
        to target: CMTime
    ) -> CMReadySampleBuffer<CMSampleBuffer.DynamicContent>? {
        guard let previous else { return next }
        guard let next else { return previous }
        let previousDistance = abs(CMTimeGetSeconds(
            CMTimeSubtract(previous.presentationTimeStamp, target)
        ))
        let nextDistance = abs(CMTimeGetSeconds(
            CMTimeSubtract(next.presentationTimeStamp, target)
        ))
        return previousDistance <= nextDistance ? previous : next
    }
}
