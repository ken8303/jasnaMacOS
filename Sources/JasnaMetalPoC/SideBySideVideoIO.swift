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
        readerOutput.alwaysCopiesSampleData = false
        guard reader.canAdd(readerOutput) else {
            throw DeformConvError.commandFailed("AVAssetReader rejected the video output")
        }
        reader.add(readerOutput)

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let bitRate = min(160_000_000, max(8_000_000, inputInfo.dimensions.pixelCount * 5 / 2))
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
        writerInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: inputInfo.dimensions.width,
                kCVPixelBufferHeightKey as String: inputInfo.dimensions.height,
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
        )
        guard writer.canAdd(writerInput) else {
            throw DeformConvError.commandFailed("AVAssetWriter rejected the HEVC video input")
        }
        writer.add(writerInput)
        guard writer.startWriting(), reader.startReading() else {
            throw writer.error ?? reader.error
                ?? DeformConvError.commandFailed("video reader/writer failed to start")
        }
        writer.startSession(atSourceTime: .zero)

        var previous: CMSampleBuffer?
        var next = readerOutput.copyNextSampleBuffer()
        var written = 0
        for outputIndex in 0..<plan.frameRate.outputFrameCount {
            let outputTime = CMTime(value: CMTimeValue(outputIndex), timescale: 30)
            while let candidate = next,
                  CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(candidate), outputTime) < 0 {
                previous = candidate
                next = readerOutput.copyNextSampleBuffer()
            }
            guard let selected = closestSample(previous: previous, next: next, to: outputTime),
                  let decodedBuffer = CMSampleBufferGetImageBuffer(selected)
            else {
                throw reader.error
                    ?? DeformConvError.commandFailed("decoder ended before output frame \(outputIndex)")
            }
            while !writerInput.isReadyForMoreMediaData {
                if writer.status == .failed {
                    throw writer.error ?? DeformConvError.commandFailed("video writer failed")
                }
                try await Task.sleep(for: .milliseconds(1))
            }
            let outputBuffer: CVPixelBuffer
            if verifyTiledPixelPath {
                var accumulator = try TileFrameAccumulator(dimensions: plan.dimensions)
                for tile in plan.tiles {
                    let planar = try TilePixelPipeline.extractPlanarRGB(
                        from: decodedBuffer, tile: tile
                    )
                    try accumulator.accumulate(tile: tile, planarRGB: planar)
                }
                guard let pool = adaptor.pixelBufferPool else {
                    throw DeformConvError.commandFailed("video writer has no pixel-buffer pool")
                }
                var optionalOutput: CVPixelBuffer?
                let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optionalOutput)
                guard status == kCVReturnSuccess, let created = optionalOutput else {
                    throw DeformConvError.commandFailed(
                        "failed allocating output pixel buffer (CoreVideo \(status))"
                    )
                }
                CVBufferPropagateAttachments(decodedBuffer, created)
                try accumulator.writeBGRA(to: created)
                outputBuffer = created
            } else {
                outputBuffer = decodedBuffer
            }
            guard adaptor.append(outputBuffer, withPresentationTime: outputTime) else {
                throw writer.error
                    ?? DeformConvError.commandFailed("failed writing output frame \(outputIndex)")
            }
            written += 1
        }

        reader.cancelReading()
        writerInput.markAsFinished()
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
        previous: CMSampleBuffer?,
        next: CMSampleBuffer?,
        to target: CMTime
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
