import AVFoundation
import CoreVideo
import Foundation

@available(macOS 27.0, *)
extension SideBySideRestoration {
    final class FrameDecoder {
        private let reader: AVAssetReader
        private let provider: AVAssetReaderOutput.Provider<
            CMReadySampleBuffer<CMSampleBuffer.DynamicContent>
        >
        private let dimensions: VideoDimensions
        private let cropX: Int
        private var previous: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?
        private var next: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?

        init(
            inputURL: URL,
            plan: SideBySideVideoPlan,
            sourceDimensions: VideoDimensions? = nil,
            cropX: Int = 0
        ) async throws {
            let asset = AVURLAsset(url: inputURL)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw DeformConvError.commandFailed("video has no video track")
            }
            let naturalSize = try await track.load(.naturalSize)
            let expectedSource = sourceDimensions ?? plan.dimensions
            guard Int(abs(naturalSize.width).rounded()) == expectedSource.width,
                  Int(abs(naturalSize.height).rounded()) == expectedSource.height,
                  cropX >= 0,
                  cropX + plan.dimensions.width <= expectedSource.width,
                  plan.dimensions.height == expectedSource.height
            else {
                throw DeformConvError.commandFailed(
                    "video dimensions/crop do not match the restoration plan; "
                        + "rotated tracks are not supported yet"
                )
            }
            reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferMetalCompatibilityKey as String: true,
                ]
            )
            dimensions = plan.dimensions
            self.cropX = cropX
            guard reader.canAdd(output) else {
                throw DeformConvError.commandFailed("video reader rejected BGRA output")
            }
            provider = reader.outputProvider(for: output)
            try reader.start()
            next = try await provider.next()
        }

        deinit { reader.cancelReading() }

        func copyFrame(outputIndex: Int) async throws -> CVPixelBuffer {
            let target = CMTime(value: CMTimeValue(outputIndex), timescale: 30)
            while let candidate = next,
                  CMTimeCompare(candidate.presentationTimeStamp, target) < 0 {
                previous = candidate
                next = try await provider.next()
            }
            guard let sample = Self.closest(previous: previous, next: next, to: target) else {
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
            guard case .pixelBuffer(let source) = sample.content else {
                throw DeformConvError.commandFailed("decoded video sample has no pixel buffer")
            }
            var copiedFrame: CVPixelBuffer?
            try source.withUnsafeBuffer {
                copiedFrame = try Self.copyBGRA($0, dimensions: dimensions, cropX: cropX)
            }
            guard let copiedFrame else {
                throw DeformConvError.commandFailed("decoded frame copy was not created")
            }
            return copiedFrame
        }

        private static func closest(
            previous: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?,
            next: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?,
            to target: CMTime
        ) -> CMReadySampleBuffer<CMSampleBuffer.DynamicContent>? {
            guard let previous else { return next }
            guard let next else { return previous }
            let priorDistance = abs(CMTimeGetSeconds(
                CMTimeSubtract(previous.presentationTimeStamp, target)
            ))
            let nextDistance = abs(CMTimeGetSeconds(
                CMTimeSubtract(next.presentationTimeStamp, target)
            ))
            return priorDistance <= nextDistance ? previous : next
        }

        private static func copyBGRA(
            _ source: CVPixelBuffer, dimensions: VideoDimensions, cropX: Int
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
                    sourceBase.advanced(
                        by: row * CVPixelBufferGetBytesPerRow(source) + cropX * 4
                    ),
                    dimensions.width * 4
                )
            }
            CVBufferPropagateAttachments(source, destination)
            return destination
        }
    }
}
