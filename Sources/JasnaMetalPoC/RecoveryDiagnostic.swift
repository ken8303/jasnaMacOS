import Foundation

@available(macOS 27.0, *)
extension SideBySideRestoration {
    static func writeRecoveryDiagnosticIfRequested(
        restored: CompletedRegionRestoration,
        region: MosaicRegion,
        samplingMap: MosaicCropSamplingMap,
        windowIndex: Int,
        workDirectoryURL: URL?
    ) throws {
        let environment = ProcessInfo.processInfo.environment
        guard let directoryPath = environment["JASNA_RECOVERY_DIAGNOSTIC_DIR"],
              !directoryPath.isEmpty
        else { return }
        let selectedText = environment["JASNA_RECOVERY_DIAGNOSTIC_REGION"] ?? "0"
        if selectedText != "all" {
            guard let selectedRegion = Int(selectedText), selectedRegion >= 0 else {
                throw DeformConvError.commandFailed(
                    "JASNA_RECOVERY_DIAGNOSTIC_REGION must be a non-negative integer or all"
                )
            }
            guard restored.prepared.regionIndex == selectedRegion else { return }
        }
        let requestedFrame = Int(environment["JASNA_RECOVERY_DIAGNOSTIC_FRAME"] ?? "")
        let activeFrame = min(
            restored.prepared.activeFrameCount - 1,
            max(0, requestedFrame ?? restored.prepared.activeFrameCount / 2)
        )
        let tensorFrame = min(
            restored.prepared.restoredFrameOffset + activeFrame,
            restored.frames.count - 1
        )
        let input = restored.prepared.inputFrames[tensorFrame]
        let output = restored.frames[tensorFrame]
        guard input.count == tileElements, output.count == tileElements else {
            throw DeformConvError.invalidShape
        }

        let directory = URL(fileURLWithPath: directoryPath, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let eye = sanitizedDiagnosticComponent(workDirectoryURL?.lastPathComponent ?? "eye")
        let stem = String(
            format: "window-%05d-%@-region-%03d-frame-%03d",
            windowIndex,
            eye,
            restored.prepared.regionIndex,
            restored.prepared.localStart + activeFrame
        )
        try planarRGBPPM(input).write(
            to: directory.appendingPathComponent("\(stem)-input.ppm"), options: .atomic
        )
        try planarRGBPPM(output).write(
            to: directory.appendingPathComponent("\(stem)-output.ppm"), options: .atomic
        )
        try planarDifferencePPM(input: input, output: output, amplification: 4).write(
            to: directory.appendingPathComponent("\(stem)-difference-x4.ppm"),
            options: .atomic
        )
        let absoluteFrame = windowIndex * SideBySideVideoPlan.temporalWindowFrames
            + restored.prepared.localStart + activeFrame
        try modelMaskPGM(
            region: region.resolvingSegmentationMask(at: absoluteFrame),
            samplingMap: samplingMap
        ).write(
            to: directory.appendingPathComponent("\(stem)-mask.pgm"), options: .atomic
        )

        let difference = zip(input, output).map { abs(Float($0) - Float($1)) }
        let metadata: [String: Any] = [
            "window": windowIndex,
            "eye": eye,
            "region": restored.prepared.regionIndex,
            "activeFrame": activeFrame,
            "absoluteFrame": absoluteFrame,
            "regionX": region.x,
            "regionY": region.y,
            "regionWidth": region.width,
            "regionHeight": region.height,
            "meanAbsoluteModelDelta": difference.reduce(0, +) / Float(difference.count),
            "maximumAbsoluteModelDelta": difference.max() ?? 0,
        ]
        let metadataData = try JSONSerialization.data(
            withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]
        )
        try metadataData.write(
            to: directory.appendingPathComponent("\(stem)-metadata.json"), options: .atomic
        )
        report("Recovery diagnostic written: \(directory.path)/\(stem)-*")
    }

    static func planarRGBPPM(_ values: [Float16]) throws -> Data {
        let size = SideBySideVideoPlan.modelTileSize
        let plane = size * size
        guard values.count == 3 * plane else { throw DeformConvError.invalidShape }
        var data = Data("P6\n\(size) \(size)\n255\n".utf8)
        var pixels = [UInt8](repeating: 0, count: 3 * plane)
        for pixel in 0..<plane {
            pixels[3 * pixel] = normalizedByte(Float(values[pixel]))
            pixels[3 * pixel + 1] = normalizedByte(Float(values[plane + pixel]))
            pixels[3 * pixel + 2] = normalizedByte(Float(values[2 * plane + pixel]))
        }
        data.append(contentsOf: pixels)
        return data
    }

    static func planarDifferencePPM(
        input: [Float16], output: [Float16], amplification: Float
    ) throws -> Data {
        guard input.count == output.count else { throw DeformConvError.invalidShape }
        return try planarRGBPPM(zip(input, output).map {
            Float16(min(1, abs(Float($0) - Float($1)) * amplification))
        })
    }

    static func modelMaskPGM(
        region: MosaicRegion, samplingMap: MosaicCropSamplingMap
    ) -> Data {
        let size = SideBySideVideoPlan.modelTileSize
        var data = Data("P5\n\(size) \(size)\n255\n".utf8)
        var pixels = [UInt8](repeating: 0, count: size * size)
        for modelY in 0..<size {
            for modelX in 0..<size {
                let source = samplingMap.sourceCoordinate(modelX: modelX, modelY: modelY)
                let alpha = region.segmentationMaskAlpha(
                    x: Int(source.x.rounded()), y: Int(source.y.rounded())
                )
                pixels[modelY * size + modelX] = normalizedByte(alpha)
            }
        }
        data.append(contentsOf: pixels)
        return data
    }

    private static func normalizedByte(_ value: Float) -> UInt8 {
        UInt8(clamping: Int((min(1, max(0, value)) * 255).rounded()))
    }

    private static func sanitizedDiagnosticComponent(_ value: String) -> String {
        String(value.map { character in
            character.isLetter || character.isNumber || character == "-" ? character : "-"
        })
    }
}
