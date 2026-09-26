import Foundation

@available(macOS 27.0, *)
enum MLXRestorationBridge {
    private static let worker = Worker()

    private final class Worker: @unchecked Sendable {
        private let lock = NSLock()
        private var process: Process?
        private var input: FileHandle?
        private var output: FileHandle?
        private var loadedArchive: URL?

        func restore(
            python: URL, root: URL, library: URL, script: URL, archive: URL,
            inputPath: URL, outputPath: URL, frameCount: Int
        ) throws -> String {
            lock.lock()
            defer { lock.unlock() }
            if process?.isRunning == true && loadedArchive != archive {
                input?.closeFile()
                process?.terminate()
                process?.waitUntilExit()
                process = nil
            }
            if process?.isRunning != true {
                let child = Process()
                child.executableURL = python
                child.arguments = [
                    "-c",
                    "import runpy,sys;sys.path[:0]=[sys.argv[1],sys.argv[2]];"
                        + "script=sys.argv[3];sys.argv=sys.argv[3:];"
                        + "runpy.run_path(script,run_name='__main__')",
                    library.path, root.appendingPathComponent("tools").path, script.path,
                    "--archive", archive.path, "--serve"
                ]
                child.currentDirectoryURL = root
                let stdinPipe = Pipe()
                let stdoutPipe = Pipe()
                child.standardInput = stdinPipe
                child.standardOutput = stdoutPipe
                child.standardError = FileHandle.nullDevice
                try child.run()
                process = child
                loadedArchive = archive
                input = stdinPipe.fileHandleForWriting
                output = stdoutPipe.fileHandleForReading
                guard try readLine() == "READY" else {
                    child.terminate()
                    throw DeformConvError.commandFailed("MLX worker failed to start")
                }
            }
            let command = "\(inputPath.path)\t\(outputPath.path)\t\(frameCount)\n"
            try input?.write(contentsOf: Data(command.utf8))
            guard let response = try readLine() else {
                throw DeformConvError.commandFailed("MLX worker stopped during crop restoration")
            }
            guard response.hasPrefix("OK\t") else {
                throw DeformConvError.commandFailed("MLX crop restoration failed: \(response)")
            }
            return response
        }

        private func readLine() throws -> String? {
            guard let output else { return nil }
            var bytes = [UInt8]()
            while bytes.count < 4096 {
                guard let data = try output.read(upToCount: 1), !data.isEmpty else {
                    return nil
                }
                if data[0] == 10 { return String(decoding: bytes, as: UTF8.self) }
                bytes.append(data[0])
            }
            throw DeformConvError.commandFailed("MLX worker response is too long")
        }
    }

    static var isSelected: Bool {
        ProcessInfo.processInfo.environment["JASNA_RESTORATION_BACKEND"] == "mlx"
    }

    static func restore(
        inputFrames: [[Float16]],
        modelsURL: URL
    ) throws -> (frames: [[Float16]], gpuMilliseconds: Double) {
        let root = modelsURL.deletingLastPathComponent().deletingLastPathComponent()
        let archive = root.appendingPathComponent("Models/MLX/basicvsrpp-v1.2.safetensors")
        let library = root.appendingPathComponent("Models/MLXRuntime", isDirectory: true)
        let script = root.appendingPathComponent("tools/restore_mlx_crop.py")
        let python = root.appendingPathComponent(".venv-rfdetr/bin/python")
        for required in [archive, library, script, python] {
            guard FileManager.default.fileExists(atPath: required.path) else {
                throw DeformConvError.commandFailed(
                    "MLX runtime component is missing: \(required.path)"
                )
            }
        }
        let elementsPerFrame = SideBySideRestoration.tileElements
        guard (3...40).contains(inputFrames.count),
              inputFrames.allSatisfy({ $0.count == elementsPerFrame }) else {
            throw DeformConvError.invalidShape
        }
        let scratchRoot = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["TMPDIR"] ?? NSTemporaryDirectory(),
            isDirectory: true
        )
        let scratch = scratchRoot.appendingPathComponent(
            "jasna-mlx-crop-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let input = scratch.appendingPathComponent("input.fp16")
        let output = scratch.appendingPathComponent("output.fp16")
        let values = inputFrames.flatMap { $0 }
        try values.withUnsafeBytes { bytes in
            try Data(bytes).write(to: input, options: .atomic)
        }

        let started = ContinuousClock.now
        let message = try worker.restore(
            python: python, root: root, library: library, script: script,
            archive: archive, inputPath: input, outputPath: output,
            frameCount: inputFrames.count
        )
        let data = try Data(contentsOf: output)
        let expectedBytes = inputFrames.count * elementsPerFrame * MemoryLayout<Float16>.stride
        guard data.count == expectedBytes else {
            throw DeformConvError.commandFailed("MLX crop output has an invalid size")
        }
        let restored = data.withUnsafeBytes { bytes -> [Float16] in
            var result = [Float16](repeating: 0, count: bytes.count / 2)
            result.withUnsafeMutableBytes { destination in
                destination.copyMemory(from: bytes)
            }
            return result
        }
        guard restored.allSatisfy({ Float($0).isFinite }) else {
            throw DeformConvError.commandFailed("MLX crop output contains non-finite pixels")
        }
        let frames = (0..<inputFrames.count).map { index in
            Array(restored[(index * elementsPerFrame)..<((index + 1) * elementsPerFrame)])
        }
        let elapsed = started.duration(to: .now).components
        let wallMilliseconds = Double(elapsed.seconds) * 1_000
            + Double(elapsed.attoseconds) / 1_000_000_000_000_000
        print("MLX crop: \(inputFrames.count) frames, \(String(format: "%.1f", wallMilliseconds)) ms wall; \(message.trimmingCharacters(in: .whitespacesAndNewlines))")
        return (frames, 0)
    }
}
