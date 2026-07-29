import Foundation
import Metal

struct DeformConvWeightSet {
    static let weightElements = 128 * 9 * 64
    static let biasElements = 64
    static let expectedBytes = (weightElements + biasElements) * MemoryLayout<Float16>.stride

    let direction: String
    let data: Data

    init(direction: String, url: URL) throws {
        let data = try Data(contentsOf: url)
        guard data.count == Self.expectedBytes else {
            throw DeformConvError.commandFailed(
                "\(url.lastPathComponent) has \(data.count) bytes; expected \(Self.expectedBytes)"
            )
        }
        self.direction = direction
        self.data = data
    }

    func makeBuffers(device: MTLDevice) throws -> (weight: MTLBuffer, bias: MTLBuffer) {
        let weightBytes = Self.weightElements * MemoryLayout<Float16>.stride
        return try data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let weight = device.makeBuffer(bytes: base, length: weightBytes, options: .storageModeShared),
                  let bias = device.makeBuffer(
                    bytes: base.advanced(by: weightBytes),
                    length: Self.biasElements * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                  )
            else { throw DeformConvError.metalUnavailable }
            return (weight, bias)
        }
    }
}

func loadDeformConvWeightSets(directoryURL: URL) throws -> [DeformConvWeightSet] {
    try ["backward_1", "forward_1", "backward_2", "forward_2"].map { direction in
        try DeformConvWeightSet(
            direction: direction,
            url: directoryURL.appendingPathComponent("\(direction).dcnfp16")
        )
    }
}
