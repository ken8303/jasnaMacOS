import Foundation

struct DeformConvShape: Sendable {
    let batch: Int
    let inputChannels: Int
    let inputHeight: Int
    let inputWidth: Int
    let outputChannels: Int
    let outputHeight: Int
    let outputWidth: Int
    let kernelHeight: Int
    let kernelWidth: Int
    let padHeight: Int
    let padWidth: Int
    let strideHeight: Int
    let strideWidth: Int
    let dilationHeight: Int
    let dilationWidth: Int
    let groups: Int
    let offsetGroups: Int

    var inputCount: Int { batch * inputChannels * inputHeight * inputWidth }
    var outputCount: Int { batch * outputChannels * outputHeight * outputWidth }
    var kernelArea: Int { kernelHeight * kernelWidth }
    var offsetCount: Int { batch * 2 * offsetGroups * kernelArea * outputHeight * outputWidth }
    var maskCount: Int { batch * offsetGroups * kernelArea * outputHeight * outputWidth }
    var weightCount: Int { outputChannels * (inputChannels / groups) * kernelArea }

    func validate() throws {
        guard batch > 0, inputChannels > 0, outputChannels > 0,
              inputHeight > 0, inputWidth > 0,
              outputHeight > 0, outputWidth > 0,
              kernelHeight > 0, kernelWidth > 0,
              groups > 0, offsetGroups > 0,
              inputChannels.isMultiple(of: groups),
              outputChannels.isMultiple(of: groups),
              inputChannels.isMultiple(of: offsetGroups)
        else {
            throw DeformConvError.invalidShape
        }
    }
}

enum DeformConvError: Error, CustomStringConvertible {
    case invalidShape
    case invalidArrayCount(name: String, expected: Int, actual: Int)
    case metalUnavailable
    case shaderResourceMissing
    case nonFiniteOutput(String)
    case commandFailed(String)

    var isRecoverableNumericalFailure: Bool {
        if case .nonFiniteOutput = self { return true }
        return false
    }

    var description: String {
        switch self {
        case .invalidShape:
            return "Invalid deformable-convolution shape"
        case let .invalidArrayCount(name, expected, actual):
            return "\(name) has \(actual) elements; expected \(expected)"
        case .metalUnavailable:
            return "No Metal device is available"
        case .shaderResourceMissing:
            return "The deform_conv.metal resource is missing"
        case let .nonFiniteOutput(message):
            return "Metal model produced non-finite output: \(message)"
        case let .commandFailed(message):
            return "Metal command failed: \(message)"
        }
    }
}

enum DeformConvCPU {
    static func run(
        shape s: DeformConvShape,
        input: [Float],
        offset: [Float],
        mask: [Float],
        weight: [Float],
        bias: [Float]
    ) throws -> [Float] {
        try s.validate()
        try require(input, count: s.inputCount, name: "input")
        try require(offset, count: s.offsetCount, name: "offset")
        try require(mask, count: s.maskCount, name: "mask")
        try require(weight, count: s.weightCount, name: "weight")
        try require(bias, count: s.outputChannels, name: "bias")

        let channelsPerGroup = s.inputChannels / s.groups
        let outputsPerGroup = s.outputChannels / s.groups
        let channelsPerOffsetGroup = s.inputChannels / s.offsetGroups
        let outputPlane = s.outputHeight * s.outputWidth
        let inputPlane = s.inputHeight * s.inputWidth
        var result = [Float](repeating: 0, count: s.outputCount)

        for n in 0..<s.batch {
            for oc in 0..<s.outputChannels {
                let group = oc / outputsPerGroup
                for oy in 0..<s.outputHeight {
                    for ox in 0..<s.outputWidth {
                        var sum = bias[oc]
                        for localIC in 0..<channelsPerGroup {
                            let ic = group * channelsPerGroup + localIC
                            let offsetGroup = ic / channelsPerOffsetGroup
                            for ky in 0..<s.kernelHeight {
                                for kx in 0..<s.kernelWidth {
                                    let kernelIndex = ky * s.kernelWidth + kx
                                    let offsetChannel = 2 * (offsetGroup * s.kernelArea + kernelIndex)
                                    let spatial = oy * s.outputWidth + ox
                                    let offsetBase = n * 2 * s.offsetGroups * s.kernelArea * outputPlane
                                    let offsetY = offset[offsetBase + offsetChannel * outputPlane + spatial]
                                    let offsetX = offset[offsetBase + (offsetChannel + 1) * outputPlane + spatial]
                                    let y = Float(oy * s.strideHeight - s.padHeight + ky * s.dilationHeight) + offsetY
                                    let x = Float(ox * s.strideWidth - s.padWidth + kx * s.dilationWidth) + offsetX
                                    let sampled = bilinear(
                                        input: input,
                                        base: (n * s.inputChannels + ic) * inputPlane,
                                        height: s.inputHeight,
                                        width: s.inputWidth,
                                        y: y,
                                        x: x
                                    )
                                    let maskBase = n * s.offsetGroups * s.kernelArea * outputPlane
                                    let maskIndex = maskBase + (offsetGroup * s.kernelArea + kernelIndex) * outputPlane + spatial
                                    let weightIndex = ((oc * channelsPerGroup + localIC) * s.kernelHeight + ky) * s.kernelWidth + kx
                                    sum += sampled * mask[maskIndex] * weight[weightIndex]
                                }
                            }
                        }
                        result[((n * s.outputChannels + oc) * s.outputHeight + oy) * s.outputWidth + ox] = sum
                    }
                }
            }
        }
        return result
    }

    private static func require(_ array: [Float], count: Int, name: String) throws {
        guard array.count == count else {
            throw DeformConvError.invalidArrayCount(name: name, expected: count, actual: array.count)
        }
    }

    private static func bilinear(
        input: [Float], base: Int, height: Int, width: Int, y: Float, x: Float
    ) -> Float {
        let y0 = Int(floor(y))
        let x0 = Int(floor(x))
        let y1 = y0 + 1
        let x1 = x0 + 1
        let ly = y - Float(y0)
        let lx = x - Float(x0)

        func value(_ yy: Int, _ xx: Int) -> Float {
            guard yy >= 0, yy < height, xx >= 0, xx < width else { return 0 }
            return input[base + yy * width + xx]
        }

        return value(y0, x0) * (1 - ly) * (1 - lx)
            + value(y0, x1) * (1 - ly) * lx
            + value(y1, x0) * ly * (1 - lx)
            + value(y1, x1) * ly * lx
    }
}

struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func floats(count: Int, range: ClosedRange<Float>) -> [Float] {
        (0..<count).map { _ in Float.random(in: range, using: &self) }
    }
}
