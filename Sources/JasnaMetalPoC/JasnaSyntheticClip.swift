import Foundation

func makeJasnaSyntheticFrame(index frameIndex: Int) -> [Float16] {
    let count = 3 * 256 * 256
    return (0..<count).map { index in
        Float16(
            Float((index * (29 + frameIndex * 6) + 17 + frameIndex * 23) % 1021) / 1020
        )
    }
}

func bicubicDownsampleQuarterReference(_ input: [Float16]) throws -> [Float16] {
    let inputSize = 256
    let outputSize = 64
    let inputPlane = inputSize * inputSize
    let outputPlane = outputSize * outputSize
    guard input.count == 3 * inputPlane else { throw DeformConvError.invalidShape }

    func weight(_ distance: Float) -> Float {
        let a: Float = -0.75
        let x = abs(distance)
        if x <= 1 {
            return (a + 2) * x * x * x - (a + 3) * x * x + 1
        }
        if x < 2 {
            return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a
        }
        return 0
    }

    var output = [Float16](repeating: 0, count: 3 * outputPlane)
    for channel in 0..<3 {
        for outputY in 0..<outputSize {
            let sourceY = (Float(outputY) + 0.5) * 4 - 0.5
            let baseY = Int(floor(sourceY))
            for outputX in 0..<outputSize {
                let sourceX = (Float(outputX) + 0.5) * 4 - 0.5
                let baseX = Int(floor(sourceX))
                var value: Float = 0
                for yy in -1...2 {
                    let wy = weight(sourceY - Float(baseY + yy))
                    for xx in -1...2 {
                        let wx = weight(sourceX - Float(baseX + xx))
                        let source = channel * inputPlane
                            + (baseY + yy) * inputSize + baseX + xx
                        value += Float(input[source]) * wy * wx
                    }
                }
                output[channel * outputPlane + outputY * outputSize + outputX] = Float16(value)
            }
        }
    }
    return output
}
