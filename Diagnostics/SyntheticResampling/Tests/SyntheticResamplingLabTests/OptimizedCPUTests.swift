import Testing
@testable import SyntheticResamplingLab

@Suite("Optimized CPU sampler", .serialized)
struct OptimizedCPUTests {
    @Test(arguments: [1, 4]) func knownInterpolationIncludesAlpha(workers: Int) throws {
        let source = Raster(width: 2, height: 2, bytes: [
            0, 0, 0, 0, 255, 0, 0, 255,
            0, 255, 0, 128, 255, 255, 255, 64,
        ])
        let points: [SIMD2<Float>] = [SIMD2(0.5, 0.5), SIMD2(-100, -100), SIMD2(100, 100)]
        let output = try optimizedCPUSample(source, coordinates: points, workers: workers)
        let centre = SIMD4<Float>(0.5, 0.5, 0.25, Float(447) / 1_020)
        #expect(try maximumError([output[0]], [centre]) < 1e-6)
        #expect(try maximumError(output, referenceSample(source, coordinates: points)) < 1e-6)
    }

    @Test func chunksCoverOddAndSmallOutputsExactlyOnce() throws {
        for count in Array(1...73) + [127, 257, 1_027] {
            for workers in [1, 4] {
                let ranges = try cpuChunkRanges(count: count, workers: workers)
                #expect(ranges.count == min(count, workers))
                #expect(ranges.flatMap { Array($0) } == Array(0..<count))
                #expect(ranges.allSatisfy { !$0.isEmpty })
            }
        }
        #expect(throws: LabError.self) { try cpuChunkRanges(count: 0, workers: 4) }
        #expect(throws: LabError.self) { try cpuChunkRanges(count: 7, workers: 2) }
    }

    @Test func oddOutputsAndOnePixelSourcesAreSafe() throws {
        let source = Raster.generated(width: 31, height: 19, frame: 3)
        for count in [1, 2, 3, 5, 1_003] {
            let points = (0..<count).map { SIMD2<Float>(Float($0 % 43) - 5.25, Float($0 % 29) * 0.69 - 2.5) }
            let serial = try optimizedCPUSample(source, coordinates: points)
            let parallel = try optimizedCPUSample(source, coordinates: points, workers: 4)
            #expect(try maximumError(serial, parallel) == 0)
            #expect(try maximumError(parallel, referenceSample(source, coordinates: points)) < 1e-6)
        }
        let pixel = Raster(width: 1, height: 1, bytes: [17, 93, 201, 128])
        let extremes: [SIMD2<Float>] = [.zero, SIMD2(repeating: .greatestFiniteMagnitude), SIMD2(repeating: -.greatestFiniteMagnitude)]
        let output = try optimizedCPUSample(pixel, coordinates: extremes, workers: 4)
        #expect(try maximumError(output, referenceSample(pixel, coordinates: extremes)) < 1e-6)
    }

    @Test func movingPatternsMatchOracleAndRepeatAcrossWorkers() throws {
        for frame in 0..<24 {
            let source = Raster.generated(width: 257, height: 193, frame: frame)
            let points = movingCoordinates(source: source, width: 96, height: 80, frame: frame)
            let serial = try optimizedCPUSample(source, coordinates: points)
            let parallel = try optimizedCPUSample(source, coordinates: points, workers: 4)
            let repeated = try optimizedCPUSample(source, coordinates: points, workers: 4)
            #expect(try maximumError(parallel, serial) == 0)
            #expect(try maximumError(parallel, repeated) == 0)
            #expect(try maximumError(parallel, referenceSample(source, coordinates: points)) < 1e-6)
        }
    }

    @Test func malformedInputsAreRejectedBeforePointerAccess() {
        let good = Raster.generated(width: 3, height: 3, frame: 0)
        for source in [Raster(width: 0, height: 1, bytes: []), Raster(width: 2, height: 2, bytes: [0]),
                       Raster(width: Int.max, height: 2, bytes: [])] {
            #expect(throws: LabError.self) { try optimizedCPUSample(source, coordinates: [.zero]) }
        }
        for points: [SIMD2<Float>] in [[], [SIMD2(.nan, 0)], [SIMD2(0, .infinity)]] {
            #expect(throws: LabError.self) { try optimizedCPUSample(good, coordinates: points) }
        }
        #expect(throws: LabError.self) { try optimizedCPUSample(good, coordinates: [.zero], workers: 2) }
    }
}
