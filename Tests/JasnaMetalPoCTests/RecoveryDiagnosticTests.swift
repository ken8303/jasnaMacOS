import Foundation
import Testing
@testable import JasnaMetalPoC

@Test @available(macOS 27.0, *)
func recoveryDiagnosticEncodesPlanarRGBAsPPM() throws {
    let plane = SideBySideVideoPlan.modelTileSize * SideBySideVideoPlan.modelTileSize
    let values = [Float16](repeating: 1, count: plane)
        + [Float16](repeating: 0.5, count: plane)
        + [Float16](repeating: 0, count: plane)

    let data = try SideBySideRestoration.planarRGBPPM(values)
    let header = "P6\n256 256\n255\n"

    #expect(data.starts(with: Data(header.utf8)))
    #expect(data.count == header.utf8.count + 3 * plane)
    #expect(data[header.utf8.count] == 255)
    #expect(data[header.utf8.count + 1] == 128)
    #expect(data[header.utf8.count + 2] == 0)
}

@Test @available(macOS 27.0, *)
func recoveryDiagnosticDifferenceUsesConfiguredAmplification() throws {
    let plane = SideBySideVideoPlan.modelTileSize * SideBySideVideoPlan.modelTileSize
    let input = [Float16](repeating: 0.25, count: 3 * plane)
    let output = [Float16](repeating: 0.50, count: 3 * plane)

    let data = try SideBySideRestoration.planarDifferencePPM(
        input: input, output: output, amplification: 4
    )
    let headerSize = "P6\n256 256\n255\n".utf8.count

    #expect(data[headerSize] == 255)
}
