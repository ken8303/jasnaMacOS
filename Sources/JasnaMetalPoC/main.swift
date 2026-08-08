import Foundation

do {
    try await runJasnaCLI()
} catch {
    FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
    exit(1)
}
