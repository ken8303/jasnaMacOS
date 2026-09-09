import AppKit
import UniformTypeIdentifiers

@MainActor
enum FilePanelService {
    static func chooseInputVideo() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose Side-by-Side VR Video"
        panel.prompt = "Choose Video"
        panel.allowedContentTypes = [.movie]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }

    static func chooseOutputVideo(suggestedName: String) -> URL? {
        let panel = NSSavePanel()
        panel.title = "Choose Restored Video Output"
        panel.prompt = "Choose Output"
        panel.allowedContentTypes = [.quickTimeMovie, .mpeg4Movie]
        panel.nameFieldStringValue = suggestedName
        panel.canCreateDirectories = true
        return panel.runModal() == .OK ? panel.url : nil
    }
}
