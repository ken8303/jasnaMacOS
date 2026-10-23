import AppKit
import UniformTypeIdentifiers

@MainActor
enum FilePanelService {
    static func chooseInputVideos() -> [URL] {
        let panel = NSOpenPanel()
        panel.title = "Choose Side-by-Side VR Video"
        panel.prompt = "Choose Video"
        panel.allowedContentTypes = [.movie]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        return panel.runModal() == .OK ? panel.urls : []
    }

}
