import AppKit
import SwiftUI

@MainActor
private final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        NSApp.mainMenu?.items.first?.title = "Jasna VR Restoration"
        NSApp.windows.first?.makeKeyAndOrderFront(nil)
    }
}

@main
struct JasnaMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var session = RestorationSession()

    var body: some Scene {
        WindowGroup("Jasna VR Restoration", id: "restoration") {
            ContentView(session: session)
        }
        .defaultSize(width: 820, height: 900)
        .windowResizability(.contentMinSize)
    }
}
