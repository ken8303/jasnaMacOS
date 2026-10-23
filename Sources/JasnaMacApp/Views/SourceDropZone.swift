import JasnaAppSupport
import SwiftUI

struct SourceDropZone: View {
    @Bindable var session: RestorationSession
    @State private var isTargeted = false

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "tray.and.arrow.down")
                        .font(.title3)
                    Text("Drop SBS videos here")
                        .font(.headline)
                    Spacer()
                    Button("Add Videos…", action: session.chooseInput)
                }
                .padding(4)
                if !session.sourceURLs.isEmpty {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(session.sourceURLs.enumerated()), id: \.element) { index, url in
                                HStack {
                                    Image(systemName: session.isRunning && index == session.activeSourceIndex ? "play.circle" : "film")
                                    Text(url.lastPathComponent)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                        .help(url.path + " → " + RestorationSourceQueue.outputURL(for: url).lastPathComponent)
                                    Spacer()
                                    Button { session.removeSource(url) } label: {
                                        Image(systemName: "xmark.circle")
                                    }
                                    .buttonStyle(.borderless)
                                    .help("Remove from queue")
                                    .accessibilityLabel("Remove \(url.lastPathComponent)")
                                }
                            }
                        }
                    }
                    .frame(height: min(78, CGFloat(session.sourceURLs.count) * 26))
                    HStack {
                        Text("\(session.sourceURLs.count) video(s) · Saved beside each source with _restored added to the name")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Clear", action: session.clearSources)
                    }
                } else {
                    Text("Results are saved in the source folder as filename_restored.mp4 (or .mov).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Source Videos", systemImage: "film.stack")
        }
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(isTargeted ? Color.accentColor : Color.clear, lineWidth: 2)
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard !session.isRunning else { return false }
            session.addSources(urls)
            return urls.contains { RestorationSourceQueue.accepts($0) }
        } isTargeted: { isTargeted = $0 }
    }
}
