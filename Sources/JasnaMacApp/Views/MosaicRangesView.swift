import SwiftUI

struct MosaicRangesView: View {
    @Bindable var session: RestorationSession

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Mosaic times (optional)").font(.caption)
                Text("Blank = automatic detection for every video")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Add Range", systemImage: "plus") { session.addRange() }
                    .controlSize(.small)
                    .disabled(session.isRunning)
            }
            ScrollView {
                VStack(spacing: 4) {
                    ForEach($session.ranges) { $range in
                        HStack {
                            TextField("Example: 11:00-30:00", text: $range.value)
                                .textFieldStyle(.roundedBorder)
                                .monospacedDigit()
                            Button { session.removeRange(id: range.id) } label: {
                                Image(systemName: "minus.circle")
                            }
                            .buttonStyle(.borderless)
                            .help("Remove range")
                            .disabled(session.ranges.count == 1 || session.isRunning)
                        }
                    }
                }
            }
            .frame(height: min(64, CGFloat(session.ranges.count) * 28))
        }
    }
}
