import SwiftUI

struct MosaicRangesView: View {
    @Bindable var session: RestorationSession

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                ForEach($session.ranges) { $range in
                    HStack {
                        TextField("11:00-30:00", text: $range.value)
                            .textFieldStyle(.roundedBorder)
                            .monospacedDigit()
                        Button {
                            session.removeRange(id: range.id)
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.borderless)
                        .help("Remove range")
                        .disabled(session.ranges.count == 1 || session.isRunning)
                    }
                }

                Text(
                    "Leave blank to detect mosaic across the full video, or enter start-end, "
                        + "for example 11:00-30:00."
                )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack {
                    Button("Add Range", systemImage: "plus") {
                        session.addRange()
                    }
                    .disabled(session.isRunning)
                    Spacer()
                    if let preview = session.normalizedRangePreview {
                        Text(preview)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
            .padding(6)
        } label: {
            Label("Times Containing Mosaic", systemImage: "clock.badge.checkmark")
        }
    }
}
