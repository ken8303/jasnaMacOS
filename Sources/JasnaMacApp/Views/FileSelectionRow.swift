import SwiftUI

struct FileSelectionRow: View {
    let title: String
    let systemImage: String
    let url: URL?
    let placeholder: String
    let action: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: systemImage)
                .frame(width: 22)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.headline)
                Text(url?.path ?? placeholder)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(url == nil ? .tertiary : .secondary)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 12)
            Button("Choose…", action: action)
        }
        .padding(.vertical, 4)
    }
}
