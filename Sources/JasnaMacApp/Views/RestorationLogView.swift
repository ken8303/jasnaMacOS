import SwiftUI

struct RestorationLogView: View {
    @Bindable var session: RestorationSession
    @State private var followLatest = true

    var body: some View {
        let text = session.logText
        GroupBox {
            ScrollViewReader { proxy in
                ScrollView {
                    Text(text.isEmpty ? "Restoration output will appear here." : text)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(text.isEmpty ? .tertiary : .secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                        .padding(8)
                    Color.clear.frame(height: 1).id("log-end")
                }
                .onChange(of: text) {
                    if followLatest { proxy.scrollTo("log-end", anchor: .bottom) }
                }
                .onChange(of: followLatest) {
                    if followLatest { proxy.scrollTo("log-end", anchor: .bottom) }
                }
            }
            .frame(minHeight: 150)
        } label: {
            HStack {
                Text("Live Log")
                Spacer()
                Toggle("Follow latest", isOn: $followLatest)
                    .toggleStyle(.checkbox)
                    .font(.caption)
            }
        }
    }
}
