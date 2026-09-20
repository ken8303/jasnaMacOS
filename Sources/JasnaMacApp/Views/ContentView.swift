import JasnaAppSupport
import SwiftUI

struct ContentView: View {
    @Bindable var session: RestorationSession
    @AppStorage("restorationPerformanceProfile") private var performanceProfileRawValue =
        RestorationPerformanceProfile.fast.rawValue

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Jasna VR Mosaic Restoration")
                    .font(.largeTitle.weight(.semibold))
                Text(
                    "Restore only the time ranges you identify. The rest of the original "
                        + "side-by-side video is copied directly into the final output."
                )
                .foregroundStyle(.secondary)
            }

            GroupBox {
                VStack(spacing: 8) {
                    FileSelectionRow(
                        title: "Source SBS video",
                        systemImage: "film.stack",
                        url: session.inputURL,
                        placeholder: "Choose the original VR video",
                        action: session.chooseInput
                    )
                    Divider()
                    FileSelectionRow(
                        title: "Restored output",
                        systemImage: "square.and.arrow.down",
                        url: session.outputURL,
                        placeholder: "Choose where to save the restored video",
                        action: session.chooseOutput
                    )
                }
                .padding(6)
            } label: {
                Label("Video Files", systemImage: "movieclapper")
            }
            .disabled(session.isRunning)

            MosaicRangesView(session: session)
                .disabled(session.isRunning)

            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Performance", selection: performanceProfileBinding) {
                        ForEach(RestorationPerformanceProfile.allCases) { profile in
                            Text(profile.title).tag(profile)
                        }
                    }
                    .pickerStyle(.segmented)

                    Label(
                        performanceProfile.detail,
                        systemImage: "gauge.with.dots.needle.67percent"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .padding(6)
            } label: {
                Label("Performance", systemImage: "speedometer")
            }
            .disabled(session.isRunning)

            GroupBox {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(session.progress.phases) { phase in
                        HStack {
                            Text(phase.title)
                                .frame(width: 145, alignment: .leading)
                            ProgressView(value: phase.fraction)
                            Text(phase.fraction, format: .percent.precision(.fractionLength(0)))
                                .monospacedDigit()
                                .frame(width: 42, alignment: .trailing)
                        }
                    }
                    HStack {
                        Text("Overall (approx.)")
                        Spacer()
                        Text(session.progress.overallFraction, format: .percent.precision(.fractionLength(0)))
                            .monospacedDigit()
                        if let estimatedCompletion = session.progress.estimatedCompletion,
                           session.isRunning {
                            Text("Current phase finishes around \(estimatedCompletion.formatted(date: .abbreviated, time: .shortened))")
                                .foregroundStyle(.secondary)
                        } else if session.isRunning {
                            Text("Phase finish time: estimating…")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .font(.caption)
                }
                .padding(6)
            } label: {
                Label("Progress", systemImage: "chart.bar.fill")
            }

            Toggle(
                "Shut down this Mac after successful completion",
                isOn: $session.autoShutdownAfterCompletion
            )
            .help("Only shuts down after the final output passes validation. A 60-second cancellation period is provided.")
            .disabled(session.isRunning)

            if let validationMessage = session.validationMessage {
                Label(validationMessage, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            if let logWarning = session.logWarning {
                Label(logWarning, systemImage: "externaldrive.badge.exclamationmark")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }

            HStack(spacing: 12) {
                if session.isRunning {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: statusSymbol)
                        .foregroundStyle(statusColor)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(session.state.title).font(.headline)
                    Text(session.activity)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()
                if session.canRevealOutput {
                    Button("Show Output") { session.revealOutput() }
                }
                if session.shutdownCountdownSeconds != nil {
                    Button("Cancel Shutdown") { session.cancelAutomaticShutdown() }
                }
                if session.isRunning {
                    Button("Stop", role: .destructive) { session.stop() }
                } else {
                    Button("Start Restoration") {
                        session.start(performanceProfile: performanceProfile)
                    }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.return, modifiers: [.command])
                        .disabled(!session.canStart)
                }
            }

            // Let the log leaf observe logText. Reading it here would invalidate and rebuild
            // this entire form on every live-log refresh.
            RestorationLogView(session: session)
        }
        .padding(24)
        .frame(minWidth: 720, minHeight: 820)
    }

    private var performanceProfile: RestorationPerformanceProfile {
        RestorationPerformanceProfile(rawValue: performanceProfileRawValue) ?? .fast
    }

    private var performanceProfileBinding: Binding<RestorationPerformanceProfile> {
        Binding(
            get: { performanceProfile },
            set: { performanceProfileRawValue = $0.rawValue }
        )
    }

    private var statusSymbol: String {
        switch session.state {
        case .completed: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        default: "circle"
        }
    }

    private var statusColor: Color {
        switch session.state {
        case .completed: .green
        case .failed: .red
        default: .secondary
        }
    }
}
