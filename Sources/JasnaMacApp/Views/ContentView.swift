import JasnaAppSupport
import SwiftUI

struct ContentView: View {
    @Bindable var session: RestorationSession
    @AppStorage("restorationPerformanceProfile") private var performanceProfileRawValue =
        RestorationPerformanceProfile.fast.rawValue
    @AppStorage("restorationBackend") private var backendRawValue =
        RestorationBackend.metal.rawValue
    @AppStorage("hybridDetectorDefaultApplied") private var hybridDetectorDefaultApplied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Jasna VR Restoration")
                    .font(.title.weight(.semibold))
                Text("Drop your videos, choose a model, and start restoration.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            SourceDropZone(session: session)
                .disabled(session.isRunning)

            GroupBox {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 16) {
                        Text("Restoration")
                        Picker("Restoration model", selection: backendBinding) {
                            ForEach(RestorationBackend.allCases) { backend in
                                Text(backend.title).tag(backend)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .frame(width: 210)
                        .help(backend.detail)
                        Spacer()
                        Text("Performance")
                        Picker("Performance", selection: performanceProfileBinding) {
                            ForEach(RestorationPerformanceProfile.allCases) { profile in
                                Text(profile.title).tag(profile)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .frame(width: 180)
                        .help(performanceProfile.detail)
                    }
                    MosaicRangesView(session: session)
                    Toggle("Shut down after all videos finish successfully", isOn: $session.autoShutdownAfterCompletion)
                        .font(.caption)
                        .help("Shutdown starts only after all queued videos pass validation, with 60 seconds to cancel.")
                }
                .padding(4)
            } label: {
                Text("Settings")
            }
            .disabled(session.isRunning)

            GroupBox {
                VStack(spacing: 8) {
                    HStack(spacing: 14) {
                        ForEach(session.progress.phases) { phase in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(phase.title).lineLimit(1)
                                    Spacer(minLength: 3)
                                    Text(phase.fraction, format: .percent.precision(.fractionLength(0)))
                                        .monospacedDigit()
                                }
                                .font(.caption)
                                ProgressView(value: phase.fraction)
                            }
                            .frame(maxWidth: .infinity)
                        }
                    }
                    HStack {
                        Text("Overall (approx.)")
                        Text(session.progress.overallFraction, format: .percent.precision(.fractionLength(0)))
                            .monospacedDigit()
                        Spacer()
                        if session.isRunning, let estimatedCompletion = session.progress.estimatedCompletion {
                            Text("Current phase finishes around \(estimatedCompletion.formatted(date: .omitted, time: .shortened))")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .padding(4)
            } label: {
                Text("Progress")
            }

            if let validationMessage = session.validationMessage {
                Text(validationMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .help(validationMessage)
            }
            if let logWarning = session.logWarning {
                Text(logWarning)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .lineLimit(1)
                    .help(logWarning)
            }

            HStack(spacing: 10) {
                Image(systemName: session.isRunning ? "play.circle" : statusSymbol)
                    .foregroundStyle(statusColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text(session.isRunning ? "Video \(session.activeSourceIndex + 1) of \(session.sourceURLs.count)" : session.state.title)
                        .font(.headline)
                    Text(session.implementationResumeAvailable ? "App updated. Resume matching saved work." : session.activity)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .help(session.activity)
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
                    if session.implementationResumeAvailable {
                        Button("Resume Saved Work") {
                            session.start(performanceProfile: performanceProfile, backend: backend, allowImplementationResume: true)
                        }
                        .disabled(!session.canStart)
                    }
                    Button(session.freshStartAvailable ? "Start Fresh" : "Start Restoration") {
                        session.start(performanceProfile: performanceProfile, backend: backend, startFresh: session.freshStartAvailable)
                    }
                    .help(session.freshStartAvailable
                          ? "Preserve this video's previous output and saved work in a backup folder, then restart with the selected settings."
                          : "Restore the queued videos, resuming compatible saved work.")
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(!session.canStart)
                }
            }

            RestorationLogView(session: session)
        }
        .padding(18)
        .frame(minWidth: 760, minHeight: 720)
        .onAppear {
            if !hybridDetectorDefaultApplied {
                backendRawValue = RestorationBackend.metal.rawValue
                hybridDetectorDefaultApplied = true
            }
            session.selectBackend(backend)
        }
    }

    private var performanceProfile: RestorationPerformanceProfile {
        RestorationPerformanceProfile(rawValue: performanceProfileRawValue) ?? .fast
    }

    private var backend: RestorationBackend {
        RestorationBackend(rawValue: backendRawValue) ?? .metal
    }

    private var backendBinding: Binding<RestorationBackend> {
        Binding(
            get: { backend },
            set: {
                backendRawValue = $0.rawValue
                session.selectBackend($0)
            }
        )
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
