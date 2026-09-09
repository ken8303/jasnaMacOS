import Foundation
import Testing
@testable import JasnaMetalPoC

@Test
func runtimePerformanceTelemetryFormatsStableStates() {
    #expect(RuntimePerformanceTelemetry.thermalStateName(.nominal) == "nominal")
    #expect(RuntimePerformanceTelemetry.thermalStateName(.fair) == "fair")
    #expect(RuntimePerformanceTelemetry.thermalStateName(.serious) == "serious")
    #expect(RuntimePerformanceTelemetry.thermalStateName(.critical) == "critical")
    #expect(RuntimePerformanceTelemetry.message(
        phase: "process end",
        thermalState: .serious,
        lowPowerModeEnabled: true
    ) == "Runtime performance state (process end): thermal serious, "
        + "low power mode enabled")
}
