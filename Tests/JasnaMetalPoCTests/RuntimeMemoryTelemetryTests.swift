import Testing
@testable import JasnaMetalPoC

@Test func runtimeMemoryTelemetryFormatsBinaryGigabytes() {
    #expect(RuntimeMemoryTelemetry.formattedGiB(bytes: 0) == "0.00 GiB")
    #expect(RuntimeMemoryTelemetry.formattedGiB(bytes: 1_073_741_824) == "1.00 GiB")
    #expect(RuntimeMemoryTelemetry.formattedGiB(bytes: 26_843_545_600) == "25.00 GiB")
}
