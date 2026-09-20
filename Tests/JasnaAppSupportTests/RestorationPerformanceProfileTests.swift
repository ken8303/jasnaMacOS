import Testing
@testable import JasnaAppSupport

struct RestorationPerformanceProfileTests {
    @Test func fastProfileSelectsTheOptimizedRolloutPath() {
        let profile = RestorationPerformanceProfile.fast

        #expect(profile.processEnvironment["JASNA_PERFORMANCE_PROFILE"] == "fast")
        #expect(profile.processEnvironment["JASNA_METAL_WINDOWS_PER_PROCESS"] == "8")
        #expect(profile.processEnvironment["JASNA_DETECT_BATCH_SIZE"] == "2")
        #expect(profile.processEnvironment["JASNA_IN_MEMORY_CROP_CACHE"] == "1")
        #expect(profile.processEnvironment["JASNA_IN_MEMORY_CACHE_LIMIT_MB"] == "512")
        #expect(profile.processEnvironment["JASNA_STEREO_WRITER_DEPTH"] == "2")
        #expect(profile.processEnvironment["JASNA_REGION_PREPARE_DEPTH"] == "2")
        #expect(profile.detail.contains("512 MiB"))
    }

    @Test func balancedProfileSelectsTheLowerMemoryRolloutPath() {
        let profile = RestorationPerformanceProfile.balanced

        #expect(profile.processEnvironment["JASNA_PERFORMANCE_PROFILE"] == "balanced")
        #expect(profile.processEnvironment["JASNA_METAL_WINDOWS_PER_PROCESS"] == "2")
        #expect(profile.processEnvironment["JASNA_DETECT_BATCH_SIZE"] == "1")
        #expect(profile.processEnvironment["JASNA_IN_MEMORY_CROP_CACHE"] == "0")
        #expect(profile.processEnvironment["JASNA_IN_MEMORY_CACHE_LIMIT_MB"] == "128")
        #expect(profile.processEnvironment["JASNA_STEREO_WRITER_DEPTH"] == "1")
        #expect(profile.processEnvironment["JASNA_REGION_PREPARE_DEPTH"] == "1")
        #expect(profile.detail.contains("disk-backed"))
    }
}
