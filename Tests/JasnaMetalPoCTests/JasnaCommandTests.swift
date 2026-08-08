import Testing
@testable import JasnaMetalPoC

@Test
func commandCatalogContainsUniqueFlags() {
    let flags = JasnaCommand.allCases.map(\.rawValue)

    #expect(Set(flags).count == flags.count)
    #expect(flags.allSatisfy { $0.hasPrefix("--") })
}

@Test
func commandLineFindsTypedCommandsWithoutGlobalState() {
    let commandLine = JasnaCommandLine(arguments: [
        "JasnaMetalPoC", "--restore-eye-video", "input.mov", "output.mov",
    ])

    #expect(!commandLine.hasNoArguments)
    #expect(commandLine.contains(.restoreEyeVideo))
    #expect(commandLine.index(of: .restoreEyeVideo) == 1)
    #expect(!commandLine.contains(.benchmark))
}

@Test
func emptyCommandLineSelectsDefaultMode() {
    let commandLine = JasnaCommandLine(arguments: ["JasnaMetalPoC"])

    #expect(commandLine.hasNoArguments)
}

@Test
func dispatcherRunsOnlyRequestedAndDefaultCommands() async {
    let requested = JasnaCommandLine(arguments: ["JasnaMetalPoC", "--schedule", "3"])
    var requestedIndexes = [Int]()
    await requested.dispatch(.benchmark, whenNoArguments: true) { index in
        requestedIndexes.append(index)
    }
    await requested.dispatch(.schedule) { index in
        requestedIndexes.append(index)
    }
    #expect(requestedIndexes == [1])

    let empty = JasnaCommandLine(arguments: ["JasnaMetalPoC"])
    var defaultIndexes = [Int]()
    await empty.dispatch(.selfTest, whenNoArguments: true) { index in
        defaultIndexes.append(index)
    }
    #expect(defaultIndexes == [0])
}

@Test
func dispatcherPreservesAliasPriority() async {
    let commandLine = JasnaCommandLine(arguments: [
        "JasnaMetalPoC", "--restore-sbs-window", "--restore-sbs-video",
    ])
    var selectedIndex: Int?

    await commandLine.dispatch(
        firstOf: [.restoreSBSVideo, .restoreSBSWindow]
    ) { index in
        selectedIndex = index
    }

    #expect(selectedIndex == 2)
}
