enum LabScope: String {
    case all
    case mixedOnly = "mixed-only"
    case heldOutOnly = "held-out-only"
    case jobProfileOnly = "job-profile-only"
    case selectorCandidateOnly = "selector-candidate-only"
    case boundaryHoldoutOnly = "boundary-holdout-only"
}

struct LabOptions {
    let reportPath: String
    let scope: LabScope
    var mixedOnly: Bool { scope == .mixedOnly }
    var heldOutOnly: Bool { scope == .heldOutOnly }

    init(arguments: [String]) throws {
        guard (arguments.count == 2 || arguments.count == 3), arguments[0] == "--report",
              !arguments[1].isEmpty, !arguments[1].hasPrefix("--"),
              arguments.count == 2 || ["--mixed-only", "--held-out-only", "--job-profile-only", "--selector-candidate-only", "--boundary-holdout-only"].contains(arguments[2]) else {
            throw LabError.invalid("usage: SyntheticResamplingLab --report NEW_REPORT.json [--mixed-only | --held-out-only | --job-profile-only | --selector-candidate-only | --boundary-holdout-only]")
        }
        reportPath = arguments[1]
        scope = arguments.count == 2 ? .all : LabScope(rawValue: String(arguments[2].dropFirst(2)))!
    }
}
