import Foundation

enum PropagationDirection: String, Sendable {
    case backward
    case forward
}

struct PropagationBranch: Sendable {
    let name: String
    let direction: PropagationDirection
    let pass: Int
    let branchIndex: Int

    var offsetPackage: String { "offset_\(name).mtlpackage" }
    var backbonePackage: String { "backbone_\(name).mtlpackage" }
    var backboneInputChannels: Int { (2 + branchIndex) * 64 }
}

struct PropagationStep: Sendable {
    let branch: PropagationBranch
    let position: Int
    let frameIndex: Int
    let flowIndex: Int?
    let secondOrderFrameIndex: Int?
}

struct TemporalSchedule: Sendable {
    let frameCount: Int
    let branches: [PropagationBranch]
    let steps: [PropagationStep]

    init(frameCount: Int) throws {
        guard frameCount > 0 else { throw DeformConvError.invalidShape }
        self.frameCount = frameCount
        branches = [
            PropagationBranch(name: "backward_1", direction: .backward, pass: 1, branchIndex: 0),
            PropagationBranch(name: "forward_1", direction: .forward, pass: 1, branchIndex: 1),
            PropagationBranch(name: "backward_2", direction: .backward, pass: 2, branchIndex: 2),
            PropagationBranch(name: "forward_2", direction: .forward, pass: 2, branchIndex: 3),
        ]
        var generated = [PropagationStep]()
        for branch in branches {
            let traversal = branch.direction == .backward
                ? Array((0..<frameCount).reversed())
                : Array(0..<frameCount)
            for (position, frameIndex) in traversal.enumerated() {
                let flowIndex: Int?
                if position == 0 {
                    flowIndex = nil
                } else if branch.direction == .forward {
                    flowIndex = frameIndex - 1
                } else {
                    flowIndex = frameIndex
                }
                generated.append(PropagationStep(
                    branch: branch,
                    position: position,
                    frameIndex: frameIndex,
                    flowIndex: flowIndex,
                    secondOrderFrameIndex: position > 1 ? traversal[position - 2] : nil
                ))
            }
        }
        steps = generated
        try validate()
    }

    var flowCountPerDirection: Int { max(frameCount - 1, 0) }

    var persistentTensorBytes: Int {
        let fp16Bytes = 2
        let lq = frameCount * 3 * 256 * 256 * fp16Bytes
        let features = frameCount * 5 * 64 * 64 * 64 * fp16Bytes
        let bidirectionalFlows = 2 * flowCountPerDirection * 2 * 64 * 64 * fp16Bytes
        return lq + features + bidirectionalFlows
    }

    func validate() throws {
        guard branches.count == 4,
              steps.count == branches.count * frameCount
        else { throw DeformConvError.commandFailed("invalid propagation schedule size") }
        for branch in branches {
            let branchSteps = steps.filter { $0.branch.name == branch.name }
            guard branchSteps.count == frameCount,
                  branch.backboneInputChannels == (2 + branch.branchIndex) * 64
            else { throw DeformConvError.commandFailed("invalid branch \(branch.name)") }
            let expectedFrames = branch.direction == .backward
                ? Array((0..<frameCount).reversed())
                : Array(0..<frameCount)
            guard branchSteps.map(\.frameIndex) == expectedFrames else {
                throw DeformConvError.commandFailed("incorrect traversal for \(branch.name)")
            }
            for step in branchSteps {
                if let flowIndex = step.flowIndex,
                   !(0..<flowCountPerDirection).contains(flowIndex) {
                    throw DeformConvError.commandFailed("flow index out of range for \(branch.name)")
                }
                if step.position < 2, step.secondOrderFrameIndex != nil {
                    throw DeformConvError.commandFailed("unexpected second-order feature")
                }
            }
        }
    }
}
