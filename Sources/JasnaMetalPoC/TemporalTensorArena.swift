import Foundation
import Metal

@available(macOS 27.0, *)
struct TemporalTensorSlot {
    let name: String
    let dimensions: [Int]
    let tensor: any MTLTensor
    let buffer: MTLBuffer
}

@available(macOS 27.0, *)
final class TemporalTensorArena {
    let schedule: TemporalSchedule
    let slots: [TemporalTensorSlot]

    init(device: MTLDevice, schedule: TemporalSchedule) throws {
        self.schedule = schedule
        var allocated = [TemporalTensorSlot]()

        func extents(_ values: [Int]) throws -> MTLTensorExtents {
            var mutable = values
            guard let result = mutable.withUnsafeMutableBufferPointer({
                MTLTensorExtents(__rank: $0.count, values: $0.baseAddress)
            }) else { throw DeformConvError.invalidShape }
            return result
        }

        func allocate(name: String, dimensions: [Int]) throws {
            var strides = [Int]()
            var elements = 1
            for size in dimensions {
                strides.append(elements)
                elements *= size
            }
            guard let buffer = device.makeBuffer(length: elements * 2, options: .storageModeShared)
            else { throw DeformConvError.metalUnavailable }
            let descriptor = MTLTensorDescriptor()
            descriptor.dimensions = try extents(dimensions)
            descriptor.strides = try extents(strides)
            descriptor.dataType = .float16
            descriptor.usage = [.compute, .machineLearning]
            descriptor.storageMode = .shared
            let attachments = MTLTensorBufferAttachments()
            attachments.setBuffer(buffer, offset: 0, for: .data)
            let tensor = try device.makeTensor(descriptor: descriptor, attachments: attachments)
            allocated.append(TemporalTensorSlot(
                name: name, dimensions: dimensions, tensor: tensor, buffer: buffer
            ))
        }

        for frame in 0..<schedule.frameCount {
            try allocate(name: "lq[\(frame)]", dimensions: [256, 256, 3, 1])
            try allocate(name: "spatial[\(frame)]", dimensions: [64, 64, 64, 1])
        }
        for branch in schedule.branches {
            for frame in 0..<schedule.frameCount {
                try allocate(name: "\(branch.name)[\(frame)]", dimensions: [64, 64, 64, 1])
            }
        }
        for direction in [PropagationDirection.forward, .backward] {
            for flow in 0..<schedule.flowCountPerDirection {
                try allocate(name: "flow_\(direction.rawValue)[\(flow)]", dimensions: [64, 64, 2, 1])
            }
        }
        slots = allocated
        guard allocatedBytes == schedule.persistentTensorBytes else {
            throw DeformConvError.commandFailed(
                "tensor arena size \(allocatedBytes) differs from schedule \(schedule.persistentTensorBytes)"
            )
        }
    }

    var allocatedBytes: Int { slots.reduce(0) { $0 + $1.buffer.length } }

    func slot(named name: String) -> TemporalTensorSlot? {
        slots.first { $0.name == name }
    }
}
