import Foundation

struct MosaicRegionSubdivisionConfiguration: Equatable, Sendable {
    let maximumBlendDimension: Int
    let overlap: Int
    let splitLimit: Int
    let maximumAxisCrops: Int
    let maskGrowthFraction: Double
    let maskFeatherFraction: Double
    let blockResidualGrowthFraction: Double
    let maskTemporalRadius: Int
    let detailCropDimension: Int
    let detailCropCount: Int

    static let disabled = Self(
        maximumBlendDimension: 0,
        overlap: 0,
        splitLimit: 0,
        maximumAxisCrops: 1,
        maskGrowthFraction: 0,
        maskFeatherFraction: 0,
        blockResidualGrowthFraction: 0,
        maskTemporalRadius: 0,
        detailCropDimension: 0,
        detailCropCount: 0
    )

    static func fromEnvironment(_ environment: [String: String]) -> Self {
        let maximumBlendDimension = max(
            0, Int(environment["JASNA_LARGE_REGION_MAX_BLEND"] ?? "") ?? 768
        )
        guard maximumBlendDimension > 0 else { return .disabled }
        return Self(
            maximumBlendDimension: maximumBlendDimension,
            overlap: max(0, Int(environment["JASNA_LARGE_REGION_OVERLAP"] ?? "") ?? 96),
            splitLimit: max(0, Int(environment["JASNA_LARGE_REGION_SPLIT_LIMIT"] ?? "") ?? 1),
            maximumAxisCrops: min(
                4, max(2, Int(environment["JASNA_LARGE_REGION_MAX_AXIS_CROPS"] ?? "") ?? 4)
            ),
            maskGrowthFraction: boundedFraction(
                environment["JASNA_LARGE_REGION_MASK_GROWTH"], default: 0.05
            ),
            maskFeatherFraction: boundedFraction(
                environment["JASNA_LARGE_REGION_MASK_FEATHER"], default: 0.025
            ),
            blockResidualGrowthFraction: boundedFraction(
                environment["JASNA_LARGE_REGION_BLOCK_GROWTH"], default: 0.04
            ),
            maskTemporalRadius: min(
                2,
                max(
                    0,
                    Int(environment["JASNA_LARGE_REGION_MASK_TEMPORAL_RADIUS"] ?? "") ?? 1
                )
            ),
            detailCropDimension: min(
                maximumBlendDimension,
                max(
                    256,
                    Int(environment["JASNA_LARGE_REGION_DETAIL_DIMENSION"] ?? "") ?? 576
                )
            ),
            detailCropCount: min(
                2,
                max(0, Int(environment["JASNA_LARGE_REGION_DETAIL_CROPS"] ?? "") ?? 2)
            )
        )
    }

    private static func boundedFraction(_ value: String?, default defaultValue: Double) -> Double {
        guard let value, let parsed = Double(value), parsed.isFinite else {
            return defaultValue
        }
        return min(0.25, max(0, parsed))
    }
}

enum MosaicRegionSubdivision {
    struct Result: Equatable, Sendable {
        let regions: [MosaicRegion]
        let splitRegionCount: Int
        let addedModelCropCount: Int
    }

    static func expand(
        _ regions: [MosaicRegion],
        configuration: MosaicRegionSubdivisionConfiguration
    ) -> Result {
        guard configuration.maximumBlendDimension > 0,
              !regions.isEmpty
        else {
            return Result(regions: regions, splitRegionCount: 0, addedModelCropCount: 0)
        }
        let candidates = regions.indices.filter { index in
            max(
                regions[index].effectiveBlendWidth,
                regions[index].effectiveBlendHeight
            ) > configuration.maximumBlendDimension
        }.sorted { left, right in
            let leftRegion = regions[left]
            let rightRegion = regions[right]
            let leftArea = leftRegion.effectiveBlendWidth * leftRegion.effectiveBlendHeight
            let rightArea = rightRegion.effectiveBlendWidth * rightRegion.effectiveBlendHeight
            if leftArea == rightArea { return left < right }
            return leftArea > rightArea
        }
        let selected = Set(candidates.prefix(configuration.splitLimit))
        var expanded = [MosaicRegion]()
        expanded.reserveCapacity(
            regions.count + selected.count * (3 + configuration.detailCropCount)
        )
        var added = 0
        for (index, region) in regions.enumerated() {
            let expandedMaskRegion = expandingMaskCoverage(
                of: region, configuration: configuration
            )
            guard selected.contains(index) else {
                expanded.append(expandedMaskRegion)
                continue
            }
            let children = subdivide(
                expandedMaskRegion,
                configuration: configuration,
                subdivisionGroup: index + 1,
                gridReference: region
            )
            let detailCrops = lowerResidualDetailCrops(
                from: expandedMaskRegion,
                focusReference: region,
                configuration: configuration,
                subdivisionGroup: index + 1
            )
            expanded.append(contentsOf: children)
            expanded.append(contentsOf: detailCrops)
            added += children.count + detailCrops.count - 1
        }
        return Result(
            regions: expanded,
            splitRegionCount: selected.count,
            addedModelCropCount: added
        )
    }

    static func expandedMask(
        _ data: Data,
        width: Int,
        height: Int,
        growthFraction: Double,
        featherFraction: Double,
        blockResidualGrowthFraction: Double = 0
    ) -> Data? {
        guard width > 1,
              height > 1,
              data.count == width * height,
              growthFraction > 0 || blockResidualGrowthFraction > 0
        else { return data.count == width * height ? data : nil }
        let scale = max(width, height)
        let growth = max(1, Int(ceil(Double(scale) * growthFraction)))
        let feather = max(1, Int(ceil(Double(scale) * featherFraction)))
        let original = [UInt8](data)
        let infinity = Int.max / 8
        var distance = original.map { $0 >= 128 ? 0 : infinity }
        guard distance.contains(0) else { return data }

        @inline(__always)
        func relaxed(_ current: Int, _ neighbour: Int, cost: Int) -> Int {
            neighbour >= infinity - cost ? current : min(current, neighbour + cost)
        }

        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                var value = distance[index]
                if x > 0 { value = relaxed(value, distance[index - 1], cost: 3) }
                if y > 0 {
                    value = relaxed(value, distance[index - width], cost: 3)
                    if x > 0 {
                        value = relaxed(value, distance[index - width - 1], cost: 4)
                    }
                    if x + 1 < width {
                        value = relaxed(value, distance[index - width + 1], cost: 4)
                    }
                }
                distance[index] = value
            }
        }
        for y in stride(from: height - 1, through: 0, by: -1) {
            for x in stride(from: width - 1, through: 0, by: -1) {
                let index = y * width + x
                var value = distance[index]
                if x + 1 < width { value = relaxed(value, distance[index + 1], cost: 3) }
                if y + 1 < height {
                    value = relaxed(value, distance[index + width], cost: 3)
                    if x + 1 < width {
                        value = relaxed(value, distance[index + width + 1], cost: 4)
                    }
                    if x > 0 {
                        value = relaxed(value, distance[index + width - 1], cost: 4)
                    }
                }
                distance[index] = value
            }
        }

        let innerDistance = growth * 3
        let outerDistance = (growth + feather) * 3
        let transition = max(1, outerDistance - innerDistance)
        var output = original
        for index in output.indices {
            let candidate: Int
            if distance[index] <= innerDistance {
                candidate = 255
            } else if distance[index] < outerDistance {
                candidate = Int(
                    (255.0 * Double(outerDistance - distance[index]) / Double(transition)).rounded()
                )
            } else {
                candidate = 0
            }
            output[index] = max(output[index], UInt8(clamping: candidate))
        }
        if blockResidualGrowthFraction > 0 {
            addBlockResidualCoverage(
                source: original,
                output: &output,
                width: width,
                height: height,
                innerRadius: max(
                    1,
                    Int(ceil(Double(scale) * (growthFraction + blockResidualGrowthFraction)))
                ),
                featherRadius: max(0, Int(ceil(Double(scale) * featherFraction)))
            )
        }
        return Data(output)
    }

    /// Adds an axis-aligned halo around the semantic mask. Mosaic residuals are
    /// square blocks, so this reaches corners that rounded dilation can miss.
    private static func addBlockResidualCoverage(
        source: [UInt8],
        output: inout [UInt8],
        width: Int,
        height: Int,
        innerRadius: Int,
        featherRadius: Int
    ) {
        var prefix = [Int](repeating: 0, count: (width + 1) * (height + 1))
        for y in 0..<height {
            var row = 0
            for x in 0..<width {
                if source[y * width + x] >= 128 { row += 1 }
                prefix[(y + 1) * (width + 1) + x + 1]
                    = prefix[y * (width + 1) + x + 1] + row
            }
        }

        @inline(__always)
        func containsSeed(x: Int, y: Int, radius: Int) -> Bool {
            let left = max(0, x - radius)
            let right = min(width, x + radius + 1)
            let top = max(0, y - radius)
            let bottom = min(height, y + radius + 1)
            let stride = width + 1
            let count = prefix[bottom * stride + right]
                - prefix[top * stride + right]
                - prefix[bottom * stride + left]
                + prefix[top * stride + left]
            return count > 0
        }

        let outerRadius = innerRadius + featherRadius
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                if containsSeed(x: x, y: y, radius: innerRadius) {
                    output[index] = 255
                    continue
                }
                guard featherRadius > 0,
                      containsSeed(x: x, y: y, radius: outerRadius)
                else { continue }
                var firstHit = outerRadius
                for radius in (innerRadius + 1)...outerRadius {
                    if containsSeed(x: x, y: y, radius: radius) {
                        firstHit = radius
                        break
                    }
                }
                let remaining = outerRadius - firstHit + 1
                let candidate = UInt8(clamping: Int(
                    (255.0 * Double(remaining) / Double(featherRadius + 1)).rounded()
                ))
                output[index] = max(output[index], candidate)
            }
        }
    }

    static func temporallyStabilizedKeyframes(
        _ keyframes: [MosaicMaskKeyframe],
        expectedByteCount: Int,
        radius: Int
    ) -> [MosaicMaskKeyframe] {
        guard radius > 0, keyframes.count > 1 else { return keyframes }
        return keyframes.indices.map { index in
            var output = [UInt8](keyframes[index].maskData)
            guard output.count == expectedByteCount else { return keyframes[index] }
            let lower = max(keyframes.startIndex, index - radius)
            let upper = min(keyframes.index(before: keyframes.endIndex), index + radius)
            for neighbourIndex in lower...upper where neighbourIndex != index {
                let neighbour = [UInt8](keyframes[neighbourIndex].maskData)
                guard neighbour.count == expectedByteCount else { continue }
                for byteIndex in output.indices {
                    // Keep the neighbour below the 128 expansion threshold:
                    // it contributes alpha but does not seed another full halo.
                    let temporalCandidate = UInt8(
                        clamping: Int(neighbour[byteIndex]) / 2
                    )
                    output[byteIndex] = max(output[byteIndex], temporalCandidate)
                }
            }
            return MosaicMaskKeyframe(
                frame: keyframes[index].frame,
                maskData: Data(output)
            )
        }
    }

    private static func expandingMaskCoverage(
        of region: MosaicRegion,
        configuration: MosaicRegionSubdivisionConfiguration
    ) -> MosaicRegion {
        guard let maskWidth = region.maskWidth,
              let maskHeight = region.maskHeight,
              configuration.maskGrowthFraction > 0
                || configuration.blockResidualGrowthFraction > 0
                || configuration.maskTemporalRadius > 0
        else { return region }
        let expandedStaticMask = region.maskData.flatMap {
            expandedMask(
                $0,
                width: maskWidth,
                height: maskHeight,
                growthFraction: configuration.maskGrowthFraction,
                featherFraction: configuration.maskFeatherFraction,
                blockResidualGrowthFraction: configuration.blockResidualGrowthFraction
            )
        }
        let stabilizedKeyframes = region.maskKeyframes.map {
            temporallyStabilizedKeyframes(
                $0,
                expectedByteCount: maskWidth * maskHeight,
                radius: configuration.maskTemporalRadius
            )
        }
        let expandedKeyframes = stabilizedKeyframes?.compactMap { keyframe in
            expandedMask(
                keyframe.maskData,
                width: maskWidth,
                height: maskHeight,
                growthFraction: configuration.maskGrowthFraction,
                featherFraction: configuration.maskFeatherFraction,
                blockResidualGrowthFraction: configuration.blockResidualGrowthFraction
            ).map { MosaicMaskKeyframe(frame: keyframe.frame, maskData: $0) }
        }
        let coverageFraction = configuration.maskGrowthFraction
            + configuration.blockResidualGrowthFraction
            + configuration.maskFeatherFraction
        let horizontalGrowth = Int(ceil(Double(region.width) * coverageFraction))
        let verticalGrowth = Int(ceil(Double(region.height) * coverageFraction))
        let blendLeft = max(region.x, region.effectiveBlendX - horizontalGrowth)
        let blendTop = max(region.y, region.effectiveBlendY - verticalGrowth)
        let blendRight = min(
            region.x + region.width,
            region.effectiveBlendX + region.effectiveBlendWidth + horizontalGrowth
        )
        let blendBottom = min(
            region.y + region.height,
            region.effectiveBlendY + region.effectiveBlendHeight + verticalGrowth
        )
        return MosaicRegion(
            startFrame: region.startFrame,
            endFrame: region.endFrame,
            x: region.x,
            y: region.y,
            width: region.width,
            height: region.height,
            confidence: region.confidence,
            blendX: blendLeft,
            blendY: blendTop,
            blendWidth: blendRight - blendLeft,
            blendHeight: blendBottom - blendTop,
            maskWidth: expandedStaticMask == nil ? nil : maskWidth,
            maskHeight: expandedStaticMask == nil ? nil : maskHeight,
            maskData: expandedStaticMask,
            maskKeyframes: expandedKeyframes?.isEmpty == false ? expandedKeyframes : nil,
            subdivisionGroup: region.subdivisionGroup,
            detailBlendFeather: region.detailBlendFeather
        )
    }

    static func subdivide(
        _ region: MosaicRegion,
        configuration: MosaicRegionSubdivisionConfiguration,
        subdivisionGroup: Int = 1,
        gridReference: MosaicRegion? = nil
    ) -> [MosaicRegion] {
        let maximum = configuration.maximumBlendDimension
        guard maximum > 0 else { return [region] }
        // Mask coverage may enlarge the blend envelope, but it must not silently
        // turn a 2x2 inference grid into 3x3. Choose model density from the
        // detector's original envelope and let the expanded mask use that grid.
        let gridRegion = gridReference ?? region
        let columns = min(
            configuration.maximumAxisCrops,
            max(1, (gridRegion.effectiveBlendWidth + maximum - 1) / maximum)
        )
        let rows = min(
            configuration.maximumAxisCrops,
            max(1, (gridRegion.effectiveBlendHeight + maximum - 1) / maximum)
        )
        guard columns > 1 || rows > 1 else { return [region] }

        let originalBlendLeft = region.effectiveBlendX
        let originalBlendTop = region.effectiveBlendY
        let originalBlendRight = originalBlendLeft + region.effectiveBlendWidth
        let originalBlendBottom = originalBlendTop + region.effectiveBlendHeight
        let extractionRight = region.x + region.width
        let extractionBottom = region.y + region.height
        let blendOverlap = max(1, configuration.overlap / 2)
        var result = [MosaicRegion]()
        result.reserveCapacity(columns * rows)

        for row in 0..<rows {
            let partitionTop = originalBlendTop + row * region.effectiveBlendHeight / rows
            let partitionBottom = originalBlendTop
                + (row + 1) * region.effectiveBlendHeight / rows
            let childBlendTop = max(
                originalBlendTop, partitionTop - (row == 0 ? 0 : blendOverlap)
            )
            let childBlendBottom = min(
                originalBlendBottom,
                partitionBottom + (row + 1 == rows ? 0 : blendOverlap)
            )
            for column in 0..<columns {
                let partitionLeft = originalBlendLeft
                    + column * region.effectiveBlendWidth / columns
                let partitionRight = originalBlendLeft
                    + (column + 1) * region.effectiveBlendWidth / columns
                let childBlendLeft = max(
                    originalBlendLeft, partitionLeft - (column == 0 ? 0 : blendOverlap)
                )
                let childBlendRight = min(
                    originalBlendRight,
                    partitionRight + (column + 1 == columns ? 0 : blendOverlap)
                )
                let childX = max(region.x, childBlendLeft - configuration.overlap)
                let childY = max(region.y, childBlendTop - configuration.overlap)
                let childRight = min(
                    extractionRight, childBlendRight + configuration.overlap
                )
                let childBottom = min(
                    extractionBottom, childBlendBottom + configuration.overlap
                )
                result.append(
                    croppedRegion(
                        region,
                        x: childX,
                        y: childY,
                        width: childRight - childX,
                        height: childBottom - childY,
                        blendX: childBlendLeft,
                        blendY: childBlendTop,
                        blendWidth: childBlendRight - childBlendLeft,
                        blendHeight: childBlendBottom - childBlendTop,
                        subdivisionGroup: subdivisionGroup
                    )
                )
            }
        }
        return result
    }

    /// Adds a compact high-resolution crop centred on the lower semantic-mask
    /// boundary. The large VR failures concentrate there; one extra crop raises
    /// sampling density without changing every 2x2 grid into an expensive 3x3.
    static func lowerResidualDetailCrops(
        from region: MosaicRegion,
        focusReference: MosaicRegion,
        configuration: MosaicRegionSubdivisionConfiguration,
        subdivisionGroup: Int
    ) -> [MosaicRegion] {
        guard configuration.detailCropCount > 0,
              configuration.detailCropDimension > 0,
              let bounds = maskPixelBounds(in: focusReference)
        else { return [] }

        let blendLeftLimit = region.effectiveBlendX
        let blendTopLimit = region.effectiveBlendY
        let blendRightLimit = blendLeftLimit + region.effectiveBlendWidth
        let blendBottomLimit = blendTopLimit + region.effectiveBlendHeight
        let detailWidth = min(configuration.detailCropDimension, region.effectiveBlendWidth)
        let detailHeight = min(configuration.detailCropDimension, region.effectiveBlendHeight)
        guard detailWidth > 0, detailHeight > 0 else { return [] }

        let focusWidth = max(1, bounds.right - bounds.left)
        // One focused crop remains enough for compact regions. Long moving VR
        // boundaries need a second sample point; otherwise most of the strip is
        // still reconstructed only by the coarse subdivision grid.
        let count = min(
            configuration.detailCropCount,
            focusWidth > 3 * detailWidth ? 2 : 1
        )
        let extractionRight = region.x + region.width
        let extractionBottom = region.y + region.height
        return (0..<count).map { detailIndex in
            let numerator = 2 * detailIndex + 1
            let focusX = bounds.left + focusWidth * numerator / (2 * count)
            let desiredLeft = focusX - detailWidth / 2
            let desiredTop = bounds.bottom - detailHeight / 2
            let detailBlendLeft = min(
                max(blendLeftLimit, desiredLeft), blendRightLimit - detailWidth
            )
            let detailBlendTop = min(
                max(blendTopLimit, desiredTop), blendBottomLimit - detailHeight
            )
            let detailBlendRight = detailBlendLeft + detailWidth
            let detailBlendBottom = detailBlendTop + detailHeight
            let detailX = max(region.x, detailBlendLeft - configuration.overlap)
            let detailY = max(region.y, detailBlendTop - configuration.overlap)
            let detailRight = min(
                extractionRight, detailBlendRight + configuration.overlap
            )
            let detailBottom = min(
                extractionBottom, detailBlendBottom + configuration.overlap
            )
            return croppedRegion(
                region,
                x: detailX,
                y: detailY,
                width: detailRight - detailX,
                height: detailBottom - detailY,
                blendX: detailBlendLeft,
                blendY: detailBlendTop,
                blendWidth: detailWidth,
                blendHeight: detailHeight,
                subdivisionGroup: subdivisionGroup,
                detailBlendFeather: max(
                    48,
                    min(128, min(detailWidth, detailHeight) / 8)
                )
            )
        }
    }

    private static func maskPixelBounds(
        in region: MosaicRegion
    ) -> (left: Int, top: Int, right: Int, bottom: Int)? {
        guard let maskWidth = region.maskWidth,
              let maskHeight = region.maskHeight,
              maskWidth > 1,
              maskHeight > 1
        else { return nil }
        var masks = [Data]()
        if let maskData = region.maskData { masks.append(maskData) }
        if masks.isEmpty, let keyframes = region.maskKeyframes {
            masks.append(contentsOf: keyframes.map(\.maskData))
        }
        var minimumX = maskWidth
        var minimumY = maskHeight
        var maximumX = -1
        var maximumY = -1
        for data in masks where data.count == maskWidth * maskHeight {
            data.withUnsafeBytes { bytes in
                let values = bytes.bindMemory(to: UInt8.self)
                for y in 0..<maskHeight {
                    for x in 0..<maskWidth where values[y * maskWidth + x] >= 128 {
                        minimumX = min(minimumX, x)
                        minimumY = min(minimumY, y)
                        maximumX = max(maximumX, x)
                        maximumY = max(maximumY, y)
                    }
                }
            }
        }
        guard maximumX >= minimumX, maximumY >= minimumY else { return nil }
        return (
            left: region.x + minimumX * region.width / maskWidth,
            top: region.y + minimumY * region.height / maskHeight,
            right: region.x + (maximumX + 1) * region.width / maskWidth,
            bottom: region.y + (maximumY + 1) * region.height / maskHeight
        )
    }

    private static func croppedRegion(
        _ region: MosaicRegion,
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        blendX: Int,
        blendY: Int,
        blendWidth: Int,
        blendHeight: Int,
        subdivisionGroup: Int,
        detailBlendFeather: Int? = nil
    ) -> MosaicRegion {
        let croppedMask = region.maskData.flatMap {
            resampleMask($0, from: region, x: x, y: y, width: width, height: height)
        }
        let croppedKeyframes = region.maskKeyframes?.compactMap { keyframe in
            resampleMask(
                keyframe.maskData,
                from: region,
                x: x,
                y: y,
                width: width,
                height: height
            ).map { MosaicMaskKeyframe(frame: keyframe.frame, maskData: $0) }
        }
        return MosaicRegion(
            startFrame: region.startFrame,
            endFrame: region.endFrame,
            x: x,
            y: y,
            width: width,
            height: height,
            confidence: region.confidence,
            blendX: blendX,
            blendY: blendY,
            blendWidth: blendWidth,
            blendHeight: blendHeight,
            maskWidth: croppedMask == nil ? nil : region.maskWidth,
            maskHeight: croppedMask == nil ? nil : region.maskHeight,
            maskData: croppedMask,
            maskKeyframes: croppedKeyframes?.isEmpty == false ? croppedKeyframes : nil,
            subdivisionGroup: subdivisionGroup,
            detailBlendFeather: detailBlendFeather
        )
    }

    private static func resampleMask(
        _ data: Data,
        from region: MosaicRegion,
        x: Int,
        y: Int,
        width: Int,
        height: Int
    ) -> Data? {
        guard let maskWidth = region.maskWidth,
              let maskHeight = region.maskHeight,
              maskWidth > 1,
              maskHeight > 1,
              data.count == maskWidth * maskHeight,
              width > 0,
              height > 0
        else { return nil }
        var output = Data(count: maskWidth * maskHeight)
        output.withUnsafeMutableBytes { outputBytes in
            data.withUnsafeBytes { inputBytes in
                let destination = outputBytes.bindMemory(to: UInt8.self)
                let source = inputBytes.bindMemory(to: UInt8.self)
                for maskY in 0..<maskHeight {
                    let pixelY = Float(y)
                        + Float(maskY) * Float(max(height - 1, 0)) / Float(maskHeight - 1)
                    let sourceY = (pixelY - Float(region.y)) * Float(maskHeight - 1)
                        / Float(max(region.height - 1, 1))
                    let y0 = min(maskHeight - 1, max(0, Int(floor(sourceY))))
                    let y1 = min(maskHeight - 1, y0 + 1)
                    let fy = min(1, max(0, sourceY - Float(y0)))
                    for maskX in 0..<maskWidth {
                        let pixelX = Float(x)
                            + Float(maskX) * Float(max(width - 1, 0)) / Float(maskWidth - 1)
                        let sourceX = (pixelX - Float(region.x)) * Float(maskWidth - 1)
                            / Float(max(region.width - 1, 1))
                        let x0 = min(maskWidth - 1, max(0, Int(floor(sourceX))))
                        let x1 = min(maskWidth - 1, x0 + 1)
                        let fx = min(1, max(0, sourceX - Float(x0)))
                        let top = Float(source[y0 * maskWidth + x0]) * (1 - fx)
                            + Float(source[y0 * maskWidth + x1]) * fx
                        let bottom = Float(source[y1 * maskWidth + x0]) * (1 - fx)
                            + Float(source[y1 * maskWidth + x1]) * fx
                        destination[maskY * maskWidth + maskX] = UInt8(clamping: Int(
                            (top * (1 - fy) + bottom * fy).rounded()
                        ))
                    }
                }
            }
        }
        return output
    }
}
