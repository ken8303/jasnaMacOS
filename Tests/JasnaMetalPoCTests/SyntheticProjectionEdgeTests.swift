import Foundation
import Testing
@testable import JasnaMetalPoC

// Test-target-only spherical ray oracle. It shares the patch's bounds, but not
// production's sourceCoordinate/modelCoordinate implementations. Integer raster
// coordinates are pixel centres; the first/last pixel edges are -0.5/size-0.5.
func pixelCentreReferenceSource(
    modelX: Double, modelY: Double, transform: FisheyeMosaicCropTransform
) -> (x: Double, y: Double) {
    let u = transform.fisheyeMinU + (transform.fisheyeMaxU - transform.fisheyeMinU)
        * (modelX + 0.5) / Double(transform.modelSize)
    let v = transform.fisheyeMinV + (transform.fisheyeMaxV - transform.fisheyeMinV)
        * (modelY + 0.5) / Double(transform.modelSize)
    let azimuth = atan2(1 - 2 * v, 2 * u - 1)
    let angle = hypot(2 * u - 1, 1 - 2 * v) * .pi / 2
    let longitude = atan2(sin(angle) * cos(azimuth), cos(angle))
    let latitude = asin(max(-1, min(1, sin(angle) * sin(azimuth))))
    return (
        (longitude / .pi + 0.5) * Double(transform.eyeWidth) - 0.5,
        (0.5 - latitude / .pi) * Double(transform.eyeHeight) - 0.5
    )
}

func pixelCentreReferenceModel(
    x: Double, y: Double, transform: FisheyeMosaicCropTransform
) -> (x: Double, y: Double) {
    let longitude = ((x + 0.5) / Double(transform.eyeWidth) - 0.5) * .pi
    let latitude = (0.5 - (y + 0.5) / Double(transform.eyeHeight)) * .pi
    let rayX = cos(latitude) * sin(longitude)
    let rayY = sin(latitude)
    let rayZ = cos(latitude) * cos(longitude)
    let angle = acos(max(-1, min(1, rayZ)))
    let azimuth = atan2(rayY, rayX)
    let u = 0.5 + angle / .pi * cos(azimuth)
    let v = 0.5 - angle / .pi * sin(azimuth)
    return (
        (u - transform.fisheyeMinU) / (transform.fisheyeMaxU - transform.fisheyeMinU)
            * Double(transform.modelSize) - 0.5,
        (v - transform.fisheyeMinV) / (transform.fisheyeMaxV - transform.fisheyeMinV)
            * Double(transform.modelSize) - 0.5
    )
}

@Test func syntheticReferenceMatchesAnalyticAxes() {
    let transform = FisheyeMosaicCropTransform(
        region: MosaicRegion(
            startFrame: 0, endFrame: 1, x: 0, y: 0, width: 4_096, height: 4_096, confidence: 1
        ), eyeWidth: 4_096, eyeHeight: 4_096
    )
    // Equator and central meridian have closed-form equidistant coordinates.
    // Anchor these explicitly: two mutually wrong functions can round-trip.
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
        for vertical in [false, true] {
            let u = vertical ? 0.5 : fraction
            let v = vertical ? fraction : 0.5
            let x = u * 4_096 - 0.5
            let y = v * 4_096 - 0.5
            let expectedX = (u - transform.fisheyeMinU)
                / (transform.fisheyeMaxU - transform.fisheyeMinU) * 256 - 0.5
            let expectedY = (v - transform.fisheyeMinV)
                / (transform.fisheyeMaxV - transform.fisheyeMinV) * 256 - 0.5
            let model = pixelCentreReferenceModel(x: x, y: y, transform: transform)
            #expect(abs(model.x - expectedX) < 1e-9)
            #expect(abs(model.y - expectedY) < 1e-9)
            let source = pixelCentreReferenceSource(modelX: expectedX, modelY: expectedY, transform: transform)
            #expect(abs(source.y - y) < 0.0001)
            // Longitude is undefined at the exact poles, not at row 0/last row
            // pixel centres. Do not claim an inverse for this singularity.
            if !(vertical && (fraction == 0 || fraction == 1)) {
                #expect(abs(source.x - x) < 0.0001)
            }
        }
    }
    print("Synthetic reference axes: 10 anchors checked")
}

@Test(arguments: [512, 4_096])
func syntheticReferenceRoundTripsBorderPixelCentres(size: Int) {
    let transform = FisheyeMosaicCropTransform(
        region: MosaicRegion(
            startFrame: 0, endFrame: 1, x: 0, y: 0, width: size, height: size, confidence: 1
        ), eyeWidth: size, eyeHeight: size
    )
    let last = Double(size - 1)
    var maximumError: Double = 0
    var checked = 0
    for step in 0...64 {
        let along = last * Double(step) / 64
        for inset in [0.0, 0.25, 1.0] {
            for point in [(along, inset), (along, last - inset), (inset, along), (last - inset, along)] {
                let model = pixelCentreReferenceModel(x: point.0, y: point.1, transform: transform)
                let source = pixelCentreReferenceSource(modelX: model.x, modelY: model.y, transform: transform)
                #expect(source.x.isFinite && source.y.isFinite)
                maximumError = max(maximumError, abs(source.x - point.0), abs(source.y - point.1))
                checked += 1
            }
        }
    }
    #expect(checked == 780)
    #expect(maximumError < 0.0001)
    print("Synthetic reference borders \(size)px: \(checked) points, maximum displacement \(maximumError) source pixels")
}

@Test func syntheticReferenceSubpixelMotionSurvivesCropChanges() {
    var maximumError: Double = 0
    var maximumMotionError: Double = 0
    var previousErrors = [Double]()
    var cropTransitions = 0
    for frame in 0..<60 {
        let stage = (frame / 10) % 3
        let region = MosaicRegion(
            startFrame: frame, endFrame: frame + 1,
            x: [1_024, 992, 1_048][stage], y: [1_760, 1_728, 1_792][stage],
            width: [1_280, 1_361, 1_241][stage], height: [1_024, 1_113, 1_049][stage], confidence: 1
        )
        let transform = FisheyeMosaicCropTransform(region: region, eyeWidth: 4_096, eyeHeight: 4_096)
        var errors = [Double]()
        for offset in [0.0, 100.0, 250.0] {
            let x = 1_440 + offset + Double(frame) * 0.25
            let y = 2_080 + offset + Double(frame) * 0.125
            let model = pixelCentreReferenceModel(x: x, y: y, transform: transform)
            #expect(model.x > 1 && model.x < 254 && model.y > 1 && model.y < 254)
            for axis in 0..<2 {
                // Exercise discrete resampling, not just exact analytic inversion.
                let actual = interpolateSynthetic(x: model.x, y: model.y) {
                    let source = pixelCentreReferenceSource(modelX: Double($0), modelY: Double($1), transform: transform)
                    return axis == 0 ? source.x : source.y
                }
                let error = actual - (axis == 0 ? x : y)
                errors.append(error)
                maximumError = max(maximumError, abs(error))
            }
        }
        if !previousErrors.isEmpty {
            for (error, previous) in zip(errors, previousErrors) {
                maximumMotionError = max(maximumMotionError, abs(error - previous))
            }
            if frame % 10 == 0 { cropTransitions += 1 }
        }
        previousErrors = errors
    }
    #expect(cropTransitions == 5)
    #expect(maximumError < 0.01)
    #expect(maximumMotionError < 0.01)
    print("Synthetic reference motion: 60 frames, \(cropTransitions) crop changes, displacement \(maximumError), frame-difference error \(maximumMotionError) source pixels")
}
