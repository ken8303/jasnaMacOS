#!/usr/bin/env python3
"""Focused tests for sparse mosaic region post-processing."""

import base64
import json
from pathlib import Path
import tempfile
import unittest

from rfdetr_mps_detector import valid_prefix_counts
from scan_mosaic_regions import (
    active_frame_intervals,
    box_polygon_groups,
    choose_device,
    coarse_sample_indices,
    detector_coverage_metrics,
    dense_mask_keyframe_box_groups,
    has_box_polygons,
    mask_expansion_radius,
    mask_keyframe_box_groups,
    mask_source_boxes,
    refinement_sample_indices,
    reusable_gate_boxes,
    samples_in_intervals,
    samples_without,
    stereo_sample_eyes,
    suppress_duplicate_regions,
    suppress_nested_regions,
    write_manifest,
)


class ManifestWritingTests(unittest.TestCase):
    def test_writes_atomic_eye_manifest_with_expected_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "left.json"
            regions = [{"startFrame": 0, "endFrame": 30}]

            write_manifest(path, 4096, 4096, 30, regions)

            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["width"], 4096)
            self.assertEqual(manifest["height"], 4096)
            self.assertEqual(manifest["frameCount"], 30)
            self.assertEqual(manifest["regions"], regions)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


class DeviceSelectionTests(unittest.TestCase):
    class FakeMPS:
        def __init__(self, available):
            self.available = available

        def is_available(self):
            return self.available

    class FakeBackends:
        def __init__(self, available):
            self.mps = DeviceSelectionTests.FakeMPS(available)

    class FakeTorch:
        def __init__(self, available):
            self.backends = DeviceSelectionTests.FakeBackends(available)

    def test_auto_prefers_mps_when_available(self):
        self.assertEqual(choose_device(self.FakeTorch(True), "auto"), "mps")

    def test_auto_uses_cpu_without_mps(self):
        self.assertEqual(choose_device(self.FakeTorch(False), "auto"), "cpu")

    def test_explicit_cpu_overrides_available_mps(self):
        self.assertEqual(choose_device(self.FakeTorch(True), "cpu"), "cpu")


class RFDetrMaskPackingTests(unittest.TestCase):
    def test_counts_only_score_sorted_valid_prefix(self):
        self.assertEqual(
            valid_prefix_counts(
                [[True, True, False, False], [True, False, False, False]]
            ),
            [2, 1],
        )

    def test_accepts_empty_valid_selection(self):
        self.assertEqual(valid_prefix_counts([[False, False]]), [0])

    def test_rejects_non_prefix_valid_selection(self):
        with self.assertRaises(ValueError):
            valid_prefix_counts([[True, False, True]])


class AdaptiveScanScheduleTests(unittest.TestCase):
    def test_coarse_scan_uses_center_frame_once_per_second(self):
        self.assertEqual(coarse_sample_indices(95, 30.0, 1.0), [15, 45, 75])

    def test_refines_complete_detected_second_with_padding(self):
        indices = refinement_sample_indices(150, 30.0, 0.1, [45], 1.0)

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 87)
        self.assertEqual(len(indices), 30)

    def test_merges_overlapping_refinement_intervals(self):
        indices = refinement_sample_indices(180, 30.0, 0.1, [45, 75], 1.0)

        self.assertEqual(indices, list(range(0, 120, 3)))

    def test_empty_gate_performs_no_dense_refinement(self):
        self.assertEqual(refinement_sample_indices(900, 30.0, 0.1, [], 1.0), [])

    def test_manual_ranges_filter_dense_samples(self):
        intervals = active_frame_intervals("1.0/2.0", 30.0, 90)
        self.assertEqual(intervals, [(30, 60)])
        self.assertEqual(samples_in_intervals(range(0, 90, 3), intervals), list(range(30, 60, 3)))

    def test_short_manual_range_gets_an_anchored_detector_sample(self):
        intervals = [(1, 2)]

        self.assertEqual(
            samples_in_intervals(
                range(0, 90, 3), intervals, ensure_each_interval=True
            ),
            [1],
        )

    def test_empty_adaptive_refinement_schedule_stays_empty(self):
        self.assertEqual(samples_in_intervals([], [(30, 60)]), [])

    def test_removes_only_samples_already_inferred_by_the_gate(self):
        self.assertEqual(
            samples_without([0, 3, 6, 9, 12, 15], [15, 45]),
            [0, 3, 6, 9, 12],
        )

    def test_removes_gate_sample_even_when_it_had_no_final_detection(self):
        self.assertEqual(samples_without([12, 15, 18], [15]), [12, 18])

    def test_reuses_only_gate_boxes_that_pass_the_final_threshold(self):
        low = (0, 0, 10, 10, 0.05, 15, [])
        final = (0, 0, 10, 10, 0.15, 15, [])
        strong = (0, 0, 10, 10, 0.90, 45, [])

        self.assertEqual(
            reusable_gate_boxes(
                {"left": [low, final], "right": [strong]}, 0.15
            ),
            {"left": [final], "right": [strong]},
        )


class StereoSampleScheduleTests(unittest.TestCase):
    def test_paired_mode_scans_both_eyes_at_every_sample(self):
        self.assertEqual(stereo_sample_eyes(0, "paired"), ("left", "right"))
        self.assertEqual(stereo_sample_eyes(7, "paired"), ("left", "right"))

    def test_alternating_mode_interleaves_real_eye_observations(self):
        self.assertEqual(
            [stereo_sample_eyes(index, "alternating") for index in range(4)],
            [("left",), ("right",), ("left",), ("right",)],
        )

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            stereo_sample_eyes(0, "unknown")


class DetectorCoverageMetricsTests(unittest.TestCase):
    def test_reports_average_active_regions_and_scheduled_blend_area(self):
        regions = [
            {
                "startFrame": 0,
                "endFrame": 30,
                "blendWidth": 100,
                "blendHeight": 50,
            },
            {
                "startFrame": 15,
                "endFrame": 30,
                "blendWidth": 200,
                "blendHeight": 100,
            },
        ]

        active, area = detector_coverage_metrics(regions, 1_000, 500, 30)

        self.assertEqual(active, 1.5)
        self.assertEqual(area, 3.0)

    def test_clamps_region_ranges_and_accepts_legacy_crop_dimensions(self):
        regions = [
            {
                "startFrame": -10,
                "endFrame": 40,
                "width": 100,
                "height": 100,
            }
        ]

        active, area = detector_coverage_metrics(regions, 1_000, 1_000, 30)

        self.assertEqual(active, 1.0)
        self.assertEqual(area, 1.0)

    def test_returns_zero_for_invalid_video_dimensions(self):
        self.assertEqual(detector_coverage_metrics([], 0, 1_000, 30), (0.0, 0.0))


class PolygonGroupTests(unittest.TestCase):
    def test_normalizes_legacy_single_polygon(self):
        polygon = [[1.0, 2.0], [3.0, 2.0], [2.0, 4.0]]
        box = (0, 0, 4, 4, 0.9, 0, polygon)

        self.assertEqual(box_polygon_groups(box), [polygon])
        self.assertTrue(has_box_polygons(box))

    def test_preserves_disconnected_rfdetr_mask_islands(self):
        first = [[1.0, 2.0], [3.0, 2.0], [2.0, 4.0]]
        second = [[10.0, 20.0], [13.0, 20.0], [12.0, 24.0]]
        box = (0, 0, 16, 32, 0.9, 0, [first, second])

        self.assertEqual(box_polygon_groups(box), [first, second])
        self.assertTrue(has_box_polygons(box))


class MaskExpansionTests(unittest.TestCase):
    def test_scales_with_mask_resolution(self):
        self.assertEqual(mask_expansion_radius(64, 0.10), 7)
        self.assertEqual(mask_expansion_radius(128, 0.10), 13)

    def test_never_collapses_to_zero(self):
        self.assertEqual(mask_expansion_radius(8, 0.01), 1)


class MaskKeyframeTests(unittest.TestCase):
    def test_clamps_nearby_samples_and_groups_duplicate_boundary_frames(self):
        polygon = [[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]
        cluster = [
            (10, 20, 30, 40, 0.8, 7, polygon),
            (11, 21, 31, 41, 0.8, 10, polygon),
            (12, 22, 32, 42, 0.8, 13, polygon),
            (13, 23, 33, 43, 0.8, 40, polygon),
        ]

        groups = mask_keyframe_box_groups(cluster, 10, 40, 3)

        self.assertEqual([frame for frame, _ in groups], [10, 13, 39])
        self.assertEqual(len(groups[0][1]), 2)

    def test_uses_nearest_polygon_for_temporal_padding_only_segment(self):
        polygon = [[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]
        cluster = [(10, 20, 30, 40, 0.8, 5, polygon)]

        groups = mask_keyframe_box_groups(cluster, 20, 23, 3)

        self.assertEqual(groups, [(20, cluster)])

    def test_dense_masks_translate_polygon_between_detector_samples(self):
        left_polygon = [[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]
        right_polygon = [[40.0, 20.0], [60.0, 20.0], [50.0, 40.0]]
        cluster = [
            (10, 20, 30, 40, 0.8, 0, left_polygon),
            (40, 20, 60, 40, 0.8, 3, right_polygon),
        ]

        groups = dense_mask_keyframe_box_groups(cluster, 0, 4, 3)

        self.assertEqual([frame for frame, _ in groups], [0, 1, 2, 3])
        middle = box_polygon_groups(groups[1][1][0])[0]
        self.assertAlmostEqual(middle[0][0], 20.0)
        self.assertAlmostEqual(middle[1][0], 40.0)

    def test_dense_masks_hold_nearest_polygon_through_padding(self):
        polygon = [[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]
        cluster = [(10, 20, 30, 40, 0.8, 5, polygon)]

        groups = dense_mask_keyframe_box_groups(cluster, 8, 11, 3)

        self.assertEqual([frame for frame, _ in groups], [8, 9, 10])
        self.assertTrue(all(has_box_polygons(boxes[0]) for _, boxes in groups))


class MaskSourceBoxesTests(unittest.TestCase):
    def test_uses_nearest_polygon_for_padded_segment(self):
        polygon = [[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]
        cluster = [
            (10, 20, 30, 40, 0.8, 10, polygon),
            (12, 22, 32, 42, 0.9, 20, polygon),
        ]
        interpolated = [(11, 21, 31, 41, 0.85, 14)]

        self.assertEqual(
            mask_source_boxes(cluster, interpolated, 13, 16), [cluster[0]]
        )

    def test_keeps_nearby_polygon_samples(self):
        polygon = [[10.0, 20.0], [30.0, 20.0], [20.0, 40.0]]
        nearby = [(10, 20, 30, 40, 0.8, 10, polygon)]

        self.assertIs(mask_source_boxes(nearby, nearby, 10, 11), nearby)


def region(x, y, width, height, start=0, end=30, confidence=0.9):
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "blendX": x,
        "blendY": y,
        "blendWidth": width,
        "blendHeight": height,
        "startFrame": start,
        "endFrame": end,
        "confidence": confidence,
    }


def with_mask(item, values, width=2, height=2):
    result = dict(item)
    result.update(
        {
            "maskWidth": width,
            "maskHeight": height,
            "maskData": base64.b64encode(bytes(values)).decode("ascii"),
        }
    )
    return result


class SuppressNestedRegionsTests(unittest.TestCase):
    def test_removes_inner_crop_that_would_overwrite_large_crop(self):
        large = region(100, 100, 1_300, 1_300)
        inner = region(500, 500, 256, 256)

        self.assertEqual(suppress_nested_regions([large, inner]), [large])

    def test_removes_inner_crop_when_large_mask_covers_its_mask(self):
        large = with_mask(region(0, 0, 1_000, 1_000), [255] * 16, 4, 4)
        inner = with_mask(region(400, 400, 200, 200), [255] * 4)

        self.assertEqual(suppress_nested_regions([large, inner]), [large])

    def test_preserves_inner_subject_outside_large_segmentation_mask(self):
        large = with_mask(
            region(0, 0, 1_000, 1_000),
            [255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            4,
            4,
        )
        inner = with_mask(region(700, 700, 200, 200), [255] * 4)

        self.assertEqual(suppress_nested_regions([large, inner]), [large, inner])

    def test_preserves_unmasked_inner_subject_inside_masked_large_crop(self):
        large = with_mask(region(0, 0, 1_000, 1_000), [255, 0, 0, 0])
        inner = region(400, 400, 200, 200)

        self.assertEqual(suppress_nested_regions([large, inner]), [large, inner])

    def test_removes_strongly_overlapping_track_for_same_active_time(self):
        winner = region(100, 100, 1_300, 1_300, confidence=0.95)
        duplicate = region(220, 180, 1_300, 1_300, confidence=0.60)

        self.assertEqual(suppress_duplicate_regions([duplicate, winner]), [winner])

    def test_preserves_overlapping_track_with_distinct_mask_support(self):
        first = with_mask(
            region(100, 100, 1_300, 1_300, confidence=0.95),
            [255, 0, 255, 0],
        )
        second = with_mask(
            region(100, 100, 1_300, 1_300, confidence=0.60),
            [0, 255, 0, 255],
        )

        self.assertEqual(suppress_duplicate_regions([second, first]), [second, first])

    def test_preserves_overlapping_track_outside_winner_time_range(self):
        first = region(100, 100, 1_300, 1_300, start=0, end=15, confidence=0.95)
        continuation = region(
            220, 180, 1_300, 1_300, start=10, end=30, confidence=0.60
        )

        self.assertEqual(
            suppress_duplicate_regions([first, continuation]), [first, continuation]
        )

    def test_stable_track_wins_over_higher_confidence_one_frame_blip(self):
        blip = region(
            2_721, 2_067, 171, 156, start=60, end=61, confidence=0.72
        )
        stable = region(
            2_757, 2_102, 155, 159, start=60, end=90, confidence=0.69
        )

        self.assertEqual(suppress_duplicate_regions([blip, stable]), [stable])

    def test_preserves_nearby_partially_overlapping_subject(self):
        large = region(100, 100, 900, 900)
        nearby = region(850, 400, 512, 512)

        self.assertEqual(suppress_nested_regions([large, nearby]), [large, nearby])
        self.assertEqual(suppress_duplicate_regions([large, nearby]), [large, nearby])

    def test_preserves_inner_crop_active_outside_large_time_range(self):
        large = region(100, 100, 1_300, 1_300, start=10, end=20)
        inner = region(500, 500, 256, 256, start=0, end=30)

        self.assertEqual(suppress_nested_regions([large, inner]), [large, inner])

    def test_uses_blend_rectangle_for_visible_overlap(self):
        large = region(100, 100, 1_300, 1_300)
        inner = region(1_500, 1_500, 512, 512)
        inner.update({"blendX": 500, "blendY": 500, "blendWidth": 256, "blendHeight": 256})

        self.assertEqual(suppress_nested_regions([large, inner]), [large])

    def test_collapses_observed_v5_end_frame_stack(self):
        regions = [
            region(2_030, 2_166, 1_512, 1_512, confidence=0.943),
            region(1_947, 1_846, 1_659, 1_659, confidence=0.946),
            region(2_087, 1_859, 1_542, 1_542, confidence=0.478),
            region(2_288, 1_857, 1_528, 1_527, confidence=0.585),
        ]
        blends = [
            (2_134, 2_247, 1_303, 1_350),
            (2_118, 1_934, 1_317, 1_483),
            (2_169, 2_011, 1_378, 1_237),
            (2_369, 1_948, 1_366, 1_344),
        ]
        for item, (x, y, width, height) in zip(regions, blends):
            item.update(
                {"blendX": x, "blendY": y, "blendWidth": width, "blendHeight": height}
            )

        self.assertEqual(suppress_duplicate_regions(regions), [regions[1]])


if __name__ == "__main__":
    unittest.main()
