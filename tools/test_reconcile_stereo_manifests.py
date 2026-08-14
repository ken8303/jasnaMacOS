#!/usr/bin/env python3

import unittest

from reconcile_stereo_manifests import estimate_disparity, reconcile_manifests


def region(start, end, x, y, width=256, height=256):
    return {
        "startFrame": start,
        "endFrame": end,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "blendX": x + 40,
        "blendY": y + 50,
        "blendWidth": 120,
        "blendHeight": 100,
        "confidence": 0.5,
    }


def manifest(regions):
    return {
        "version": 1,
        "width": 4096,
        "height": 4096,
        "framesPerSecond": 30.0,
        "frameCount": 90,
        "regions": regions,
    }


class StereoManifestReconciliationTests(unittest.TestCase):
    def test_estimates_right_eye_disparity_from_nearby_tracks(self):
        left = [region(30, 60, 1000, 2000), region(60, 90, 1200, 2200)]
        right = [region(30, 60, 920, 1995), region(60, 90, 1120, 2195)]

        self.assertEqual(estimate_disparity(left, right, 4096, 4096), (-80, -5))

    def test_fills_only_a_completely_empty_counterpart_window(self):
        left_regions = [region(0, 30, 1000, 2000), region(30, 60, 1100, 2100)]
        right_regions = [region(30, 60, 1020, 2090)]

        left, right, dx, dy, inferred_left, inferred_right = reconcile_manifests(
            manifest(left_regions), manifest(right_regions)
        )

        self.assertEqual((dx, dy), (-80, -10))
        self.assertEqual((inferred_left, inferred_right), (0, 1))
        self.assertEqual(len(left["regions"]), 2)
        inferred = right["regions"][-1]
        self.assertEqual((inferred["x"], inferred["y"]), (920, 1990))
        self.assertTrue(inferred["stereoInferred"])

    def test_does_not_duplicate_windows_where_both_eyes_are_active(self):
        left, right, _, _, inferred_left, inferred_right = reconcile_manifests(
            manifest([region(0, 30, 1000, 2000)]),
            manifest([region(0, 30, 920, 1995)]),
        )

        self.assertEqual((inferred_left, inferred_right), (0, 0))
        self.assertEqual((len(left["regions"]), len(right["regions"])), (1, 1))


if __name__ == "__main__":
    unittest.main()
