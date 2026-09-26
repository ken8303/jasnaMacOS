#!/usr/bin/env python3

import unittest

from manifest_window_runs import activity_run, combined_window_activity


def manifest(frame_count, regions):
    return {"frameCount": frame_count, "regions": regions}


class ManifestWindowActivityTests(unittest.TestCase):
    def test_combines_both_eyes_and_keeps_empty_windows(self):
        left = manifest(150, [{"startFrame": 35, "endFrame": 60}])
        right = manifest(150, [{"startFrame": 120, "endFrame": 150}])

        self.assertEqual(
            combined_window_activity([left, right]),
            [False, True, False, False, True],
        )

    def test_reports_maximal_run_from_requested_window(self):
        active = [False, False, True, True, False]

        self.assertEqual(activity_run(active, 0), ("empty", 2))
        self.assertEqual(activity_run(active, 2), ("active", 2))
        self.assertEqual(activity_run(active, 4), ("empty", 1))

    def test_rejects_manifest_frame_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "different frame counts"):
            combined_window_activity([manifest(30, []), manifest(60, [])])

    def test_rejects_start_outside_manifest(self):
        with self.assertRaisesRegex(ValueError, "outside the manifest"):
            activity_run([False], 1)


if __name__ == "__main__":
    unittest.main()
