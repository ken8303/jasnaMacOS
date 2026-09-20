#!/usr/bin/env python3

import argparse
import random
from pathlib import Path
import tempfile
import unittest

from prepare_finetune_dataset import (
    allowed_start_intervals,
    choose_sample,
    discover_videos,
    extraction_command,
    parse_excluded_ranges,
    parse_timestamp,
    source_split_order,
    write_json_atomic,
)


class DatasetDiscoveryTests(unittest.TestCase):
    def test_discovers_supported_videos_recursively(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            first = root / "a.mov"
            second = root / "nested" / "b.MP4"
            ignored = root / "notes.txt"
            for path in (first, second, ignored):
                path.touch()

            self.assertEqual(discover_videos([root]), [first.resolve(), second.resolve()])


class SamplePlanningTests(unittest.TestCase):
    def test_each_source_split_has_train_and_validation(self):
        splits = source_split_order(10, 0.1, random.Random(8303))

        self.assertEqual(splits.count("validation"), 1)
        self.assertEqual(splits.count("train"), 9)

    def test_source_split_requires_isolated_clips(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            source_split_order(1, 0.1, random.Random(8303))

    def test_parses_manual_mosaic_range_syntax(self):
        self.assertEqual(parse_timestamp("00:25:57"), 1557.0)
        self.assertEqual(
            parse_excluded_ranges("00:25:57-00:26:27,30:00-30:05"),
            [(1557.0, 1587.0), (1800.0, 1805.0)],
        )

    def test_allowed_starts_cannot_overlap_padded_mosaic_range(self):
        intervals = allowed_start_intervals(
            duration=100.0,
            clip_seconds=1.0,
            excluded=[(25.0, 30.0)],
            padding=2.0,
        )

        self.assertEqual(intervals, [(0.0, 22.0), (32.0, 99.0)])

    def test_merges_overlapping_excluded_ranges(self):
        intervals = allowed_start_intervals(
            duration=20.0,
            clip_seconds=2.0,
            excluded=[(5.0, 8.0), (7.0, 10.0)],
            padding=0.0,
        )

        self.assertEqual(intervals, [(0.0, 3.0), (10.0, 18.0)])

    def test_sbs_crop_never_crosses_the_eye_boundary(self):
        metadata = {"width": 8192, "height": 4096, "duration": 30.0}
        rng = random.Random(8303)

        for _ in range(100):
            sample = choose_sample(metadata, 30, 30.0, 256, True, rng)
            if sample["eye"] == "left":
                self.assertGreaterEqual(sample["x"], 0)
                self.assertLessEqual(sample["x"] + 256, 4096)
            else:
                self.assertGreaterEqual(sample["x"], 4096)
                self.assertLessEqual(sample["x"] + 256, 8192)

    def test_extraction_uses_exact_frame_limit_and_crop(self):
        command = extraction_command(
            Path("input.mov"),
            Path("dataset/train/clip-000000"),
            {"start": 12.5, "x": 100, "y": 200},
            30,
            30.0,
            256,
        )

        self.assertIn("fps=30,crop=256:256:100:200", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "30")

    def test_large_source_crop_is_scaled_to_model_size(self):
        command = extraction_command(
            Path("input.mov"),
            Path("dataset/train/clip-000000"),
            {"start": 12.5, "x": 100, "y": 200, "sourceSize": 1024},
            30,
            30.0,
            256,
        )

        self.assertIn(
            "fps=30,crop=1024:1024:100:200,scale=256:256:flags=lanczos",
            command,
        )


class AtomicMetadataTests(unittest.TestCase):
    def test_writes_without_leaving_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "dataset.json"
            write_json_atomic(path, {"version": 1})

            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "version": 1\n}\n')
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
