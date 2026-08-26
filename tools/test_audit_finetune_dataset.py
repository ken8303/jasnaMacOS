#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

from audit_finetune_dataset import (
    discover_samples,
    frame_quality,
    quarantine_clips,
    rebalance_dataset_splits,
    sample_frame_paths,
    write_json_atomic,
)


class AuditSamplingTests(unittest.TestCase):
    def test_samples_stride_and_always_includes_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            frames = [clip / f"frame-{index:04d}.png" for index in range(1, 11)]
            for frame in frames:
                frame.touch()

            self.assertEqual(sample_frame_paths(clip, 4), [frames[0], frames[4], frames[8], frames[9]])

    def test_discovers_train_and_validation_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            for split in ("train", "validation"):
                clip = dataset / split / "clip-000000"
                clip.mkdir(parents=True)
                (clip / "frame-0001.png").touch()

            samples = discover_samples(dataset, 5)

            self.assertEqual([name for name, _ in samples], ["train/clip-000000", "validation/clip-000000"])


class AuditReportTests(unittest.TestCase):
    @staticmethod
    def write_manifest(dataset: Path, clips: list[dict]) -> None:
        write_json_atomic(dataset / "dataset.json", {"version": 1, "clips": clips})

    def test_writes_report_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit" / "report.json"
            write_json_atomic(path, {"accepted": False})
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_quarantine_moves_clips_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            clip = dataset / "train" / "clip-000001"
            clip.mkdir(parents=True)
            (clip / "frame-0001.png").touch()

            result = quarantine_clips(dataset, ["train/clip-000001"])

            destination = dataset / "rejected" / "train" / "clip-000001"
            self.assertFalse(clip.exists())
            self.assertTrue((destination / "frame-0001.png").exists())
            self.assertEqual(
                result,
                [
                    {
                        "clip": "train/clip-000001",
                        "destination": "rejected/train/clip-000001",
                    }
                ],
            )

    def test_quarantine_preflights_all_destinations_before_moving(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            first = dataset / "train" / "clip-000001"
            second = dataset / "train" / "clip-000002"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (dataset / "rejected" / "train" / "clip-000002").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "destination already exists"):
                quarantine_clips(
                    dataset,
                    ["train/clip-000001", "train/clip-000002"],
                )

            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_quarantine_rejects_paths_outside_active_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "invalid active clip"):
                quarantine_clips(Path(directory), ["../clip-000001"])

    def test_rebalances_missing_validation_source_and_updates_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            clips = []
            for index in range(2):
                name = f"clip-{index:06d}"
                (dataset / "train" / name).mkdir(parents=True)
                clips.append(
                    {
                        "name": name,
                        "split": "train",
                        "source": "source-a.mp4",
                    }
                )
            self.write_manifest(dataset, clips)

            changes = rebalance_dataset_splits(dataset)

            self.assertEqual(
                changes,
                [{"clip": "clip-000000", "from": "train", "to": "validation"}],
            )
            updated = __import__("json").loads(
                (dataset / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated["clips"][0]["split"], "validation")
            self.assertEqual(updated["clips"][0]["originalSplit"], "train")
            self.assertTrue((dataset / "validation" / "clip-000000").is_dir())

    def test_rebalance_refuses_source_with_only_one_survivor(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "train" / "clip-000000").mkdir(parents=True)
            self.write_manifest(
                dataset,
                [
                    {
                        "name": "clip-000000",
                        "split": "train",
                        "source": "source-a.mp4",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "too few surviving clips"):
                rebalance_dataset_splits(dataset)


class FrameQualityTests(unittest.TestCase):
    class FakeCV2:
        COLOR_BGR2YCrCb = 1
        CV_64F = 2

        @staticmethod
        def cvtColor(frame, conversion):
            return frame

        @staticmethod
        def Laplacian(frame, depth):
            return frame.astype("float64") * 0

    def test_black_frame_has_no_skin_and_zero_luma(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is available in the RF-DETR environment")

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        skin, luma, detail = frame_quality(self.FakeCV2, frame)

        self.assertEqual(skin, 0.0)
        self.assertEqual(luma, 0.0)
        self.assertEqual(detail, 0.0)

    def test_skin_colored_ycrcb_frame_is_retained(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is available in the RF-DETR environment")

        frame = np.empty((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = 150
        frame[:, :, 1] = 155
        frame[:, :, 2] = 105
        skin, luma, detail = frame_quality(self.FakeCV2, frame)

        self.assertEqual(skin, 1.0)
        self.assertGreater(luma, 0.5)
        self.assertEqual(detail, 0.0)


if __name__ == "__main__":
    unittest.main()
