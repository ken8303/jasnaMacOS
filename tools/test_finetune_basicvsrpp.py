#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import random
import tempfile
import unittest

from finetune_basicvsrpp import (
    ema_stage_fraction,
    finite_metric_or_none,
    list_clips,
    masked_spatial_gradient_loss,
    passes_quality_gate,
    quality_baseline,
    select_frame_window,
    synthetic_mosaic,
    validate_dataset_audit,
    write_json_atomic,
)


class ClipSelectionTests(unittest.TestCase):
    def test_selects_contiguous_reproducible_window(self):
        paths = [Path(f"frame-{index:04d}.png") for index in range(30)]

        selected = select_frame_window(paths, 10, random.Random(8303))

        indices = [int(path.stem.split("-")[-1]) for path in selected]
        self.assertEqual(indices, list(range(indices[0], indices[0] + 10)))

    def test_rejects_short_clip(self):
        with self.assertRaisesRegex(ValueError, "fewer than requested"):
            select_frame_window([Path("one.png")], 3, random.Random(1))

    def test_lists_only_clip_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train" / "clip-000001").mkdir(parents=True)
            (root / "train" / "ignored").mkdir()
            self.assertEqual(
                list_clips(root, "train"), [root / "train" / "clip-000001"]
            )


class QualityGateTests(unittest.TestCase):
    def test_accepts_psnr_gain_without_temporal_regression(self):
        baseline = {"maskedPSNR": 30.0, "temporalMAE": 0.020}
        candidate = {"maskedPSNR": 30.2, "temporalMAE": 0.0203}
        self.assertTrue(passes_quality_gate(baseline, candidate, 0.1, 1.02))

    def test_rejects_sharper_but_less_stable_candidate(self):
        baseline = {"maskedPSNR": 30.0, "temporalMAE": 0.020}
        candidate = {"maskedPSNR": 31.0, "temporalMAE": 0.021}
        self.assertFalse(passes_quality_gate(baseline, candidate, 0.1, 1.02))

    def test_resume_preserves_original_quality_baseline(self):
        measured = {"maskedPSNR": 31.0, "temporalMAE": 0.01}
        original = {"maskedPSNR": 30.0, "temporalMAE": 0.02}

        self.assertEqual(quality_baseline(measured, {"baseline": original}), original)

    def test_fresh_run_uses_measured_quality_baseline(self):
        measured = {"maskedPSNR": 30.0, "temporalMAE": 0.02}

        self.assertIs(quality_baseline(measured, None), measured)

    def test_new_recipe_can_reset_resume_quality_baseline(self):
        measured = {"maskedPSNR": 28.0, "temporalMAE": 0.03}
        original = {"maskedPSNR": 30.0, "temporalMAE": 0.02}

        self.assertIs(quality_baseline(measured, {"baseline": original}, True), measured)

    def test_rejects_candidate_without_required_psnr_gain(self):
        baseline = {"maskedPSNR": 30.0, "temporalMAE": 0.020}
        candidate = {"maskedPSNR": 30.05, "temporalMAE": 0.019}
        self.assertFalse(passes_quality_gate(baseline, candidate, 0.1, 1.02))


class DatasetAuditGateTests(unittest.TestCase):
    def test_accepts_matching_passing_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            train = dataset / "train" / "clip-000001"
            validation = dataset / "validation" / "clip-000002"
            train.mkdir(parents=True)
            validation.mkdir(parents=True)
            write_json_atomic(
                dataset / "mosaic-audit.json",
                {
                    "accepted": True,
                    "clipQuality": {
                        "train/clip-000001": {},
                        "validation/clip-000002": {},
                    },
                },
            )

            report = validate_dataset_audit(dataset, [train], [validation])

            self.assertTrue(report["accepted"])

    def test_rejects_dataset_changed_after_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            train = dataset / "train" / "clip-000001"
            validation = dataset / "validation" / "clip-000002"
            train.mkdir(parents=True)
            validation.mkdir(parents=True)
            write_json_atomic(
                dataset / "mosaic-audit.json",
                {"accepted": True, "clipQuality": {"train/clip-000001": {}}},
            )

            with self.assertRaisesRegex(ValueError, "changed after its audit"):
                validate_dataset_audit(dataset, [train], [validation])

class AtomicMetricsTests(unittest.TestCase):
    def test_unset_best_metric_is_strict_json_null(self):
        self.assertIsNone(finite_metric_or_none(float("-inf")))
        self.assertEqual(finite_metric_or_none(23.5), 23.5)

    def test_writes_metrics_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics" / "latest.json"
            write_json_atomic(path, {"accepted": False})
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".json.tmp").exists())


class EMATelemetryTests(unittest.TestCase):
    def test_reports_small_contribution_for_short_slow_ema_stage(self):
        self.assertAlmostEqual(ema_stage_fraction(0.999, 500, 750, 1), 0.2213, places=3)

    def test_accounts_for_gradient_accumulation(self):
        self.assertAlmostEqual(ema_stage_fraction(0.9, 100, 120, 2), 1 - 0.9**10)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class RecoveryLossTests(unittest.TestCase):
    def test_block_edges_cost_more_than_matching_clean_gradients(self):
        import torch

        clean = torch.linspace(0, 1, 16).reshape(1, 1, 1, 4, 4).repeat(1, 3, 3, 1, 1)
        mask = torch.ones(1, 3, 1, 4, 4)
        blocky = clean.clone()
        blocky[..., :2] = blocky[..., :2].mean()
        blocky[..., 2:] = blocky[..., 2:].mean()

        matching = masked_spatial_gradient_loss(torch, clean, clean, mask)
        retained_blocks = masked_spatial_gradient_loss(torch, blocky, clean, mask)

        self.assertGreater(retained_blocks.item(), matching.item() * 5)

    def test_lada_vr_v4_creates_masked_moving_corruption(self):
        import torch
        import torch.nn.functional as functional

        clean = torch.rand(1, 3, 3, 64, 64)
        low_quality, mask = synthetic_mosaic(
            torch, functional, clean, random.Random(8303), "lada-vr-v4"
        )

        self.assertEqual(low_quality.shape, clean.shape)
        self.assertEqual(mask.shape, (1, 3, 1, 64, 64))
        self.assertGreater(mask.sum().item(), 0)
        self.assertGreater(((low_quality - clean).abs() * mask).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
