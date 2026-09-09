"""Exercise A/B orchestration with stub media and no model/video execution."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CropDensityABTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jasna-crop-ab-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scripts = self.root / "script"
        self.scripts.mkdir()
        self.harness = self.scripts / "test_vr_crop_density_ab_30s.sh"
        shutil.copyfile(ROOT / "script" / self.harness.name, self.harness)
        self.harness.chmod(0o755)
        self.handoff = self.scripts / "test_vr_crop_handoff_ab_30s.sh"
        shutil.copyfile(ROOT / "script" / self.handoff.name, self.handoff)
        child = self.scripts / "test_vr_restore_only_30s.sh"
        child.write_text(
            '#!/bin/bash\nset -eu\n'
            '[[ "$JASNA_RESTORE_ONLY_VALIDATE" == 1 ]] || exit 99\n'
            'printf "DISPATCH %s %s %s %s phases=%s trace=%s\\n" '
            '"$JASNA_LARGE_REGION_MAX_BLEND" "$JASNA_RESTORE_ONLY_START_SECOND" '
            '"$JASNA_RESTORE_ONLY_AUDIO_START_SECOND" "$JASNA_RESTORE_ONLY_SECONDS" '
            '"$JASNA_GRAPH_PHASE_TELEMETRY" "$JASNA_GRAPH_TRACE"\n'
            'printf "HANDOFF %s %s batch=%s windows=%s retries=%s\\n" '
            '"$JASNA_IN_MEMORY_CROP_CACHE" "$JASNA_IN_MEMORY_CACHE_LIMIT_MB" '
            '"$JASNA_MODEL_BATCH" "$JASNA_METAL_WINDOWS_PER_PROCESS" "$JASNA_GPU_TIMEOUT_RETRIES"\n',
            encoding="utf-8",
        )
        child.chmod(0o755)
        self.reference = self.root / "reference work"
        self.reference.mkdir()
        (self.reference / "run-config.txt").write_text(
            "model_batch=2\nmetal_windows_per_process=4\n", encoding="utf-8"
        )
        for segment in (9, 10):
            self.asset(f"shared-sbs-source/shared-{segment:05d}.mp4")
            for eye in ("left", "right"):
                self.asset(f"stereo-reconciled-manifests/{eye}-{segment:05d}.json")
        # No real FFmpeg or GPU command should be invoked in these tests.
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("ffmpeg", "ffprobe"):
            stub = self.bin / name
            stub.write_text("#!/bin/bash\nexit 99\n", encoding="utf-8")
            stub.chmod(0o755)
        self.prefix = self.root / "output with spaces"
        self.work = Path(f"{self.prefix}.jasna-crop-density-ab-work")

    def asset(self, relative):
        target = self.reference / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")

    def run_harness(self, handoff=False, **overrides):
        env = {key: value for key, value in os.environ.items() if not key.startswith("JASNA_")}
        env.update(
            PATH=f"{self.bin}:/usr/bin:/bin",
            JASNA_RESTORE_ONLY_VALIDATE="1",
        )
        env.update(overrides)
        return subprocess.run(
            ["/bin/bash", str(self.handoff if handoff else self.harness), str(self.reference), str(self.prefix), "00:19:50"],
            env=env, capture_output=True, text=True, timeout=10,
        )

    @staticmethod
    def dispatches(result):
        return [line for line in result.stdout.splitlines() if line.startswith("DISPATCH ")]

    def test_reverse_order_keeps_absolute_timeline_across_segment_boundary(self):
        result = self.run_harness(JASNA_AB_ORDER="candidate-first")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.dispatches(result), [
            "DISPATCH 1024 110 1190 10 phases=1 trace=1",
            "DISPATCH 1024 0 1200 20 phases=1 trace=1",
            "DISPATCH 768 110 1190 10 phases=1 trace=1",
            "DISPATCH 768 0 1200 20 phases=1 trace=1",
        ])
        self.assertFalse(self.work.exists(), "validation must not create output work")

    def test_baseline_first_remains_default(self):
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.dispatches(result)[0].startswith("DISPATCH 768 "))
        self.assertTrue(self.dispatches(result)[2].startswith("DISPATCH 1024 "))

    def test_diagnostics_can_be_disabled_equally_for_both_variants(self):
        result = self.run_harness(JASNA_GRAPH_PHASE_TELEMETRY="0", JASNA_GRAPH_TRACE="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.dispatches(result)), 4)
        self.assertTrue(all(line.endswith("phases=0 trace=0") for line in self.dispatches(result)))

    def test_invalid_order_is_rejected_before_dispatch(self):
        result = self.run_harness(JASNA_AB_ORDER="reverse")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JASNA_AB_ORDER", result.stderr)
        self.assertEqual(self.dispatches(result), [])

    def test_handoff_comparison_changes_only_storage_for_identical_grid_and_range(self):
        result = self.run_harness(handoff=True, JASNA_IN_MEMORY_CACHE_LIMIT_MB="99999")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.dispatches(result), [
            "DISPATCH 1024 110 1190 10 phases=1 trace=1",
            "DISPATCH 1024 0 1200 20 phases=1 trace=1",
        ] * 2)
        handoffs = [line for line in result.stdout.splitlines() if line.startswith("HANDOFF ")]
        self.assertEqual(handoffs, [
            "HANDOFF 0 512 batch=2 windows=4 retries=0",
            "HANDOFF 0 512 batch=2 windows=4 retries=0",
            "HANDOFF 1 512 batch=2 windows=4 retries=0",
            "HANDOFF 1 512 batch=2 windows=4 retries=0",
        ])
        self.assertFalse(Path(f"{self.prefix}.jasna-crop-handoff-ab-work").exists())

    def test_handoff_comparison_can_run_memory_first(self):
        result = self.run_harness(handoff=True, JASNA_AB_ORDER="candidate-first")
        self.assertEqual(result.returncode, 0, result.stderr)
        handoffs = [line for line in result.stdout.splitlines() if line.startswith("HANDOFF ")]
        self.assertTrue(handoffs[0].startswith("HANDOFF 1 512 "))
        self.assertTrue(handoffs[2].startswith("HANDOFF 0 512 "))

    def test_comparison_disables_timeout_retries_even_when_inherited(self):
        result = self.run_harness(JASNA_GPU_TIMEOUT_RETRIES="9")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(
            line.endswith("retries=0") for line in result.stdout.splitlines()
            if line.startswith("HANDOFF ")
        ))

    def test_invalid_diagnostic_flag_is_rejected(self):
        result = self.run_harness(JASNA_GRAPH_TRACE="yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trace flags", result.stderr)

    def test_old_work_is_never_reused_as_a_fresh_measurement(self):
        self.work.mkdir()
        marker = self.work / "previous-measurement.txt"
        marker.write_text("preserve me", encoding="utf-8")
        result = self.run_harness(JASNA_RESTORE_ONLY_VALIDATE="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh output prefix", result.stderr)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
        self.assertEqual(self.dispatches(result), [])

    def test_detector_manifest_fallback_returns_success_when_found(self):
        shutil.rmtree(self.reference / "stereo-reconciled-manifests")
        for segment in (9, 10):
            for eye in ("left", "right"):
                self.asset(f"{eye}-restored.{eye}-segments-work/detector/{eye}-{segment:05d}-mosaic-regions.json")
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.dispatches(result)), 4)

    def test_physical_mp4_inputs_are_supported(self):
        shutil.rmtree(self.reference / "shared-sbs-source")
        for segment in (9, 10):
            for eye in ("left", "right"):
                self.asset(f"{eye}-restored.{eye}-segments-work/source/{eye}-{segment:05d}.mp4")
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.dispatches(result)), 4)

    def test_missing_segment_is_not_replaced_by_another(self):
        (self.reference / "shared-sbs-source/shared-00010.mp4").unlink()
        result = self.run_harness()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("segment 10 is unavailable", result.stderr)
        self.assertEqual(len(self.dispatches(result)), 1)

    def test_missing_explicit_input_fails_closed_in_real_child(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith("JASNA_")}
        env.update(
            JASNA_RESTORE_ONLY_VALIDATE="1",
            JASNA_RESTORE_ONLY_LEFT_INPUT=str(self.reference / "missing-left.mp4"),
            JASNA_RESTORE_ONLY_RIGHT_INPUT=str(self.reference / "missing-right.mp4"),
        )
        result = subprocess.run(
            ["/bin/bash", str(ROOT / "script/test_vr_restore_only_30s.sh"),
             str(self.reference), str(self.prefix) + ".mov"],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to substitute a different source segment", result.stderr)


if __name__ == "__main__":
    unittest.main()
