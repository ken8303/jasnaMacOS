#!/usr/bin/env python3

from pathlib import Path
import os
import subprocess
import tempfile
import unittest
import zipfile


class RunConfigurationSafetyTests(unittest.TestCase):
    def test_persistent_workflows_include_diagnostic_passthrough_in_resume_identity(self):
        root = Path(__file__).resolve().parents[1]
        expected = "allow_passthrough=${JASNA_ALLOW_PASSTHROUGH:-0}"
        scripts = (
            root / "script" / "test_vr_sparse_30s.sh",
            root / "script" / "restore_vr_eye_segments.sh",
        )

        for script in scripts:
            with self.subTest(script=script.name):
                self.assertIn(expected, script.read_text(encoding="utf-8"))

    def test_rollout_defaults_to_validated_batch_two_profile(self):
        root = Path(__file__).resolve().parents[1]
        rollout = (root / "script" / "restore_vr_rollout.sh").read_text(encoding="utf-8")

        self.assertIn('JASNA_METAL_WINDOWS_PER_PROCESS:-2', rollout)
        self.assertIn('JASNA_MODEL_BATCH:-auto', rollout)
        self.assertIn('JASNA_IN_MEMORY_CROP_CACHE:-0', rollout)
        self.assertIn('JASNA_IN_MEMORY_CACHE_LIMIT_MB:-128', rollout)
        self.assertIn('JASNA_COMPOSITE_CONCURRENCY:-1', rollout)
        self.assertIn('JASNA_RUNTIME_SCRATCH_ON_OUTPUT:-1', rollout)
        self.assertNotIn('JASNA_CLEAN_WORK_ON_SUCCESS', rollout)
        self.assertIn('JASNA_ALLOW_IMPLEMENTATION_RESUME:-0', rollout)
        self.assertIn('JASNA_DETECT_BATCH_SIZE:-1', rollout)
        self.assertIn('JASNA_DETECT_DECODE_MODE:-sequential', rollout)
        self.assertIn("metal_model_packages_available", rollout)
        self.assertIn("offset_backward_1", rollout)
        self.assertIn("backbone_forward_2", rollout)
        self.assertIn("RECORDED_MODEL_BATCH", rollout)
        self.assertIn(
            "Resume profile: retaining recorded model batch", rollout
        )
        batch_two_branch = rollout.split(
            'elif [[ "$REQUESTED_MODEL_BATCH" == "2" ]]', 1
        )[1].split(
            'elif [[ "$REQUESTED_MODEL_BATCH" == "1" ]]', 1
        )[0]
        self.assertIn("export JASNA_MODEL_BATCH=2", batch_two_branch)

    def test_rollout_fast_profile_uses_measured_memory_accelerators(self):
        root = Path(__file__).resolve().parents[1]
        rollout = (root / "script" / "restore_vr_rollout.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('JASNA_PERFORMANCE_PROFILE:-balanced', rollout)
        fast_branch = rollout.split('fast)', 1)[1].split('balanced)', 1)[0]
        self.assertIn('JASNA_METAL_WINDOWS_PER_PROCESS:-8', fast_branch)
        self.assertIn('JASNA_REGION_PREPARE_DEPTH:-2', fast_branch)
        self.assertIn('JASNA_DETECT_BATCH_SIZE:-2', fast_branch)
        self.assertIn('JASNA_IN_MEMORY_CROP_CACHE:-1', fast_branch)
        self.assertIn('JASNA_IN_MEMORY_CACHE_LIMIT_MB:-512', fast_branch)
        balanced_branch = rollout.split('balanced)', 1)[1].split('*)', 1)[0]
        self.assertIn('JASNA_METAL_WINDOWS_PER_PROCESS:-2', balanced_branch)
        self.assertIn('JASNA_REGION_PREPARE_DEPTH:-1', balanced_branch)
        self.assertIn('JASNA_DETECT_BATCH_SIZE:-1', balanced_branch)
        self.assertIn('JASNA_IN_MEMORY_CROP_CACHE:-0', balanced_branch)
        self.assertIn('JASNA_IN_MEMORY_CACHE_LIMIT_MB:-128', balanced_branch)

    def test_metal_window_ab_reuses_identical_restoration_inputs(self):
        root = Path(__file__).resolve().parents[1]
        profile = (root / "script" / "test_vr_metal_windows_ab_30s.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("test_vr_restore_only_30s.sh", profile)
        self.assertIn("export JASNA_MODEL_BATCH=2", profile)
        self.assertIn("export JASNA_IN_MEMORY_CROP_CACHE=1", profile)
        self.assertIn("export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512", profile)
        self.assertIn('JASNA_WINDOWS_AB_CONTROL:-4', profile)
        self.assertIn('JASNA_WINDOWS_AB_CANDIDATE:-8', profile)
        self.assertIn("export JASNA_GPU_TIMEOUT_RETRIES=0", profile)

    def test_runtime_scratch_is_scoped_to_output_work_directory(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "script" / "test_vr_sparse_30s.sh").read_text(encoding="utf-8")

        self.assertIn('RUNTIME_SCRATCH_DIR="$WORK_DIR/runtime-scratch"', workflow)
        self.assertIn('export TMPDIR="$RUNTIME_SCRATCH_DIR/tmp/"', workflow)
        self.assertIn('export XDG_CACHE_HOME="$RUNTIME_SCRATCH_DIR/cache"', workflow)
        self.assertIn('cleanup_runtime_scratch', workflow)

    def test_sparse_workflow_sources_stage_modules(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "script" / "test_vr_sparse_30s.sh").read_text(
            encoding="utf-8"
        )
        expected_modules = (
            "vr_sparse_media.sh",
            "vr_sparse_direct.sh",
            "vr_sparse_eye_pair.sh",
        )

        for module in expected_modules:
            with self.subTest(module=module):
                self.assertIn(f'source "$ROOT_DIR/script/lib/{module}"', workflow)
                self.assertTrue((root / "script" / "lib" / module).is_file())

        self.assertIn("run_direct_sbs_pipeline", workflow)
        self.assertIn("run_eye_pair_pipeline", workflow)

        identity = (root / "script" / "restoration_identity.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"$root_dir/script/lib"', identity)

    def test_direct_test_defaults_to_validated_batch_two_profile(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "script" / "test_vr_sparse_30s.sh").read_text(encoding="utf-8")

        self.assertIn('JASNA_METAL_WINDOWS_PER_PROCESS:-2', workflow)
        self.assertIn('JASNA_MODEL_BATCH:-2', workflow)
        self.assertIn('JASNA_IN_MEMORY_CROP_CACHE:-0', workflow)
        self.assertIn('JASNA_IN_MEMORY_CACHE_LIMIT_MB:-128', workflow)
        self.assertIn('JASNA_COMPOSITE_CONCURRENCY:-1', workflow)
        self.assertIn("composite_concurrency=$COMPOSITE_CONCURRENCY", workflow)

    def test_direct_writer_emits_the_join_timescale_without_normalization(self):
        root = Path(__file__).resolve().parents[1]
        writer = (
            root / "Sources" / "JasnaMetalPoC" / "SideBySideFrameWriter.swift"
        ).read_text(encoding="utf-8")

        self.assertIn("input.mediaTimeScale = 600", writer)

    def test_empty_window_bypass_accounts_for_a_partial_final_window(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "script" / "lib" / "vr_sparse_direct.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "EXPECTED_BYPASS_FRAMES=$((JOB_FRAME_COUNT - WINDOW_START * 30))",
            workflow,
        )
        self.assertIn('-t "$BYPASS_DURATION"', workflow)

    def test_best_five_minute_profile_is_quality_focused_and_bounded(self):
        root = Path(__file__).resolve().parents[1]
        profile = (root / "script" / "test_vr_best_5min.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'export JASNA_TEST_SECONDS="${JASNA_TEST_SECONDS:-300}"', profile
        )
        self.assertIn("export JASNA_DETECTOR=rfdetr-vr-v1", profile)
        self.assertIn("export JASNA_DETECT_SAMPLE_STRIDE=0.1", profile)
        self.assertIn("export JASNA_DETECT_BATCH_SIZE=2", profile)
        self.assertIn("export JASNA_MODEL_BATCH=2", profile)
        self.assertIn("export JASNA_METAL_WINDOWS_PER_PROCESS=8", profile)
        self.assertIn("export JASNA_REGION_PREPARE_DEPTH=2", profile)
        self.assertIn("export JASNA_IN_MEMORY_CROP_CACHE=1", profile)
        self.assertIn("export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512", profile)
        self.assertIn("export JASNA_COMPOSITE_CONCURRENCY=1", profile)
        self.assertIn("export JASNA_RUNTIME_SCRATCH_ON_OUTPUT=1", profile)
        self.assertNotIn("JASNA_CLEAN_WORK_ON_SUCCESS", profile)

    def test_restore_only_replays_recorded_settings_but_keeps_explicit_overrides(self):
        root = Path(__file__).resolve().parents[1]
        helper = root / "script" / "lib" / "run_configuration.sh"
        with tempfile.TemporaryDirectory() as directory:
            configuration = Path(directory) / "run-config.txt"
            configuration.write_text(
                "model_batch=2\nlarge_region_mask_growth=0.075\n",
                encoding="utf-8",
            )
            command = f"""
source {str(helper)!r}
jasna_adopt_recorded_environment {str(configuration)!r} model_batch JASNA_MODEL_BATCH
jasna_adopt_recorded_environment {str(configuration)!r} large_region_mask_growth JASNA_LARGE_REGION_MASK_GROWTH
printf '%s|%s\n' "$JASNA_MODEL_BATCH" "$JASNA_LARGE_REGION_MASK_GROWTH"
"""
            inherited = subprocess.run(
                ["/bin/bash", "-c", command],
                check=True,
                capture_output=True,
                text=True,
                env={key: value for key, value in os.environ.items() if key != "JASNA_MODEL_BATCH"},
            )
            self.assertEqual(inherited.stdout.strip(), "2|0.075")

            override_environment = os.environ.copy()
            override_environment["JASNA_MODEL_BATCH"] = "1"
            overridden = subprocess.run(
                ["/bin/bash", "-c", command],
                check=True,
                capture_output=True,
                text=True,
                env=override_environment,
            )
            self.assertEqual(overridden.stdout.strip(), "1|0.075")

    def test_temporal_mask_ab_builds_one_shared_executable(self):
        root = Path(__file__).resolve().parents[1]
        profile = (root / "script" / "test_vr_temporal_mask_ab.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("Building one optimized executable for both temporal-mask runs", profile)
        self.assertIn("export JASNA_APP_BINARY", profile)
        self.assertEqual(profile.count("test_vr_restore_only_30s.sh"), 2)

    def test_packager_publishes_validated_zip_and_preserves_apps(self):
        self.check_archive_publication(valid=True)

    def test_packager_keeps_existing_release_when_validation_fails(self):
        self.check_archive_publication(valid=False)

    def check_archive_publication(self, valid):
        root = Path(__file__).resolve().parents[1]
        packager = (root / "script/package_jasna_test_app.sh").read_text()
        start = packager.index("publish_archive() (")
        end = packager.index("\n)\n", start) + 3
        function = packager[start:end]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            current = destination / "Jasna-VR-Restoration-macOS27-Test.zip"
            previous = destination / "Jasna-VR-Restoration-macOS27-Test.previous.zip"
            current.write_bytes(b"current release")
            previous.write_bytes(b"older release")
            app = destination / "Jasna VR Restoration.app"
            app.mkdir()
            (app / "sentinel").write_text("working app")
            staged = destination / "staged.zip"
            if valid:
                with zipfile.ZipFile(staged, "w") as archive:
                    archive.writestr("build.txt", "new release")
            else:
                staged.write_bytes(b"invalid archive")
            result = subprocess.run(
                ["bash", "-c", function + '\npublish_archive "$1" "$2"',
                 "test", str(staged), str(destination)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode == 0, valid, result.stderr)
            self.assertEqual(current.read_bytes(),
                             staged.read_bytes() if valid else b"current release")
            self.assertEqual(previous.read_bytes(),
                             b"current release" if valid else b"older release")
            self.assertEqual((app / "sentinel").read_text(), "working app")
            self.assertEqual(list(destination.glob(".jasna-*")), [])

    def test_rfdetr_setup_and_packager_pin_validated_version(self):
        root = Path(__file__).resolve().parents[1]
        setup = (root / "script" / "setup_rfdetr_detector.sh").read_text(
            encoding="utf-8"
        )
        packager = (root / "script" / "package_jasna_test_app.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('RFDETR_VERSION="1.10.0"', setup)
        self.assertIn('"rfdetr==$RFDETR_VERSION"', setup)
        self.assertIn('EXPECTED_RFDETR_VERSION="1.10.0"', packager)
        self.assertIn(
            '[[ "$INSTALLED_RFDETR_VERSION" == "$EXPECTED_RFDETR_VERSION" ]]',
            packager,
        )
        self.assertIn('export CLANG_MODULE_CACHE_PATH="$MODULE_CACHE_DIR"', packager)
        self.assertIn('export SWIFTPM_MODULECACHE_OVERRIDE="$MODULE_CACHE_DIR"', packager)

    def test_quality_speed_five_minute_profile_keeps_quality_settings(self):
        root = Path(__file__).resolve().parents[1]
        profile = (root / "script" / "test_vr_quality_speed_5min.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'export JASNA_TEST_SECONDS="${JASNA_TEST_SECONDS:-300}"', profile
        )
        self.assertIn("export JASNA_DETECTOR=rfdetr-vr-v1", profile)
        self.assertIn("export JASNA_DETECT_SAMPLE_STRIDE=0.1", profile)
        self.assertIn("export JASNA_ADAPTIVE_DETECT=0", profile)
        self.assertIn("export JASNA_MOSAIC_MASK_RECOVERY_ALL_REGIONS=1", profile)
        self.assertIn("export JASNA_TEMPORAL_WARMUP_FRAMES=5", profile)
        self.assertIn("export JASNA_DETECT_BATCH_SIZE=2", profile)
        self.assertIn("export JASNA_MODEL_BATCH=2", profile)
        self.assertIn(
            'export JASNA_BATCH2_MODELS_DIR="$ROOT_DIR/Models/MetalMLBatch2"',
            profile,
        )
        self.assertIn("export JASNA_METAL_WINDOWS_PER_PROCESS=8", profile)
        self.assertIn("export JASNA_REGION_PREPARE_DEPTH=2", profile)
        self.assertIn("export JASNA_IN_MEMORY_CROP_CACHE=1", profile)
        self.assertIn("export JASNA_IN_MEMORY_CACHE_LIMIT_MB=512", profile)
        self.assertIn("export JASNA_COMPOSITE_CONCURRENCY=1", profile)

    def test_detector_batch_two_comparison_changes_only_submission_batching(self):
        root = Path(__file__).resolve().parents[1]
        profile = (root / "script" / "test_detector_batch2_reuse.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("export JASNA_DETECTOR=rfdetr-vr-v1", profile)
        self.assertIn("export JASNA_DETECT_SAMPLE_STRIDE=0.1", profile)
        self.assertIn("export JASNA_DETECT_BATCH_SIZE=2", profile)
        self.assertIn("export JASNA_DETECT_DECODE_MODE=sequential", profile)
        self.assertIn("export JASNA_STEREO_SAMPLE_MODE=paired", profile)
        self.assertIn("export JASNA_ADAPTIVE_DETECT=0", profile)
        self.assertIn("export JASNA_DETECT_CONFIDENCE=0.15", profile)
        self.assertIn("export JASNA_MASK_SIZE=128", profile)
        self.assertNotIn("build_and_run.sh", profile)

    def test_physical_eye_preparation_is_main8_and_has_no_model_work(self):
        root = Path(__file__).resolve().parents[1]
        profile = (root / "script" / "prepare_eye_by_eye_5min.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("crop=4096:4096:0:0,fps=30,format=yuv420p", profile)
        self.assertIn("crop=4096:4096:4096:0,fps=30,format=yuv420p", profile)
        self.assertEqual(profile.count("-c:v hevc_videotoolbox -profile:v main"), 2)
        self.assertIn('[[ "$(video_value "$path" nb_frames)" == "1800" ]]', profile)
        self.assertIn("-segment_time 60", profile)
        self.assertNotIn("scan_mosaic_regions.sh", profile)
        self.assertNotIn("build_and_run.sh", profile)

    def test_physical_eye_detector_batch_four_has_no_restoration_work(self):
        root = Path(__file__).resolve().parents[1]
        profile = (root / "script" / "test_detector_batch4_eye_files.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("export JASNA_DETECTOR=rfdetr-vr-v1", profile)
        self.assertIn("export JASNA_DETECT_SAMPLE_STRIDE=0.1", profile)
        self.assertIn("export JASNA_DETECT_BATCH_SIZE=4", profile)
        self.assertIn("export JASNA_DETECT_DECODE_MODE=sequential", profile)
        self.assertIn("export JASNA_ADAPTIVE_DETECT=0", profile)
        self.assertIn("scan_mosaic_regions.sh", profile)
        self.assertIn("Archived incomplete detector attempt at:", profile)
        self.assertNotIn("build_and_run.sh", profile)

    def test_single_eye_detector_does_not_expand_an_empty_stereo_array(self):
        root = Path(__file__).resolve().parents[1]
        scanner = (root / "script" / "scan_mosaic_regions.sh").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("STEREO_ARGUMENTS", scanner)
        self.assertIn('DETECT_COMMAND+=(--stereo-right-manifest "$3")', scanner)
        self.assertIn('"${DETECT_COMMAND[@]}"', scanner)

    def test_restoration_only_accepts_longer_reusable_sources(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "script" / "test_vr_restore_only_30s.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('SOURCE_FRAME_COUNT" -ge $(((AUDIO_START_SECOND + SELECTED_SECONDS) * 30))', workflow)
        self.assertIn('INPUT_SECONDS=$((LEFT_INPUT_FRAME_COUNT / 30))', workflow)
        self.assertIn('END_WINDOW" -le "$INPUT_SECONDS"', workflow)
        self.assertNotIn('SOURCE_FRAME_COUNT" == "900"', workflow)

    def test_restoration_only_clean_bypass_uses_prepared_source_timeline(self):
        workflow = (Path(__file__).resolve().parents[1] / "script/test_vr_restore_only_30s.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('BYPASS_SOURCE_SECOND=$((AUDIO_START_SECOND + WINDOW_START - START_WINDOW))', workflow)
        self.assertIn('-ss "$BYPASS_SOURCE_SECOND" -i "$SOURCE_SBS"', workflow)

    def test_model_batch_ab_reuses_assets_and_changes_only_model_batch(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / "script" / "test_vr_model_batch_ab_30s.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("export JASNA_RESTORE_ONLY_SECONDS=", workflow)
        self.assertIn(
            'export JASNA_METAL_WINDOWS_PER_PROCESS="${JASNA_METAL_WINDOWS_PER_PROCESS:-2}"',
            workflow,
        )
        self.assertIn("export JASNA_TEMPORAL_WARMUP_FRAMES=5", workflow)
        self.assertIn("export JASNA_GPU_TIMEOUT_RETRIES=0", workflow)
        self.assertIn("Building one shared optimized Swift executable", workflow)
        self.assertIn("export JASNA_APP_BINARY", workflow)
        self.assertIn("JASNA_MODEL_BATCH=1", workflow)
        self.assertIn("JASNA_MODEL_BATCH=2", workflow)
        self.assertIn("test_vr_restore_only_30s.sh", workflow)
        self.assertNotIn("scan_mosaic_regions.sh", workflow)

    def test_mpsgraph_cleanup_is_scoped_to_the_child_process(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "script" / "build_and_run.sh").read_text(encoding="utf-8")

        self.assertIn('JASNA_RUNTIME_PID_FILE="$JASNA_PROCESS_LOCK/app-pid"', launcher)
        self.assertIn('EXISTING_APP_PID=', launcher)
        self.assertIn('cleanup_mpsgraph_temp_for_pid "$EXISTING_APP_PID"', launcher)
        self.assertIn('-name "mpsgraph-${app_pid}-*"', launcher)
        self.assertIn(
            'expected_root="$(getconf DARWIN_USER_TEMP_DIR)com.apple.MetalPerformanceShadersGraph"',
            launcher,
        )

    def test_general_restoration_wrappers_do_not_force_batch_two(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("restore_vr_sparse_sbs.sh", "restore_vr_eye_by_eye.sh"):
            script = (root / "script" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn('JASNA_MODEL_BATCH:-1', script)
                self.assertNotIn('JASNA_MODEL_BATCH:-2', script)

        detector = (root / "script" / "scan_mosaic_regions.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('JASNA_DETECT_BATCH_SIZE:-1', detector)

    def test_swift_restoration_has_no_system_temporary_fallback(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("SideBySideRestoration.swift", "SparseRegionRestoration.swift"):
            source = (root / "Sources" / "JasnaMetalPoC" / name).read_text(
                encoding="utf-8"
            )
            with self.subTest(source=name):
                self.assertNotIn("FileManager.default.temporaryDirectory", source)

    def test_app_always_preserves_restart_work(self):
        root = Path(__file__).resolve().parents[1]
        session = (
            root / "Sources" / "JasnaMacApp" / "Stores" / "RestorationSession.swift"
        ).read_text(encoding="utf-8")
        workflow = (root / "script" / "test_vr_sparse_30s.sh").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("keepRestartFilesAfterCompletion", session)
        self.assertNotIn("JASNA_CLEAN_WORK_ON_SUCCESS", session)
        self.assertIn("Persistent segments and caches", workflow)
        self.assertNotIn("/bin/rm -rf -- \"$WORK_DIR\"", workflow)


if __name__ == "__main__":
    unittest.main()
