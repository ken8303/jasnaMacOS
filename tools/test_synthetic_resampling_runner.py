"""Test result/exit handling with a fake compiler; never invokes Swift or media tools."""

from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MARKERS = [
    "Synthetic raw moving ramp:",
    "Synthetic fisheye 512px moving ramp:",
    "Synthetic fisheye 4096px moving ramp:",
    "Synthetic reference axes: 10 anchors checked",
    "Synthetic reference borders 512px: 780 points",
    "Synthetic reference borders 4096px: 780 points",
    "Synthetic reference motion: 60 frames, 5 crop changes",
] + [
    f"Synthetic {projection} moving detail at {origin}:"
    for projection in ("raw", "fisheye")
    for origin in (0, 64, 128)
]


class SyntheticRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jasna-synthetic-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / "script"
        scripts.mkdir()
        self.script = scripts / "test_synthetic_resampling.sh"
        shutil.copyfile(ROOT / "script" / self.script.name, self.script)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        stub = self.bin / "swift"
        stub.write_text(
            '#!/bin/bash\nset -eu\n'
            '[[ "$1" == test ]] || exit 99\n'
            '[[ "$JASNA_SYNTHETIC_STRICT_ALIGNMENT" == 1 ]] || exit 98\n'
            'printf "STRICT %s\\n" "$JASNA_SYNTHETIC_STRICT_ALIGNMENT"\n'
            'if [[ "${SYNTHETIC_STUB_MARKERS:-all}" == all ]]; then\n'
            + "".join(f"printf '%s\\n' '{marker}'\n" for marker in MARKERS)
            + 'fi\nexit "${SYNTHETIC_STUB_EXIT:-0}"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        self.output = self.root / "results with spaces"

    def run_harness(self, *arguments, **overrides):
        env = {key: value for key, value in os.environ.items() if not key.startswith("JASNA_")}
        env.update(
            PATH=f"{self.bin}:/usr/bin:/bin", JASNA_SYNTHETIC_STRICT_ALIGNMENT="0",
            JASNA_SYNTHETIC_BUILD_DIR=str(self.output / "build"),
        )
        env.update(overrides)
        return subprocess.run(
            ["/bin/bash", str(self.script), *(arguments or (str(self.output),))],
            env=env, capture_output=True, text=True, timeout=10,
        )

    def summary(self):
        summaries = list(self.output.glob("run-*/summary.txt"))
        self.assertEqual(len(summaries), 1)
        return summaries[0].read_text(encoding="utf-8")

    def signing_fixture(self, *, external_directory=False, repeat_failure=False, first_attempt_markers=False):
        products = self.output / "build/out/Products/Debug"
        products.parent.mkdir(parents=True)
        if external_directory:
            external = self.root / "unrelated data"
            external.mkdir()
            products.symlink_to(external, target_is_directory=True)
        else:
            products.mkdir()
        for name in ("JasnaAppSupportTests.xctest", "JasnaMetalPoCTests.xctest"):
            bundle = products / name
            bundle.mkdir()
            for attribute in ("com.apple.FinderInfo", "com.apple.ResourceFork", "com.apple.quarantine"):
                (bundle / attribute).write_text("simulated metadata", encoding="utf-8")
        xattr = self.bin / "xattr"
        xattr.write_text(
            '#!/bin/bash\nset -eu\n'
            'if [[ $# == 1 ]]; then\n'
            '  for attribute in com.apple.FinderInfo com.apple.ResourceFork com.apple.quarantine; do\n'
            '    if [[ -f "$1/$attribute" ]]; then printf "%s\\n" "$attribute"; fi\n'
            '  done\n'
            'elif [[ $# == 3 && "$1" == -d ]]; then\n'
            '  [[ "$2" == com.apple.FinderInfo || "$2" == com.apple.ResourceFork ]] || exit 97\n'
            '  /bin/rm -- "$3/$2"\n'
            'else exit 96; fi\n',
            encoding="utf-8",
        )
        xattr.chmod(0o755)
        compiler = self.bin / "swift"
        content = compiler.read_text(encoding="utf-8")
        signing_check = (
            'bundle="$CLANG_MODULE_CACHE_PATH/../out/Products/Debug/JasnaAppSupportTests.xctest"\n'
            'printf "call\\n" >> "$CLANG_MODULE_CACHE_PATH/calls"\n'
            + ('if true; then\n' if repeat_failure else 'if [[ -f "$bundle/com.apple.FinderInfo" ]]; then\n')
            + ("".join(f"printf '%s\\n' '{marker}'\n" for marker in MARKERS) if first_attempt_markers else "")
            + 'echo "resource fork, Finder information, or similar detritus not allowed" >&2\n'
            'exit 1\nfi\n'
        )
        compiler.write_text(content.replace('printf "STRICT', signing_check + 'printf "STRICT'), encoding="utf-8")
        return products

    def test_pass_requires_complete_cases_and_forces_strict_mode(self):
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = self.summary()
        self.assertIn("status=PASS\n", summary)
        self.assertIn("missing_cases=0\n", summary)
        self.assertIn("strict_alignment=1\n", summary)
        log = next(self.output.glob("run-*/synthetic-tests.log")).read_text(encoding="utf-8")
        self.assertIn("STRICT 1", log)
        self.assertIn("Synthetic diagnostic: PASS", log)
        self.assertTrue((self.output / "build").is_dir())

    def default_cache_path(self):
        package_id = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:16]
        return Path(f"/private/tmp/jasna-synthetic-build-{os.getuid()}-{package_id}")

    def test_default_cache_is_external_reusable_and_keeps_reports(self):
        cache = self.default_cache_path()
        self.assertFalse(cache.exists())
        self.addCleanup(lambda: shutil.rmtree(cache) if cache.exists() else None)
        first = self.run_harness(JASNA_SYNTHETIC_BUILD_DIR="")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn(f"build_directory={cache}\n", self.summary())
        self.assertEqual(cache.stat().st_mode & 0o777, 0o700)
        self.assertFalse((self.output / "build").exists())
        second = self.run_harness(JASNA_SYNTHETIC_BUILD_DIR="")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        summaries = list(self.output.glob("run-*/summary.txt"))
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(f"build_directory={cache}\n" in item.read_text() for item in summaries))

    def test_default_cache_refuses_preexisting_symlink(self):
        cache = self.default_cache_path()
        self.assertFalse(cache.exists())
        protected = self.root / "unrelated data"
        protected.mkdir()
        sentinel = protected / "keep"
        sentinel.write_text("unchanged", encoding="utf-8")
        cache.symlink_to(protected, target_is_directory=True)
        self.addCleanup(cache.unlink)
        result = self.run_harness(JASNA_SYNTHETIC_BUILD_DIR="")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("unsafe default build-cache directory", result.stderr)
        self.assertEqual(sentinel.read_text(), "unchanged")
        self.assertEqual(list(protected.iterdir()), [sentinel])

    def test_test_failure_survives_successful_tee(self):
        result = self.run_harness(SYNTHETIC_STUB_EXIT="1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("status=FAIL\n", self.summary())
        self.assertIn("swift_exit=1\n", self.summary())
        self.assertIn("logging_exit=0\n", self.summary())

    def test_zero_tests_cannot_report_pass(self):
        result = self.run_harness(SYNTHETIC_STUB_MARKERS="none")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("status=INCOMPLETE\n", self.summary())
        self.assertIn("missing_cases=13\n", self.summary())

    def test_build_failure_is_incomplete_not_a_geometry_result(self):
        result = self.run_harness(SYNTHETIC_STUB_MARKERS="none", SYNTHETIC_STUB_EXIT="42")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("status=INCOMPLETE\n", self.summary())
        self.assertIn("swift_exit=42\n", self.summary())

    def test_logging_failure_cannot_report_pass(self):
        tee = self.bin / "tee"
        tee.write_text(
            '#!/bin/bash\n/usr/bin/tee "$@"\n'
            'if [[ "${1:-}" == -a ]]; then exit 1; fi\n',
            encoding="utf-8",
        )
        tee.chmod(0o755)
        result = self.run_harness()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("status=INCOMPLETE\n", self.summary())
        self.assertNotIn("status=PASS\n", self.summary())

    def test_signing_repair_is_targeted_and_keeps_quarantine(self):
        products = self.signing_fixture()
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("attempts=2\n", self.summary())
        self.assertIn("metadata_repair_status=0\n", self.summary())
        for bundle in products.iterdir():
            self.assertFalse((bundle / "com.apple.FinderInfo").exists())
            self.assertFalse((bundle / "com.apple.ResourceFork").exists())
            self.assertEqual((bundle / "com.apple.quarantine").read_text(), "simulated metadata")
        self.assertEqual(len(list(self.output.glob("run-*/attempt-*.log"))), 2)

    def test_signing_repair_retries_only_once(self):
        self.signing_fixture(repeat_failure=True)
        result = self.run_harness()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("status=INCOMPLETE\n", self.summary())
        self.assertIn("attempts=2\n", self.summary())
        calls = (self.output / "build/ModuleCache/calls").read_text().splitlines()
        self.assertEqual(calls, ["call", "call"])

    def test_signing_repair_refuses_symlink_escape(self):
        products = self.signing_fixture(external_directory=True)
        result = self.run_harness()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("metadata_repair_status=2\n", self.summary())
        self.assertIn("attempts=1\n", self.summary())
        self.assertIn("Refusing metadata repair through a symlink", result.stdout)
        for bundle in products.iterdir():
            self.assertTrue((bundle / "com.apple.FinderInfo").exists())

    def test_retry_cannot_borrow_completion_markers_from_failed_attempt(self):
        self.signing_fixture(first_attempt_markers=True)
        result = self.run_harness(SYNTHETIC_STUB_MARKERS="none")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("status=INCOMPLETE\n", self.summary())
        self.assertIn("swift_exit=0\n", self.summary())
        self.assertIn("missing_cases=13\n", self.summary())

    def test_repeat_preserves_prior_results(self):
        self.assertEqual(self.run_harness(SYNTHETIC_STUB_EXIT="1").returncode, 1)
        first = {item: item.read_bytes() for item in self.output.glob("run-*/*")}
        self.assertEqual(self.run_harness().returncode, 0)
        self.assertEqual(len(list(self.output.glob("run-*/summary.txt"))), 2)
        for item, contents in first.items():
            self.assertEqual(item.read_bytes(), contents)

    def test_empty_argument_is_rejected_without_results(self):
        result = self.run_harness("")
        self.assertEqual(result.returncode, 2)
        self.assertIn("empty or ends with whitespace", result.stderr)
        self.assertFalse(self.output.exists())

    def test_help_and_invalid_flags_do_not_start_a_run(self):
        self.assertEqual(self.run_harness("--help").returncode, 0)
        self.assertEqual(self.run_harness("--unknown").returncode, 2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
