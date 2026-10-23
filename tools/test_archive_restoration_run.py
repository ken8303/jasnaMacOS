from pathlib import Path
import os
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from archive_restoration_run import archive_run, LOCK, MARKER


class ArchiveRestorationRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / 'clip_restored.jasna-vr-full-v22-work'
        self.lock = self.work / LOCK
        self.lock.mkdir(parents=True)
        (self.lock / 'pid').write_text(str(os.getpid()))
        (self.work / 'source').mkdir()
        (self.work / 'source/video.mp4').write_bytes(b'cached source')
        (self.work / 'run-config.txt').write_text('restoration_backend=metal')
        (self.work / '.joined-sbs-writing.mp4').write_bytes(b'pending output')
        self.output = self.root / 'clip_restored.mp4'
        self.output.write_bytes(b'previous restored output')
        (self.root / 'clip.mp4').write_bytes(b'original')

    def test_preserves_output_and_all_work_while_retaining_lock(self):
        backup = archive_run(self.work, self.output, os.getpid())
        self.assertEqual((backup / self.output.name).read_bytes(), b'previous restored output')
        self.assertEqual((backup / 'work/source/video.mp4').read_bytes(), b'cached source')
        self.assertEqual((backup / 'work/.joined-sbs-writing.mp4').read_bytes(), b'pending output')
        self.assertEqual((backup / 'work/run-config.txt').read_text(), 'restoration_backend=metal')
        self.assertEqual(list(self.work.iterdir()), [self.lock])
        self.assertFalse(self.output.exists())
        self.assertEqual((self.root / 'clip.mp4').read_bytes(), b'original')

    def test_does_not_archive_another_workflows_files(self):
        with self.assertRaises(ValueError):
            archive_run(self.work, self.output, os.getpid() + 1)
        self.assertTrue(self.output.exists())
        self.assertTrue((self.work / 'run-config.txt').exists())
        self.assertEqual(list(self.root.glob('*.jasna-previous-*')), [])

    def test_interrupted_archive_is_resumable_without_losing_work(self):
        real_rename = os.rename
        def fail_for_output(source, destination):
            if Path(source) == self.output:
                raise OSError('simulated interruption')
            real_rename(source, destination)
        with patch('archive_restoration_run.os.rename', side_effect=fail_for_output):
            with self.assertRaises(OSError):
                archive_run(self.work, self.output, os.getpid())
        self.assertTrue((self.work / MARKER).exists())
        backup = archive_run(self.work, self.output, os.getpid())
        self.assertEqual((backup / self.output.name).read_bytes(), b'previous restored output')
        self.assertEqual((backup / 'work/source/video.mp4').read_bytes(), b'cached source')
        self.assertFalse((self.work / MARKER).exists())
        self.assertEqual(len(list(self.root.glob('*.jasna-previous-*'))), 1)

    def test_missing_output_and_repeated_fresh_runs(self):
        self.output.unlink()
        first = archive_run(self.work, self.output, os.getpid())
        (self.work / 'run-config.txt').write_text('restoration_backend=mlx')
        second = archive_run(self.work, self.output, os.getpid())
        self.assertNotEqual(first, second)
        self.assertEqual((first / 'work/run-config.txt').read_text(), 'restoration_backend=metal')
        self.assertEqual((second / 'work/run-config.txt').read_text(), 'restoration_backend=mlx')

    def test_workflow_fresh_start_replaces_conflicting_configuration_after_backup(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / 'script/test_vr_sparse_30s.sh').read_text()
        fresh_block = 'if [[ "${JASNA_START_FRESH:-0}" == "1" ]]; then' + script.split(
            'if [[ "${JASNA_START_FRESH:-0}" == "1" ]]; then', 1
        )[1].split('prepare_runtime_scratch', 1)[0]
        config_block = 'if [[ -s "$RUN_CONFIG_PATH" ]]; then' + script.split(
            'if [[ -s "$RUN_CONFIG_PATH" ]]; then', 1
        )[1].split('LOG_SESSION_START_LINE=0', 1)[0]
        original = 'input=clip.mp4\ndetector=rfdetr-vr-v1\nimplementation_fingerprint=same\n'
        (self.work / 'run-config.txt').write_text(original)
        current = original.rstrip() + '\nrestoration_backend=mlx'
        command = ('set -euo pipefail\nprintf "%s\\n" "$$" > "$WORK_DIR/.jasna-workflow-lock/pid"\n'
                   + fresh_block + '\n' + config_block)
        environment = dict(os.environ, ROOT_DIR=str(root), WORK_DIR=str(self.work), OUTPUT_PATH=str(self.output),
                           RUN_CONFIG_PATH=str(self.work / 'run-config.txt'), RUN_CONFIG=current,
                           ALLOW_IMPLEMENTATION_RESUME='0')
        conflict = subprocess.run(['bash', '-c', command], env=dict(environment, JASNA_START_FRESH='0'),
                                  capture_output=True, text=True)
        self.assertEqual(conflict.returncode, 1)
        self.assertIn('different test configuration', conflict.stderr)
        self.assertTrue(self.output.exists())
        fresh = subprocess.run(['bash', '-c', command], env=dict(environment, JASNA_START_FRESH='1'),
                               capture_output=True, text=True)
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertEqual((self.work / 'run-config.txt').read_text().strip(), current)
        self.assertFalse(self.output.exists())
        backup = next(self.root.glob('*.jasna-previous-*'))
        self.assertEqual((backup / self.output.name).read_bytes(), b'previous restored output')
        self.assertEqual((backup / 'work/run-config.txt').read_text(), original)

    def test_normal_resume_refuses_partially_archived_work(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / 'script/test_vr_sparse_30s.sh').read_text()
        block = 'if [[ "${JASNA_START_FRESH:-0}" == "1" ]]; then' + script.split(
            'if [[ "${JASNA_START_FRESH:-0}" == "1" ]]; then', 1
        )[1].split('prepare_runtime_scratch', 1)[0]
        (self.work / MARKER).write_text('{}')
        result = subprocess.run(['bash', '-c', block], capture_output=True, text=True,
                                env=dict(os.environ, WORK_DIR=str(self.work), JASNA_START_FRESH='0'))
        self.assertEqual(result.returncode, 1)
        self.assertIn('a previous fresh start was interrupted', result.stderr)
        self.assertTrue(self.output.exists())

    def test_fresh_rollout_ignores_previous_batch_and_detector(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / 'script/restore_vr_rollout.sh').read_text()
        batch = 'RECORDED_MODEL_BATCH=""' + script.split('RECORDED_MODEL_BATCH=""', 1)[1].split(
            '# Batch 2 is the validated M4 default.', 1)[0]
        detector = 'source "$ROOT_DIR/script/lib/run_configuration.sh"' + script.split(
            'source "$ROOT_DIR/script/lib/run_configuration.sh"', 1)[1].split('export JASNA_DETECT_DECODE_MODE=', 1)[0]
        config = self.work / 'run-config.txt'
        config.write_text('model_batch=1\ndetect_device=mps\n')
        environment = dict(os.environ, ROOT_DIR=str(root), ROLLOUT_RESUME_CONFIG=str(config))
        environment.pop('JASNA_DETECT_DEVICE', None)
        for fresh, expected in [('0', '1|mps'), ('1', '|mlx')]:
            result = subprocess.run(['bash', '-c', batch + detector + '\nprintf "%s|%s" "$RECORDED_MODEL_BATCH" "$JASNA_DETECT_DEVICE"'],
                                    env=dict(environment, JASNA_START_FRESH=fresh), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)


if __name__ == '__main__':
    unittest.main()
