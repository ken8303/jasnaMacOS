import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from source_video_bitrate import video_bitrate


class SourceVideoBitrateTests(unittest.TestCase):
    def test_measured_video_excludes_audio(self):
        probe = {
            "format": {"size": "2615193508", "duration": "1793.429833"},
            "streams": [
                {"codec_type": "video", "bit_rate": "11402072"},
                {"codec_type": "audio", "bit_rate": "255999"},
            ],
        }
        self.assertEqual(video_bitrate(probe), 11409664)

    def test_uses_video_stream_rate_when_audio_rate_missing(self):
        probe = {
            "format": {"size": "2615193508", "duration": "1793.429833"},
            "streams": [
                {"codec_type": "video", "bit_rate": "11402072"},
                {"codec_type": "audio", "bit_rate": "N/A"},
            ],
        }
        self.assertEqual(video_bitrate(probe), 11402072)

    def test_rejects_unmeasurable_video_with_audio(self):
        with self.assertRaises(ValueError):
            video_bitrate({"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]})

    def test_command_reports_full_and_eye_targets(self):
        tool = Path(__file__).with_name("source_video_bitrate.py")
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "ffprobe"
            payload = {
                "format": {"size": "12500000", "duration": "10"},
                "streams": [
                    {"codec_type": "video", "bit_rate": "9000000"},
                    {"codec_type": "audio", "bit_rate": "200000"},
                ],
            }
            probe.write_text("#!/bin/sh\ncat <<'EOF'\n" + json.dumps(payload) + "\nEOF\n")
            probe.chmod(0o755)
            for option, expected in (([], "9800000"), (["--eye"], "4900000")):
                result = subprocess.run(
                    [sys.executable, str(tool), str(probe), "source.mp4", *option],
                    check=True, capture_output=True, text=True,
                )
                self.assertEqual(result.stdout.strip(), expected)
