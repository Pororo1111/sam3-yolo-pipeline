from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline import extractor, media


ROOT = Path(__file__).resolve().parents[1]


class ExtractorTests(unittest.TestCase):
    def test_video_capture_yields_rgb_preview_and_saves_frame(self):
        sample_video = ROOT / "samples" / "sample.mp4"
        self.assertTrue(sample_video.is_file())

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "raw_frames"
            with mock.patch.object(extractor, "OUT_DIR", output_dir):
                stream = extractor.capture(
                    media.SOURCE_VIDEO,
                    "",
                    5,
                    video_file=sample_video,
                )
                try:
                    frame, status = next(stream)
                finally:
                    stream.close()

            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
            self.assertIn("미리보기", status)
            self.assertEqual(len(list(output_dir.glob("frame_*.jpg"))), 1)

    def test_invalid_source_does_not_delete_existing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "raw_frames"
            output_dir.mkdir()
            existing = output_dir / "frame_00000.jpg"
            existing.write_bytes(b"existing")

            with mock.patch.object(extractor, "OUT_DIR", output_dir):
                outputs = list(
                    extractor.capture(
                        media.SOURCE_VIDEO,
                        "",
                        5,
                        video_file=None,
                    )
                )

            self.assertTrue(existing.exists())
            self.assertIn("업로드", outputs[-1][1])

    def test_unreadable_open_source_preserves_existing_frames(self):
        class UnreadableCapture:
            def read(self):
                return False, None

        @contextmanager
        def open_capture(_source):
            yield UnreadableCapture()

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "raw_frames"
            output_dir.mkdir()
            existing = output_dir / "frame_00000.jpg"
            existing.write_bytes(b"existing")

            source = media.VideoSource("camera", media.SOURCE_WEBCAM, False)
            with (
                mock.patch.object(extractor, "OUT_DIR", output_dir),
                mock.patch.object(media, "resolve_video_source", return_value=source),
                mock.patch.object(media, "open_video_capture", side_effect=open_capture),
            ):
                outputs = list(
                    extractor.capture(
                        media.SOURCE_WEBCAM,
                        "",
                        5,
                        webcam_index="0",
                    )
                )

            self.assertEqual(existing.read_bytes(), b"existing")
            self.assertIn("첫 프레임", outputs[-1][1])

    def test_stop_during_first_read_preserves_existing_frames(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        class StopDuringReadCapture:
            def read(self):
                extractor._controller.stop_event.set()
                return True, frame

        @contextmanager
        def open_capture(_source):
            yield StopDuringReadCapture()

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "raw_frames"
            output_dir.mkdir()
            existing = output_dir / "frame_00000.jpg"
            existing.write_bytes(b"existing")
            source = media.VideoSource("camera", media.SOURCE_WEBCAM, False)
            with (
                mock.patch.object(extractor, "OUT_DIR", output_dir),
                mock.patch.object(media, "resolve_video_source", return_value=source),
                mock.patch.object(media, "open_video_capture", side_effect=open_capture),
            ):
                outputs = list(
                    extractor.capture(
                        media.SOURCE_WEBCAM,
                        "",
                        5,
                        webcam_index="0",
                    )
                )

            self.assertEqual(existing.read_bytes(), b"existing")
            self.assertIn("중지됨", outputs[-1][1])


if __name__ == "__main__":
    unittest.main()
