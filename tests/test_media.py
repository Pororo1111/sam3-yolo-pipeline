from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import media


class _FakeCapture:
    def __init__(self, opened: bool = True):
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True


class MediaTests(unittest.TestCase):
    def test_youtube_resolver_prefers_video_only_http_mp4(self):
        captured_options = {}

        class _FakeYoutubeDL:
            def __init__(self, options):
                captured_options.update(options)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def extract_info(self, url, download):
                self.url = url
                self.download = download
                return {"url": "https://example.com/video.mp4"}

        resolver = media.YouTubeStreamResolver()
        with mock.patch.object(media.yt_dlp, "YoutubeDL", _FakeYoutubeDL):
            stream_url = resolver.resolve("https://youtu.be/example")

        self.assertEqual(stream_url, "https://example.com/video.mp4")
        self.assertTrue(captured_options["format"].startswith("bestvideo[ext=mp4]"))

    def test_uploaded_video_path_validates_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "video.txt"
            invalid.touch()

            with self.assertRaisesRegex(
                media.MediaSourceError,
                "지원하지 않는 비디오 형식",
            ):
                media.uploaded_video_path(invalid)

    def test_filter_image_paths_keeps_supported_files_in_name_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.jpg"
            second = root / "b.png"
            ignored = root / "c.txt"
            for path in (second, ignored, first):
                path.touch()

            self.assertEqual(
                media.filter_image_paths([second, ignored, first]),
                [first, second],
            )

    def test_open_video_capture_always_releases_handle(self):
        capture = _FakeCapture()
        source = media.VideoSource(
            value="video.mp4",
            source_type=media.SOURCE_VIDEO,
            pace_reads=True,
        )

        with mock.patch.object(media.cv2, "VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with media.open_video_capture(source):
                    raise RuntimeError("cancelled")

        self.assertTrue(capture.released)

    def test_open_video_capture_releases_failed_handle(self):
        capture = _FakeCapture(opened=False)
        source = media.VideoSource(
            value="missing.mp4",
            source_type=media.SOURCE_VIDEO,
            pace_reads=True,
        )

        with mock.patch.object(media.cv2, "VideoCapture", return_value=capture):
            with self.assertRaises(media.MediaSourceError):
                with media.open_video_capture(source):
                    pass

        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
