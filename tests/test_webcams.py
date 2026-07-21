from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline import webcams


class _FakeCapture:
    def __init__(self, opened: bool, frames=None):
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        self.released = True

    def set(self, _prop, _value):
        return True

    def get(self, _prop):
        return 30.0

    def getBackendName(self):
        return "fake"


class WebcamTests(unittest.TestCase):
    def test_missing_selection_is_not_silently_changed_to_zero(self):
        with self.assertRaisesRegex(webcams.WebcamOpenError, "선택"):
            webcams.coerce_webcam_index(None)

    def test_backend_fallback_preserves_warmup_frame(self):
        failed = _FakeCapture(False)
        frame = np.full((4, 5, 3), 7, dtype=np.uint8)
        working = _FakeCapture(True, [frame])

        with (
            mock.patch.object(webcams.platform, "system", return_value="Windows"),
            mock.patch.object(
                webcams.cv2,
                "VideoCapture",
                side_effect=[failed, working],
            ) as constructor,
        ):
            capture = webcams.open_webcam("2", warmup_timeout=0.1)
            ok, actual = capture.read()
            capture.release()

        self.assertEqual(constructor.call_count, 2)
        self.assertTrue(failed.released)
        self.assertTrue(ok)
        np.testing.assert_array_equal(actual, frame)
        self.assertTrue(working.released)

    def test_linux_high_device_indices_are_not_treated_as_index_limit(self):
        def fake_glob(path, pattern):
            if str(path).replace("\\", "/") == "/dev":
                return [Path("/dev/video2"), Path("/dev/video12")]
            return []

        with (
            mock.patch.object(webcams.platform, "system", return_value="Linux"),
            mock.patch.object(webcams.Path, "glob", new=fake_glob),
        ):
            self.assertEqual(webcams._candidate_indices(10), [2, 12])

    def test_linux_default_scan_keeps_camera_after_many_low_video_nodes(self):
        paths = [Path(f"/dev/video{index}") for index in range(10)]
        paths.append(Path("/dev/video12"))

        def fake_candidates(max_devices):
            self.assertEqual(max_devices, 64)
            return [int(path.name.removeprefix("video")) for path in paths]

        with (
            mock.patch.object(webcams.platform, "system", return_value="Linux"),
            mock.patch.object(webcams, "_candidate_indices", side_effect=fake_candidates),
            mock.patch.object(webcams, "webcam_in_use", return_value=True),
        ):
            choices = webcams.list_webcams()
        self.assertIn("12", [value for _, value in choices])


if __name__ == "__main__":
    unittest.main()
