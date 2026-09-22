from __future__ import annotations

import json
import threading
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

    def test_dropdown_puts_server_cameras_before_browser_cameras(self):
        payload = json.dumps(
            {
                "devices": [
                    {"id": "front-id", "label": "Front Camera"},
                    {"id": "back-id", "label": "Back Camera"},
                ]
            }
        )
        with mock.patch.object(
            webcams,
            "list_webcams",
            return_value=[("카메라 0", "0")],
        ):
            choices, value = webcams.refresh_webcam_dropdown(payload, "capture")

        self.assertEqual(value, "0")
        self.assertEqual(choices[0], ("서버 · 카메라 0", "0"))
        self.assertEqual([label for label, _value in choices[1:]], [
            "접속 기기 · Front Camera",
            "접속 기기 · Back Camera",
        ])
        parsed = webcams.parse_browser_webcam_value(choices[1][1], "session-a")
        self.assertEqual(parsed.channel, "capture")
        self.assertEqual(parsed.device_id, "front-id")

    def test_browser_capture_receives_new_rgb_frame_as_bgr(self):
        session_id = webcams.create_browser_session()
        value = webcams.browser_webcam_value("inference", "phone-camera")
        source = webcams.parse_browser_webcam_value(value, session_id)
        capture = webcams.open_browser_webcam(source)
        rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)

        sender = threading.Timer(
            0.02,
            webcams.push_browser_frame,
            args=(session_id, value, rgb),
        )
        sender.start()
        try:
            ok, frame_bgr = capture.read()
        finally:
            sender.join()
            capture.release()
            webcams.delete_browser_session(session_id)

        self.assertTrue(ok)
        np.testing.assert_array_equal(
            frame_bgr,
            np.array([[[30, 20, 10]]], dtype=np.uint8),
        )

    def test_browser_payload_deduplicates_and_limits_untrusted_devices(self):
        payload = json.dumps(
            {
                "devices": [
                    {"id": "same", "label": "First"},
                    {"id": "same", "label": "Duplicate"},
                    {"id": "other", "label": ""},
                ],
                "error": "permission warning",
            }
        )
        devices, error = webcams.parse_browser_devices_payload(payload)
        self.assertEqual(devices, [("same", "First"), ("other", "카메라 3")])
        self.assertEqual(error, "permission warning")


if __name__ == "__main__":
    unittest.main()
