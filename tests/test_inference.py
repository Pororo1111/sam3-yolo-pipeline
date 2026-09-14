from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline import inference


class _EmptyCapture:
    def read(self):
        return False, None


class _SingleFrameCapture:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        if self.frame is None:
            return False, None
        frame, self.frame = self.frame, None
        return True, frame


class InferenceTests(unittest.TestCase):
    def setUp(self):
        inference._stop_event.clear()

    def test_default_detection_uses_elapsed_seconds_and_reuses_boxes(self):
        clock = [0.0]
        calls = []
        box = SimpleNamespace(xyxy=np.array([[1, 1, 5, 5]]), cls=np.array([0]), conf=np.array([0.9]), id=None)
        class Capture:
            def __init__(self):
                self.times = iter([0.0, 0.4, 0.9, 1.0, 1.7, 2.0])
            def read(self):
                try:
                    clock[0] = next(self.times)
                    return True, np.zeros((10, 10, 3), dtype=np.uint8)
                except StopIteration:
                    return False, None
        def model(_frame, **_kwargs):
            calls.append(clock[0])
            return [SimpleNamespace(boxes=[box])]
        with patch.object(inference.time, "perf_counter", side_effect=lambda: clock[0]):
            outputs = list(inference._predict_video(model, {0: "object"}, [0], Capture(), 0.5))
        self.assertEqual(calls, [0.0, 1.0, 2.0])
        previews = [output for output in outputs if output[0] is not None]
        self.assertGreater(len(previews), len(calls))
        self.assertTrue(all("감지 1개" in output[1] for output in previews))
        self.assertTrue(all("탐지 주기=1초" in output[1] for output in previews))

    def test_browser_webcam_timeout_reports_actionable_status(self):
        outputs = list(
            inference._predict_video(
                model=None,
                names={},
                class_ids=[],
                capture=_EmptyCapture(),
                conf=0.25,
                detection_interval=3,
                browser_webcam=True,
            )
        )

        self.assertEqual(outputs[0], (None, "추론 시작...", ""))
        self.assertIn("카메라 프레임을 받지 못했습니다", outputs[-1][1])
        self.assertNotIn("총 0프레임", outputs[-1][1])

    def test_browser_overlay_is_svg_without_a_video_frame_payload(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        box = SimpleNamespace(
            xyxy=np.array([[10, 20, 80, 90]], dtype=float),
            cls=np.array([0]),
            conf=np.array([0.91]),
            id=None,
        )

        svg = inference._browser_overlay_svg(frame, [box], {0: "person"})

        self.assertIn('viewBox="0 0 200 100"', svg)
        self.assertIn('x="10.0"', svg)
        self.assertIn('stroke-width="4.0"', svg)
        self.assertIn('font-size="28.0"', svg)
        self.assertIn("person 0.91", svg)

    def test_browser_webcam_sends_only_overlay_not_rendered_frame(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        box = SimpleNamespace(
            xyxy=np.array([[10, 20, 80, 90]], dtype=float),
            cls=np.array([0]),
            conf=np.array([0.91]),
            id=None,
        )

        class FakeModel:
            def __call__(self, _frame, **_kwargs):
                return [SimpleNamespace(boxes=[box])]

        outputs = list(
            inference._predict_video(
                model=FakeModel(),
                names={0: "person"},
                class_ids=[0],
                capture=_SingleFrameCapture(frame),
                conf=0.25,
                detection_interval=1,
                browser_webcam=True,
            )
        )

        frame_update = outputs[1]
        self.assertIsNone(frame_update[0])
        self.assertIn("감지 1개", frame_update[1])
        self.assertIn("<svg", frame_update[2])

if __name__ == "__main__":
    unittest.main()
