from __future__ import annotations

import unittest
from types import SimpleNamespace

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
    def test_browser_webcam_timeout_reports_actionable_status(self):
        outputs = list(
            inference._predict_video(
                model=None,
                names={},
                class_ids=[],
                capture=_EmptyCapture(),
                conf=0.25,
                infer_every=3,
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
                infer_every=1,
                browser_webcam=True,
            )
        )

        frame_update = outputs[1]
        self.assertIsNone(frame_update[0])
        self.assertIn("감지 1개", frame_update[1])
        self.assertIn("<svg", frame_update[2])

if __name__ == "__main__":
    unittest.main()
