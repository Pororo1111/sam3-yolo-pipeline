from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline import vision


class VisionTests(unittest.TestCase):
    def test_draw_boxes_scales_stroke_and_text_for_display_resize(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        box = SimpleNamespace(
            xyxy=np.array([[100, 120, 600, 800]], dtype=float),
            cls=np.array([0]),
            conf=np.array([0.91]),
            id=None,
        )
        expected_scale = 1920 / vision.DISPLAY_MAX_WIDTH

        with (
            patch("pipeline.vision.cv2.rectangle") as rectangle,
            patch("pipeline.vision.cv2.putText") as put_text,
        ):
            vision.draw_boxes(frame, [box], {0: "person"})

        self.assertEqual(
            rectangle.call_args.args[-1],
            round(vision.DEFAULT_BOX_THICKNESS * expected_scale),
        )
        self.assertAlmostEqual(
            put_text.call_args.args[4],
            vision.DEFAULT_FONT_SCALE * expected_scale,
        )
        self.assertEqual(
            put_text.call_args.args[-1],
            round(vision.DEFAULT_TEXT_THICKNESS * expected_scale),
        )


if __name__ == "__main__":
    unittest.main()
