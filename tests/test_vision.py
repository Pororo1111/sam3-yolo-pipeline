from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline import vision


class VisionTests(unittest.TestCase):
    def test_inference_class_ids_excludes_no_prefix(self):
        names = {
            0: "Person",
            1: "NO-Hardhat",
            2: "iiac_vest",
            3: " no-mask ",
        }

        self.assertEqual(vision.inference_class_ids(names), [0, 2])

    def test_person_containing_iiac_detection_is_displayed_as_worker(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [
            SimpleNamespace(
                xyxy=np.array([[10, 10, 90, 90]], dtype=float),
                cls=np.array([0]),
                conf=np.array([0.95]),
                id=None,
            ),
            SimpleNamespace(
                xyxy=np.array([[30, 20, 50, 40]], dtype=float),
                cls=np.array([1]),
                conf=np.array([0.88]),
                id=None,
            ),
        ]

        with patch("pipeline.vision.cv2.putText") as put_text:
            vision.draw_boxes(frame, boxes, {0: "Person", 1: "iiac_helmet"})

        self.assertTrue(put_text.call_args_list[0].args[1].startswith("worker "))
        self.assertTrue(put_text.call_args_list[1].args[1].startswith("iiac_helmet "))

    def test_person_containing_hardhat_is_displayed_as_worker(self):
        detections = [
            ("Person", (0.1, 0.1, 0.9, 1.0)),
            ("Hardhat", (0.35, 0.1, 0.6, 0.3)),
        ]

        self.assertEqual(vision.worker_person_indexes(detections), {0})

    def test_no_hardhat_does_not_mark_person_as_worker(self):
        detections = [
            ("Person", (0.1, 0.1, 0.9, 1.0)),
            ("NO-Hardhat", (0.35, 0.1, 0.6, 0.3)),
        ]

        self.assertEqual(vision.worker_person_indexes(detections), set())

    def test_person_without_contained_iiac_keeps_original_label(self):
        detections = [
            ("person", (0.0, 0.0, 0.4, 1.0)),
            ("iiac_vest", (0.6, 0.2, 0.8, 0.5)),
        ]

        self.assertEqual(vision.worker_person_indexes(detections), set())

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
