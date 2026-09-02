from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pipeline import zone_monitor


class ZoneMonitorTests(unittest.TestCase):
    def setUp(self):
        self.session_id = zone_monitor.create_session()
        self.runtime = zone_monitor._runtime(self.session_id)
        with self.runtime.lock:
            self.runtime.edit_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def tearDown(self):
        zone_monitor.delete_session(self.session_id)

    def test_observation_marks_person_with_contained_iiac_as_worker(self):
        class Box:
            def __init__(self, xyxy, class_id, track_id):
                self.xyxy = np.array([xyxy], dtype=float)
                self.cls = np.array([class_id])
                self.conf = np.array([0.9])
                self.id = np.array([track_id])

        observations = zone_monitor._observations_from_boxes(
            [
                Box((10, 10, 90, 90), 0, 1),
                Box((30, 20, 50, 40), 1, 2),
            ],
            {0: "person", 1: "iiac_vest"},
            (100, 100, 3),
        )

        self.assertEqual(observations[0].class_name, "woker")
        self.assertEqual(observations[1].class_name, "iiac_vest")

    def test_manual_polygon_uses_normalized_click_coordinates(self):
        for point in ([10, 10], [90, 10], [50, 90]):
            _, status = zone_monitor.select_editor_point(
                self.session_id,
                zone_monitor.MODE_MANUAL,
                point,
            )
            self.assertIn("수동 꼭짓점", status)

        _, status = zone_monitor.finish_draft(
            self.session_id,
            zone_monitor.MODE_MANUAL,
            "Entrance",
        )
        self.assertIn("Entrance", status)
        with self.runtime.lock:
            zone = self.runtime.zones[0]
        self.assertEqual(zone["mode"], "fixed")
        self.assertAlmostEqual(zone["points"][0][0], 10 / 99)
        self.assertAlmostEqual(zone["points"][2][1], 90 / 99)

    def test_tracked_zone_moves_anchors_and_excludes_them_from_intruders(self):
        anchors = [
            {"track_id": 1, "class_id": 0, "class_name": "cone", "point": [0.1, 0.8], "missing": 0},
            {"track_id": 2, "class_id": 0, "class_name": "cone", "point": [0.5, 0.2], "missing": 0},
            {"track_id": 3, "class_id": 0, "class_name": "cone", "point": [0.9, 0.8], "missing": 0},
        ]
        with self.runtime.lock:
            self.runtime.zones = [
                {
                    "id": "zone",
                    "label": "Cone zone",
                    "mode": zone_monitor.MODE_TRACKED,
                    "anchors": anchors,
                }
            ]

        observations = [
            zone_monitor.TrackObservation(1, 0, 0.9, "cone", (0.05, 0.6, 0.15, 0.82)),
            zone_monitor.TrackObservation(2, 0, 0.9, "cone", (0.45, 0.1, 0.55, 0.22)),
            zone_monitor.TrackObservation(3, 0, 0.9, "cone", (0.85, 0.6, 0.95, 0.82)),
            zone_monitor.TrackObservation(50, 1, 0.9, "person", (0.45, 0.5, 0.55, 0.7)),
        ]
        with self.runtime.lock:
            zone_monitor._update_tracked_zones_locked(self.runtime, observations)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, intruders, zone_count, missing = zone_monitor._render_zones(
            self.runtime,
            frame,
            observations,
        )

        self.assertEqual(zone_count, 1)
        self.assertEqual(intruders, 1)
        self.assertEqual(missing, 0)
        with self.runtime.lock:
            self.assertAlmostEqual(self.runtime.zones[0]["anchors"][0]["point"][1], 0.82)

    def test_detection_boxes_and_labels_scale_for_display_resize(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        observation = zone_monitor.TrackObservation(
            7,
            1,
            0.9,
            "person",
            (0.1, 0.2, 0.5, 0.8),
        )
        expected_scale = 1920 / zone_monitor.vision.DISPLAY_MAX_WIDTH

        with (
            patch("pipeline.zone_monitor.cv2.rectangle") as rectangle,
            patch("pipeline.zone_monitor.cv2.putText") as put_text,
        ):
            zone_monitor._draw_track_observations(frame, [observation])

        self.assertEqual(
            rectangle.call_args.args[-1],
            round(zone_monitor._DETECTION_BOX_THICKNESS * expected_scale),
        )
        self.assertAlmostEqual(
            put_text.call_args.args[4],
            zone_monitor._DETECTION_FONT_SCALE * expected_scale,
        )
        self.assertEqual(
            put_text.call_args.args[-1],
            round(zone_monitor._DETECTION_TEXT_THICKNESS * expected_scale),
        )

    def test_tracked_mode_auto_selects_cone_tracks_in_polygon_order(self):
        with self.runtime.lock:
            self.runtime.edit_tracks = [
                zone_monitor.TrackObservation(1, 0, 0.9, "Safety Cone", (0.05, 0.6, 0.15, 0.8)),
                zone_monitor.TrackObservation(2, 0, 0.9, "Safety Cone", (0.45, 0.1, 0.55, 0.3)),
                zone_monitor.TrackObservation(3, 0, 0.9, "Safety Cone", (0.85, 0.6, 0.95, 0.8)),
                zone_monitor.TrackObservation(4, 1, 0.99, "person", (0.4, 0.4, 0.6, 0.9)),
            ]

        _, status = zone_monitor.auto_select_tracked_anchors(
            self.session_id,
            zone_monitor.MODE_TRACKED,
        )

        self.assertIn("3개를 자동 선택", status)
        with self.runtime.lock:
            self.assertEqual(
                {anchor["track_id"] for anchor in self.runtime.draft_anchors},
                {1, 2, 3},
            )
            points = [tuple(anchor["point"]) for anchor in self.runtime.draft_anchors]
        self.assertGreater(zone_monitor._polygon_area(points), 0.0001)

    def test_auto_selection_ignores_non_safety_cone_class(self):
        with self.runtime.lock:
            self.runtime.edit_tracks = [
                zone_monitor.TrackObservation(
                    track_id, 7, 0.8, "safety marker", xyxy
                )
                for track_id, xyxy in (
                    (10, (0.0, 0.6, 0.2, 0.8)),
                    (11, (0.4, 0.1, 0.6, 0.3)),
                    (12, (0.8, 0.6, 1.0, 0.8)),
                )
            ]

        zone_monitor.auto_select_tracked_anchors(
            self.session_id,
            zone_monitor.MODE_TRACKED,
        )
        with self.runtime.lock:
            self.assertEqual(self.runtime.draft_anchors, [])

    def test_stream_update_creates_and_refreshes_safety_cone_zone(self):
        run_id, stop_event = zone_monitor._begin_stream(self.runtime)
        observations = [
            zone_monitor.TrackObservation(1, 0, 0.9, "Safety Cone", (0.0, 0.6, 0.2, 0.8)),
            zone_monitor.TrackObservation(2, 0, 0.9, "Safety Cone", (0.4, 0.1, 0.6, 0.3)),
            zone_monitor.TrackObservation(3, 0, 0.9, "Safety Cone", (0.8, 0.6, 1.0, 0.8)),
        ]

        updated = zone_monitor._update_tracking_and_latest(
            self.runtime,
            run_id,
            stop_event,
            np.zeros((100, 100, 3), dtype=np.uint8),
            observations,
        )

        self.assertTrue(updated)
        with self.runtime.lock:
            self.assertEqual(len(self.runtime.zones), 1)
            self.assertEqual(self.runtime.zones[0]["label"], "Safety Cone zone")
            first_zone_id = self.runtime.zones[0]["id"]

        moved = [
            zone_monitor.TrackObservation(
                item.track_id,
                item.class_id,
                item.confidence,
                item.class_name,
                tuple(value + 0.01 for value in item.xyxy),
            )
            for item in observations
        ]
        zone_monitor._update_tracking_and_latest(
            self.runtime,
            run_id,
            stop_event,
            np.zeros((100, 100, 3), dtype=np.uint8),
            moved,
        )
        with self.runtime.lock:
            self.assertEqual(len(self.runtime.zones), 1)
            self.assertEqual(self.runtime.zones[0]["id"], first_zone_id)
            moved_by_id = {
                anchor["track_id"]: anchor["point"]
                for anchor in self.runtime.zones[0]["anchors"]
            }
            self.assertAlmostEqual(moved_by_id[1][0], 0.11)

    def test_track_id_reuse_by_another_class_marks_anchor_missing(self):
        anchor = {
            "track_id": 1,
            "class_id": 0,
            "class_name": "cone",
            "point": [0.1, 0.8],
            "missing": 0,
        }
        with self.runtime.lock:
            self.runtime.zones = [
                {
                    "id": "tracked",
                    "label": "Cone zone",
                    "mode": zone_monitor.MODE_TRACKED,
                    "anchors": [anchor],
                }
            ]
            zone_monitor._update_tracked_zones_locked(
                self.runtime,
                [
                    zone_monitor.TrackObservation(
                        1, 1, 0.95, "person", (0.7, 0.1, 0.9, 0.9)
                    )
                ],
            )

        self.assertEqual(anchor["class_name"], "cone")
        self.assertEqual(anchor["point"], [0.1, 0.8])
        self.assertEqual(anchor["missing"], 1)

    def test_folder_loop_boundary_invalidates_only_tracked_zones(self):
        with self.runtime.lock:
            self.runtime.zones = [
                {
                    "id": "tracked",
                    "label": "Tracked",
                    "mode": zone_monitor.MODE_TRACKED,
                    "anchors": [],
                },
                {
                    "id": "fixed",
                    "label": "Fixed",
                    "mode": "fixed",
                    "points": [[0, 0], [1, 0], [0, 1]],
                },
            ]
            self.runtime.draft_anchors = [{"track_id": 9}]
            invalidated = zone_monitor._invalidate_tracked_zones_locked(self.runtime)

        self.assertEqual(invalidated, 1)
        self.assertEqual([zone["id"] for zone in self.runtime.zones], ["fixed"])
        self.assertEqual(self.runtime.draft_anchors, [])

    def test_render_is_pure_and_overlapping_zones_count_unique_tracks(self):
        points_a = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]]
        points_b = [[0.2, 0.2], [0.9, 0.2], [0.9, 0.9], [0.2, 0.9]]
        with self.runtime.lock:
            self.runtime.zones = [
                {"id": "a", "label": "A", "mode": "fixed", "points": points_a},
                {"id": "b", "label": "B", "mode": "fixed", "points": points_b},
            ]
        observation = zone_monitor.TrackObservation(
            9,
            1,
            0.9,
            "person",
            (0.4, 0.4, 0.6, 0.7),
        )
        _, intruders, _, _ = zone_monitor._render_zones(
            self.runtime,
            np.zeros((100, 100, 3), dtype=np.uint8),
            [observation],
        )
        self.assertEqual(intruders, 1)

        with self.runtime.lock:
            self.runtime.zones = [
                {
                    "id": "tracked",
                    "label": "Tracked",
                    "mode": zone_monitor.MODE_TRACKED,
                    "anchors": [
                        {"track_id": 1, "point": [0.1, 0.1], "missing": 0},
                        {"track_id": 2, "point": [0.8, 0.1], "missing": 0},
                        {"track_id": 3, "point": [0.5, 0.8], "missing": 0},
                    ],
                }
            ]
        for _ in range(3):
            zone_monitor._render_zones(
                self.runtime,
                np.zeros((100, 100, 3), dtype=np.uint8),
                [],
            )
        with self.runtime.lock:
            self.assertEqual(
                [anchor["missing"] for anchor in self.runtime.zones[0]["anchors"]],
                [0, 0, 0],
            )

    def test_sessions_do_not_share_zones(self):
        other_id = zone_monitor.create_session()
        try:
            with self.runtime.lock:
                self.runtime.zones.append(
                    {"id": "one", "label": "One", "mode": "fixed", "points": []}
                )
            self.assertEqual(zone_monitor._runtime(other_id).zones, [])
        finally:
            zone_monitor.delete_session(other_id)

    def test_reset_prevents_old_stream_from_restoring_last_frame(self):
        run_id, stop_event = zone_monitor._begin_stream(self.runtime)
        zone_monitor.reset(self.session_id)
        updated = zone_monitor._update_tracking_and_latest(
            self.runtime,
            run_id,
            stop_event,
            np.ones((10, 10, 3), dtype=np.uint8),
            [],
        )
        self.assertFalse(updated)
        with self.runtime.lock:
            self.assertIsNone(self.runtime.last_frame)

    def test_prepare_stream_clears_old_fixed_zones_and_frames(self):
        with self.runtime.lock:
            self.runtime.last_frame = np.ones((5, 5, 3), dtype=np.uint8)
            self.runtime.edit_frame = self.runtime.last_frame.copy()
            self.runtime.zones = [
                {
                    "id": "old",
                    "label": "Old",
                    "mode": "fixed",
                    "points": [[0, 0], [1, 0], [0, 1]],
                }
            ]
        zone_monitor.prepare_stream(self.session_id)
        with self.runtime.lock:
            self.assertIsNone(self.runtime.last_frame)
            self.assertIsNone(self.runtime.edit_frame)
            self.assertEqual(self.runtime.zones, [])


if __name__ == "__main__":
    unittest.main()
