"""Tab 7: 세션별 수동/ByteTrack 추적 영역 침입 감시."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline import media, models, vision


MODE_MANUAL = "manual"
MODE_TRACKED = "tracked"
TRACKER_CONFIG = "bytetrack.yaml"
_TRACK_CONFIDENCE = 0.1
_DISPLAY_INTERVAL = 1.0 / 15
_DETECTION_BOX_THICKNESS = 4
_DETECTION_FONT_SCALE = 0.9
_DETECTION_TEXT_THICKNESS = 3


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    class_id: int
    confidence: float
    class_name: str
    xyxy: tuple[float, float, float, float]

    @property
    def anchor(self) -> tuple[float, float]:
        """라바콘이 지면에 닿는 bbox 바닥 중앙 정규화 좌표."""

        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class ZoneRuntime:
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    run_id: str = ""
    zones: list[dict] = field(default_factory=list)
    draft_points: list[tuple[float, float]] = field(default_factory=list)
    draft_anchors: list[dict] = field(default_factory=list)
    last_frame: np.ndarray | None = None
    last_tracks: list[TrackObservation] = field(default_factory=list)
    edit_frame: np.ndarray | None = None
    edit_tracks: list[TrackObservation] = field(default_factory=list)


_sessions: dict[str, ZoneRuntime] = {}
_sessions_lock = threading.Lock()


def create_session() -> str:
    session_id = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = ZoneRuntime()
    return session_id


def delete_session(session_id: str) -> None:
    with _sessions_lock:
        runtime = _sessions.pop(str(session_id), None)
    if runtime is not None:
        runtime.stop_event.set()


def _runtime(session_id: str) -> ZoneRuntime:
    key = str(session_id or "").strip()
    if not key:
        raise ValueError("침입 감지 세션이 없습니다. 페이지를 새로고침하세요.")
    with _sessions_lock:
        runtime = _sessions.get(key)
        if runtime is None:
            runtime = ZoneRuntime()
            _sessions[key] = runtime
        return runtime


def _scalar(value, default=0):
    if value is None:
        return default
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return value.item()
        array = np.asarray(value).reshape(-1)
        return array[0].item() if array.size else default
    except Exception:
        return default


def _observations_from_boxes(boxes, names: dict, frame_shape) -> list[TrackObservation]:
    if boxes is None:
        return []
    height, width = frame_shape[:2]
    observations: list[TrackObservation] = []
    for box in boxes:
        try:
            xyxy = box.xyxy[0]
            if hasattr(xyxy, "detach"):
                xyxy = xyxy.detach()
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu()
            x1, y1, x2, y2 = map(float, np.asarray(xyxy).reshape(-1)[:4])
            class_id = int(_scalar(getattr(box, "cls", None), 0))
            confidence = float(_scalar(getattr(box, "conf", None), 0.0))
            track_id = int(_scalar(getattr(box, "id", None), -1))
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    class_id=class_id,
                    confidence=confidence,
                    class_name=str(names.get(class_id, class_id)),
                    xyxy=(
                        float(np.clip(x1 / width, 0.0, 1.0)),
                        float(np.clip(y1 / height, 0.0, 1.0)),
                        float(np.clip(x2 / width, 0.0, 1.0)),
                        float(np.clip(y2 / height, 0.0, 1.0)),
                    ),
                )
            )
        except (TypeError, ValueError, IndexError):
            continue
    worker_indexes = vision.worker_person_indexes(
        (item.class_name, item.xyxy) for item in observations
    )
    return [
        TrackObservation(
            track_id=item.track_id,
            class_id=item.class_id,
            confidence=item.confidence,
            class_name="woker" if index in worker_indexes else item.class_name,
            xyxy=item.xyxy,
        )
        for index, item in enumerate(observations)
    ]


def _point_pixels(point, width: int, height: int) -> tuple[int, int]:
    return (
        int(round(float(point[0]) * max(width - 1, 1))),
        int(round(float(point[1]) * max(height - 1, 1))),
    )


def _polygon_pixels(points, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [_point_pixels(point, width, height) for point in points],
        dtype=np.int32,
    )


def _zone_points(zone: dict) -> list[tuple[float, float]]:
    if zone.get("mode") == MODE_TRACKED:
        return [tuple(anchor["point"]) for anchor in zone.get("anchors", [])]
    return [tuple(point) for point in zone.get("points", [])]


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    x = array[:, 0]
    y = array[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def _objects_in_zone(
    observations: list[TrackObservation],
    polygon: np.ndarray,
    excluded_track_ids: set[int],
    width: int,
    height: int,
) -> set[tuple[str, int]]:
    objects: set[tuple[str, int]] = set()
    for index, observation in enumerate(observations):
        if observation.track_id >= 0 and observation.track_id in excluded_track_ids:
            continue
        cx, cy = observation.center
        point = (cx * max(width - 1, 1), cy * max(height - 1, 1))
        if cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0:
            key = (
                ("track", observation.track_id)
                if observation.track_id >= 0
                else ("detection", index)
            )
            objects.add(key)
    return objects


def _zone_box_color(class_id: int) -> tuple[int, int, int]:
    return (
        (class_id * 67 + 100) % 256,
        (class_id * 113 + 50) % 256,
        (class_id * 41 + 200) % 256,
    )


def _update_tracked_zones_locked(
    runtime: ZoneRuntime,
    observations: list[TrackObservation],
) -> None:
    by_id = {
        observation.track_id: observation
        for observation in observations
        if observation.track_id >= 0
    }
    for zone in runtime.zones:
        if zone.get("mode") != MODE_TRACKED:
            continue
        for anchor in zone.get("anchors", []):
            observation = by_id.get(int(anchor.get("track_id", -1)))
            if observation is None:
                anchor["missing"] = int(anchor.get("missing", 0)) + 1
                continue
            expected_class_id = anchor.get("class_id")
            if (
                expected_class_id is not None
                and int(expected_class_id) != observation.class_id
            ):
                # ByteTrack을 재시작하거나 장면이 급변하면 ID가 다른 클래스에
                # 재할당될 수 있다. 라바콘 앵커가 사람 등을 따라가지 않게 한다.
                anchor["missing"] = int(anchor.get("missing", 0)) + 1
                continue
            anchor["point"] = list(observation.anchor)
            if expected_class_id is None:
                anchor["class_id"] = observation.class_id
                anchor["class_name"] = observation.class_name
            anchor["missing"] = 0


def _invalidate_tracked_zones_locked(runtime: ZoneRuntime) -> int:
    """시간축이 끊긴 입력 경계에서 재사용될 수 있는 Track ID를 폐기한다."""

    before = len(runtime.zones)
    runtime.zones = [
        zone for zone in runtime.zones if zone.get("mode") != MODE_TRACKED
    ]
    runtime.draft_anchors.clear()
    runtime.last_tracks.clear()
    runtime.edit_tracks.clear()
    return before - len(runtime.zones)


def _render_zone_list(
    annotated: np.ndarray,
    zones: list[dict],
    observations: list[TrackObservation],
) -> tuple[np.ndarray, int, int, int]:
    """영역을 그리고 (프레임, 침입수, 영역수, 유실 anchor수)를 반환한다."""

    height, width = annotated.shape[:2]
    all_intruders: set[tuple[str, int]] = set()
    total_missing = 0
    # 영역 경계용 라바콘은 겹치는 다른 영역에서도 침입 객체로 세지 않는다.
    all_anchor_ids = {
        int(anchor.get("track_id", -1))
        for zone in zones
        if zone.get("mode") == MODE_TRACKED
        for anchor in zone.get("anchors", [])
        if int(anchor.get("track_id", -1)) >= 0
    }
    for zone in zones:
        points = _zone_points(zone)
        if len(points) < 3:
            continue
        polygon = _polygon_pixels(points, width, height)
        anchors = zone.get("anchors", []) if zone.get("mode") == MODE_TRACKED else []
        missing = sum(1 for anchor in anchors if int(anchor.get("missing", 0)) > 0)
        total_missing += missing
        zone_intruders = _objects_in_zone(
            observations,
            polygon,
            all_anchor_ids,
            width,
            height,
        )
        count = len(zone_intruders)
        all_intruders.update(zone_intruders)

        if count > 0:
            color = (0, 0, 220)
        elif missing > 0:
            color = (0, 165, 255)
        else:
            color = (0, 220, 80)
        thickness = 3 if count > 0 else 2

        if count > 0:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], (0, 0, 180))
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        cv2.polylines(annotated, [polygon], True, color, thickness)
        for anchor, point in zip(anchors, polygon):
            anchor_color = (
                (0, 165, 255) if int(anchor.get("missing", 0)) > 0 else (255, 220, 0)
            )
            cv2.circle(annotated, tuple(point), 6, anchor_color, -1)
            cv2.putText(
                annotated,
                f"#{anchor.get('track_id', '?')}",
                (int(point[0]) + 5, max(int(point[1]) - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                anchor_color,
                1,
            )

        label = f"{zone.get('label', 'zone')} ({count})"
        if missing:
            label += f" lost:{missing}"
        tx = int(polygon[0][0])
        ty = max(int(polygon[0][1]) - 8, 0)
        cv2.putText(
            annotated,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    return annotated, len(all_intruders), len(zones), total_missing


def _render_zones(
    runtime: ZoneRuntime,
    annotated: np.ndarray,
    observations: list[TrackObservation],
):
    with runtime.lock:
        zones = copy.deepcopy(runtime.zones)
    return _render_zone_list(annotated, zones, observations)


def _draw_track_observations(
    frame: np.ndarray,
    observations: list[TrackObservation],
) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = vision.annotation_scale(frame)
    box_thickness = max(1, round(_DETECTION_BOX_THICKNESS * scale))
    font_scale = _DETECTION_FONT_SCALE * scale
    text_thickness = max(1, round(_DETECTION_TEXT_THICKNESS * scale))
    label_offset = max(8, round(8 * scale))
    minimum_baseline = max(24, round(24 * scale))
    for observation in observations:
        x1, y1 = _point_pixels(observation.xyxy[:2], width, height)
        x2, y2 = _point_pixels(observation.xyxy[2:], width, height)
        color = _zone_box_color(observation.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        track = f" #{observation.track_id}" if observation.track_id >= 0 else ""
        cv2.putText(
            frame,
            f"{observation.class_name}{track} {observation.confidence:.2f}",
            (x1, max(y1 - label_offset, minimum_baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
        )
    return frame


def _render_editor_locked(runtime: ZoneRuntime) -> np.ndarray | None:
    if runtime.edit_frame is None:
        return None
    frame = runtime.edit_frame.copy()
    frame = _draw_track_observations(frame, runtime.edit_tracks)
    frame, _, _, _ = _render_zone_list(frame, runtime.zones, runtime.edit_tracks)
    height, width = frame.shape[:2]

    if runtime.draft_points:
        points = _polygon_pixels(runtime.draft_points, width, height)
        if len(points) >= 2:
            cv2.polylines(frame, [points], False, (0, 255, 255), 2)
        for index, point in enumerate(points, start=1):
            cv2.circle(frame, tuple(point), 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                str(index),
                (int(point[0]) + 6, int(point[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

    if runtime.draft_anchors:
        points = _polygon_pixels(
            [anchor["point"] for anchor in runtime.draft_anchors],
            width,
            height,
        )
        if len(points) >= 2:
            cv2.polylines(frame, [points], False, (255, 220, 0), 2)
        for anchor, point in zip(runtime.draft_anchors, points):
            cv2.circle(frame, tuple(point), 7, (255, 220, 0), -1)
            cv2.putText(
                frame,
                f"#{anchor['track_id']}",
                (int(point[0]) + 6, int(point[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2,
            )
    return vision.to_rgb(frame)


def _auto_anchor_candidates(
    observations: list[TrackObservation],
) -> tuple[list[dict], str | None]:
    """같은 클래스 Track 중 라바콘 후보를 골라 볼록 다각형 순서로 반환한다."""

    grouped: dict[int, list[TrackObservation]] = {}
    for observation in observations:
        if observation.track_id >= 0:
            grouped.setdefault(observation.class_id, []).append(observation)

    eligible = [
        items
        for items in grouped.values()
        if len(items) >= 3
        and "safety cone"
        in items[0].class_name.casefold().replace("_", " ").replace("-", " ")
    ]
    if not eligible:
        return [], None

    def group_score(items: list[TrackObservation]):
        mean_confidence = sum(item.confidence for item in items) / len(items)
        return (len(items), mean_confidence)

    selected = max(eligible, key=group_score)
    points = np.asarray([item.anchor for item in selected], dtype=np.float32)
    hull_indices = cv2.convexHull(points, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return [], selected[0].class_name

    anchors = []
    for index in hull_indices.reshape(-1):
        observation = selected[int(index)]
        anchors.append(
            {
                "track_id": observation.track_id,
                "class_id": observation.class_id,
                "class_name": observation.class_name,
                "point": list(observation.anchor),
                "missing": 0,
            }
        )
    return anchors, selected[0].class_name


def _ensure_auto_cone_zone_locked(
    runtime: ZoneRuntime,
    observations: list[TrackObservation],
) -> None:
    """Safety Cone Track 3개 이상을 실시간 자동 추적 영역으로 유지한다."""

    anchors, _ = _auto_anchor_candidates(observations)
    if len(anchors) < 3:
        return

    tracked_zone = next(
        (zone for zone in runtime.zones if zone.get("mode") == MODE_TRACKED),
        None,
    )
    if tracked_zone is None:
        runtime.zones.append(
            {
                "id": uuid.uuid4().hex,
                "label": "Safety Cone zone",
                "mode": MODE_TRACKED,
                "anchors": anchors,
            }
        )
    else:
        tracked_zone["anchors"] = anchors


def auto_select_tracked_anchors(session_id: str, mode: str):
    """추적 모드 진입 시 현재 편집 프레임의 라바콘 경계를 자동 선택한다."""

    runtime = _runtime(session_id)
    with runtime.lock:
        if mode != MODE_TRACKED:
            runtime.draft_anchors.clear()
            return _render_editor_locked(runtime), "수동 다각형 모드입니다."
        if runtime.edit_frame is None:
            return None, "먼저 현재 프레임 가져오기를 눌러 편집 프레임을 고정하세요."

        runtime.draft_points.clear()
        runtime.draft_anchors, class_name = _auto_anchor_candidates(
            runtime.edit_tracks
        )
        count = len(runtime.draft_anchors)
        image = _render_editor_locked(runtime)
        if count < 3:
            runtime.draft_anchors.clear()
            return image, (
                "동일 클래스의 Track ID 객체가 3개 이상 필요합니다. "
                "라바콘이 모두 검출된 프레임에서 다시 시도하세요."
            )
        return image, (
            f"{class_name} 라바콘 Track {count}개를 자동 선택했습니다. "
            "경계를 확인한 뒤 다각형 완료를 누르세요."
        )


def capture_editor_frame(session_id: str, mode: str = MODE_MANUAL):
    """현재 영상과 Track 목록을 고정해 안전한 클릭 편집 프레임을 만든다."""

    runtime = _runtime(session_id)
    with runtime.lock:
        if runtime.last_frame is None:
            return None, "먼저 스트림을 시작해 프레임을 표시하세요."
        runtime.edit_frame = runtime.last_frame.copy()
        runtime.edit_tracks = list(runtime.last_tracks)
        runtime.draft_points.clear()
        runtime.draft_anchors.clear()
        track_count = sum(1 for item in runtime.edit_tracks if item.track_id >= 0)
        if mode == MODE_TRACKED:
            runtime.draft_anchors, class_name = _auto_anchor_candidates(
                runtime.edit_tracks
            )
            image = _render_editor_locked(runtime)
            if len(runtime.draft_anchors) >= 3:
                return image, (
                    f"{class_name} 라바콘 Track {len(runtime.draft_anchors)}개를 "
                    "자동 선택했습니다. 경계를 확인한 뒤 다각형 완료를 누르세요."
                )
        image = _render_editor_locked(runtime)
    return image, (
        f"편집 프레임을 고정했습니다. Track ID가 있는 객체 {track_count}개 · "
        "모드를 고르고 꼭짓점을 순서대로 클릭하세요."
    )


def _event_point(event_index, frame: np.ndarray) -> tuple[float, float]:
    if not isinstance(event_index, (list, tuple)) or len(event_index) < 2:
        raise ValueError("이미지 클릭 좌표를 읽을 수 없습니다.")
    display = vision.resize_for_display(frame)
    height, width = display.shape[:2]
    x = float(event_index[0])
    y = float(event_index[1])
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError("이미지 내부를 클릭하세요.")
    return (
        float(np.clip(x / max(width - 1, 1), 0.0, 1.0)),
        float(np.clip(y / max(height - 1, 1), 0.0, 1.0)),
    )


def _find_clicked_track(
    point: tuple[float, float],
    observations: list[TrackObservation],
) -> TrackObservation | None:
    x, y = point
    containing = []
    for observation in observations:
        if observation.track_id < 0:
            continue
        x1, y1, x2, y2 = observation.xyxy
        if x1 <= x <= x2 and y1 <= y <= y2:
            containing.append(((x2 - x1) * (y2 - y1), observation))
    if containing:
        return min(containing, key=lambda item: item[0])[1]
    return None


def select_editor_point(session_id: str, mode: str, event_index):
    runtime = _runtime(session_id)
    with runtime.lock:
        if runtime.edit_frame is None:
            return None, "먼저 ‘현재 프레임 가져오기’를 누르세요."
        try:
            point = _event_point(event_index, runtime.edit_frame)
        except ValueError as exc:
            return _render_editor_locked(runtime), str(exc)

        if mode == MODE_TRACKED:
            observation = _find_clicked_track(point, runtime.edit_tracks)
            if observation is None:
                return (
                    _render_editor_locked(runtime),
                    "Track ID가 있는 검출 bbox 안을 클릭하세요. 추론 간격 1을 권장합니다.",
                )
            if any(
                int(anchor["track_id"]) == observation.track_id
                for anchor in runtime.draft_anchors
            ):
                return (
                    _render_editor_locked(runtime),
                    f"Track #{observation.track_id}은(는) 이미 선택했습니다.",
                )
            if (
                runtime.draft_anchors
                and int(runtime.draft_anchors[0]["class_id"]) != observation.class_id
            ):
                expected = runtime.draft_anchors[0]["class_name"]
                return (
                    _render_editor_locked(runtime),
                    f"첫 앵커와 같은 클래스({expected})만 선택할 수 있습니다.",
                )
            runtime.draft_anchors.append(
                {
                    "track_id": observation.track_id,
                    "class_id": observation.class_id,
                    "class_name": observation.class_name,
                    "point": list(observation.anchor),
                    "missing": 0,
                }
            )
            status = (
                f"라바콘/앵커 {len(runtime.draft_anchors)}개 선택 · "
                f"{observation.class_name} Track #{observation.track_id}"
            )
        else:
            runtime.draft_points.append(point)
            status = f"수동 꼭짓점 {len(runtime.draft_points)}개 선택"
        return _render_editor_locked(runtime), status


def undo_draft_point(session_id: str, mode: str):
    runtime = _runtime(session_id)
    with runtime.lock:
        draft = runtime.draft_anchors if mode == MODE_TRACKED else runtime.draft_points
        if draft:
            draft.pop()
        return _render_editor_locked(runtime), f"현재 선택: {len(draft)}개"


def clear_draft(session_id: str):
    runtime = _runtime(session_id)
    with runtime.lock:
        runtime.draft_points.clear()
        runtime.draft_anchors.clear()
        return _render_editor_locked(runtime), "작성 중인 꼭짓점을 모두 지웠습니다."


def finish_draft(session_id: str, mode: str, label: str):
    runtime = _runtime(session_id)
    with runtime.lock:
        if mode == MODE_TRACKED:
            if len(runtime.draft_anchors) < 3:
                return _render_editor_locked(runtime), "서로 다른 Track 객체를 3개 이상 선택하세요."
            points = [tuple(anchor["point"]) for anchor in runtime.draft_anchors]
        else:
            if len(runtime.draft_points) < 3:
                return _render_editor_locked(runtime), "꼭짓점을 3개 이상 선택하세요."
            points = list(runtime.draft_points)

        if _polygon_area(points) < 0.0001:
            return _render_editor_locked(runtime), "다각형 면적이 너무 작습니다. 점을 다시 선택하세요."

        zone_number = len(runtime.zones) + 1
        zone_label = (label or "").strip() or (
            f"Tracked zone {zone_number}"
            if mode == MODE_TRACKED
            else f"Manual zone {zone_number}"
        )
        if mode == MODE_TRACKED:
            runtime.zones.append(
                {
                    "id": uuid.uuid4().hex,
                    "label": zone_label,
                    "mode": MODE_TRACKED,
                    "anchors": copy.deepcopy(runtime.draft_anchors),
                }
            )
            runtime.draft_anchors.clear()
        else:
            runtime.zones.append(
                {
                    "id": uuid.uuid4().hex,
                    "label": zone_label,
                    "mode": "fixed",
                    "points": [list(point) for point in runtime.draft_points],
                }
            )
            runtime.draft_points.clear()
        return _render_editor_locked(runtime), f"영역 설정 완료: {zone_label}"


def clear_zones(session_id: str):
    runtime = _runtime(session_id)
    with runtime.lock:
        runtime.zones.clear()
        runtime.draft_points.clear()
        runtime.draft_anchors.clear()
        return _render_editor_locked(runtime), "모든 감시 영역을 지웠습니다."


def stop(session_id: str) -> None:
    _runtime(session_id).stop_event.set()


def reset(session_id: str):
    runtime = _runtime(session_id)
    runtime.stop_event.set()
    with runtime.lock:
        runtime.run_id = uuid.uuid4().hex
        runtime.zones.clear()
        runtime.draft_points.clear()
        runtime.draft_anchors.clear()
        runtime.last_frame = None
        runtime.last_tracks.clear()
        runtime.edit_frame = None
        runtime.edit_tracks.clear()
    return (
        None,
        None,
        "스트림 중지 / 자동 라바콘 영역 초기화",
        "수동 영역 편집을 초기화했습니다.",
    )


def prepare_stream(session_id: str):
    """새 스트림 전에 이전 작업과 화면/영역 상태를 모두 무효화한다."""

    runtime = _runtime(session_id)
    runtime.stop_event.set()
    with runtime.lock:
        runtime.run_id = uuid.uuid4().hex
        runtime.zones.clear()
        runtime.draft_points.clear()
        runtime.draft_anchors.clear()
        runtime.last_frame = None
        runtime.last_tracks.clear()
        runtime.edit_frame = None
        runtime.edit_tracks.clear()
    return (
        None,
        None,
        "Safety Cone 자동 추적 스트림을 준비합니다.",
        "수동 영역 편집을 초기화했습니다.",
    )


def _begin_stream(runtime: ZoneRuntime) -> tuple[str, threading.Event]:
    runtime.stop_event.set()
    current_event = threading.Event()
    run_id = uuid.uuid4().hex
    with runtime.lock:
        runtime.stop_event = current_event
        runtime.run_id = run_id
        # 새 소스/모델의 좌표와 Track ID는 이전 스트림과 호환되지 않는다.
        runtime.zones.clear()
        runtime.draft_points.clear()
        runtime.draft_anchors.clear()
        runtime.edit_frame = None
        runtime.last_frame = None
        runtime.last_tracks.clear()
        runtime.edit_tracks.clear()
    return run_id, current_event


def _is_current(runtime: ZoneRuntime, run_id: str, stop_event: threading.Event) -> bool:
    with runtime.lock:
        return (
            runtime.run_id == run_id
            and runtime.stop_event is stop_event
            and not stop_event.is_set()
        )


def _track_frame(model, frame_bgr: np.ndarray, visible_conf: float):
    results = model.track(
        frame_bgr,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=min(_TRACK_CONFIDENCE, max(0.01, float(visible_conf))),
        verbose=False,
    )
    return results[0].boxes if results and results[0].boxes is not None else None


def _update_tracking_and_latest(
    runtime: ZoneRuntime,
    run_id: str,
    stop_event: threading.Event,
    frame_bgr: np.ndarray,
    observations: list[TrackObservation],
) -> bool:
    with runtime.lock:
        if (
            runtime.run_id != run_id
            or runtime.stop_event is not stop_event
            or stop_event.is_set()
        ):
            return False
        _update_tracked_zones_locked(runtime, observations)
        _ensure_auto_cone_zone_locked(runtime, observations)
        runtime.last_frame = frame_bgr.copy()
        runtime.last_tracks = list(observations)
        return True


def _tracking_active(runtime: ZoneRuntime) -> bool:
    with runtime.lock:
        return bool(runtime.draft_anchors) or any(
            zone.get("mode") == MODE_TRACKED for zone in runtime.zones
        )


def stream(
    session_id: str,
    source_type: str,
    youtube_url: str,
    model_path: str,
    conf: float,
    infer_every: int,
    folder_files=None,
    webcam_index=None,
    video_file=None,
):
    runtime = _runtime(session_id)
    run_id, stop_event = _begin_stream(runtime)

    if not (model_path or "").strip():
        model_path = models.latest_trained_model()
    if model_path is None or not Path(model_path).is_file():
        yield None, f"모델 파일을 찾을 수 없습니다: {model_path}"
        return

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
    except Exception as exc:
        yield None, f"모델 로딩 실패: {exc}"
        return

    names = model.names or {}
    interval = max(1, int(infer_every))

    if source_type == media.SOURCE_IMAGES:
        try:
            yield from _stream_folder(
                runtime,
                run_id,
                stop_event,
                model,
                names,
                folder_files,
                float(conf),
                interval,
            )
        except GeneratorExit:
            raise
        except Exception as exc:
            if _is_current(runtime, run_id, stop_event):
                yield None, f"이미지 폴더 스트림 오류: {exc}"
        return

    try:
        source = media.resolve_video_source(
            source_type,
            youtube_url=youtube_url,
            webcam_index=webcam_index,
            video_file=video_file,
        )
    except media.MediaSourceError as exc:
        yield None, str(exc)
        return

    yield None, "ByteTrack 스트림 시작..."
    last_yield = 0.0
    frame_index = 0
    observations: list[TrackObservation] = []

    try:
        with media.open_video_capture(source) as capture:
            while (
                not stop_event.is_set()
                and _is_current(runtime, run_id, stop_event)
            ):
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                effective_interval = 1
                if frame_index % effective_interval == 0:
                    boxes = _track_frame(model, frame_bgr, float(conf))
                    observations = _observations_from_boxes(
                        boxes,
                        names,
                        frame_bgr.shape,
                    )
                    # 편집 스냅샷은 Track bbox와 정확히 같은 프레임이어야 한다.
                    if not _update_tracking_and_latest(
                        runtime,
                        run_id,
                        stop_event,
                        frame_bgr,
                        observations,
                    ):
                        break

                now = time.perf_counter()
                if now - last_yield < _DISPLAY_INTERVAL:
                    frame_index += 1
                    continue
                last_yield = now

                visible_observations = [
                    item for item in observations if item.confidence >= float(conf)
                ]
                annotated = _draw_track_observations(
                    frame_bgr.copy(),
                    visible_observations,
                )
                annotated, intruders, zone_count, missing = _render_zones(
                    runtime,
                    annotated,
                    visible_observations,
                )
                frame_index += 1
                status = (
                    f"프레임 {frame_index} | 영역: {zone_count}개 | "
                    f"침입 객체: {intruders}개 | ByteTrack skip={effective_interval}"
                )
                if missing:
                    status += f" | 추적 유실 anchor: {missing}개"
                if not _is_current(runtime, run_id, stop_event):
                    break
                yield vision.to_rgb(annotated), status

    except media.MediaSourceError as exc:
        yield None, str(exc)
    except GeneratorExit:
        raise
    except Exception as exc:
        yield None, f"스트림 오류: {exc}"

    if _is_current(runtime, run_id, stop_event):
        yield None, "스트림 종료"


def _reset_model_trackers(model) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        reset_method = getattr(tracker, "reset", None)
        if callable(reset_method):
            reset_method()


def _stream_folder(
    runtime: ZoneRuntime,
    run_id: str,
    stop_event: threading.Event,
    model,
    names: dict,
    folder_files,
    conf: float,
    infer_every: int,
):
    images = media.filter_image_paths(folder_files)
    if not images:
        yield None, "업로드된 이미지가 없습니다. 이미지 폴더를 업로드하세요."
        return

    yield None, (
        f"이미지 폴더 ByteTrack 감시 시작 — {len(images)}장 "
        "(시간순 프레임 폴더에서만 추적 ID가 유효합니다.)"
    )
    shown = 0
    frame_index = 0
    observations: list[TrackObservation] = []
    boundary_notice = ""
    while not stop_event.is_set() and _is_current(runtime, run_id, stop_event):
        for path in images:
            if stop_event.is_set() or not _is_current(runtime, run_id, stop_event):
                break
            frame_bgr = media.read_image(path)
            if frame_bgr is None:
                continue
            effective_interval = 1
            if frame_index % effective_interval == 0:
                boxes = _track_frame(model, frame_bgr, conf)
                observations = _observations_from_boxes(boxes, names, frame_bgr.shape)
                if not _update_tracking_and_latest(
                    runtime,
                    run_id,
                    stop_event,
                    frame_bgr,
                    observations,
                ):
                    break

            visible_observations = [
                item for item in observations if item.confidence >= conf
            ]
            annotated = _draw_track_observations(
                frame_bgr.copy(),
                visible_observations,
            )
            annotated, intruders, zone_count, missing = _render_zones(
                runtime,
                annotated,
                visible_observations,
            )
            shown += 1
            frame_index += 1
            status = (
                f"{path.name} | 영역: {zone_count}개 | 침입 객체: {intruders}개 "
                f"| ByteTrack skip={effective_interval} ({shown})"
            )
            if missing:
                status += f" | 추적 유실 anchor: {missing}개"
            if boundary_notice:
                status += f" | {boundary_notice}"
                boundary_notice = ""
            if not _is_current(runtime, run_id, stop_event):
                break
            yield vision.to_rgb(annotated), status
            stop_event.wait(0.4)
        # 폴더의 끝→처음은 실제 시간축이 아니며 ByteTrack ID가 재사용될 수 있다.
        # 고정 영역은 유지하되 추적 영역은 폐기해 다른 객체로 이동하지 않게 한다.
        _reset_model_trackers(model)
        with runtime.lock:
            invalidated = _invalidate_tracked_zones_locked(runtime)
        if invalidated:
            boundary_notice = (
                f"폴더 반복 경계에서 추적 영역 {invalidated}개를 해제했습니다. "
                "라바콘을 다시 선택하세요."
            )
        frame_index = 0
        observations = []

    if _is_current(runtime, run_id, stop_event):
        yield None, "스트림 종료"
