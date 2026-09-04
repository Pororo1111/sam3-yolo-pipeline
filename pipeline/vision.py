"""OpenCV 프레임을 Gradio 표시 형식으로 변환하는 공용 유틸리티."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import cv2
import numpy as np


Color = tuple[int, int, int]
ColorProvider = Callable[[int], Color]
Detection = tuple[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class BoxAnnotation:
    """프레임에 표시할 하나의 감지 박스 정보."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    color: Color

DISPLAY_MAX_WIDTH = 854
DEFAULT_BOX_THICKNESS = 4
DEFAULT_FONT_SCALE = 0.9
DEFAULT_TEXT_THICKNESS = 3

_PALETTE: tuple[Color, ...] = (
    (255, 100, 0),
    (0, 220, 255),
    (0, 200, 0),
    (100, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (200, 200, 0),
    (128, 0, 128),
)


def class_color(class_id: int) -> Color:
    return _PALETTE[class_id % len(_PALETTE)]


def inference_class_ids(names: Mapping) -> list[int]:
    """추론 결과에 포함할 클래스 ID를 반환한다.

    ``NO-`` 접두어 클래스는 PPE 미착용 보조 라벨이므로 추론·침입 감시에서
    제외한다. 대소문자와 이름 앞뒤 공백은 무시한다.
    """

    return sorted(
        int(class_id)
        for class_id, class_name in names.items()
        if not str(class_name).strip().casefold().startswith("no-")
    )


def worker_person_indexes(detections: Iterable[Detection]) -> set[int]:
    """보호구를 착용한 ``person`` 인덱스를 반환한다.

    보호구 bbox 전체가 사람 bbox 안에 꼭 들어맞지 않는 실제 검출 오차를 고려해
    ``iiac...`` 또는 ``Hardhat`` 검출 bbox의 중심점 포함 여부를 사용한다.
    """

    items = list(detections)
    equipment_centers = []
    for class_name, (x1, y1, x2, y2) in items:
        normalized_name = str(class_name).strip().casefold()
        if normalized_name.startswith("iiac") or normalized_name == "hardhat":
            equipment_centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    workers: set[int] = set()
    for index, (class_name, (x1, y1, x2, y2)) in enumerate(items):
        if str(class_name).strip().casefold() != "person":
            continue
        if any(
            x1 <= cx <= x2 and y1 <= cy <= y2
            for cx, cy in equipment_centers
        ):
            workers.add(index)
    return workers


def annotation_scale(frame: np.ndarray, max_width: int = DISPLAY_MAX_WIDTH) -> float:
    """표시 단계의 축소를 감안한 오버레이 배율을 반환한다."""

    width = frame.shape[1]
    return max(1.0, width / max(1, int(max_width)))


def box_annotations(
    boxes: Iterable | None,
    names: Mapping,
    *,
    color_provider: ColorProvider = class_color,
) -> list[BoxAnnotation]:
    """Ultralytics 결과를 OpenCV와 브라우저가 공유하는 표시 정보로 변환한다."""

    if boxes is None:
        return []

    box_list = list(boxes)
    detections = []
    for box in box_list:
        class_id = int(box.cls[0])
        detections.append(
            (str(names.get(class_id, class_id)), tuple(map(float, box.xyxy[0])))
        )
    worker_indexes = worker_person_indexes(detections)

    annotations = []
    for index, box in enumerate(box_list):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        track_id = getattr(box, "id", None)
        if track_id is not None:
            try:
                value = track_id[0]
                if hasattr(value, "item"):
                    value = value.item()
                track_id = int(value)
            except (TypeError, ValueError, IndexError, RuntimeError):
                track_id = None
        track_label = f" #{track_id}" if track_id is not None else ""
        class_label = "worker" if index in worker_indexes else names.get(class_id, class_id)
        annotations.append(
            BoxAnnotation(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=f"{class_label}{track_label} {confidence:.2f}",
                color=color_provider(class_id),
            )
        )
    return annotations


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: Iterable | None,
    names: Mapping,
    *,
    color_provider: ColorProvider = class_color,
    font_scale: float = DEFAULT_FONT_SCALE,
    box_thickness: int = DEFAULT_BOX_THICKNESS,
    text_thickness: int = DEFAULT_TEXT_THICKNESS,
) -> np.ndarray:
    """Ultralytics box 결과를 BGR 프레임에 그린다."""

    annotated = frame_bgr.copy()
    scale = annotation_scale(frame_bgr)
    scaled_font = float(font_scale) * scale
    scaled_box_thickness = max(1, round(int(box_thickness) * scale))
    scaled_text_thickness = max(1, round(int(text_thickness) * scale))
    label_offset = max(8, round(8 * scale))
    minimum_baseline = max(24, round(24 * scale))

    for annotation in box_annotations(
        boxes,
        names,
        color_provider=color_provider,
    ):
        x1, y1, x2, y2 = map(
            int,
            (annotation.x1, annotation.y1, annotation.x2, annotation.y2),
        )
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            annotation.color,
            scaled_box_thickness,
        )
        cv2.putText(
            annotated,
            annotation.label,
            (x1, max(y1 - label_offset, minimum_baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scaled_font,
            annotation.color,
            scaled_text_thickness,
        )
    return annotated


def resize_for_display(
    frame: np.ndarray,
    max_width: int = DISPLAY_MAX_WIDTH,
) -> np.ndarray:
    """웹소켓 전송량을 제한하도록 큰 프레임만 비율 유지 축소한다."""

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def to_rgb(frame_bgr: np.ndarray, max_width: int = DISPLAY_MAX_WIDTH) -> np.ndarray:
    """BGR 프레임을 Gradio ``gr.Image(type="numpy")`` 출력으로 변환한다."""

    display_frame = resize_for_display(frame_bgr, max_width=max_width)
    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
