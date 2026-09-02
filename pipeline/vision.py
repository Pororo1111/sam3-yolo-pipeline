"""OpenCV 프레임을 Gradio 표시 형식으로 변환하는 공용 유틸리티."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import cv2
import numpy as np


Color = tuple[int, int, int]
ColorProvider = Callable[[int], Color]

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


def annotation_scale(frame: np.ndarray, max_width: int = DISPLAY_MAX_WIDTH) -> float:
    """표시 단계의 축소를 감안한 오버레이 배율을 반환한다."""

    width = frame.shape[1]
    return max(1.0, width / max(1, int(max_width)))


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
    if boxes is None:
        return annotated

    scale = annotation_scale(frame_bgr)
    scaled_font = float(font_scale) * scale
    scaled_box_thickness = max(1, round(int(box_thickness) * scale))
    scaled_text_thickness = max(1, round(int(text_thickness) * scale))
    label_offset = max(8, round(8 * scale))
    minimum_baseline = max(24, round(24 * scale))

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        color = color_provider(class_id)
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
        label = f"{names.get(class_id, class_id)}{track_label} {confidence:.2f}"
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            scaled_box_thickness,
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - label_offset, minimum_baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scaled_font,
            color,
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
