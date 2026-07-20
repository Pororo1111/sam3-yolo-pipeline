"""OpenCV 프레임을 Gradio 표시 형식으로 변환하는 공용 유틸리티."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import cv2
import numpy as np


Color = tuple[int, int, int]
ColorProvider = Callable[[int], Color]

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


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: Iterable | None,
    names: Mapping,
    *,
    color_provider: ColorProvider = class_color,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Ultralytics box 결과를 BGR 프레임에 그린다."""

    annotated = frame_bgr.copy()
    if boxes is None:
        return annotated

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        color = color_provider(class_id)
        label = f"{names.get(class_id, class_id)} {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )
    return annotated


def resize_for_display(frame: np.ndarray, max_width: int = 854) -> np.ndarray:
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


def to_rgb(frame_bgr: np.ndarray, max_width: int = 854) -> np.ndarray:
    """BGR 프레임을 Gradio ``gr.Image(type="numpy")`` 출력으로 변환한다."""

    display_frame = resize_for_display(frame_bgr, max_width=max_width)
    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
