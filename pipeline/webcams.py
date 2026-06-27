"""웹캠 장비 감지 유틸리티."""

from __future__ import annotations

import platform
from pathlib import Path

import cv2


def _camera_label(index: int) -> str:
    """OS에서 얻을 수 있는 단서를 이용해 사용자용 카메라 이름을 만든다."""
    if platform.system() == "Linux":
        name_path = Path(f"/sys/class/video4linux/video{index}/name")
        if name_path.exists():
            name = name_path.read_text(encoding="utf-8", errors="ignore").strip()
            if name:
                return f"카메라 {index} · {name}"
    return f"카메라 {index}"


def list_webcams(max_devices: int = 10) -> list[tuple[str, str]]:
    """열 수 있는 웹캠 인덱스를 감지해 Gradio Dropdown choices 형태로 반환한다."""
    choices: list[tuple[str, str]] = []
    for index in range(max(0, int(max_devices))):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if ok:
                choices.append((_camera_label(index), str(index)))
        finally:
            cap.release()
    return choices


def refresh_webcam_dropdown():
    """웹캠 드롭다운 갱신용 choices/value 튜플을 반환한다."""
    choices = list_webcams()
    return choices, (choices[0][1] if choices else None)


def coerce_webcam_index(value) -> int:
    """드롭다운 값이 비어 있거나 잘못되어도 기본 0번 카메라로 안전하게 변환한다."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
