"""웹캠 장비 감지 유틸리티."""

from __future__ import annotations

import contextlib
import os
import platform
from pathlib import Path

import cv2


@contextlib.contextmanager
def _suppress_native_stderr():
    """OpenCV 네이티브 백엔드가 stderr에 직접 쓰는 감지 오류를 숨긴다."""
    try:
        stderr_fd = sys_stderr_fd = 2
        saved_fd = os.dup(sys_stderr_fd)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(saved_fd, stderr_fd)
                os.close(saved_fd)
    except OSError:
        yield


def _camera_label(index: int) -> str:
    """OS에서 얻을 수 있는 단서를 이용해 사용자용 카메라 이름을 만든다."""
    if platform.system() == "Linux":
        name_path = Path(f"/sys/class/video4linux/video{index}/name")
        if name_path.exists():
            name = name_path.read_text(encoding="utf-8", errors="ignore").strip()
            if name:
                return f"카메라 {index} · {name}"
    return f"카메라 {index}"


def _candidate_indices(max_devices: int) -> list[int]:
    """무의미한 범위 스캔을 줄이기 위해 실제 단서가 있는 카메라 인덱스만 우선 반환한다."""
    max_devices = max(0, int(max_devices))
    if platform.system() == "Linux":
        indices: set[int] = set()
        for path in Path("/dev").glob("video*"):
            suffix = path.name.removeprefix("video")
            if suffix.isdigit():
                index = int(suffix)
                if index < max_devices:
                    indices.add(index)
        for path in Path("/sys/class/video4linux").glob("video*"):
            suffix = path.name.removeprefix("video")
            if suffix.isdigit():
                index = int(suffix)
                if index < max_devices:
                    indices.add(index)
        return sorted(indices)
    return list(range(max_devices))


def _open_camera(index: int):
    """플랫폼별 권장 백엔드로 카메라를 연다."""
    system = platform.system()
    if system == "Windows":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if system == "Linux":
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def list_webcams(max_devices: int = 10) -> list[tuple[str, str]]:
    """열 수 있는 웹캠 인덱스를 감지해 Gradio Dropdown choices 형태로 반환한다."""
    choices: list[tuple[str, str]] = []
    for index in _candidate_indices(max_devices):
        with _suppress_native_stderr():
            cap = _open_camera(index)
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
