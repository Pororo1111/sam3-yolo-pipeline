"""웹캠 장비 감지 유틸리티."""

from __future__ import annotations

import contextlib
import os
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class WebcamOpenError(RuntimeError):
    """선택한 서버 웹캠을 열거나 첫 프레임을 읽지 못했을 때 발생한다."""


class _PrefetchedCapture:
    """워밍업에서 읽은 첫 프레임을 버리지 않는 ``VideoCapture`` 프록시."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        first_frame: np.ndarray,
        release_device,
    ):
        self._capture = capture
        self._first_frame = first_frame
        self._release_device = release_device
        self._released = False

    def read(self):
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            return True, frame
        return self._capture.read()

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._first_frame = None
        try:
            self._capture.release()
        finally:
            self._release_device()

    def get(self, prop_id: int) -> float:
        return self._capture.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        return self._capture.set(prop_id, value)

    def getBackendName(self) -> str:
        return self._capture.getBackendName()


_device_locks: dict[int, threading.Lock] = {}
_device_locks_guard = threading.Lock()


def _device_lock(index: int) -> threading.Lock:
    with _device_locks_guard:
        return _device_locks.setdefault(index, threading.Lock())


def webcam_in_use(index) -> bool:
    try:
        lock = _device_lock(coerce_webcam_index(index))
    except WebcamOpenError:
        return False
    acquired = lock.acquire(blocking=False)
    if acquired:
        lock.release()
        return False
    return True


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
                indices.add(int(suffix))
        for path in Path("/sys/class/video4linux").glob("video*"):
            suffix = path.name.removeprefix("video")
            if suffix.isdigit():
                indices.add(int(suffix))
        # video10 같은 높은 인덱스도 실제 장치라면 제외하지 않는다. max_devices는
        # 인덱스 상한이 아니라 탐색할 장치 개수의 상한이다.
        return sorted(indices)[:max_devices]
    return list(range(max_devices))


def _backend_candidates() -> list[tuple[int | None, str]]:
    """플랫폼별 권장 순서로 OpenCV 카메라 백엔드를 반환한다."""

    system = platform.system()
    if system == "Windows":
        return [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (None, "기본"),
        ]
    if system == "Linux":
        return [(cv2.CAP_V4L2, "V4L2"), (None, "기본")]
    if system == "Darwin":
        return [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (None, "기본")]
    return [(None, "기본")]


def _open_with_backend(index: int, backend: int | None):
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def _read_warmup_frame(
    capture: cv2.VideoCapture,
    timeout_seconds: float,
) -> np.ndarray | None:
    """초기화가 느린 카메라를 잠시 재시도해 첫 정상 프레임을 얻는다."""

    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        time.sleep(0.05)
    return None


def open_webcam(index, warmup_timeout: float = 1.5):
    """서버 카메라를 백엔드 폴백과 첫 프레임 검증을 거쳐 연다.

    이 앱의 웹캠은 브라우저 카메라가 아니라 앱이 실행되는 서버(라즈베리파이
    포함)에 연결된 카메라다. 워밍업에서 얻은 첫 프레임은 프록시에 보관해 실제
    캡처 루프가 그대로 소비한다.
    """

    index = coerce_webcam_index(index)
    device_lock = _device_lock(index)
    if not device_lock.acquire(blocking=False):
        raise WebcamOpenError(
            f"웹캠 {index}은(는) 다른 캡처/추론 작업에서 사용 중입니다. "
            "해당 작업을 먼저 중지하세요."
        )

    reserved = True

    def release_device() -> None:
        nonlocal reserved
        if reserved:
            reserved = False
            device_lock.release()

    failures: list[str] = []
    success = False
    try:
        for backend, backend_name in _backend_candidates():
            capture = None
            try:
                capture = _open_with_backend(index, backend)
                if not capture.isOpened():
                    failures.append(f"{backend_name}: 열기 실패")
                    capture.release()
                    continue

                # 지원하지 않는 백엔드에서는 False만 반환하므로 안전하게 무시한다.
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                frame = _read_warmup_frame(capture, warmup_timeout)
                if frame is None:
                    failures.append(f"{backend_name}: 프레임 읽기 실패")
                    capture.release()
                    continue
                success = True
                return _PrefetchedCapture(capture, frame, release_device)
            except Exception as exc:
                failures.append(f"{backend_name}: {exc}")
                if capture is not None:
                    capture.release()

        detail = "; ".join(failures) if failures else "사용 가능한 백엔드 없음"
        raise WebcamOpenError(
            f"웹캠 {index}을(를) 사용할 수 없습니다 ({detail}). "
            "장치 권한과 다른 프로세스의 카메라 점유 여부를 확인하세요."
        )
    finally:
        # 성공 시에는 반환된 프록시가 장치 수명을 소유한다.
        if reserved and not success:
            release_device()


def list_webcams(max_devices: int | None = None) -> list[tuple[str, str]]:
    """열 수 있는 웹캠 인덱스를 감지해 Gradio Dropdown choices 형태로 반환한다."""
    if max_devices is None:
        # Raspberry Pi는 codec/ISP 노드가 다수 생길 수 있어 높은 video 인덱스까지 본다.
        max_devices = 64 if platform.system() == "Linux" else 10
    choices: list[tuple[str, str]] = []
    for index in _candidate_indices(max_devices):
        if webcam_in_use(index):
            choices.append((_camera_label(index) + " · 사용 중", str(index)))
            continue
        with _suppress_native_stderr():
            try:
                cap = open_webcam(index, warmup_timeout=0.75)
            except WebcamOpenError:
                if webcam_in_use(index):
                    choices.append((_camera_label(index) + " · 사용 중", str(index)))
                continue
            try:
                choices.append((_camera_label(index), str(index)))
            finally:
                cap.release()
    return choices


def refresh_webcam_dropdown():
    """웹캠 드롭다운 갱신용 choices/value 튜플을 반환한다."""
    choices = list_webcams()
    return choices, (choices[0][1] if choices else None)


def coerce_webcam_index(value) -> int:
    """드롭다운 값을 정수 카메라 인덱스로 변환한다."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise WebcamOpenError("웹캠 장비를 선택하세요.")
