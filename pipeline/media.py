"""영상 및 이미지 입력을 정규화하는 공용 미디어 유틸리티."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yt_dlp

from pipeline import webcams


SOURCE_YOUTUBE = "YouTube URL"
SOURCE_WEBCAM = "웹캠"
SOURCE_VIDEO = "비디오 파일"
SOURCE_IMAGES = "이미지 폴더"

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}
)


class MediaSourceError(RuntimeError):
    """사용자가 선택한 미디어 소스를 열 수 없을 때 발생한다."""


@dataclass(frozen=True)
class VideoSource:
    """OpenCV가 열 수 있는 영상 소스와 재생 특성."""

    value: str | int
    source_type: str
    pace_reads: bool


class YouTubeStreamResolver:
    """yt-dlp로 얻은 직접 스트림 URL을 제한된 시간 동안 재사용한다."""

    def __init__(self, ttl_seconds: float = 600.0):
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def resolve(self, url: str) -> str:
        normalized = (url or "").strip()
        if not normalized:
            raise MediaSourceError("YouTube URL을 입력하세요.")

        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(normalized)
            if cached and now - cached[0] < self._ttl_seconds:
                return cached[1]

        options = {
            # 프레임 처리에는 음성이 필요 없다. 영상/음성이 분리된 YouTube
            # 영상에서도 열 수 있도록 영상 전용 MP4를 우선 선택한다.
            "format": (
                "bestvideo[ext=mp4][protocol^=http]/"
                "bestvideo[protocol^=http]/bestvideo/"
                "best[ext=mp4][protocol^=http]/best[protocol^=http]/best"
            ),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(normalized, download=False)
        except Exception as exc:
            raise MediaSourceError(f"YouTube URL 처리 실패: {exc}") from exc

        if info and info.get("_type") == "playlist":
            info = next((entry for entry in info.get("entries") or [] if entry), None)

        stream_url = (info or {}).get("url")
        if not stream_url:
            raise MediaSourceError("YouTube 영상 스트림 URL을 찾지 못했습니다.")

        with self._lock:
            self._cache[normalized] = (now, stream_url)
        return stream_url


youtube_streams = YouTubeStreamResolver()


def upload_path(value) -> Path | None:
    """Gradio 파일 값과 일반 경로를 ``Path``로 정규화한다."""

    if not value:
        return None
    if isinstance(value, (str, Path)):
        return Path(value)
    return Path(getattr(value, "name", value))


def uploaded_video_path(value) -> Path:
    """업로드된 비디오 경로를 검증해 반환한다."""

    path = upload_path(value)
    if path is None:
        raise MediaSourceError("비디오 파일을 업로드하세요.")
    if not path.is_file():
        raise MediaSourceError(f"비디오 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise MediaSourceError(
            f"지원하지 않는 비디오 형식입니다 (지원 형식: {supported})"
        )
    return path


def filter_image_paths(values) -> list[Path]:
    """Gradio 파일 목록에서 지원하는 이미지 파일만 이름순으로 반환한다."""

    if not values:
        return []

    paths = []
    for value in values:
        path = upload_path(value)
        if (
            path is not None
            and path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            paths.append(path)
    return sorted(paths, key=lambda path: path.name)


def read_image(path: Path) -> np.ndarray | None:
    """한글 경로도 처리할 수 있도록 이미지 바이트를 OpenCV로 디코딩한다."""

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resolve_video_source(
    source_type: str,
    youtube_url: str = "",
    webcam_index=None,
    video_file=None,
) -> VideoSource:
    """UI 입력을 OpenCV용 영상 소스로 변환한다."""

    if source_type == SOURCE_WEBCAM:
        try:
            index = webcams.coerce_webcam_index(webcam_index)
        except webcams.WebcamOpenError as exc:
            raise MediaSourceError(str(exc)) from exc
        return VideoSource(value=index, source_type=source_type, pace_reads=False)
    if source_type == SOURCE_VIDEO:
        return VideoSource(
            value=str(uploaded_video_path(video_file)),
            source_type=source_type,
            pace_reads=True,
        )
    if source_type == SOURCE_YOUTUBE:
        return VideoSource(
            value=youtube_streams.resolve(youtube_url),
            source_type=source_type,
            pace_reads=True,
        )
    raise MediaSourceError(f"지원하지 않는 영상 소스입니다: {source_type}")


@contextmanager
def open_video_capture(source: VideoSource) -> Iterator[cv2.VideoCapture]:
    """영상 소스를 열고 성공·실패·취소 여부와 무관하게 핸들을 해제한다."""

    if source.source_type == SOURCE_WEBCAM:
        try:
            capture = webcams.open_webcam(source.value)
        except webcams.WebcamOpenError as exc:
            raise MediaSourceError(str(exc)) from exc
    else:
        capture = cv2.VideoCapture(source.value)

    try:
        if not capture.isOpened():
            raise MediaSourceError("영상 소스를 열 수 없습니다.")
        yield capture
    finally:
        capture.release()


def capture_fps(capture: cv2.VideoCapture, fallback: float = 30.0) -> float:
    """OpenCV가 유효한 FPS를 주지 않을 때 안전한 기본값을 사용한다."""

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps >= 1.0 else fallback
