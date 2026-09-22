"""추출 프레임 파일명에서 원본 소스 그룹을 식별하는 공용 유틸리티."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse, parse_qs


LEGACY_SOURCE_ID = "legacy"


def source_id_from_stem(stem: str) -> str:
    """``frame_<source>_<index>``에서 source를 반환한다.

    기존 ``frame_00001`` 형식은 하나의 legacy 그룹으로 취급해 이전 데이터도
    미리보기와 라벨링에는 계속 사용할 수 있게 한다.
    """

    prefix = "frame_"
    if not stem.startswith(prefix):
        return LEGACY_SOURCE_ID
    body = stem[len(prefix) :]
    source_id, separator, frame_index = body.rpartition("_")
    if separator and source_id and frame_index.isdigit():
        return source_id
    return LEGACY_SOURCE_ID


def source_id_from_path(path: Path) -> str:
    return source_id_from_stem(path.stem)


def group_frames(frames: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for frame in sorted(frames):
        groups[source_id_from_path(frame)].append(frame)
    return dict(sorted(groups.items()))


def source_choices(frames_dir: Path) -> list[tuple[str, str]]:
    groups = group_frames(list(frames_dir.glob("frame_*.jpg")))
    return [
        (f"{source_id} — {len(frames)}장", source_id)
        for source_id, frames in groups.items()
    ]


def filter_frames(frames: list[Path], selected_sources=None) -> list[Path]:
    selected = {str(value) for value in (selected_sources or []) if value}
    if not selected:
        return sorted(frames)
    return [
        frame
        for frame in sorted(frames)
        if source_id_from_path(frame) in selected
    ]


def youtube_identity(value: str) -> str:
    """공유/shorts/watch URL과 시간 매개변수를 같은 영상 ID로 정규화한다."""
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0]
    if host == "youtube.com" or host.endswith(".youtube.com"):
        parts = parsed.path.strip("/").split("/")
        if parts[0] in {"shorts", "embed", "live"} and len(parts) > 1:
            return parts[1]
        video_id = parse_qs(parsed.query).get("v")
        if video_id:
            return video_id[0]
    return value


def source_origin(record):
    source_type = record.get("type")
    value = record.get("origin", record.get("value", record["id"]))
    if source_type == "YouTube URL":
        value = youtube_identity(value)
    elif source_type != "비디오 파일":
        value = record["id"]
    return source_type, value
