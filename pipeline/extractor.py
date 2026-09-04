"""Tab 1 프레임 추출 서비스."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import cv2

from pipeline import media, vision


OUT_DIR = Path("dataset/raw_frames")

_MAX_PREVIEW_FPS = 10
_FOLDER_PREVIEW_INTERVAL = 0.2


def _valid_frame(frame) -> bool:
    return frame is not None and getattr(frame, "size", 0) > 0


class _CaptureController:
    """현재 캡처 작업의 취소 신호와 진행량을 동기화한다."""

    def __init__(self):
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._saved_count = 0

    def begin(self) -> None:
        self.stop_event.clear()
        with self._lock:
            self._saved_count = 0

    def saved(self, count: int) -> None:
        with self._lock:
            self._saved_count = count

    def stop(self) -> int:
        with self._lock:
            self.stop_event.set()
            return self._saved_count

    def prepare_output(self) -> bool:
        """중지 요청과 출력 초기화가 엇갈리지 않도록 원자적으로 준비한다."""

        with self._lock:
            if self.stop_event.is_set():
                return False
            _reset_output_directory()
            return True


_controller = _CaptureController()


def stop() -> str:
    """현재 캡처를 중지하고 마지막 저장 장수를 반환한다."""

    saved_count = _controller.stop()
    return f"중지됨 — {saved_count}장 저장 완료  →  {OUT_DIR.resolve()}"


def capture(
    source_type: str,
    youtube_url: str,
    capture_fps: int,
    folder_files=None,
    webcam_index=None,
    video_file=None,
    browser_session_id: str = "",
):
    """프레임과 상태 문자열을 연속으로 생성한다."""

    _controller.begin()

    try:
        if source_type == media.SOURCE_IMAGES:
            images = media.filter_image_paths(folder_files)
            if not folder_files:
                yield None, "이미지 폴더(또는 파일)를 업로드하세요."
                return
            if not images:
                supported = ", ".join(sorted(media.IMAGE_EXTENSIONS))
                yield None, f"이미지 파일이 없습니다 (지원 형식: {supported})"
                return

            if not _controller.prepare_output():
                yield None, stop()
                return
            source_id = "images001"
            copied_count = yield from _import_images(images, source_id)
            _write_sources_manifest([
                {
                    "id": source_id,
                    "type": source_type,
                    "value": "업로드 이미지",
                    "frame_count": copied_count,
                }
            ])
            return

        if source_type == media.SOURCE_YOUTUBE:
            urls = _parse_youtube_urls(youtube_url)
            if not urls:
                yield None, "YouTube URL을 한 줄에 하나씩 입력하세요."
                return
            yield from _capture_youtube_urls(urls, max(1, int(capture_fps)))
            return

        try:
            source = media.resolve_video_source(
                source_type,
                youtube_url=youtube_url,
                webcam_index=webcam_index,
                video_file=video_file,
                browser_session_id=browser_session_id,
            )
            with media.open_video_capture(source) as video:
                ok, first_frame = video.read()
                if not ok or not _valid_frame(first_frame):
                    raise media.MediaSourceError(
                        "소스는 열렸지만 첫 프레임을 읽지 못했습니다. "
                        "웹캠 권한·장치 점유 또는 영상 파일 상태를 확인하세요."
                    )
                if not _controller.prepare_output():
                    yield None, stop()
                    return
                source_id = (
                    "webcam001"
                    if source_type == media.SOURCE_WEBCAM
                    else "video001"
                )
                saved_count, _last_preview = yield from _capture_video(
                    video,
                    source,
                    max(1, int(capture_fps)),
                    source_id,
                    first_frame=first_frame,
                )
                _write_sources_manifest([
                    {
                        "id": source_id,
                        "type": source_type,
                        "value": str(video_file or webcam_index or ""),
                        "frame_count": saved_count,
                    }
                ])
        except media.MediaSourceError as exc:
            yield None, str(exc)
    except GeneratorExit:
        raise
    except Exception as exc:
        yield None, f"프레임 추출 오류: {exc}"


def _reset_output_directory() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("frame_*.jpg"):
        path.unlink()
    # raw_frames 전체를 새 캡처 묶음으로 교체하므로 대응하는 평면 라벨도 제거한다.
    # labels/train, labels/val 하위의 이미 구성된 데이터셋은 보존한다.
    labels_dir = OUT_DIR.parent / "labels"
    if labels_dir.exists():
        for path in labels_dir.glob("*.txt"):
            path.unlink()
    manifest_path = OUT_DIR.parent / "sources.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _parse_youtube_urls(value: str) -> list[str]:
    """한 줄에 하나씩 입력된 URL을 순서 유지 중복 제거한다."""

    urls = []
    seen = set()
    for line in (value or "").splitlines():
        url = line.strip()
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _write_sources_manifest(records: list[dict]) -> None:
    path = OUT_DIR.parent / "sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "sources": records}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _capture_youtube_urls(urls: list[str], target_fps: int):
    """여러 YouTube URL을 소스별 파일명으로 차례대로 추출한다."""

    records: list[dict] = []
    errors: list[str] = []
    prepared = False
    total_saved = 0
    last_preview = None

    for index, url in enumerate(urls, start=1):
        if _controller.stop_event.is_set():
            break

        source_id = f"yt{index:03d}"
        yield last_preview, (
            f"YouTube {index}/{len(urls)} 스트림 URL 추출 중...  |  {source_id}"
        )
        try:
            source = media.resolve_video_source(
                media.SOURCE_YOUTUBE,
                youtube_url=url,
            )
            with media.open_video_capture(source) as video:
                ok, first_frame = video.read()
                if not ok or not _valid_frame(first_frame):
                    raise media.MediaSourceError(
                        "소스는 열렸지만 첫 프레임을 읽지 못했습니다."
                    )
                if not prepared:
                    if not _controller.prepare_output():
                        yield last_preview, stop()
                        return
                    prepared = True

                result = yield from _capture_video(
                    video,
                    source,
                    target_fps,
                    source_id,
                    saved_offset=total_saved,
                    first_frame=first_frame,
                    source_position=(index, len(urls)),
                )
                saved_count, last_preview = result
                total_saved += saved_count
                records.append(
                    {
                        "id": source_id,
                        "type": media.SOURCE_YOUTUBE,
                        "value": url,
                        "frame_count": saved_count,
                    }
                )
                _write_sources_manifest(records)
        except media.MediaSourceError as exc:
            errors.append(f"{source_id}: {exc}")
            yield last_preview, f"{source_id} 처리 실패 — {exc}"

    if not prepared:
        detail = "\n".join(errors) if errors else "열 수 있는 URL이 없습니다."
        yield last_preview, f"YouTube 프레임을 추출하지 못했습니다.\n{detail}"
        return

    prefix = "중지됨" if _controller.stop_event.is_set() else "전체 완료"
    error_text = f"  |  실패 {len(errors)}개" if errors else ""
    yield last_preview, (
        f"{prefix} — {len(records)}/{len(urls)}개 소스, "
        f"총 {total_saved}장 저장{error_text}  →  {OUT_DIR.resolve()}"
    )


def _capture_video(
    video: cv2.VideoCapture,
    source: media.VideoSource,
    target_fps: int,
    source_id: str,
    saved_offset: int = 0,
    first_frame=None,
    source_position: tuple[int, int] | None = None,
):
    source_fps = media.capture_fps(video)
    save_every = max(1, round(source_fps / target_fps))
    preview_fps = max(1, min(_MAX_PREVIEW_FPS, round(source_fps)))
    preview_interval = 1.0 / preview_fps
    read_interval = 1.0 / min(max(source_fps, 1.0), 30.0)

    frame_index = 0
    saved_count = 0
    last_preview_at = 0.0
    last_preview = None
    next_read_at = time.perf_counter()
    pending_frame = first_frame

    while not _controller.stop_event.is_set():
        if source.pace_reads:
            wait = next_read_at - time.perf_counter()
            if wait > 0:
                _controller.stop_event.wait(wait)
            if _controller.stop_event.is_set():
                break
            next_read_at = max(
                next_read_at + read_interval,
                time.perf_counter(),
            )

        if pending_frame is not None:
            ok, frame_bgr = True, pending_frame
            pending_frame = None
        else:
            ok, frame_bgr = video.read()
            if (not ok or not _valid_frame(frame_bgr)) and source.source_type == media.SOURCE_WEBCAM:
                # USB/V4L2 카메라는 일시적으로 빈 프레임을 반환할 수 있다.
                for _ in range(4):
                    if _controller.stop_event.wait(0.05):
                        break
                    ok, frame_bgr = video.read()
                    if ok and _valid_frame(frame_bgr):
                        break
        if not ok or not _valid_frame(frame_bgr):
            if (
                source.source_type == media.SOURCE_WEBCAM
                and not _controller.stop_event.is_set()
            ):
                raise media.MediaSourceError(
                    "웹캠 프레임을 연속으로 읽지 못했습니다. 장치 연결과 점유 상태를 확인하세요."
                )
            break

        if frame_index % save_every == 0:
            output_path = OUT_DIR / f"frame_{source_id}_{saved_count:05d}.jpg"
            if not cv2.imwrite(str(output_path), frame_bgr):
                raise OSError(f"프레임 저장 실패: {output_path}")
            saved_count += 1
            _controller.saved(saved_offset + saved_count)

        now = time.perf_counter()
        if last_preview_at == 0.0 or now - last_preview_at >= preview_interval:
            last_preview_at = now
            last_preview = vision.to_rgb(frame_bgr)
            source_text = (
                f"소스 {source_position[0]}/{source_position[1]}  |  "
                if source_position
                else ""
            )
            yield (
                last_preview,
                f"{source_text}{source_id} {saved_count}장 저장 중...  "
                f"|  미리보기 {preview_fps}fps",
            )

        frame_index += 1

    prefix = "중지됨" if _controller.stop_event.is_set() else "완료"
    yield (
        last_preview,
        f"{prefix} — {source_id} {saved_count}장 저장 완료  →  {OUT_DIR.resolve()}",
    )
    return saved_count, last_preview


def _import_images(images: list[Path], source_id: str):
    total = len(images)
    yield None, f"{total}장 발견 — 복사 시작..."

    copied_count = 0
    last_preview_at = 0.0
    last_preview = None

    for source_path in images:
        if _controller.stop_event.is_set():
            break

        output_path = OUT_DIR / f"frame_{source_id}_{copied_count:05d}.jpg"
        try:
            frame_bgr = None
            if source_path.suffix.lower() in {".jpg", ".jpeg"}:
                shutil.copyfile(source_path, output_path)
            else:
                frame_bgr = media.read_image(source_path)
                if frame_bgr is None:
                    yield (
                        last_preview,
                        f"건너뜀 (읽기 실패): {source_path.name}",
                    )
                    continue
                if not cv2.imwrite(str(output_path), frame_bgr):
                    raise OSError(f"이미지 저장 실패: {output_path}")

            copied_count += 1
            _controller.saved(copied_count)

            now = time.perf_counter()
            should_preview = (
                copied_count == 1
                or copied_count == total
                or now - last_preview_at >= _FOLDER_PREVIEW_INTERVAL
            )
            if should_preview:
                last_preview_at = now
                if frame_bgr is None:
                    frame_bgr = media.read_image(source_path)
                if frame_bgr is not None:
                    last_preview = vision.to_rgb(frame_bgr)
                yield last_preview, f"{copied_count} / {total} 복사 완료"
        except OSError as exc:
            yield last_preview, f"{source_path.name} 처리 실패: {exc}"

    prefix = "중지됨" if _controller.stop_event.is_set() else "완료"
    yield (
        last_preview,
        f"{prefix} — {copied_count}장 복사 완료  →  {OUT_DIR.resolve()}",
    )
    return copied_count
