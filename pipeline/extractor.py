import threading
import shutil
import time
import cv2
import numpy as np
import yt_dlp
from pathlib import Path


OUT_DIR = Path("dataset/raw_frames")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}

_stop_event = threading.Event()
_saved_count = 0  # 현재 캡처 세션에서 저장한 장수 (중지 시 최종 메시지에 사용)


_webcam_last_save = 0.0
_webcam_saved = 0

def start_browser_webcam_capture():
    """브라우저 webcam 스트림 캡처 세션을 초기화한다."""
    global _webcam_last_save, _webcam_saved, _saved_count
    _stop_event.clear()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("frame_*.jpg"):
        f.unlink()
    _webcam_last_save = 0.0
    _webcam_saved = 0
    _saved_count = 0
    return True, None, "브라우저 웹캠 대기 중... 카메라 권한을 허용하면 프레임 저장을 시작합니다."


def capture_browser_webcam_frame(frame, active: bool, capture_fps: int):
    """브라우저에서 전달된 webcam 프레임을 지정 FPS로 저장한다."""
    global _webcam_last_save, _webcam_saved, _saved_count
    if not active or frame is None or _stop_event.is_set():
        return frame, stop() if _stop_event.is_set() else "대기 중"

    now = time.perf_counter()
    interval = 1.0 / max(1, int(capture_fps or 1))
    if _webcam_last_save and now - _webcam_last_save < interval:
        return frame, f"{_webcam_saved}장 저장 중..."

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb = frame.astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    path = OUT_DIR / f"frame_{_webcam_saved:05d}.jpg"
    cv2.imwrite(str(path), bgr)
    _webcam_saved += 1
    _saved_count = _webcam_saved
    _webcam_last_save = now
    return frame, f"{_webcam_saved}장 저장 중... (브라우저 웹캠)"


def stop_browser_webcam_capture():
    _stop_event.set()
    return False, stop()


def stop():
    """중지 요청 + 최종 상태 문자열 반환.

    중지 버튼은 `cancels`로 캡처 제너레이터를 강제 종료하므로 제너레이터
    내부의 마지막 yield(완료 메시지)가 실행되지 못한다. 따라서 중지 버튼이
    직접 상태창을 갱신하도록 여기서 최종 메시지를 반환한다.
    """
    _stop_event.set()
    return f"중지됨 — {_saved_count}장 저장 완료  →  {OUT_DIR.resolve()}"


def _get_youtube_stream_url(url: str) -> str:
    ydl_opts = {
        "format": "best[ext=mp4]/bestvideo[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # live stream or regular video both return 'url'
        formats = info.get("formats") or []
        if formats:
            # pick best mp4
            mp4 = [f for f in formats if f.get("ext") == "mp4" and f.get("url")]
            if mp4:
                return mp4[-1]["url"]
        return info["url"]


def capture(source_type: str, youtube_url: str, capture_fps: int, folder_files=None, webcam_index=None, video_file=None):
    """
    Generator — yields (rgb_frame | None, status_str) until done or stopped.
    source_type: "YouTube URL" | "웹캠" | "비디오 파일" | "이미지 폴더"
    folder_files: 이미지 폴더 모드에서 gr.File 업로드 결과(파일 경로 리스트).
    webcam_index: 더 이상 사용하지 않음(브라우저 웹캠은 gr.Image 스트림으로 전달).
    video_file: 비디오 파일 모드에서 gr.File 업로드 결과.
    """
    global _saved_count
    _stop_event.clear()
    _saved_count = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 프레임 삭제
    for f in OUT_DIR.glob("frame_*.jpg"):
        f.unlink()

    # ── 이미지 폴더 임포트 ─────────────────────────────────────
    if source_type == "이미지 폴더":
        yield from _import_from_files(folder_files)
        return

    # ── 소스 열기 ──────────────────────────────────────────────
    if source_type == "YouTube URL":
        url = youtube_url.strip()
        if not url:
            yield None, "YouTube URL을 입력하세요."
            return
        yield None, "YouTube 스트림 URL 추출 중..."
        try:
            stream_url = _get_youtube_stream_url(url)
        except Exception as e:
            yield None, f"URL 추출 실패: {e}"
            return
        cap = cv2.VideoCapture(stream_url)
    elif source_type == "비디오 파일":
        video_path, err = _uploaded_video_path(video_file)
        if err:
            yield None, err
            return
        cap = cv2.VideoCapture(str(video_path))
    else:
        yield None, "웹캠은 브라우저 기반 입력을 사용합니다. 화면의 웹캠 미리보기 권한을 허용한 뒤 캡처를 시작하세요."
        return

    if not cap.isOpened():
        yield None, "소스를 열 수 없습니다."
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 원본 FPS가 0으로 잡히는 경우(라이브) 30 가정
    if src_fps < 1:
        src_fps = 30.0
    frame_interval = max(1, round(src_fps / max(1, capture_fps)))

    frame_idx = 0
    saved = 0
    # 저장 FPS와 미리보기 FPS를 분리한다. 기존에는 capture_fps=5이면
    # 미리보기도 5fps로만 갱신되어 YouTube/비디오 소스가 끊겨 보였다.
    # 저장은 capture_fps를 따르되, 화면 미리보기는 최대 15fps로 별도 갱신한다.
    preview_fps = 15
    display_interval = 1.0 / preview_fps
    last_yield = 0.0
    last_preview_rgb = None
    # 일부 YouTube/파일 소스는 cap.read()가 실제 재생 속도보다 빠르게 반환된다.
    # UI가 중간 프레임을 받을 수 있도록 원본 FPS 기준으로 읽기 루프도 페이싱한다.
    read_interval = 1.0 / min(max(src_fps, 1.0), 30.0)
    next_read_at = time.perf_counter()

    while True:
        if _stop_event.is_set():
            break

        now = time.perf_counter()
        wait = next_read_at - now
        if wait > 0:
            time.sleep(wait)
        next_read_at = max(next_read_at + read_interval, time.perf_counter())

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            path = OUT_DIR / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            _saved_count = saved

        now = time.perf_counter()
        if last_yield == 0.0 or now - last_yield >= display_interval:
            last_yield = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_preview_rgb = rgb
            yield rgb, f"{saved}장 저장 중...  |  미리보기 {preview_fps}fps"

        frame_idx += 1

    cap.release()

    if _stop_event.is_set():
        yield last_preview_rgb, f"중지됨 — {saved}장 저장 완료  →  {OUT_DIR.resolve()}"
    else:
        yield last_preview_rgb, f"완료 — {saved}장 저장 완료  →  {OUT_DIR.resolve()}"


def _uploaded_video_path(file) -> "tuple[Path | None, str | None]":
    """gr.File 업로드 결과를 비디오 파일 Path로 정규화한다."""
    if not file:
        return None, "비디오 파일을 업로드하세요."
    path = Path(getattr(file, "name", file))
    if not path.is_file():
        return None, f"비디오 파일을 찾을 수 없습니다: {path}"
    if path.suffix.lower() not in _VIDEO_EXTS:
        return None, f"지원하지 않는 비디오 형식입니다 (지원 형식: {', '.join(sorted(_VIDEO_EXTS))})"
    return path, None


def _imread_any_path(path: Path) -> "np.ndarray | None":
    """cv2.imread는 한글/유니코드 경로를 못 읽으므로 np.fromfile 경유."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _filter_image_paths(files) -> "list[Path]":
    """gr.File 업로드 결과를 받아 지원 형식의 이미지 경로만 정렬해 반환."""
    if not files:
        return []
    paths = []
    for f in files:
        # gr.File(type="filepath")는 경로 문자열을 주지만, 방어적으로 .name도 처리
        p = Path(getattr(f, "name", f))
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            paths.append(p)
    return sorted(paths, key=lambda p: p.name)


def _import_from_files(files):
    """gr.File 업로드된 이미지 파일들을 raw_frames로 복사."""
    global _saved_count
    images = _filter_image_paths(files)

    if not files:
        yield None, "이미지 폴더(또는 파일)를 업로드하세요."
        return

    if not images:
        yield None, f"이미지 파일이 없습니다 (지원 형식: {', '.join(sorted(_IMAGE_EXTS))})"
        return

    total = len(images)
    yield None, f"{total}장 발견 — 복사 시작..."

    copied = 0
    for idx, src_path in enumerate(images):
        if _stop_event.is_set():
            yield None, f"중지됨 — {copied}장 복사 완료  →  {OUT_DIR.resolve()}"
            return

        dst_path = OUT_DIR / f"frame_{copied:05d}.jpg"

        try:
            bgr = _imread_any_path(src_path)
            if bgr is None:
                yield None, f"{idx + 1}/{total}  건너뜀 (읽기 실패): {src_path.name}"
                continue
            cv2.imwrite(str(dst_path), bgr)
            copied += 1
            _saved_count = copied
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield rgb, f"{copied} / {total} 복사 완료"
        except Exception as e:
            yield None, f"{src_path.name} 처리 실패: {e}"

    yield None, f"완료 — {copied}장 복사  →  {OUT_DIR.resolve()}"
