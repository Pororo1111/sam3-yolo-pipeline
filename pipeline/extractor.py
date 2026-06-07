import threading
import cv2
import yt_dlp
from pathlib import Path

OUT_DIR = Path("dataset/raw_frames")
TARGET_FRAMES = 500

_stop_event = threading.Event()


def stop():
    _stop_event.set()


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


def capture(source_type: str, youtube_url: str, capture_fps: int):
    """
    Generator — yields (rgb_frame | None, status_str) until done or stopped.
    source_type: "YouTube URL" | "웹캠"
    """
    _stop_event.clear()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 프레임 삭제
    for f in OUT_DIR.glob("frame_*.jpg"):
        f.unlink()

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
    else:
        cap = cv2.VideoCapture(0)

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

    while saved < TARGET_FRAMES:
        if _stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            path = OUT_DIR / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield rgb, f"{saved} / {TARGET_FRAMES} 프레임 저장"

        frame_idx += 1

    cap.release()

    if _stop_event.is_set():
        yield None, f"중지됨 — {saved}장 저장 완료  →  {OUT_DIR.resolve()}"
    else:
        yield None, f"완료 — {saved}장 저장 완료  →  {OUT_DIR.resolve()}"
