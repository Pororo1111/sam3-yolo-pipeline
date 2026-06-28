import base64
import json
import threading
import time
from pathlib import Path

from pipeline import webcams

import cv2
import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}


def _find_best_pt() -> str | None:
    candidates = sorted(Path("runs/detect").glob("*/weights/best.pt"))
    return str(candidates[-1]) if candidates else None


def _filter_image_paths(files) -> "list[Path]":
    """gr.File 업로드 결과에서 지원 형식의 이미지 경로만 정렬해 반환."""
    if not files:
        return []
    paths = []
    for f in files:
        p = Path(getattr(f, "name", f))
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            paths.append(p)
    return sorted(paths, key=lambda p: p.name)


def _count_objects_in_zone(boxes, polygon: np.ndarray) -> int:
    count = 0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
        if cv2.pointPolygonTest(polygon, (cx, cy), measureDist=False) >= 0:
            count += 1
    return count


def _overlay_boxes(frame_bgr: np.ndarray, boxes, names: dict) -> np.ndarray:
    img = frame_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
        color = ((cls_id * 67 + 100) % 256, (cls_id * 113 + 50) % 256, (cls_id * 41 + 200) % 256)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img

def _render_zones(annotated: np.ndarray, last_boxes):
    """현재 zone들을 annotated 위에 그리고 (annotated, 침입수, zone수) 반환."""
    with _zone_lock:
        current_zones = list(_zones)

    total_intruders = 0
    for zone in current_zones:
        count = _count_objects_in_zone(last_boxes or [], zone["polygon"])
        total_intruders += count
        color = (0, 0, 220) if count > 0 else (0, 220, 80)  # BGR: 빨강 or 초록
        thickness = 3 if count > 0 else 2

        if count > 0:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [zone["polygon"]], (0, 0, 180))
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        cv2.polylines(annotated, [zone["polygon"]], True, color, thickness)
        label = f"{zone['label']} ({count})"
        tx = zone["polygon"][0][0]
        ty = max(zone["polygon"][0][1] - 8, 0)
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated, total_intruders, len(current_zones)


_stop_event = threading.Event()
_zones: list = []
_zone_lock = threading.Lock()
_last_frame: np.ndarray | None = None
_last_frame_lock = threading.Lock()


def stop():
    _stop_event.set()


def reset():
    global _zones, _last_frame
    _stop_event.set()
    with _zone_lock:
        _zones = []
    with _last_frame_lock:
        _last_frame = None


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


def _resolve_source(source_type: str, youtube_url: str, webcam_index=None, video_file=None):
    if source_type == "웹캠":
        return webcams.coerce_webcam_index(webcam_index), None
    if source_type == "비디오 파일":
        video_path, err = _uploaded_video_path(video_file)
        return (str(video_path) if video_path else None), err
    if not youtube_url.strip():
        return None, "YouTube URL을 입력하세요."
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "format": "best[ext=mp4]/best"}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info["url"], None
    except Exception as e:
        return None, f"YouTube URL 처리 실패: {e}"


def stream(source_type: str, youtube_url: str, model_path: str, conf: float,
           infer_every: int, folder_files=None, webcam_index=None, video_file=None):
    global _last_frame
    _stop_event.clear()

    # YOLO 모델 로딩
    if not (model_path or "").strip():
        model_path = _find_best_pt()
    if model_path is None or not Path(model_path).exists():
        yield None, f"모델 파일을 찾을 수 없습니다: {model_path}"
        return

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        yield None, f"모델 로딩 실패: {e}"
        return

    names = model.names or {}

    if source_type == "이미지 폴더":
        yield from _stream_folder(model, names, folder_files, conf)
        return

    cap_source, err = _resolve_source(source_type, youtube_url, webcam_index, video_file)
    if err:
        yield None, err
        return

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        yield None, "영상 소스를 열 수 없습니다."
        return

    yield None, "스트림 시작..."

    display_interval = 1.0 / 15
    last_yield = 0.0
    frame_idx = 0
    last_boxes = None

    try:
        while not _stop_event.is_set():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            with _last_frame_lock:
                _last_frame = frame_bgr.copy()

            if frame_idx % infer_every == 0:
                results = model(frame_bgr, conf=conf, verbose=False)
                last_boxes = results[0].boxes if results[0].boxes is not None else None

            now = time.perf_counter()
            if now - last_yield < display_interval:
                frame_idx += 1
                continue
            last_yield = now

            # bbox 오버레이
            annotated = _overlay_boxes(frame_bgr, last_boxes or [], names)

            # zone 오버레이
            annotated, total_intruders, n_zones = _render_zones(annotated, last_boxes)

            h, w = annotated.shape[:2]
            if w > 854:
                scale = 854 / w
                annotated = cv2.resize(annotated, (854, int(h * scale)))

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_idx += 1
            status = (
                f"프레임 {frame_idx} | 영역: {n_zones}개 | "
                f"침입 객체: {total_intruders}개"
            )
            yield rgb, status

    except Exception as e:
        yield None, f"스트림 오류: {e}"
    finally:
        cap.release()

    yield None, "스트림 종료"


def _stream_folder(model, names: dict, folder_files, conf: float):
    """업로드된 이미지들을 순회하며 zone 감시 (중지 전까지 반복)."""
    global _last_frame

    images = _filter_image_paths(folder_files)
    if not images:
        yield None, "업로드된 이미지가 없습니다. 이미지 폴더를 업로드하세요."
        return

    yield None, f"이미지 폴더 감시 시작 — {len(images)}장 (영역 설정 후 침입 판별)"

    shown = 0
    while not _stop_event.is_set():
        for p in images:
            if _stop_event.is_set():
                break
            buf = np.fromfile(str(p), dtype=np.uint8)
            frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            with _last_frame_lock:
                _last_frame = frame_bgr.copy()

            results = model(frame_bgr, conf=conf, verbose=False)
            last_boxes = results[0].boxes if results[0].boxes is not None else None

            annotated = _overlay_boxes(frame_bgr, last_boxes or [], names)
            annotated, total_intruders, n_zones = _render_zones(annotated, last_boxes)

            h, w = annotated.shape[:2]
            if w > 854:
                scale = 854 / w
                annotated = cv2.resize(annotated, (854, int(h * scale)))

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            shown += 1
            yield rgb, (
                f"{p.name} | 영역: {n_zones}개 | "
                f"침입 객체: {total_intruders}개 ({shown})"
            )
            time.sleep(0.4)

    yield None, "스트림 종료"


def set_zone(prompt: str, model: str):
    """Returns (status, llm_raw_response)."""
    global _zones

    if not prompt.strip():
        return "프롬프트를 입력하세요.", ""

    with _last_frame_lock:
        if _last_frame is None:
            return "먼저 스트림을 시작하세요.", ""
        frame = _last_frame.copy()

    h, w = frame.shape[:2]
    _, buf = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    system_prompt = (
        "You are a zone detection assistant. Analyze the image and return monitoring zones "
        "as JSON only — no explanation, no markdown.\n"
        'Schema: {"zones": [{"label": "string", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}]}\n'
        "All coordinates normalized 0.0–1.0. x1,y1=top-left corner, x2,y2=bottom-right corner.\n"
        "IMPORTANT: The label field must always be in English, regardless of the input language."
    )

    payload = {
        "model": model.strip() or "gemma4:e4b",
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Identify the monitoring zone(s) for: {prompt}",
                "images": [img_b64],
            },
        ],
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        parsed = json.loads(content)
    except requests.exceptions.ConnectionError:
        return "Ollama 연결 실패 (localhost:11434). Ollama가 실행 중인지 확인하세요.", ""
    except json.JSONDecodeError as e:
        return f"LLM 응답 파싱 실패: {e}", content if "content" in dir() else ""
    except Exception as e:
        return f"영역 설정 실패: {e}", ""

    formatted = json.dumps(parsed, ensure_ascii=False, indent=2)

    zones = []
    for z in parsed.get("zones", []):
        try:
            x1 = int(z["x1"] * w)
            y1 = int(z["y1"] * h)
            x2 = int(z["x2"] * w)
            y2 = int(z["y2"] * h)
            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            zones.append({"label": z.get("label", "zone"), "polygon": polygon})
        except (KeyError, TypeError):
            continue

    if not zones:
        return "LLM이 영역을 반환하지 않았습니다. 프롬프트를 바꿔서 다시 시도하세요.", formatted

    with _zone_lock:
        _zones = zones

    labels = [z["label"] for z in zones]
    return f"영역 설정 완료: {', '.join(labels)}", formatted
