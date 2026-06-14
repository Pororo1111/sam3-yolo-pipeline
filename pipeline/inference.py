import logging
import threading
import time
import cv2
import numpy as np
from pathlib import Path

logging.getLogger("ultralytics").setLevel(logging.ERROR)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

_stop_event = threading.Event()


def stop():
    _stop_event.set()


def _find_best_pt() -> str | None:
    candidates = sorted(Path("runs/detect").glob("*/weights/best.pt"))
    return str(candidates[-1]) if candidates else None


def _overlay_boxes(frame_bgr: np.ndarray, boxes, names: dict) -> np.ndarray:
    img = frame_bgr.copy()
    if boxes is None:
        return img
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = f"{names.get(cls_id, cls_id)} {conf:.2f}"
        color  = _cls_color(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def _cls_color(cls_id: int) -> tuple:
    palette = [
        (255, 100,   0), (  0, 220, 255), (  0, 200,   0), (100,   0, 255),
        (255,   0, 128), (  0, 128, 255), (200, 200,   0), (128,   0, 128),
    ]
    return palette[cls_id % len(palette)]


def predict(model_path: str, source_type: str, youtube_url: str,
            conf: float, infer_every: int, folder_path: str = ""):
    """
    Generator — yields (rgb_frame | None, status_str)
    infer_every: N프레임마다 1회 추론, 나머지는 마지막 bbox 재사용
    """
    _stop_event.clear()

    if not model_path.strip():
        model_path = _find_best_pt()
        if model_path is None:
            yield None, "best.pt를 찾을 수 없습니다. 먼저 Tab 4에서 학습을 완료하거나 경로를 직접 입력하세요."
            return

    if not Path(model_path).exists():
        yield None, f"모델 파일 없음: {model_path}"
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        yield None, "ultralytics 패키지를 찾을 수 없습니다."
        return

    yield None, f"모델 로딩 중... ({model_path})"

    try:
        model = YOLO(model_path)
    except Exception as e:
        yield None, f"모델 로딩 실패: {e}"
        return

    names = model.names or {}

    if source_type == "이미지 폴더":
        yield from _predict_folder(model, names, folder_path, conf)
        return

    if source_type == "웹캠":
        cap_source = 0
    else:
        if not youtube_url.strip():
            yield None, "YouTube URL을 입력하세요."
            return
        try:
            import yt_dlp
            ydl_opts = {"quiet": True, "format": "best[ext=mp4]/best"}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                cap_source = info["url"]
        except Exception as e:
            yield None, f"YouTube URL 처리 실패: {e}"
            return

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        yield None, "영상 소스를 열 수 없습니다."
        return

    yield None, "추론 시작..."

    frame_idx        = 0
    last_boxes       = None
    last_n_det       = 0
    t_infer          = 0.0
    display_interval = 1.0 / 15
    last_yield       = 0.0

    try:
        while not _stop_event.is_set():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % infer_every == 0:
                t0         = time.perf_counter()
                results    = model(frame_bgr, conf=conf, verbose=False)
                t_infer    = (time.perf_counter() - t0) * 1000
                last_boxes = results[0].boxes if results[0].boxes is not None else None
                last_n_det = len(last_boxes) if last_boxes else 0

            now = time.perf_counter()
            if now - last_yield < display_interval:
                frame_idx += 1
                continue

            last_yield = now
            annotated  = _overlay_boxes(frame_bgr, last_boxes, names)

            h, w = annotated.shape[:2]
            if w > 854:
                scale = 854 / w
                annotated = cv2.resize(annotated, (854, int(h * scale)),
                                       interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_idx += 1
            yield (
                rgb,
                f"프레임 {frame_idx}  |  감지 {last_n_det}개  "
                f"|  추론 {t_infer:.0f}ms  |  skip={infer_every}",
            )

    except Exception as e:
        yield None, f"추론 오류: {e}"
    finally:
        cap.release()

    yield None, f"추론 완료 — 총 {frame_idx}프레임 처리"


def _predict_folder(model, names: dict, folder_path: str, conf: float):
    """이미지 폴더의 모든 이미지를 순회하며 추론 (중지 전까지 반복)."""
    src = Path(folder_path.strip()) if folder_path and folder_path.strip() else None
    if src is None or not src.is_dir():
        yield None, f"이미지 폴더를 찾을 수 없습니다: {folder_path}"
        return

    images = sorted(p for p in src.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        yield None, "폴더에 이미지가 없습니다."
        return

    yield None, f"이미지 폴더 추론 시작 — {len(images)}장"

    shown = 0
    while not _stop_event.is_set():
        for p in images:
            if _stop_event.is_set():
                break
            buf = np.fromfile(str(p), dtype=np.uint8)
            frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            t0 = time.perf_counter()
            results = model(frame_bgr, conf=conf, verbose=False)
            t_infer = (time.perf_counter() - t0) * 1000
            boxes = results[0].boxes if results[0].boxes is not None else None
            n_det = len(boxes) if boxes else 0

            annotated = _overlay_boxes(frame_bgr, boxes, names)
            h, w = annotated.shape[:2]
            if w > 854:
                scale = 854 / w
                annotated = cv2.resize(annotated, (854, int(h * scale)),
                                       interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            shown += 1
            yield rgb, f"{p.name}  |  감지 {n_det}개  |  추론 {t_infer:.0f}ms  ({shown})"
            time.sleep(0.4)

    yield None, f"중지됨 — {shown}장 표시"
