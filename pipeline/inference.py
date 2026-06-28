import logging
import threading
import time
import cv2
import numpy as np
from pathlib import Path


logging.getLogger("ultralytics").setLevel(logging.ERROR)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}

_stop_event = threading.Event()

_browser_model = None
_browser_names = {}
_browser_frame_idx = 0
_browser_last_boxes = None
_browser_last_n_det = 0
_browser_last_infer_ms = 0.0


def start_browser_webcam_predict(model_path: str):
    """브라우저 webcam 추론 세션을 시작하고 모델을 1회 로딩한다."""
    global _browser_model, _browser_names, _browser_frame_idx, _browser_last_boxes, _browser_last_n_det, _browser_last_infer_ms
    _stop_event.clear()
    if not (model_path or "").strip():
        model_path = _find_best_pt()
        if model_path is None:
            return False, "학습된 모델이 없습니다. 먼저 학습 탭에서 학습을 완료하세요."
    if not Path(model_path).exists():
        return False, f"모델 파일 없음: {model_path}"
    try:
        from ultralytics import YOLO
        _browser_model = YOLO(model_path)
    except ImportError:
        return False, "ultralytics 패키지를 찾을 수 없습니다."
    except Exception as e:
        return False, f"모델 로딩 실패: {e}"
    _browser_names = _browser_model.names or {}
    _browser_frame_idx = 0
    _browser_last_boxes = None
    _browser_last_n_det = 0
    _browser_last_infer_ms = 0.0
    return True, f"브라우저 웹캠 추론 시작 — {model_path}"


def predict_browser_webcam_frame(frame, active: bool, conf: float, infer_every: int):
    """브라우저에서 전달된 webcam 프레임 1장을 추론해 반환한다."""
    global _browser_frame_idx, _browser_last_boxes, _browser_last_n_det, _browser_last_infer_ms
    if not active or frame is None:
        return frame, "대기 중"
    if _browser_model is None:
        return frame, "모델이 아직 로딩되지 않았습니다."
    if _stop_event.is_set():
        return frame, "중지됨"

    frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
    infer_every = max(1, int(infer_every or 1))
    if _browser_frame_idx % infer_every == 0:
        t0 = time.perf_counter()
        results = _browser_model(frame_bgr, conf=conf, verbose=False)
        _browser_last_infer_ms = (time.perf_counter() - t0) * 1000
        _browser_last_boxes = results[0].boxes if results[0].boxes is not None else None
        _browser_last_n_det = len(_browser_last_boxes) if _browser_last_boxes else 0

    annotated = _overlay_boxes(frame_bgr, _browser_last_boxes, _browser_names)
    h, w = annotated.shape[:2]
    if w > 854:
        scale = 854 / w
        annotated = cv2.resize(annotated, (854, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    _browser_frame_idx += 1
    return rgb, (
        f"브라우저 웹캠 프레임 {_browser_frame_idx}  |  감지 {_browser_last_n_det}개  "
        f"|  추론 {_browser_last_infer_ms:.0f}ms  |  skip={infer_every}"
    )


def stop_browser_webcam_predict():
    _stop_event.set()
    return False, "브라우저 웹캠 추론 중지"


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


def predict(model_path: str, source_type: str, youtube_url: str,
            conf: float, infer_every: int, folder_files=None, webcam_index=None, video_file=None):
    """
    Generator — yields (rgb_frame | None, status_str)
    infer_every: N프레임마다 1회 추론, 나머지는 마지막 bbox 재사용
    """
    _stop_event.clear()

    if not (model_path or "").strip():
        model_path = _find_best_pt()
        if model_path is None:
            yield None, "학습된 모델이 없습니다. 먼저 학습 탭에서 학습을 완료하세요."
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
        yield from _predict_folder(model, names, folder_files, conf)
        return

    if source_type == "웹캠":
        yield None, "웹캠 추론은 UI의 브라우저 웹캠 스트림 핸들러에서 처리합니다. 서버 카메라에는 접근하지 않습니다."
        return
    elif source_type == "비디오 파일":
        cap_source, err = _uploaded_video_path(video_file)
        if err:
            yield None, err
            return
        cap_source = str(cap_source)
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


def _predict_folder(model, names: dict, folder_files, conf: float):
    """업로드된 이미지들을 순회하며 추론 (중지 전까지 반복)."""
    images = _filter_image_paths(folder_files)
    if not images:
        yield None, "업로드된 이미지가 없습니다. 이미지 폴더를 업로드하세요."
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
