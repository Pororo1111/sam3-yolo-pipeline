"""Tab 6 YOLO 추론 서비스."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from pipeline import media, models, vision


logging.getLogger("ultralytics").setLevel(logging.ERROR)

_stop_event = threading.Event()
_DISPLAY_INTERVAL = 1.0 / 15


def stop() -> None:
    _stop_event.set()


def predict(
    model_path: str,
    source_type: str,
    youtube_url: str,
    conf: float,
    infer_every: int,
    folder_files=None,
    webcam_index=None,
    video_file=None,
):
    """영상 또는 이미지 폴더에서 추론한 RGB 프레임과 상태를 생성한다."""

    _stop_event.clear()
    resolved_model_path = (model_path or "").strip() or models.latest_trained_model()
    if not resolved_model_path:
        yield None, "학습된 모델이 없습니다. 먼저 학습 탭에서 학습을 완료하세요."
        return
    if not Path(resolved_model_path).is_file():
        yield None, f"모델 파일 없음: {resolved_model_path}"
        return

    try:
        from ultralytics import YOLO

        yield None, f"모델 로딩 중... ({resolved_model_path})"
        model = YOLO(resolved_model_path)
    except Exception as exc:
        yield None, f"모델 로딩 실패: {exc}"
        return

    names = model.names or {}
    class_ids = vision.inference_class_ids(names)

    if source_type == media.SOURCE_IMAGES:
        yield from _predict_folder(
            model, names, class_ids, folder_files, float(conf)
        )
        return

    try:
        source = media.resolve_video_source(
            source_type,
            youtube_url=youtube_url,
            webcam_index=webcam_index,
            video_file=video_file,
        )
        with media.open_video_capture(source) as capture:
            yield from _predict_video(
                model,
                names,
                class_ids,
                capture,
                float(conf),
                max(1, int(infer_every)),
            )
    except media.MediaSourceError as exc:
        yield None, str(exc)
    except GeneratorExit:
        raise
    except Exception as exc:
        yield None, f"추론 오류: {exc}"


def _predict_video(
    model,
    names: dict,
    class_ids: list[int],
    capture,
    conf: float,
    infer_every: int,
):
    yield None, "추론 시작..."

    frame_index = 0
    last_boxes = None
    last_detection_count = 0
    inference_ms = 0.0
    last_preview_at = 0.0

    while not _stop_event.is_set():
        ok, frame_bgr = capture.read()
        if not ok:
            break

        if frame_index % infer_every == 0:
            started_at = time.perf_counter()
            results = model(
                frame_bgr,
                conf=conf,
                classes=class_ids,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - started_at) * 1000
            last_boxes = results[0].boxes if results[0].boxes is not None else None
            last_detection_count = (
                len(last_boxes) if last_boxes is not None else 0
            )

        now = time.perf_counter()
        if now - last_preview_at >= _DISPLAY_INTERVAL:
            last_preview_at = now
            annotated = vision.draw_boxes(frame_bgr, last_boxes, names)
            yield (
                vision.to_rgb(annotated),
                f"프레임 {frame_index + 1}  |  감지 {last_detection_count}개 "
                f"|  추론 {inference_ms:.0f}ms  |  skip={infer_every}",
            )

        frame_index += 1

    prefix = "중지됨" if _stop_event.is_set() else "완료"
    yield None, f"추론 {prefix} — 총 {frame_index}프레임 처리"


def _predict_folder(
    model,
    names: dict,
    class_ids: list[int],
    folder_files,
    conf: float,
):
    images = media.filter_image_paths(folder_files)
    if not images:
        yield None, "업로드된 이미지가 없습니다. 이미지 폴더를 업로드하세요."
        return

    yield None, f"이미지 폴더 추론 시작 — {len(images)}장"

    shown_count = 0
    while not _stop_event.is_set():
        for path in images:
            if _stop_event.is_set():
                break

            frame_bgr = media.read_image(path)
            if frame_bgr is None:
                continue

            started_at = time.perf_counter()
            results = model(
                frame_bgr,
                conf=conf,
                classes=class_ids,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - started_at) * 1000
            boxes = results[0].boxes if results[0].boxes is not None else None
            detection_count = len(boxes) if boxes is not None else 0

            annotated = vision.draw_boxes(frame_bgr, boxes, names)
            shown_count += 1
            yield (
                vision.to_rgb(annotated),
                f"{path.name}  |  감지 {detection_count}개 "
                f"|  추론 {inference_ms:.0f}ms  ({shown_count})",
            )
            _stop_event.wait(0.4)

    yield None, f"중지됨 — {shown_count}장 표시"
