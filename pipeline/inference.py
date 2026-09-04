"""Tab 6 YOLO 추론 서비스."""

from __future__ import annotations

import html
import logging
import threading
import time
from pathlib import Path

from pipeline import media, models, vision, webcams


logging.getLogger("ultralytics").setLevel(logging.ERROR)

_stop_event = threading.Event()
_DISPLAY_INTERVAL = 1.0 / 15


def stop() -> str:
    _stop_event.set()
    return ""


def _browser_overlay_svg(frame_bgr, boxes, names: dict) -> str:
    """원본 브라우저 영상 위에 겹칠 가벼운 SVG 박스를 생성한다."""

    height, width = frame_bgr.shape[:2]
    # SVG 전체가 모바일 화면 폭에 맞춰 축소되므로, 원본 좌표계에서는 크게
    # 그려야 실제 표시 크기가 충분히 읽힌다.
    stroke_width = max(4.0, width / 160.0)
    font_size = max(28.0, width / 22.0)
    text_stroke = max(2.0, font_size / 8.0)
    elements = []
    for annotation in vision.box_annotations(boxes, names):
        blue, green, red = annotation.color
        color = f"rgb({red},{green},{blue})"
        label = html.escape(annotation.label, quote=True)
        label_y = max(font_size, annotation.y1 - stroke_width * 2)
        elements.append(
            f'<rect x="{annotation.x1:.1f}" y="{annotation.y1:.1f}" '
            f'width="{max(0.0, annotation.x2 - annotation.x1):.1f}" '
            f'height="{max(0.0, annotation.y2 - annotation.y1):.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_width:.1f}"/>'
            f'<text x="{annotation.x1:.1f}" y="{label_y:.1f}" '
            f'font-family="sans-serif" font-size="{font_size:.1f}" '
            f'font-weight="700" fill="{color}" stroke="#000" '
            f'stroke-width="{text_stroke:.1f}" paint-order="stroke">{label}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
        f'aria-hidden="true">{"".join(elements)}</svg>'
    )


def predict(
    model_path: str,
    source_type: str,
    youtube_url: str,
    conf: float,
    infer_every: int,
    folder_files=None,
    webcam_index=None,
    video_file=None,
    browser_session_id: str = "",
):
    """영상 또는 이미지 폴더에서 추론한 RGB 프레임과 상태를 생성한다."""

    _stop_event.clear()
    resolved_model_path = (model_path or "").strip() or models.latest_trained_model()
    if not resolved_model_path:
        yield None, "학습된 모델이 없습니다. 먼저 학습 탭에서 학습을 완료하세요.", ""
        return
    if not Path(resolved_model_path).is_file():
        yield None, f"모델 파일 없음: {resolved_model_path}", ""
        return

    try:
        from ultralytics import YOLO

        yield None, f"모델 로딩 중... ({resolved_model_path})", ""
        model = YOLO(resolved_model_path)
    except Exception as exc:
        yield None, f"모델 로딩 실패: {exc}", ""
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
            browser_session_id=browser_session_id,
        )
        with media.open_video_capture(source) as capture:
            yield from _predict_video(
                model,
                names,
                class_ids,
                capture,
                float(conf),
                max(1, int(infer_every)),
                browser_webcam=isinstance(
                    source.value,
                    webcams.BrowserWebcamSource,
                ),
            )
    except media.MediaSourceError as exc:
        yield None, str(exc), ""
    except GeneratorExit:
        raise
    except Exception as exc:
        yield None, f"추론 오류: {exc}", ""


def _predict_video(
    model,
    names: dict,
    class_ids: list[int],
    capture,
    conf: float,
    infer_every: int,
    browser_webcam: bool = False,
):
    yield None, "추론 시작...", ""

    frame_index = 0
    last_boxes = None
    last_detection_count = 0
    inference_ms = 0.0
    last_preview_at = 0.0

    while not _stop_event.is_set():
        ok, frame_bgr = capture.read()
        if not ok:
            if frame_index == 0 and browser_webcam:
                yield (
                    None,
                    "접속 기기 카메라 프레임을 받지 못했습니다. "
                    "카메라 미리보기가 보이는지 확인한 뒤 다시 시작하세요.",
                    "",
                )
                return
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

            if browser_webcam:
                yield (
                    None,
                    f"프레임 {frame_index + 1}  |  감지 {last_detection_count}개 "
                    f"|  추론 {inference_ms:.0f}ms  |  skip={infer_every}",
                    _browser_overlay_svg(frame_bgr, last_boxes, names),
                )

        now = time.perf_counter()
        if not browser_webcam and now - last_preview_at >= _DISPLAY_INTERVAL:
            last_preview_at = now
            annotated = vision.draw_boxes(frame_bgr, last_boxes, names)
            yield (
                vision.to_rgb(annotated),
                f"프레임 {frame_index + 1}  |  감지 {last_detection_count}개 "
                f"|  추론 {inference_ms:.0f}ms  |  skip={infer_every}",
                "",
            )

        frame_index += 1

    prefix = "중지됨" if _stop_event.is_set() else "완료"
    yield None, f"추론 {prefix} — 총 {frame_index}프레임 처리", ""


def _predict_folder(
    model,
    names: dict,
    class_ids: list[int],
    folder_files,
    conf: float,
):
    images = media.filter_image_paths(folder_files)
    if not images:
        yield None, "업로드된 이미지가 없습니다. 이미지 폴더를 업로드하세요.", ""
        return

    yield None, f"이미지 폴더 추론 시작 — {len(images)}장", ""

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
                "",
            )
            _stop_event.wait(0.4)

    yield None, f"중지됨 — {shown_count}장 표시", ""
