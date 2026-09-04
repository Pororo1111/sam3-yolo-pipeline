import threading
import cv2
import numpy as np
from pathlib import Path

from pipeline import source_groups

FRAMES_DIR = Path("dataset/raw_frames")
LABELS_DIR = Path("dataset/labels")

_stop_event = threading.Event()
_predictor  = None   # 최초 1회만 로드


def _get_predictor(conf: float):
    global _predictor
    if _predictor is not None:
        _predictor.args.conf = conf
        return _predictor
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model="models/sam3.pt",
        half=False,
        save=False,
    )
    _predictor = SAM3SemanticPredictor(overrides=overrides)
    return _predictor


def stop():
    _stop_event.set()


def source_choices() -> list[tuple[str, str]]:
    """현재 추출 프레임의 소스 그룹 선택지를 반환한다."""

    return source_groups.source_choices(FRAMES_DIR)


def _frames_for_sources(selected_sources=None) -> list[Path]:
    frames = list(FRAMES_DIR.glob("frame_*.jpg"))
    if selected_sources is not None and not selected_sources:
        return []
    return source_groups.filter_frames(frames, selected_sources)


def _mask_to_yolo_bbox(mask_u8: np.ndarray, img_w: int, img_h: int):
    """bool/uint8 마스크 → (x_c, y_c, w, h) normalized"""
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    x_c = (xs.min() + xs.max()) / 2.0 / img_w
    y_c = (ys.min() + ys.max()) / 2.0 / img_h
    w   = (xs.max() - xs.min()) / img_w
    h   = (ys.max() - ys.min()) / img_h
    return x_c, y_c, w, h


def _infer_and_overlay(predictor, frame_bgr: np.ndarray, prompts: list[str]):
    """단일 프레임 SAM3 추론 → (rgb_overlay, label_lines, n_objects).

    라벨 파일은 저장하지 않는다 — 미리보기/전체 라벨링이 공유하는 순수 추론부.
    """
    img_h, img_w = frame_bgr.shape[:2]
    predictor.set_image(frame_bgr)
    results = predictor(text=prompts)

    label_lines: list[str] = []
    preview_bgr = frame_bgr.copy()

    if results and results[0].masks is not None:
        r       = results[0]
        masks   = r.masks.data.cpu().numpy().astype(np.uint8)
        cls_ids = (
            r.boxes.cls.cpu().numpy().astype(int)
            if r.boxes is not None
            else []
        )

        for mask, cls_id in zip(masks, cls_ids):
            mask_r = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            bbox = _mask_to_yolo_bbox(mask_r, img_w, img_h)
            if bbox is None:
                continue
            x_c, y_c, w, h = bbox
            label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            color = _class_color(cls_id)
            overlay = preview_bgr.copy()
            overlay[mask_r > 0] = color
            cv2.addWeighted(overlay, 0.4, preview_bgr, 0.6, 0, preview_bgr)

    rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    return rgb, label_lines, len(label_lines)


def preview(
    prompts_str: str,
    conf: float,
    n_preview: int,
    selected_sources=None,
):
    """미리보기 — 전체에서 균등 샘플링한 N장만 라벨 결과를 보여준다 (저장 안 함).

    Generator — yields (gallery_items, status_str)
    gallery_items: list of (rgb_np, caption)
    """
    _stop_event.clear()

    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    if not prompts:
        yield [], "클래스 프롬프트를 입력하세요. (예: person, car)"
        return

    frames = _frames_for_sources(selected_sources)
    if not frames:
        yield [], "선택한 소스에 추출된 프레임이 없습니다."
        return

    n = max(1, int(n_preview))
    if len(frames) <= n:
        sample = frames
    else:
        idxs = np.linspace(0, len(frames) - 1, n).astype(int)
        sample = [frames[i] for i in sorted(set(idxs.tolist()))]

    yield [], "SAM3 모델 로딩 중..."

    try:
        predictor = _get_predictor(conf)
    except Exception as e:
        yield [], f"모델 로딩 실패: {e}"
        return

    gallery: list = []
    total_obj = 0

    for i, frame_path in enumerate(sample):
        if _stop_event.is_set():
            yield gallery, f"미리보기 중지됨 — {len(gallery)}장"
            return

        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        rgb, _lines, n_obj = _infer_and_overlay(predictor, frame_bgr, prompts)
        total_obj += n_obj
        gallery.append((rgb, f"{frame_path.name} · {n_obj}개"))
        yield gallery, f"미리보기 {i + 1}/{len(sample)}  ·  누적 {total_obj}개 객체"

    yield gallery, (
        f"미리보기 완료 — {len(gallery)}장 샘플 · 총 {total_obj}개 객체. "
        f"결과가 괜찮으면 「전체 라벨링 시작」을 누르세요. (아직 라벨은 저장되지 않았습니다)"
    )


def label(prompts_str: str, conf: float, selected_sources=None):
    """
    전체 라벨링 — 모든 프레임에 추론하고 라벨 파일을 저장한다.
    Generator — yields (rgb_preview | None, status_str)
    prompts_str: "person, car, bicycle"  (쉼표 구분)
    """
    _stop_event.clear()

    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    if not prompts:
        yield None, "클래스 프롬프트를 입력하세요. (예: person, car)"
        return

    frames = _frames_for_sources(selected_sources)
    if not frames:
        yield None, "선택한 소스에 추출된 프레임이 없습니다."
        return

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 선택한 소스만 다시 라벨링한다. 다른 URL/웹캠 세션의 결과는 보존한다.
    for frame in frames:
        old = LABELS_DIR / f"{frame.stem}.txt"
        if old.exists():
            old.unlink()

    yield None, "SAM3 모델 로딩 중..."

    try:
        predictor = _get_predictor(conf)
    except Exception as e:
        yield None, f"모델 로딩 실패: {e}"
        return

    total = len(frames)
    selected_group_count = len(
        {source_groups.source_id_from_path(frame) for frame in frames}
    )
    done  = 0

    for frame_path in frames:
        if _stop_event.is_set():
            break

        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        rgb, label_lines, n_obj = _infer_and_overlay(predictor, frame_bgr, prompts)

        # 라벨 파일 저장 (마스크 없으면 빈 파일)
        label_path = LABELS_DIR / (frame_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))

        done += 1
        yield rgb, f"{done} / {total}  |  {frame_path.name}  →  {n_obj}개 객체"

    if _stop_event.is_set():
        yield None, f"중지됨 — {done}/{total} 완료  →  {LABELS_DIR.resolve()}"
    else:
        yield None, (
            f"라벨링 완료 — {selected_group_count}개 소스, {done}장  "
            f"→  {LABELS_DIR.resolve()}"
        )


def _class_color(cls_id: int) -> tuple:
    palette = [
        (255, 100,   0),
        (  0, 220, 255),
        (  0, 200,   0),
        (100,   0, 255),
        (255,   0, 128),
        (  0, 128, 255),
        (200, 200,   0),
        (128,   0, 128),
    ]
    return palette[cls_id % len(palette)]
