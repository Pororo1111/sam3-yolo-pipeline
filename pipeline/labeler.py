import threading
import cv2
import numpy as np
from pathlib import Path

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


def label(prompts_str: str, conf: float):
    """
    Generator — yields (rgb_preview | None, status_str)
    prompts_str: "person, car, bicycle"  (쉼표 구분)
    """
    _stop_event.clear()

    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    if not prompts:
        yield None, "클래스 프롬프트를 입력하세요. (예: person, car)"
        return

    frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not frames:
        yield None, f"추출된 프레임이 없습니다. 먼저 Tab 1에서 프레임을 추출하세요."
        return

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 평면 라벨 삭제 — 이전 소스의 오라벨이 새 이미지에 붙는 것 방지
    # (train/val 하위 폴더는 glob("*.txt")에 걸리지 않아 보존됨)
    for old in LABELS_DIR.glob("*.txt"):
        old.unlink()

    yield None, "SAM3 모델 로딩 중..."

    try:
        predictor = _get_predictor(conf)
    except Exception as e:
        yield None, f"모델 로딩 실패: {e}"
        return

    total = len(frames)
    done  = 0

    for frame_path in frames:
        if _stop_event.is_set():
            break

        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue
        img_h, img_w = frame_bgr.shape[:2]

        predictor.set_image(frame_bgr)
        results = predictor(text=prompts)

        label_lines = []

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

            # 라벨 시각화 (첫 마스크만 오버레이)
            preview_bgr = frame_bgr.copy()
            for mask, cls_id in zip(masks, cls_ids):
                mask_r = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                color = _class_color(cls_id)
                overlay = preview_bgr.copy()
                overlay[mask_r > 0] = color
                cv2.addWeighted(overlay, 0.4, preview_bgr, 0.6, 0, preview_bgr)
            rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 라벨 파일 저장 (마스크 없으면 빈 파일)
        label_path = LABELS_DIR / (frame_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))

        done += 1
        yield rgb, f"{done} / {total}  |  {frame_path.name}  →  {len(label_lines)}개 객체"

    if _stop_event.is_set():
        yield None, f"중지됨 — {done}/{total} 완료  →  {LABELS_DIR.resolve()}"
    else:
        yield None, f"라벨링 완료 — {done}장  →  {LABELS_DIR.resolve()}"


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
