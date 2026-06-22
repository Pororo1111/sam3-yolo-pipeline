import shutil
import random
import cv2
import numpy as np
import yaml
from pathlib import Path

FRAMES_DIR  = Path("dataset/raw_frames")
LABELS_DIR  = Path("dataset/labels")
IMAGES_DIR  = Path("dataset/images")
YAML_PATH   = Path("dataset/dataset.yaml")

_CLASS_COLORS = [
    (255, 100,   0),
    (  0, 220, 255),
    (  0, 200,   0),
    (100,   0, 255),
    (255,   0, 128),
    (  0, 128, 255),
    (200, 200,   0),
    (128,   0, 128),
]


def _draw_bboxes(frame_bgr: np.ndarray, label_path: Path, class_names: list[str]) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        color = _CLASS_COLORS[cls_id % len(_CLASS_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(img, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def _has_label(stem: str) -> bool:
    lp = LABELS_DIR / (stem + ".txt")
    return lp.exists() and lp.stat().st_size > 0


def _gallery_frames(filter_empty: bool) -> list[Path]:
    """갤러리에 표시되는 것과 '동일한 순서'의 프레임 경로 리스트.

    갤러리 클릭(select) 시 evt.index → 이 리스트로 매핑해 정확한 프레임을 찾는다.
    """
    frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if filter_empty:
        frames = [f for f in frames if _has_label(f.stem)]
    return frames


def load_preview(prompts_str: str, filter_empty: bool):
    """
    라벨링 결과 미리보기 로드.
    Returns (gallery_items, stats_str)
    gallery_items: list of (rgb_np, caption)
    """
    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    all_frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))

    if not all_frames:
        return [], "추출된 프레임이 없습니다."

    labeled = sum(1 for f in all_frames if _has_label(f.stem))
    empty = len(all_frames) - labeled

    gallery = []
    for fp in _gallery_frames(filter_empty):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        lp = LABELS_DIR / (fp.stem + ".txt")
        annotated = _draw_bboxes(bgr, lp, prompts)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        obj_count = len([l for l in (lp.read_text().strip().splitlines() if lp.exists() else []) if l])
        caption = f"{fp.name}  ({obj_count}개 객체)"
        gallery.append((rgb, caption))

    stats = (
        f"전체 {len(all_frames)}장  |  라벨 있음 {labeled}장  |  라벨 없음 {empty}장"
        + ("  (빈 프레임 숨김)" if filter_empty else "")
    )
    return gallery, stats


def _read_yaml_names() -> dict[int, str]:
    """기존 dataset.yaml 의 names 매핑을 읽어 {id: name} 으로 반환."""
    if not YAML_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
    except Exception:
        pass
    return {}


def _count_class_ids() -> dict[int, int]:
    """labels/ 최상위 txt 를 스캔해 {class_id: 해당 클래스가 들어있는 프레임 수} 반환.

    같은 프레임에 같은 클래스가 여러 개 있어도 1로 센다(프레임 단위 집계).
    """
    counts: dict[int, int] = {}
    for lf in LABELS_DIR.glob("*.txt"):
        ids_in_frame = set()
        for line in lf.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                ids_in_frame.add(int(parts[0]))
        for cid in ids_in_frame:
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def scan_classes(label_prompts: str = "") -> list[dict]:
    """labels/ 를 스캔해 클래스 목록을 [{id, name, count}] 형태로 반환.

    count 는 해당 클래스가 들어있는 프레임(이미지) 수.

    이름 우선순위 (최신 라벨링 결과를 항상 우선):
      1) Tab 2 오토라벨링 프롬프트(label_prompts) — SAM3 프롬프트 순서 = 클래스 ID
      2) 기존 dataset.yaml 의 names
      3) class_{id}

    Tab 3 에서 직접 편집한 이름의 보존은 호출부(app.py)에서 "Tab 2 프롬프트가
    바뀌지 않았으면 재스캔하지 않음"으로 처리한다. (재라벨링 시 새 이름 반영)
    """
    counts = _count_class_ids()
    if not counts:
        return []

    lab = [p.strip() for p in label_prompts.split(",") if p.strip()]
    yaml_names = _read_yaml_names()

    classes = []
    for cid in sorted(counts):
        if cid < len(lab):
            name = lab[cid]
        elif cid in yaml_names:
            name = yaml_names[cid]
        else:
            name = f"class_{cid}"
        classes.append({"id": cid, "name": name, "count": counts[cid]})
    return classes


def select_frame(prompts_str: str, filter_empty: bool, evt):
    """갤러리 클릭 핸들러 — 클릭한 인덱스로 프레임을 찾아 상세 이미지/파일명 반환.

    캡션 파싱 대신 `evt.index`를 갤러리 순서(`_gallery_frames`)에 매핑 → Gradio
    버전별 SelectData 포맷 차이에 영향받지 않음.
    """
    idx = getattr(evt, "index", None)
    frames = _gallery_frames(filter_empty)

    if not isinstance(idx, int) or idx < 0 or idx >= len(frames):
        return None, "", "선택을 인식하지 못했습니다. 이미지를 다시 클릭하세요."

    fp = frames[idx]
    lp = LABELS_DIR / (fp.stem + ".txt")

    bgr = cv2.imread(str(fp))
    if bgr is None:
        return None, "", f"이미지를 불러올 수 없습니다: {fp.name}"

    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    annotated = _draw_bboxes(bgr, lp, prompts)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return rgb, fp.stem, f"선택됨: {fp.name} — 아래 「선택한 이미지 삭제」로 제거할 수 있습니다."


def delete_frame(frame_stem: str, prompts_str: str, filter_empty: bool):
    """선택된 프레임 이미지와 라벨을 삭제하고 갤러리를 갱신."""
    if not frame_stem:
        gallery, stats = load_preview(prompts_str, filter_empty)
        return gallery, stats, None, "", "삭제할 프레임을 먼저 선택하세요."

    fp = FRAMES_DIR / (frame_stem + ".jpg")
    lp = LABELS_DIR / (frame_stem + ".txt")

    deleted = []
    if fp.exists():
        fp.unlink()
        deleted.append(fp.name)
    if lp.exists():
        lp.unlink()
        deleted.append(lp.name)

    gallery, stats = load_preview(prompts_str, filter_empty)
    msg = f"삭제 완료: {', '.join(deleted)}" if deleted else "파일을 찾을 수 없습니다."
    return gallery, stats, None, "", msg


def build_dataset(prompts_str: str, val_ratio: float, filter_empty: bool):
    """
    train/val 분할 + dataset.yaml 생성.
    Yields status_str.
    """
    prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    if not prompts:
        yield "클래스 프롬프트를 입력하세요."
        return

    frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not frames:
        yield "추출된 프레임이 없습니다."
        return

    if filter_empty:
        frames = [f for f in frames if
                  (LABELS_DIR / (f.stem + ".txt")).exists() and
                  (LABELS_DIR / (f.stem + ".txt")).stat().st_size > 0]

    if not frames:
        yield "라벨이 있는 프레임이 없습니다. 먼저 Tab 2에서 오토라벨링을 실행하세요."
        return

    # 기존 분할 결과 삭제 — 누적/train·val 누수(같은 프레임이 양쪽에 섞임) 방지
    for split_name in ("train", "val"):
        for d in (IMAGES_DIR / split_name, Path("dataset/labels") / split_name):
            if d.exists():
                shutil.rmtree(d)
    yield "기존 train/val 폴더 정리 완료"

    random.shuffle(frames)
    split = max(1, int(len(frames) * (1 - val_ratio)))
    train_frames = frames[:split]
    val_frames   = frames[split:]

    for split_name, split_frames in [("train", train_frames), ("val", val_frames)]:
        img_dir = IMAGES_DIR / split_name
        lbl_dir = Path("dataset/labels") / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for fp in split_frames:
            shutil.copy(fp, img_dir / fp.name)
            lp = LABELS_DIR / (fp.stem + ".txt")
            if lp.exists():
                shutil.copy(lp, lbl_dir / lp.name)
            else:
                (lbl_dir / (fp.stem + ".txt")).write_text("")

        yield f"{split_name}: {len(split_frames)}장 복사 완료"

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(prompts))
    yaml_content = (
        f"path: {Path('dataset').resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(prompts)}\n"
        f"names:\n{names_yaml}\n"
    )
    YAML_PATH.write_text(yaml_content, encoding="utf-8")

    yield (
        f"완료 — train {len(train_frames)}장 / val {len(val_frames)}장\n"
        f"dataset.yaml → {YAML_PATH.resolve()}"
    )
