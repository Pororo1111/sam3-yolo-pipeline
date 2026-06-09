import logging
import re
import threading
import time
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b.")


def _clean(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    if "\r" in text:
        text = text.split("\r")[-1]
    return text


_ROOT     = Path(__file__).resolve().parent.parent
YAML_PATH = _ROOT / "dataset/dataset.yaml"
BEST_PT   = _ROOT / "runs/detect/train/weights/best.pt"
_MODEL_PT = _ROOT / "models/yolo26n.pt"

_stop_event = threading.Event()


def stop():
    _stop_event.set()


def train(epochs: int, imgsz: int, batch: int, lr0: float, device: str):
    _stop_event.clear()

    if not YAML_PATH.exists():
        yield f"dataset.yaml 없음 — 먼저 Tab 3에서 데이터셋을 구성하세요.\n경로: {YAML_PATH.resolve()}"
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        yield "ultralytics 패키지를 찾을 수 없습니다."
        return

    if not _MODEL_PT.exists():
        yield f"모델 파일 없음: {_MODEL_PT}\nmodels/ 폴더에 yolo26n.pt를 배치하세요.\n"
        return

    yield f"YOLO 모델 로딩 중 ({_MODEL_PT.name})...\n"

    try:
        model = YOLO(str(_MODEL_PT))
    except Exception as e:
        yield f"모델 로딩 실패: {e}\n"
        return

    lines = []

    # ── 콜백 기반 로그 수집 ──────────────────────────────────────
    def on_train_start(trainer):
        lines.append(f"학습 시작 — 총 {trainer.epochs} 에폭")

    def on_train_epoch_end(trainer):
        if _stop_event.is_set():
            trainer.stop = True
            return
        m = trainer.metrics or {}
        loss = getattr(trainer, "loss", None)
        loss_str = f"  loss={float(loss):.4f}" if loss is not None else ""
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float))
        lines.append(
            f"Epoch {trainer.epoch + 1}/{trainer.epochs}{loss_str}"
            + (f"  {metrics_str}" if metrics_str else "")
        )

    def on_train_end(trainer):
        lines.append("── 학습 완료 ──")

    def on_val_end(validator):
        m = getattr(validator, "metrics", None)
        if m is None:
            return
        fitness = getattr(m, "fitness", None)
        map50   = getattr(m, "box", None)
        if map50 is not None:
            map50 = getattr(map50, "map50", None)
        parts = []
        if fitness is not None:
            parts.append(f"fitness={float(fitness):.4f}")
        if map50 is not None:
            parts.append(f"mAP50={float(map50):.4f}")
        if parts:
            lines.append("  Val: " + "  ".join(parts))

    # ultralytics logger → lines
    class _LogHandler(logging.Handler):
        def emit(self, record):
            msg = _clean(self.format(record)).strip()
            if msg:
                lines.append(msg)

    log_handler = _LogHandler()
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    ult_logger = logging.getLogger("ultralytics")
    ult_logger.addHandler(log_handler)

    model.add_callback("on_train_start",     on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end",       on_train_end)
    model.add_callback("on_val_end",         on_val_end)

    header = (
        f"학습 시작\n"
        f"  epochs={epochs}  imgsz={imgsz}  batch={batch}  lr0={lr0}  device={device}\n"
        f"  data={YAML_PATH}\n"
        + "─" * 60 + "\n"
    )
    accumulated = header
    yield accumulated

    result_box = [None]
    exc_box    = [None]
    done_event = threading.Event()

    def _run():
        try:
            result_box[0] = model.train(
                data=str(YAML_PATH),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                device=device if device != "auto" else None,
                verbose=True,
                plots=True,
                workers=0,
            )
        except Exception as e:
            exc_box[0] = e
        finally:
            ult_logger.removeHandler(log_handler)
            done_event.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    sent = 0
    while not done_event.is_set():
        if _stop_event.is_set():
            break
        time.sleep(0.5)
        new = lines[sent:]
        if new:
            accumulated += "\n".join(new) + "\n"
            yield accumulated
            sent += len(new)

    new = lines[sent:]
    if new:
        accumulated += "\n".join(new) + "\n"
        yield accumulated

    if exc_box[0] is not None:
        accumulated += f"\n오류 발생: {exc_box[0]}\n"
        yield accumulated
        return

    if _stop_event.is_set():
        accumulated += "\n학습 중지됨.\n"
        yield accumulated
        return

    best = _ROOT / "runs/detect/train/weights/best.pt"
    if best.exists():
        accumulated += f"\n학습 완료!\nbest.pt → {best.resolve()}\n"
    else:
        candidates = sorted((_ROOT / "runs/detect").glob("*/weights/best.pt"))
        if candidates:
            accumulated += f"\n학습 완료!\nbest.pt → {candidates[-1].resolve()}\n"
        else:
            accumulated += "\n학습 완료 (best.pt 경로를 찾지 못했습니다 — runs/detect/ 확인).\n"
    yield accumulated
