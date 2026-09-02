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


def _safe_name(name: str) -> str:
    """학습 결과 폴더명으로 안전한 문자열 반환 (경로 구분자/금지문자 제거, 빈값은 train)."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name or "train"


_ROOT     = Path(__file__).resolve().parent.parent
_PROJECT  = _ROOT / "runs" / "detect"
_MODEL_PT = _ROOT / "models/yolo26n.pt"

_stop_event = threading.Event()


def stop():
    _stop_event.set()


def train(epochs: int, imgsz: int, batch: int, patience: int, device: str,
          name: str = "train", base_model: str = "",
          dataset_yamls: list[str] | None = None):
    _stop_event.clear()

    run_name = _safe_name(name)
    yield "선택한 데이터셋 검사 및 클래스 통합 준비 중...\n"

    try:
        from pipeline.dataset_importer import (
            DatasetImportError,
            prepare_training_data,
        )

        training_yaml, dataset_description = prepare_training_data(dataset_yamls)
    except DatasetImportError as exc:
        yield f"데이터셋 검사 실패\n{exc}\n"
        return
    except Exception as exc:
        yield f"데이터셋을 준비할 수 없습니다: {exc}\n"
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        yield "ultralytics 패키지를 찾을 수 없습니다."
        return

    # 베이스 가중치 결정 — 지정 시 그 위에 이어학습(파인튜닝), 비우면 사전학습 yolo26n.pt
    base = (base_model or "").strip()
    if base:
        base_path = Path(base)
        if not base_path.exists():
            yield f"베이스 모델 파일을 찾을 수 없습니다: {base_path}\n"
            return
    else:
        if not _MODEL_PT.exists():
            yield f"모델 파일 없음: {_MODEL_PT}\nmodels/ 폴더에 yolo26n.pt를 배치하세요.\n"
            return
        base_path = _MODEL_PT

    yield f"YOLO 모델 로딩 중 ({base_path.name})...\n"

    try:
        model = YOLO(str(base_path))
    except Exception as e:
        yield f"모델 로딩 실패: {e}\n"
        return

    lines = []

    # ── 콜백 기반 로그 수집 ──────────────────────────────────────
    save_dir_box = [None]  # 실제 결과 저장 폴더 (이름 충돌 시 ultralytics가 숫자를 붙임)

    def on_train_start(trainer):
        save_dir_box[0] = Path(trainer.save_dir)
        lines.append(f"학습 시작 — 총 {trainer.epochs} 에폭  (결과 폴더: {trainer.save_dir})")

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

    base_desc = f"{base_path.name} (이어학습)" if base else f"{base_path.name} (처음부터)"
    header = (
        f"학습 시작\n"
        f"  base={base_desc}  name={run_name}\n"
        f"  epochs={epochs}  patience={patience}  imgsz={imgsz}  batch={batch}  device={device}\n"
        f"  dataset={dataset_description}\n"
        f"  data={training_yaml}\n"
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
                data=str(training_yaml),
                epochs=epochs,
                patience=patience,
                imgsz=imgsz,
                batch=batch,
                device=device if device != "auto" else None,
                project=str(_PROJECT),
                name=run_name,
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

    best = (save_dir_box[0] / "weights" / "best.pt") if save_dir_box[0] else None
    if best and best.exists():
        accumulated += f"\n학습 완료!\nbest.pt → {best.resolve()}\n"
    else:
        candidates = sorted(_PROJECT.glob("*/weights/best.pt"))
        if candidates:
            accumulated += f"\n학습 완료!\nbest.pt → {candidates[-1].resolve()}\n"
        else:
            accumulated += "\n학습 완료 (best.pt 경로를 찾지 못했습니다 — runs/detect/ 확인).\n"
    yield accumulated
