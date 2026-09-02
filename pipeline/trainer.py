from __future__ import annotations

import csv
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
_RUNTIME_DIR = _PROJECT / ".runtime"
_STATE_PATH = _RUNTIME_DIR / "training_state.json"
_LOG_PATH = _RUNTIME_DIR / "training.log"

_stop_event = threading.Event()
_runtime_lock = threading.RLock()
_worker_thread: threading.Thread | None = None
_runtime: dict[str, Any] = {
    "status": "idle",
    "message": "학습 대기 중",
    "pid": os.getpid(),
    "updated_at": "",
}

_ACTIVE_STATUSES = {"preparing", "running", "stopping"}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_state_locked() -> None:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = _STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_runtime, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(_STATE_PATH)


def _set_runtime(**values: Any) -> None:
    with _runtime_lock:
        _runtime.update(values)
        _runtime["updated_at"] = _now()
        _write_state_locked()


def _append_log(message: str) -> None:
    cleaned = _clean(str(message)).strip()
    if not cleaned:
        return
    with _runtime_lock:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(cleaned + "\n")


def _reset_log() -> None:
    with _runtime_lock:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        _LOG_PATH.write_text("", encoding="utf-8")


def _read_log(max_chars: int = 200_000) -> str:
    try:
        text = _LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > max_chars:
        return "… 이전 로그 생략 …\n" + text[-max_chars:]
    return text


def _load_runtime_state() -> None:
    global _runtime
    try:
        saved = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    if not isinstance(saved, dict):
        return
    _runtime = saved
    saved_pid = int(saved.get("pid") or 0)
    if saved.get("status") in _ACTIVE_STATUSES and saved_pid != os.getpid():
        try:
            os.kill(saved_pid, 0)
        except (OSError, ValueError):
            _runtime.update(
                status="interrupted",
                message="서버 종료로 이전 학습이 중단되었습니다.",
                updated_at=_now(),
            )
            _write_state_locked()


def _metrics_from_results(run_dir: str | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    csv_path = Path(run_dir) / "results.csv"
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    except (OSError, csv.Error):
        return {}
    if not rows:
        return {}
    row = rows[-1]

    def number(key: str) -> float | None:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "epoch": int(number("epoch") or 0),
        "box_loss": number("train/box_loss"),
        "cls_loss": number("train/cls_loss"),
        "dfl_loss": number("train/dfl_loss"),
        "precision": number("metrics/precision(B)"),
        "recall": number("metrics/recall(B)"),
        "map50": number("metrics/mAP50(B)"),
        "map50_95": number("metrics/mAP50-95(B)"),
    }


def training_snapshot() -> dict[str, Any]:
    """새 UI 세션에서도 복원할 수 있는 현재 학습 상태·로그·지표를 반환한다."""
    with _runtime_lock:
        snapshot = dict(_runtime)
    snapshot["active"] = snapshot.get("status") in _ACTIVE_STATUSES
    snapshot["log"] = _read_log()
    snapshot["metrics"] = _metrics_from_results(snapshot.get("run_dir"))
    return snapshot


_load_runtime_state()


def stop():
    _stop_event.set()
    snapshot = training_snapshot()
    if snapshot["active"]:
        _set_runtime(status="stopping", message="중지 요청을 처리하는 중입니다…")
        _append_log("중지 요청됨 — 현재 에폭 처리가 끝나면 안전하게 중지합니다.")
    return training_snapshot()


def _data_loading_options(device: str) -> tuple[int, str | bool]:
    """플랫폼과 무관하게 안정적인 단일 프로세스 로딩 설정을 사용한다."""

    return 0, False


def train(epochs: int, imgsz: int, batch: int, patience: int, device: str,
          name: str = "train", base_model: str = "",
          dataset_yamls: list[str] | None = None):
    global _worker_thread

    run_name = _safe_name(name)
    preparation_message = "선택한 데이터셋 검사 및 클래스 통합 준비 중..."
    with _runtime_lock:
        if _runtime.get("status") in _ACTIVE_STATUSES:
            blocked = True
        else:
            blocked = False
            _stop_event.clear()
            _reset_log()
            _set_runtime(
                status="preparing",
                message=preparation_message,
                pid=os.getpid(),
                started_at=_now(),
                finished_at=None,
                run_name=run_name,
                run_dir=None,
                current_epoch=0,
                total_epochs=int(epochs),
                parameters={
                    "epochs": int(epochs),
                    "imgsz": int(imgsz),
                    "batch": int(batch),
                    "patience": int(patience),
                    "device": str(device),
                    "base_model": str(base_model or ""),
                    "dataset_yamls": list(dataset_yamls or []),
                },
            )
    if blocked:
        yield "이미 학습이 진행 중입니다. 현재 학습을 중지한 뒤 다시 시도하세요."
        return

    _append_log(preparation_message)
    yield preparation_message + "\n"

    def fail(message: str):
        _append_log(message)
        _set_runtime(status="error", message=message, finished_at=_now())

    try:
        from pipeline.dataset_importer import (
            DatasetImportError,
            prepare_training_data,
        )

        training_yaml, dataset_description = prepare_training_data(dataset_yamls)
    except DatasetImportError as exc:
        message = f"데이터셋 검사 실패\n{exc}"
        fail(message)
        yield message + "\n"
        return
    except Exception as exc:
        message = f"데이터셋을 준비할 수 없습니다: {exc}"
        fail(message)
        yield message + "\n"
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        message = "ultralytics 패키지를 찾을 수 없습니다."
        fail(message)
        yield message
        return

    # 베이스 가중치 결정 — 지정 시 그 위에 이어학습(파인튜닝), 비우면 사전학습 yolo26n.pt
    base = (base_model or "").strip()
    if base:
        base_path = Path(base)
        if not base_path.exists():
            message = f"베이스 모델 파일을 찾을 수 없습니다: {base_path}"
            fail(message)
            yield message + "\n"
            return
    else:
        if not _MODEL_PT.exists():
            message = f"모델 파일 없음: {_MODEL_PT}\nmodels/ 폴더에 yolo26n.pt를 배치하세요."
            fail(message)
            yield message + "\n"
            return
        base_path = _MODEL_PT

    loading_message = f"YOLO 모델 로딩 중 ({base_path.name})..."
    _append_log(loading_message)
    _set_runtime(message=loading_message)
    yield loading_message + "\n"

    try:
        model = YOLO(str(base_path))
    except Exception as e:
        message = f"모델 로딩 실패: {e}"
        fail(message)
        yield message + "\n"
        return

    lines = []

    # ── 콜백 기반 로그 수집 ──────────────────────────────────────
    save_dir_box = [None]  # 실제 결과 저장 폴더 (이름 충돌 시 ultralytics가 숫자를 붙임)

    def on_train_start(trainer):
        save_dir_box[0] = Path(trainer.save_dir)
        message = f"학습 시작 — 총 {trainer.epochs} 에폭  (결과 폴더: {trainer.save_dir})"
        lines.append(message)
        _append_log(message)
        _set_runtime(
            status="running",
            message=f"학습 중 · 0/{trainer.epochs} epoch",
            run_dir=str(Path(trainer.save_dir).resolve()),
            total_epochs=int(trainer.epochs),
        )

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
        _set_runtime(
            current_epoch=int(trainer.epoch + 1),
            message=f"학습 중 · {trainer.epoch + 1}/{trainer.epochs} epoch",
        )

    def on_train_end(trainer):
        message = "── 학습 루프 종료 ──"
        lines.append(message)
        _append_log(message)

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
            message = "  Val: " + "  ".join(parts)
            lines.append(message)
            _append_log(message)

    # ultralytics logger → lines
    class _LogHandler(logging.Handler):
        def emit(self, record):
            msg = _clean(self.format(record)).strip()
            if msg:
                lines.append(msg)
                _append_log(msg)

    log_handler = _LogHandler()
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    ult_logger = logging.getLogger("ultralytics")
    ult_logger.addHandler(log_handler)

    model.add_callback("on_train_start",     on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end",       on_train_end)
    model.add_callback("on_val_end",         on_val_end)

    workers, cache = _data_loading_options(device)
    base_desc = f"{base_path.name} (이어학습)" if base else f"{base_path.name} (처음부터)"
    header = (
        f"학습 시작\n"
        f"  base={base_desc}  name={run_name}\n"
        f"  epochs={epochs}  patience={patience}  imgsz={imgsz}  batch={batch}  device={device}\n"
        f"  workers={workers}  cache={cache}\n"
        f"  dataset={dataset_description}\n"
        f"  data={training_yaml}\n"
        + "─" * 60 + "\n"
    )
    accumulated = header
    _append_log(header)
    _set_runtime(
        message="YOLO 학습 실행을 시작합니다…",
        dataset=str(training_yaml),
        dataset_description=dataset_description,
        base_model=str(base_path.resolve()),
    )
    yield accumulated

    result_box = [None]
    exc_box    = [None]
    done_event = threading.Event()

    def _run():
        global _worker_thread
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
                workers=workers,
                cache=cache,
            )
        except Exception as e:
            exc_box[0] = e
            message = f"오류 발생: {e}"
            _append_log(message)
            _set_runtime(status="error", message=message, finished_at=_now())
        finally:
            ult_logger.removeHandler(log_handler)
            if exc_box[0] is None:
                if _stop_event.is_set():
                    final_status = "stopped"
                    final_message = "학습이 사용자 요청으로 중지되었습니다."
                else:
                    final_status = "completed"
                    final_message = "학습이 완료되었습니다."
                _append_log(final_message)
                _set_runtime(
                    status=final_status,
                    message=final_message,
                    finished_at=_now(),
                )
            done_event.set()
            _worker_thread = None

    _worker_thread = threading.Thread(target=_run, daemon=True, name="yolo-training")
    _worker_thread.start()

    sent = 0
    while not done_event.is_set():
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
