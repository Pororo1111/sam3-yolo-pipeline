import re
import threading
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b.")


def _clean(text: str) -> str:
    """ANSI 이스케이프 제거 + \r 덮어쓰기 처리 → 최종 보이는 텍스트만 반환"""
    text = _ANSI_RE.sub("", text)
    # \r이 있으면 tqdm이 줄을 덮어쓴 것 — 마지막 조각만 남김
    if "\r" in text:
        text = text.split("\r")[-1]
    return text

YAML_PATH  = Path("dataset/dataset.yaml")
BEST_PT    = Path("runs/detect/train/weights/best.pt")

_stop_event = threading.Event()


def stop():
    _stop_event.set()


def train(epochs: int, imgsz: int, batch: int, lr0: float, device: str):
    """
    Generator — yields log_str
    """
    _stop_event.clear()

    if not YAML_PATH.exists():
        yield f"dataset.yaml 없음 — 먼저 Tab 3에서 데이터셋을 구성하세요.\n경로: {YAML_PATH.resolve()}"
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        yield "ultralytics 패키지를 찾을 수 없습니다."
        return

    yield "YOLO 모델 로딩 중 (yolo26n.pt)...\n"

    try:
        model = YOLO("models/yolo26n.pt")
    except Exception as e:
        yield f"모델 로딩 실패: {e}\n"
        return

    log_buf = []

    def on_train_epoch_end(trainer):
        if _stop_event.is_set():
            trainer.stop = True

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    yield (
        f"학습 시작\n"
        f"  epochs={epochs}  imgsz={imgsz}  batch={batch}  lr0={lr0}  device={device}\n"
        f"  data={YAML_PATH.resolve()}\n"
        + "─" * 60 + "\n"
    )

    train_kwargs = dict(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device if device != "auto" else None,
        verbose=True,
        plots=True,
        workers=0,  # Windows에서 DataLoader 멀티프로세싱 spawn 오류 방지
    )

    import io, sys

    class _LogCapture(io.TextIOBase):
        def __init__(self, callback):
            self._cb = callback
            self._buf = ""

        def write(self, s):
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                cleaned = _clean(line)
                if cleaned.strip():
                    self._cb(cleaned)
            return len(s)

        def flush(self):
            pass

    lines = []
    result_box = [None]
    exc_box    = [None]
    done_event = threading.Event()

    def _run():
        try:
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = _LogCapture(lambda l: lines.append(l))
            try:
                result_box[0] = model.train(**train_kwargs)
            finally:
                sys.stdout = old_stdout
        except Exception as e:
            exc_box[0] = e
        finally:
            done_event.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    import time
    sent = 0
    while not done_event.is_set():
        if _stop_event.is_set():
            break
        time.sleep(0.5)
        new = lines[sent:]
        if new:
            yield "\n".join(new) + "\n"
            sent += len(new)

    # 남은 로그 flush
    new = lines[sent:]
    if new:
        yield "\n".join(new) + "\n"

    if exc_box[0] is not None:
        yield f"\n오류 발생: {exc_box[0]}\n"
        return

    if _stop_event.is_set():
        yield "\n학습 중지됨.\n"
        return

    best = Path("runs/detect/train/weights/best.pt")
    if best.exists():
        yield f"\n학습 완료!\nbest.pt → {best.resolve()}\n"
    else:
        # ultralytics는 run 번호를 올려 저장하기도 함
        candidates = sorted(Path("runs/detect").glob("*/weights/best.pt"))
        if candidates:
            yield f"\n학습 완료!\nbest.pt → {candidates[-1].resolve()}\n"
        else:
            yield "\n학습 완료 (best.pt 경로를 찾지 못했습니다 — runs/detect/ 확인).\n"
