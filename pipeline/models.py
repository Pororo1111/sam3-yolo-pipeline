"""모델 가중치 확인 및 자동 다운로드 헬퍼.

앱 시작 시 `models/` 폴더에 필요한 가중치가 없으면 ultralytics 에셋에서
자동으로 내려받는다. (현재는 학습용 베이스 모델 yolo26n.pt 대상)
"""

import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _ROOT / "models"

# (파일명, ultralytics 에셋에서 자동 다운로드 가능 여부)
_AUTO_DOWNLOAD = ["yolo26n.pt"]


def ensure_yolo_model(name: str = "yolo26n.pt") -> Path:
    """`models/<name>` 이 없으면 ultralytics 에셋에서 내려받아 배치하고 경로 반환.

    이미 존재하면 그대로 경로만 반환한다.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / name

    if target.exists():
        return target

    print(f"[models] '{name}' 이(가) models/ 에 없습니다. 다운로드를 시작합니다...")

    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError:
        print("[models] ultralytics 패키지를 찾을 수 없어 다운로드를 건너뜁니다. "
              "학습 전 models/ 폴더에 직접 배치하세요.")
        return target

    try:
        # attempt_download_asset 은 basename 으로 에셋을 찾아 지정 경로에 저장한다.
        attempt_download_asset(str(target))
    except Exception as e:
        print(f"[models] attempt_download_asset 실패({e}). YOLO 자동 다운로드로 재시도합니다...")
        try:
            from ultralytics import YOLO

            # YOLO 생성 시 가중치를 현재 작업 폴더로 자동 다운로드한다.
            YOLO(name)
            downloaded = Path(name)
            if downloaded.exists() and downloaded.resolve() != target.resolve():
                shutil.move(str(downloaded), str(target))
        except Exception as e2:
            print(f"[models] 자동 다운로드 실패: {e2}\n"
                  f"        models/ 폴더에 {name} 을(를) 직접 배치하세요.")
            return target

    if target.exists():
        print(f"[models] 다운로드 완료 → {target}")
    else:
        print(f"[models] 다운로드를 확인하지 못했습니다 → {target}")

    return target


def ensure_models() -> None:
    """앱 시작 시 호출 — 자동 다운로드 대상 모델을 모두 확인/설치한다."""
    for name in _AUTO_DOWNLOAD:
        ensure_yolo_model(name)
