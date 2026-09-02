"""외부 YOLO 데이터셋 등록, 검증, 다중 학습용 YAML 구성."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import yaml


_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_YAML = _ROOT / "dataset" / "dataset.yaml"
EXTERNAL_DATASETS_DIR = _ROOT / "dataset" / "external"
ROBOFLOW_DATASETS_DIR = EXTERNAL_DATASETS_DIR / "roboflow"
COMBINED_DATASETS_DIR = _ROOT / "dataset" / ".combined"

_IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_YAML_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
_MAX_ARCHIVE_FILES = 200_000
_MAX_ARCHIVE_BYTES = 100 * 1024**3
_EXTRACT_LOCK = threading.Lock()
_STAGE_LOCK = threading.Lock()
_STAGING_SCHEMA_VERSION = 1
_ROBOFLOW_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class DatasetImportError(ValueError):
    """사용자에게 표시할 수 있는 외부 데이터셋 오류."""


def parse_roboflow_universe_url(url: str) -> tuple[str, str, int | None]:
    """Universe 프로젝트 URL에서 workspace, project, 선택 버전을 추출한다."""
    raw_url = str(url or "").strip()
    if not raw_url:
        raise DatasetImportError("Roboflow Universe URL을 입력하세요.")
    if "://" not in raw_url:
        raw_url = f"https://{raw_url}"

    parsed = urlparse(raw_url)
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise DatasetImportError("Roboflow Universe URL은 HTTP(S) 주소여야 합니다.")
    if (parsed.hostname or "").casefold() != "universe.roboflow.com":
        raise DatasetImportError(
            "universe.roboflow.com 프로젝트 URL만 사용할 수 있습니다."
        )

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise DatasetImportError(
            "URL에 workspace와 project가 필요합니다. "
            "예: https://universe.roboflow.com/workspace/project"
        )
    workspace, project = parts[:2]
    if not _ROBOFLOW_SLUG_RE.fullmatch(workspace) or not _ROBOFLOW_SLUG_RE.fullmatch(project):
        raise DatasetImportError("Roboflow workspace/project 형식이 올바르지 않습니다.")

    version: int | None = None
    if len(parts) >= 4 and parts[2].casefold() == "dataset":
        try:
            version = int(parts[3])
        except ValueError as exc:
            raise DatasetImportError("Roboflow 데이터셋 버전은 정수여야 합니다.") from exc
        if version <= 0:
            raise DatasetImportError("Roboflow 데이터셋 버전은 1 이상이어야 합니다.")
    return workspace, project, version


def _roboflow_project(api_key: str, workspace: str, project: str):
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise DatasetImportError(
            "roboflow 패키지가 설치되지 않았습니다. requirements.txt를 설치하세요."
        ) from exc
    return Roboflow(api_key=api_key).workspace(workspace).project(project)


def _roboflow_version(project: Any, requested: int | None) -> tuple[Any, int]:
    if requested is not None:
        return project.version(requested), requested

    versions = project.versions()
    numbered = []
    for version in versions:
        try:
            numbered.append((int(version.version), version))
        except (AttributeError, TypeError, ValueError):
            continue
    if not numbered:
        raise DatasetImportError("다운로드 가능한 Roboflow 데이터셋 버전이 없습니다.")
    version_number, version = max(numbered, key=lambda item: item[0])
    return version, version_number


def download_roboflow_universe(
    url: str,
    records: list[dict[str, Any]] | None = None,
    *,
    download_root: Path | None = None,
    api_key: str | None = None,
    model_format: str = "yolo26",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Universe 데이터셋을 내려받아 검사하고 기존 레지스트리에 합친다."""
    workspace, project_slug, requested_version = parse_roboflow_universe_url(url)
    secret = str(api_key or os.getenv("ROBOFLOW_API_KEY", "")).strip()
    if not secret:
        raise DatasetImportError(
            ".env 또는 환경변수에 ROBOFLOW_API_KEY를 등록하세요."
        )

    try:
        project = _roboflow_project(secret, workspace, project_slug)
        version, version_number = _roboflow_version(project, requested_version)
    except DatasetImportError:
        raise
    except Exception as exc:
        message = str(exc).replace(secret, "***")
        raise DatasetImportError(f"Roboflow 프로젝트 조회 실패: {message}") from exc

    root = (download_root or ROBOFLOW_DATASETS_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_name = f"{workspace}-{project_slug}-v{version_number}-{model_format}"
    destination = root / base_name

    # 완전히 내려받은 동일 버전은 다시 받지 않는다. 불완전한 동명 폴더가 있으면
    # 보존한 채 새 경로를 사용해 사용자 파일을 덮어쓰지 않는다.
    existing_yamls = discover_dataset_yamls(destination) if destination.is_dir() else []
    if not existing_yamls and destination.exists():
        destination = root / f"{base_name}-{uuid.uuid4().hex[:8]}"

    if not existing_yamls:
        try:
            downloaded = version.download(
                model_format,
                location=str(destination),
                overwrite=False,
            )
            downloaded_location = Path(
                getattr(downloaded, "location", destination)
            ).resolve()
        except Exception as exc:
            message = str(exc).replace(secret, "***")
            raise DatasetImportError(f"Roboflow 데이터셋 다운로드 실패: {message}") from exc
        yaml_paths = discover_dataset_yamls(downloaded_location)
    else:
        yaml_paths = existing_yamls

    if not yaml_paths:
        raise DatasetImportError("다운로드 결과에서 data.yaml을 찾지 못했습니다.")

    added: list[dict[str, Any]] = []
    errors: list[str] = []
    for yaml_path in yaml_paths:
        try:
            record = validate_dataset(yaml_path)
            record["source"] = (
                f"Roboflow Universe · {workspace}/{project_slug} "
                f"v{version_number} · {model_format}"
            )
            added.append(record)
        except DatasetImportError as exc:
            errors.append(f"{yaml_path.name}: {exc}")

    if not added:
        detail = "\n".join(f"- {error}" for error in errors)
        raise DatasetImportError(f"다운로드한 데이터셋 검증에 실패했습니다.\n{detail}")
    added = _merge_records([], added)
    return _merge_records(records, added), added, version_number


def _natural_key(value: str) -> list[str | int]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def _read_yaml(yaml_path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetImportError(f"YAML을 읽을 수 없습니다: {exc}") from exc
    if not isinstance(data, dict):
        raise DatasetImportError("데이터셋 YAML의 최상위 값은 매핑이어야 합니다.")
    return data


def _class_names(data: dict[str, Any]) -> list[str]:
    raw_names = data.get("names")
    nc = data.get("nc")

    if isinstance(raw_names, list):
        names = [str(name).strip() for name in raw_names]
    elif isinstance(raw_names, dict):
        try:
            indexed = {int(key): str(value).strip() for key, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetImportError("names 매핑의 클래스 ID는 정수여야 합니다.") from exc
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise DatasetImportError(
                f"names 클래스 ID는 0부터 연속이어야 합니다: {sorted(indexed)}"
            )
        names = [indexed[index] for index in expected]
    elif raw_names is None and isinstance(nc, int) and nc > 0:
        names = [f"class_{index}" for index in range(nc)]
    else:
        raise DatasetImportError("names 또는 양의 정수 nc 항목이 필요합니다.")

    if not names or any(not name for name in names):
        raise DatasetImportError("클래스 이름은 한 개 이상이며 빈 값이 없어야 합니다.")
    if len({name.casefold() for name in names}) != len(names):
        raise DatasetImportError("클래스 이름은 대소문자를 무시했을 때 중복될 수 없습니다.")
    if nc is not None and (not isinstance(nc, int) or nc != len(names)):
        raise DatasetImportError(
            f"nc({nc})와 names 개수({len(names)})가 일치하지 않습니다."
        )
    return names


def _dataset_root(data: dict[str, Any], yaml_path: Path) -> Path:
    raw_root = data.get("path")
    if not raw_root:
        return yaml_path.parent.resolve()

    root = Path(os.path.expandvars(os.path.expanduser(str(raw_root))))
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _resolve_split_entry(root: Path, raw_entry: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_entry.strip()))
    path = Path(expanded)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()

    # Roboflow YAML은 YAML이 데이터셋 루트에 있어도 "../train/images"를
    # 사용하는 경우가 많다. Ultralytics와 동일하게 한 단계 제거 경로도 시도한다.
    if not path.exists() and expanded.replace("\\", "/").startswith("../"):
        path = (root / expanded.replace("\\", "/")[3:]).resolve()
    return path


def _resolve_split(
    data: dict[str, Any],
    yaml_path: Path,
    split: str,
) -> list[Path]:
    raw = data.get(split)
    if split == "val" and raw is None:
        raw = data.get("validation")
    if raw is None:
        raise DatasetImportError(f"YAML에 '{split}' 항목이 없습니다.")

    values = raw if isinstance(raw, list) else [raw]
    if not values or any(not isinstance(value, (str, os.PathLike)) for value in values):
        raise DatasetImportError(f"'{split}'은 경로 문자열 또는 경로 목록이어야 합니다.")

    root = _dataset_root(data, yaml_path)
    paths = [_resolve_split_entry(root, str(value)) for value in values]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:3])
        raise DatasetImportError(f"'{split}' 이미지 경로를 찾을 수 없습니다.\n{preview}")
    return paths


def _images_from_path(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            (
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.casefold() in _IMAGE_SUFFIXES
            ),
            key=lambda item: _natural_key(str(item)),
        )

    if path.suffix.casefold() == ".txt":
        images: list[Path] = []
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            raise DatasetImportError(f"이미지 목록 파일을 읽을 수 없습니다: {path}") from exc
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            image_path = Path(os.path.expandvars(os.path.expanduser(line)))
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve()
            if image_path.is_file() and image_path.suffix.casefold() in _IMAGE_SUFFIXES:
                images.append(image_path)
            else:
                raise DatasetImportError(
                    f"이미지 목록에 존재하지 않는 파일이 있습니다: {image_path}"
                )
        return images

    if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES:
        return [path]
    raise DatasetImportError(f"지원하지 않는 이미지 경로입니다: {path}")


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [
        index for index, part in enumerate(parts) if part.casefold() == "images"
    ]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _validate_labels(images: Iterable[Path], class_count: int) -> tuple[int, int, int]:
    box_count = 0
    missing_count = 0
    empty_count = 0
    errors: list[str] = []
    error_count = 0

    def add_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 10:
            errors.append(message)

    for image_path in images:
        label_path = _label_path(image_path)
        if not label_path.exists():
            missing_count += 1
            continue
        try:
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            add_error(f"{label_path}: 읽기 실패 ({exc})")
            continue
        nonempty_lines = [line.strip() for line in lines if line.strip()]
        if not nonempty_lines:
            empty_count += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.split()
            reason = ""
            if len(parts) != 5:
                reason = f"열 {len(parts)}개 (탐지 라벨은 5개 필요)"
            else:
                try:
                    class_value = float(parts[0])
                    coords = [float(value) for value in parts[1:]]
                    values = [class_value, *coords]
                    if not all(math.isfinite(value) for value in values):
                        reason = "NaN 또는 무한대 값"
                    elif not class_value.is_integer():
                        reason = f"정수가 아닌 클래스 ID {parts[0]}"
                    elif not 0 <= int(class_value) < class_count:
                        reason = (
                            f"클래스 ID {int(class_value)}가 허용 범위 "
                            f"0~{class_count - 1} 밖입니다"
                        )
                    elif not all(0.0 <= value <= 1.0 for value in coords):
                        reason = "정규화 좌표가 0~1 범위 밖입니다"
                    elif coords[2] <= 0.0 or coords[3] <= 0.0:
                        reason = "bbox 너비 또는 높이가 0 이하입니다"
                except ValueError:
                    reason = "숫자가 아닌 값"
            if reason:
                add_error(f"{label_path}:{line_number} — {reason}")
            else:
                box_count += 1

    if error_count:
        detail = "\n".join(f"- {error}" for error in errors)
        remainder = error_count - len(errors)
        if remainder > 0:
            detail += f"\n- 그 외 {remainder}개"
        raise DatasetImportError(f"잘못된 YOLO 라벨을 발견했습니다.\n{detail}")
    return box_count, missing_count, empty_count


def validate_dataset(yaml_file: str | os.PathLike[str]) -> dict[str, Any]:
    """YOLO 탐지 데이터셋을 검사하고 UI/학습용 직렬화 가능한 정보를 반환한다."""
    yaml_path = Path(
        os.path.expandvars(os.path.expanduser(str(yaml_file).strip().strip("\"'")))
    )
    if not yaml_path.is_absolute():
        yaml_path = Path.cwd() / yaml_path
    yaml_path = yaml_path.resolve()
    if not yaml_path.is_file():
        raise DatasetImportError(f"데이터셋 YAML을 찾을 수 없습니다: {yaml_path}")

    data = _read_yaml(yaml_path)
    names = _class_names(data)
    train_paths = _resolve_split(data, yaml_path, "train")
    val_paths = _resolve_split(data, yaml_path, "val")
    test_paths = (
        _resolve_split(data, yaml_path, "test") if data.get("test") is not None else []
    )

    split_images: dict[str, list[Path]] = {}
    for split, paths in (
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths),
    ):
        images: list[Path] = []
        for path in paths:
            images.extend(_images_from_path(path))
        if split in {"train", "val"} and not images:
            raise DatasetImportError(f"'{split}' 경로에 지원되는 이미지가 없습니다.")
        unique_images = {str(path.resolve()).casefold() for path in images}
        if len(unique_images) != len(images):
            raise DatasetImportError(
                f"'{split}' 이미지 경로 목록에 중복 이미지가 "
                f"{len(images) - len(unique_images):,}장 있습니다."
            )
        split_images[split] = images

    train_set = {str(path.resolve()).casefold() for path in split_images["train"]}
    val_set = {str(path.resolve()).casefold() for path in split_images["val"]}
    overlap = train_set & val_set
    if overlap:
        sample = next(iter(overlap))
        raise DatasetImportError(
            f"train과 val에 같은 이미지가 {len(overlap):,}장 포함되어 있습니다: {sample}"
        )

    label_stats = {
        split: _validate_labels(split_images[split], len(names))
        for split in ("train", "val", "test")
    }
    if label_stats["train"][0] == 0:
        raise DatasetImportError(
            "train 데이터에 유효한 bbox가 없습니다. 빈/누락 라벨만으로는 탐지 모델을 "
            "학습할 수 없습니다."
        )
    box_count = sum(stats[0] for stats in label_stats.values())
    missing_labels = sum(stats[1] for stats in label_stats.values())
    empty_labels = sum(stats[2] for stats in label_stats.values())

    return {
        "id": str(yaml_path).casefold(),
        "name": yaml_path.parent.name,
        "yaml": str(yaml_path),
        "classes": names,
        "train_paths": [path.as_posix() for path in train_paths],
        "val_paths": [path.as_posix() for path in val_paths],
        "test_paths": [path.as_posix() for path in test_paths],
        "train_images": len(split_images["train"]),
        "val_images": len(split_images["val"]),
        "test_images": len(split_images["test"]),
        "boxes": box_count,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
    }


def discover_dataset_yamls(folder: str | os.PathLike[str]) -> list[Path]:
    raw_folder = str(folder or "").strip().strip("\"'")
    if not raw_folder:
        raise DatasetImportError("압축 해제된 데이터셋 폴더가 비어 있습니다.")
    root = Path(os.path.expandvars(os.path.expanduser(raw_folder)))
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if not root.is_dir():
        raise DatasetImportError(f"압축 해제된 데이터셋 폴더를 찾을 수 없습니다: {root}")

    direct = [root / name for name in _YAML_NAMES if (root / name).is_file()]
    candidates = direct or [
        path
        for name in _YAML_NAMES
        for path in root.rglob(name)
        if path.is_file()
    ]
    unique = sorted({path.resolve() for path in candidates}, key=lambda p: str(p))
    if not unique:
        raise DatasetImportError(
            "폴더에서 data.yaml 또는 dataset.yaml을 찾을 수 없습니다."
        )
    return unique


def _merge_records(
    records: list[dict[str, Any]] | None,
    new_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {
        str(record["yaml"]).casefold(): dict(record)
        for record in (records or [])
        if isinstance(record, dict) and record.get("yaml")
    }
    for record in new_records:
        merged[str(record["yaml"]).casefold()] = dict(record)
    return sorted(merged.values(), key=lambda record: _natural_key(record["name"]))


def _archive_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(archive: Path, extract_root: Path) -> Path:
    if not zipfile.is_zipfile(archive):
        raise DatasetImportError(f"올바른 ZIP 파일이 아닙니다: {archive.name}")

    digest = _archive_digest(archive)
    safe_stem = re.sub(r'[<>:"/\\|?*]+', "_", archive.stem).strip(" .") or "dataset"
    destination = (extract_root / f"{safe_stem}-{digest[:12]}").resolve()
    extract_root = extract_root.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        return destination

    temporary = (extract_root / f".{destination.name}.extracting").resolve()
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive) as zip_file:
            members = zip_file.infolist()
            if len(members) > _MAX_ARCHIVE_FILES:
                raise DatasetImportError(
                    f"ZIP 파일 항목이 너무 많습니다: {len(members):,}개"
                )
            total_size = sum(member.file_size for member in members)
            if total_size > _MAX_ARCHIVE_BYTES:
                raise DatasetImportError(
                    f"ZIP 압축 해제 크기가 너무 큽니다: {total_size / 1024**3:.1f}GB"
                )

            for member in members:
                member_path = PurePosixPath(member.filename.replace("\\", "/"))
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise DatasetImportError(
                        f"ZIP에 안전하지 않은 경로가 있습니다: {member.filename}"
                    )
                target = (temporary / Path(*member_path.parts)).resolve()
                if not target.is_relative_to(temporary):
                    raise DatasetImportError(
                        f"ZIP에 안전하지 않은 경로가 있습니다: {member.filename}"
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def register_archives(
    archive_files: list[str] | str | None,
    records: list[dict[str, Any]] | None = None,
    *,
    extract_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """업로드한 ZIP 하나 이상을 안전하게 풀고 포함된 데이터셋을 등록한다."""
    paths = [archive_files] if isinstance(archive_files, str) else list(archive_files or [])
    if not paths:
        raise DatasetImportError("업로드할 YOLO 데이터셋 ZIP을 선택하세요.")

    root = (extract_root or EXTERNAL_DATASETS_DIR).resolve()
    added: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw_path in paths:
        archive = Path(raw_path).resolve()
        try:
            with _EXTRACT_LOCK:
                extracted = _safe_extract_zip(archive, root)
            for yaml_path in discover_dataset_yamls(extracted):
                try:
                    record = validate_dataset(yaml_path)
                    record["source"] = f"ZIP · {archive.name}"
                    added.append(record)
                except DatasetImportError as exc:
                    errors.append(f"{archive.name}/{yaml_path.name}: {exc}")
        except (OSError, zipfile.BadZipFile, DatasetImportError) as exc:
            errors.append(f"{archive.name}: {exc}")

    if not added:
        detail = "\n".join(f"- {error}" for error in errors)
        raise DatasetImportError(f"학습 가능한 데이터셋을 불러오지 못했습니다.\n{detail}")
    added = _merge_records([], added)
    return _merge_records(records, added), added


def initial_registry() -> list[dict[str, Any]]:
    """현재 파이프라인 데이터셋이 유효하면 초기 목록에 포함한다."""
    if not DEFAULT_DATASET_YAML.is_file():
        return []
    try:
        record = validate_dataset(DEFAULT_DATASET_YAML)
    except DatasetImportError:
        return []
    record["name"] = "파이프라인 데이터셋"
    record["source"] = "데이터셋 검토 & 구성"
    return [record]


def refresh_registry(
    records: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """등록 경로를 다시 검사하고 삭제·손상된 데이터셋을 목록에서 제외한다."""
    yamls = [
        str(record["yaml"])
        for record in (records or [])
        if isinstance(record, dict) and record.get("yaml")
    ]
    if DEFAULT_DATASET_YAML.is_file():
        yamls.append(str(DEFAULT_DATASET_YAML))

    refreshed: list[dict[str, Any]] = []
    errors: list[str] = []
    for yaml_file in dict.fromkeys(path.casefold() for path in yamls):
        original = next(path for path in yamls if path.casefold() == yaml_file)
        old = next(
            (
                record
                for record in (records or [])
                if str(record.get("yaml", "")).casefold() == yaml_file
            ),
            {},
        )
        try:
            record = validate_dataset(original)
            if Path(record["yaml"]) == DEFAULT_DATASET_YAML.resolve():
                record["name"] = "파이프라인 데이터셋"
                record["source"] = "데이터셋 검토 & 구성"
            else:
                record["source"] = old.get("source", "외부 데이터셋")
            refreshed.append(record)
        except DatasetImportError as exc:
            errors.append(f"{original}: {exc}")
    return _merge_records([], refreshed), errors


def _unified_classes(records: list[dict[str, Any]]) -> tuple[list[str], list[list[int]]]:
    """데이터셋별 로컬 class ID를 통합 class ID로 매핑한다."""
    names: list[str] = []
    global_ids: dict[str, int] = {}
    mappings: list[list[int]] = []
    for record in records:
        local_mapping: list[int] = []
        for name in record["classes"]:
            key = name.casefold()
            if key not in global_ids:
                global_ids[key] = len(names)
                names.append(name)
            local_mapping.append(global_ids[key])
        mappings.append(local_mapping)
    return names, mappings


def _record_images(record: dict[str, Any], split: str) -> list[Path]:
    images: list[Path] = []
    for raw_path in record[f"{split}_paths"]:
        images.extend(_images_from_path(Path(raw_path)))
    return images


def _file_identity(path: Path) -> tuple[Any, ...]:
    resolved = path.resolve()
    stat = resolved.stat()
    if stat.st_ino:
        return ("inode", stat.st_dev, stat.st_ino)
    return ("path", str(resolved).casefold())


def _ensure_no_cross_dataset_overlap(records: list[dict[str, Any]]) -> None:
    seen: dict[tuple[Any, ...], tuple[str, str, Path]] = {}
    for split in ("train", "val", "test"):
        for record in records:
            for image_path in _record_images(record, split):
                key = _file_identity(image_path)
                if key in seen:
                    previous_split, previous_dataset, previous_path = seen[key]
                    raise DatasetImportError(
                        "선택한 데이터셋 사이에 같은 이미지가 중복되어 있습니다: "
                        f"{image_path} ({previous_path}, "
                        f"{previous_dataset}/{previous_split}, "
                        f"{record['name']}/{split})"
                    )
                seen[key] = (split, record["name"], image_path)


def _link_or_copy(source: Path, destination: Path) -> str:
    """이미지는 hardlink를 우선 사용하고 불가능하면 독립 복사한다."""
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _rewrite_label(
    source: Path,
    destination: Path,
    class_mapping: list[int],
) -> bytes:
    raw = source.read_bytes()
    text = raw.decode("utf-8-sig")
    output_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        old_id = int(float(parts[0]))
        output_lines.append(f"{class_mapping[old_id]} {' '.join(parts[1:])}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8",
    )
    return raw


def _remove_managed_tree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root = root.resolve()
    if (
        resolved != root
        and resolved.is_relative_to(root)
        and (
            resolved.name.startswith(".staging-")
            or resolved.name.startswith("staged-")
        )
        and resolved.exists()
    ):
        shutil.rmtree(resolved)


def _staging_digest(
    records: list[dict[str, Any]],
    unified_names: list[str],
    class_mappings: list[list[int]],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"stage-v{_STAGING_SCHEMA_VERSION}\0".encode("ascii"))
    digest.update("\0".join(unified_names).encode("utf-8"))
    seen: dict[tuple[Any, ...], tuple[str, str, Path]] = {}

    for dataset_index, (record, class_mapping) in enumerate(
        zip(records, class_mappings)
    ):
        yaml_path = Path(record["yaml"]).resolve()
        digest.update(str(yaml_path).encode("utf-8"))
        digest.update(hashlib.sha256(yaml_path.read_bytes()).digest())
        digest.update("\0".join(record["classes"]).encode("utf-8"))
        digest.update(",".join(str(value) for value in class_mapping).encode("ascii"))

        for split in ("train", "val", "test"):
            for image_path in _record_images(record, split):
                resolved_image = image_path.resolve()
                identity = _file_identity(resolved_image)
                if identity in seen:
                    previous_split, previous_dataset, previous_path = seen[identity]
                    raise DatasetImportError(
                        "선택한 데이터셋 사이에 같은 이미지가 중복되어 있습니다: "
                        f"{resolved_image} ({previous_path}, "
                        f"{previous_dataset}/{previous_split}, "
                        f"{record['name']}/{split})"
                    )
                seen[identity] = (split, record["name"], resolved_image)

                image_stat = resolved_image.stat()
                digest.update(
                    (
                        f"{dataset_index}\0{split}\0{resolved_image}\0"
                        f"{image_stat.st_size}\0{image_stat.st_mtime_ns}\0"
                    ).encode("utf-8")
                )
                source_label = _label_path(resolved_image)
                if source_label.is_file():
                    digest.update(hashlib.sha256(source_label.read_bytes()).digest())
                else:
                    digest.update(b"<missing-label>")
    return digest.hexdigest()


def _cached_stage_info(
    destination: Path,
    digest: str,
    records: list[dict[str, Any]],
    unified_names: list[str],
) -> dict[str, int] | None:
    marker_path = destination / ".complete.yaml"
    yaml_path = destination / "data.yaml"
    if not marker_path.is_file() or not yaml_path.is_file():
        return None
    try:
        marker = yaml.safe_load(marker_path.read_text(encoding="utf-8")) or {}
        if (
            marker.get("schema") != _STAGING_SCHEMA_VERSION
            or marker.get("digest") != digest
        ):
            return None
        report = validate_dataset(yaml_path)
        expected = {
            "train_images": sum(record["train_images"] for record in records),
            "val_images": sum(record["val_images"] for record in records),
            "test_images": sum(record["test_images"] for record in records),
            "boxes": sum(record["boxes"] for record in records),
        }
        if report["classes"] != unified_names or any(
            report[key] != value for key, value in expected.items()
        ):
            return None
        raw_counts = marker.get("link_counts", {})
        return {
            "hardlink": int(raw_counts.get("hardlink", 0)),
            "copy": int(raw_counts.get("copy", 0)),
        }
    except (OSError, TypeError, ValueError, yaml.YAMLError, DatasetImportError):
        return None


def _stage_remapped_dataset_locked(
    records: list[dict[str, Any]],
    unified_names: list[str],
    class_mappings: list[list[int]],
    output_dir: Path,
) -> tuple[Path, dict[str, int]]:
    """클래스 ID가 다른 데이터셋을 비파괴 학습용 디렉터리로 통합한다."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = _staging_digest(records, unified_names, class_mappings)
    destination = output_dir / f"staged-{digest[:16]}"
    cached_counts = _cached_stage_info(
        destination,
        digest,
        records,
        unified_names,
    )
    if cached_counts is not None:
        return destination / "data.yaml", cached_counts

    temporary = (output_dir / f".staging-{uuid.uuid4().hex}").resolve()
    temporary.mkdir(parents=True)

    link_counts = {"hardlink": 0, "copy": 0}
    staged_counts: dict[str, int] = {}

    try:
        for split in ("train", "val", "test"):
            split_count = 0
            for dataset_index, (record, class_mapping) in enumerate(
                zip(records, class_mappings)
            ):
                images = _record_images(record, split)
                for image_path in images:
                    resolved_image = image_path.resolve()

                    suffix = resolved_image.suffix.casefold()
                    relative = (
                        Path(f"dataset_{dataset_index:03d}")
                        / f"{split_count:08d}{suffix}"
                    )
                    staged_image = temporary / "images" / split / relative
                    staged_label = (
                        temporary / "labels" / split / relative.with_suffix(".txt")
                    )
                    staged_image.parent.mkdir(parents=True, exist_ok=True)
                    method = _link_or_copy(resolved_image, staged_image)
                    link_counts[method] += 1

                    source_label = _label_path(resolved_image)
                    if source_label.is_file():
                        _rewrite_label(
                            source_label,
                            staged_label,
                            class_mapping,
                        )
                    split_count += 1
            staged_counts[split] = split_count

        dataset_yaml = {
            "train": "images/train",
            "val": "images/val",
            "names": {
                index: name for index, name in enumerate(unified_names)
            },
            "nc": len(unified_names),
        }
        if staged_counts["test"] > 0:
            dataset_yaml["test"] = "images/test"
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        report = validate_dataset(temporary / "data.yaml")
        expected = {
            "train_images": sum(record["train_images"] for record in records),
            "val_images": sum(record["val_images"] for record in records),
            "test_images": sum(record["test_images"] for record in records),
            "boxes": sum(record["boxes"] for record in records),
        }
        if report["classes"] != unified_names or any(
            report[key] != value for key, value in expected.items()
        ):
            raise DatasetImportError(
                "클래스 ID 재매핑 후 자체 검증 결과가 원본 데이터 통계와 일치하지 않습니다."
            )
        (temporary / ".complete.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": _STAGING_SCHEMA_VERSION,
                    "digest": digest,
                    "link_counts": link_counts,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        if destination.exists():
            _remove_managed_tree(destination, output_dir)
        try:
            shutil.move(str(temporary), str(destination))
        except OSError:
            cached_counts = _cached_stage_info(
                destination,
                digest,
                records,
                unified_names,
            )
            if cached_counts is None:
                raise
            _remove_managed_tree(temporary, output_dir)
            return destination / "data.yaml", cached_counts
        return destination / "data.yaml", link_counts
    except Exception:
        _remove_managed_tree(temporary, output_dir)
        raise


def _stage_remapped_dataset(
    records: list[dict[str, Any]],
    unified_names: list[str],
    class_mappings: list[list[int]],
    output_dir: Path,
) -> tuple[Path, dict[str, int]]:
    with _STAGE_LOCK:
        return _stage_remapped_dataset_locked(
            records,
            unified_names,
            class_mappings,
            output_dir,
        )


def selection_summary(
    records: list[dict[str, Any]] | None,
    selected_yamls: list[str] | None,
) -> tuple[bool, str]:
    selected = {
        str(path).casefold() for path in (selected_yamls or []) if str(path).strip()
    }
    chosen = [
        record
        for record in (records or [])
        if str(record.get("yaml", "")).casefold() in selected
    ]
    if not chosen:
        return False, "학습에 사용할 데이터셋을 한 개 이상 선택하세요."

    unified_names, _ = _unified_classes(chosen)
    needs_remap = any(
        record["classes"] != chosen[0]["classes"] for record in chosen[1:]
    )
    train_images = sum(record["train_images"] for record in chosen)
    val_images = sum(record["val_images"] for record in chosen)
    boxes = sum(record["boxes"] for record in chosen)
    mode = " · 클래스 ID 자동 통합" if needs_remap else ""
    return (
        True,
        f"{len(chosen)}개 데이터셋 적용 가능 · train {train_images:,}장 · "
        f"val {val_images:,}장 · bbox {boxes:,}개 · "
        f"통합 클래스 {len(unified_names)}개 ({', '.join(unified_names)}){mode}",
    )


def prepare_training_data(
    selected_yamls: list[str] | None,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, str]:
    """선택 데이터셋을 재검증하고 Ultralytics에 전달할 YAML을 반환한다."""
    if selected_yamls is None:
        selected = [str(DEFAULT_DATASET_YAML)]
    else:
        selected = [str(path) for path in selected_yamls if str(path).strip()]
    if not selected:
        raise DatasetImportError("학습에 사용할 데이터셋을 한 개 이상 선택하세요.")

    records = [validate_dataset(path) for path in dict.fromkeys(selected)]
    unified_names, class_mappings = _unified_classes(records)
    total_train = sum(record["train_images"] for record in records)
    total_val = sum(record["val_images"] for record in records)
    needs_remap = any(
        record["classes"] != records[0]["classes"] for record in records[1:]
    )
    description = (
        f"{len(records)}개 · train {total_train:,}장 · val {total_val:,}장 · "
        f"통합 클래스 {len(unified_names)}개 ({', '.join(unified_names)})"
    )
    if len(records) == 1:
        return Path(records[0]["yaml"]), description

    target_dir = (output_dir or COMBINED_DATASETS_DIR).resolve()
    if needs_remap:
        staged_yaml, link_counts = _stage_remapped_dataset(
            records,
            unified_names,
            class_mappings,
            target_dir,
        )
        description += (
            " · 클래스 ID 재매핑"
            f" (hardlink {link_counts['hardlink']:,}, "
            f"copy {link_counts['copy']:,})"
        )
        return staged_yaml, description

    _ensure_no_cross_dataset_overlap(records)
    combined = {
        "train": [
            path for record in records for path in record["train_paths"]
        ],
        "val": [path for record in records for path in record["val_paths"]],
        "names": {index: name for index, name in enumerate(unified_names)},
        "nc": len(unified_names),
    }
    test_paths = [path for record in records for path in record["test_paths"]]
    if test_paths:
        combined["test"] = test_paths

    identity = "\n".join(record["yaml"] for record in records)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"combined-{digest}.yaml"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        prefix=f".{target.stem}-",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        yaml.safe_dump(
            combined,
            temporary_file,
            allow_unicode=True,
            sort_keys=False,
        )
        temporary = Path(temporary_file.name)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, description
