"""라즈베리파이 등 edge 노드의 중앙 YOLO 모델 자동 동기화."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from filelock import FileLock

from pipeline.model_registry import (
    API_PREFIX,
    DEFAULT_MAX_MODEL_BYTES,
    SCHEMA_VERSION,
    safe_slug,
    sha256_file,
)


@dataclass(frozen=True)
class RemoteRelease:
    release_id: str
    run_name: str
    sha256: str
    size_bytes: int
    published_at: str
    source: str
    download_path: str

    @classmethod
    def parse(cls, value: dict, max_bytes: int) -> "RemoteRelease":
        if int(value.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("지원하지 않는 원격 모델 스키마입니다.")
        release = cls(
            release_id=str(value["release_id"]),
            run_name=str(value["run_name"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            published_at=str(value["published_at"]),
            source=str(value.get("source", "remote")),
            download_path=str(value["download_path"]),
        )
        if not re.fullmatch(r"[0-9a-f]{32}", release.release_id):
            raise ValueError("원격 모델 release_id가 올바르지 않습니다.")
        if len(release.sha256) != 64:
            raise ValueError("원격 모델 식별자 또는 SHA-256이 올바르지 않습니다.")
        int(release.sha256, 16)
        if release.size_bytes <= 0 or release.size_bytes > max_bytes:
            raise ValueError("원격 모델 크기가 허용 범위를 벗어났습니다.")
        if not release.download_path.startswith(f"{API_PREFIX}/releases/"):
            raise ValueError("원격 다운로드 경로는 같은 서버의 상대 경로여야 합니다.")
        return release

    def public_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "release_id": self.release_id,
            "run_name": self.run_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "published_at": self.published_at,
            "source": self.source,
            "download_path": self.download_path,
        }


@dataclass
class SyncStatus:
    running: bool = False
    current_release: str = ""
    last_success: str = ""
    last_error: str = ""
    message: str = "동기화 대기 중"


class ModelSyncClient:
    """최신 중앙 릴리스를 검증한 뒤 기존 모델 폴더 규약으로 설치한다."""

    def __init__(
        self,
        registry_url: str,
        token: str,
        runs_dir: Path | None = None,
        *,
        max_bytes: int = DEFAULT_MAX_MODEL_BYTES,
        connect_timeout: float = 5.0,
        read_timeout: float = 300.0,
        verify: bool | str = True,
        allow_insecure_http: bool = False,
        session: requests.Session | None = None,
    ):
        self.registry_url = registry_url.rstrip("/") + "/"
        self.token = token
        root = Path(__file__).resolve().parent.parent
        self.runs_dir = (runs_dir or root / "runs" / "detect").resolve()
        self.max_bytes = max(1, int(max_bytes))
        self.timeout = (
            max(1.0, float(connect_timeout)),
            min(30.0, max(5.0, float(read_timeout))),
        )
        self.verify = verify
        self.session = session or requests.Session()
        self.state_path = self.runs_dir / ".model-sync-state.json"
        self.staging_root = self.runs_dir / ".sync-staging"
        self.lock_path = self.runs_dir / ".model-sync.lock"
        self._thread_lock = threading.Lock()
        self._cancel_event = threading.Event()

        parsed = urlparse(self.registry_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("YOLO_REGISTRY_URL은 http(s) 중앙 서버 URL이어야 합니다.")
        if (
            parsed.scheme == "http"
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            and not allow_insecure_http
        ):
            raise ValueError(
                "원격 HTTP 모델 동기화는 토큰이 노출될 수 있습니다. HTTPS를 사용하거나 "
                "YOLO_ALLOW_INSECURE_HTTP=1을 명시하세요."
            )
        if not token:
            raise ValueError("YOLO_REGISTRY_READ_TOKEN이 비어 있습니다.")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _load_state(self) -> dict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_state(self, value: dict) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(value, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _same_origin_download_url(self, path: str) -> str:
        url = urljoin(self.registry_url, path.lstrip("/"))
        base = urlparse(self.registry_url)
        parsed = urlparse(url)
        if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
            raise ValueError("중앙 서버와 다른 origin의 다운로드 URL은 허용되지 않습니다.")
        return url

    def cancel(self) -> None:
        self._cancel_event.set()
        self.session.close()

    def reset_cancel(self) -> None:
        self._cancel_event.clear()

    def _state_model_path(self, value) -> Path | None:
        if not value:
            return None
        path = Path(str(value))
        path = path if path.is_absolute() else self.runs_dir / path
        resolved = path.resolve()
        try:
            resolved.relative_to(self.runs_dir)
        except ValueError:
            return None
        return resolved

    @staticmethod
    def _valid_model_file(path: Path | None, size_bytes, checksum) -> bool:
        if path is None or not path.is_file():
            return False
        try:
            expected_size = int(size_bytes)
        except (TypeError, ValueError):
            return False
        expected_hash = str(checksum or "")
        return (
            expected_size > 0
            and len(expected_hash) == 64
            and path.stat().st_size == expected_size
            and sha256_file(path) == expected_hash
        )

    def _installed_release(self, release: RemoteRelease) -> Path | None:
        for manifest_path in self.runs_dir.glob("remote-*/release.json"):
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("release_id") != release.release_id:
                continue
            model_path = manifest_path.parent / "weights" / "best.pt"
            if self._valid_model_file(
                model_path,
                release.size_bytes,
                release.sha256,
            ):
                return model_path
        return None

    def sync_once(self) -> tuple[bool, str, str]:
        """한 번 동기화하고 ``(설치 여부, 메시지, release_id)``를 반환한다."""

        with self._thread_lock:
            if self._cancel_event.is_set():
                raise InterruptedError("모델 동기화가 취소되었습니다.")
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self.lock_path), timeout=1):
                state = self._load_state()
                headers = dict(self.headers)
                state_model_path = self._state_model_path(state.get("model_path"))
                state_model_valid = self._valid_model_file(
                    state_model_path,
                    state.get("size_bytes"),
                    state.get("sha256"),
                )
                if state.get("etag") and state_model_valid:
                    headers["If-None-Match"] = str(state["etag"])

                latest_url = urljoin(
                    self.registry_url,
                    f"{API_PREFIX.lstrip('/')}/releases/latest",
                )
                response = self.session.get(
                    latest_url,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify,
                    allow_redirects=False,
                )
                if response.status_code == 304:
                    if not state_model_valid:
                        response.close()
                        response = self.session.get(
                            latest_url,
                            headers=self.headers,
                            timeout=self.timeout,
                            verify=self.verify,
                            allow_redirects=False,
                        )
                    else:
                        release_id = str(state.get("release_id", ""))
                        response.close()
                        return False, "중앙 모델이 최신 상태입니다.", release_id
                if 300 <= response.status_code < 400:
                    response.close()
                    raise ValueError("중앙 모델 API redirect는 허용되지 않습니다.")
                try:
                    response.raise_for_status()
                    latest_etag = response.headers.get("ETag", "")
                    release = RemoteRelease.parse(response.json(), self.max_bytes)
                finally:
                    response.close()

                installed = self._installed_release(release)
                if installed is not None:
                    self._write_state(
                        {
                            "release_id": release.release_id,
                            "etag": latest_etag,
                            "model_path": str(installed.relative_to(self.runs_dir)),
                            "size_bytes": release.size_bytes,
                            "sha256": release.sha256,
                            "updated_at": time.time(),
                        }
                    )
                    return False, "중앙 모델이 이미 설치되어 있습니다.", release.release_id

                target_name = f"remote-{safe_slug(release.run_name)}-{release.release_id}"
                final_dir = self.runs_dir / target_name
                staging_dir = self.staging_root / uuid.uuid4().hex
                part_path = staging_dir / "weights" / "best.pt.part"
                completed_path = staging_dir / "weights" / "best.pt"
                part_path.parent.mkdir(parents=True, exist_ok=False)

                try:
                    download = self.session.get(
                        self._same_origin_download_url(release.download_path),
                        headers=self.headers,
                        stream=True,
                        timeout=self.timeout,
                        verify=self.verify,
                        allow_redirects=False,
                    )
                    if 300 <= download.status_code < 400:
                        raise ValueError("중앙 모델 다운로드 redirect는 허용되지 않습니다.")
                    download.raise_for_status()
                    content_length = download.headers.get("Content-Length")
                    if content_length and int(content_length) != release.size_bytes:
                        raise ValueError("중앙 모델 Content-Length가 manifest와 다릅니다.")

                    digest = hashlib.sha256()
                    written = 0
                    with part_path.open("wb") as file:
                        for chunk in download.iter_content(chunk_size=1024 * 1024):
                            if self._cancel_event.is_set():
                                raise InterruptedError("모델 동기화가 취소되었습니다.")
                            if not chunk:
                                continue
                            written += len(chunk)
                            if written > self.max_bytes or written > release.size_bytes:
                                raise ValueError("다운로드 모델이 선언된 크기를 초과했습니다.")
                            digest.update(chunk)
                            file.write(chunk)
                        file.flush()
                        os.fsync(file.fileno())

                    if written != release.size_bytes:
                        raise ValueError("다운로드 모델 크기가 manifest와 다릅니다.")
                    if digest.hexdigest() != release.sha256:
                        raise ValueError("다운로드 모델 SHA-256 검증에 실패했습니다.")

                    os.replace(part_path, completed_path)
                    manifest_path = staging_dir / "release.json"
                    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
                        json.dump(release.public_dict(), file, ensure_ascii=False, indent=2)
                        file.write("\n")
                        file.flush()
                        os.fsync(file.fileno())

                    backup_dir = None
                    if final_dir.exists():
                        backup_dir = self.staging_root / f"backup-{uuid.uuid4().hex}"
                        os.replace(final_dir, backup_dir)
                    try:
                        os.replace(staging_dir, final_dir)
                    except Exception:
                        if backup_dir is not None and not final_dir.exists():
                            os.replace(backup_dir, final_dir)
                        raise
                    if backup_dir is not None:
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    model_path = final_dir / "weights" / "best.pt"
                    self._write_state(
                        {
                            "release_id": release.release_id,
                            "etag": latest_etag,
                            "model_path": str(model_path.relative_to(self.runs_dir)),
                            "size_bytes": release.size_bytes,
                            "sha256": release.sha256,
                            "updated_at": time.time(),
                        }
                    )
                    return True, f"새 모델 다운로드 완료: {target_name}", release.release_id
                finally:
                    close_download = locals().get("download")
                    if close_download is not None:
                        close_download.close()
                    shutil.rmtree(staging_dir, ignore_errors=True)


class ModelSyncWorker:
    def __init__(self, client: ModelSyncClient, interval_seconds: float = 60.0):
        self.client = client
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.status = SyncStatus()
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.client.reset_cancel()
        self._thread = threading.Thread(target=self._run, daemon=True, name="model-sync")
        self._thread.start()

    def stop(self) -> bool:
        self._stop_event.set()
        self.client.cancel()
        if self._thread is not None:
            self._thread.join(timeout=35)
            return not self._thread.is_alive()
        return True

    def snapshot(self) -> SyncStatus:
        with self._status_lock:
            return SyncStatus(**vars(self.status))

    def _set_status(self, **values) -> None:
        with self._status_lock:
            for key, value in values.items():
                setattr(self.status, key, value)

    def _run(self) -> None:
        self._set_status(running=True, message="중앙 모델 확인 중")
        failures = 0
        while not self._stop_event.is_set():
            try:
                _, message, release_id = self.client.sync_once()
                failures = 0
                self._set_status(
                    current_release=release_id,
                    last_success=time.strftime("%Y-%m-%d %H:%M:%S"),
                    last_error="",
                    message=message,
                )
                wait_seconds = self.interval_seconds
            except Exception as exc:
                failures += 1
                wait_seconds = min(300.0, self.interval_seconds * (2 ** min(failures, 4)))
                wait_seconds += random.uniform(0, min(5.0, wait_seconds * 0.1))
                self._set_status(last_error=str(exc), message=f"모델 동기화 실패: {exc}")
            self._stop_event.wait(wait_seconds)
        self._set_status(running=False, message="모델 동기화 중지")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def worker_from_environment() -> ModelSyncWorker | None:
    if os.getenv("YOLO_NODE_ROLE", "standalone").strip().lower() != "edge":
        return None

    registry_url = os.getenv("YOLO_REGISTRY_URL", "").strip()
    token = os.getenv("YOLO_REGISTRY_READ_TOKEN", "").strip()
    if not registry_url or not token:
        raise RuntimeError(
            "edge 역할에는 YOLO_REGISTRY_URL과 YOLO_REGISTRY_READ_TOKEN이 필요합니다."
        )

    ca_bundle = os.getenv("YOLO_REGISTRY_CA_BUNDLE", "").strip()
    verify: bool | str = ca_bundle or True
    try:
        max_bytes = int(os.getenv("YOLO_MODEL_MAX_BYTES", DEFAULT_MAX_MODEL_BYTES))
    except ValueError:
        max_bytes = DEFAULT_MAX_MODEL_BYTES
    client = ModelSyncClient(
        registry_url,
        token,
        max_bytes=max_bytes,
        connect_timeout=_float_env("YOLO_SYNC_CONNECT_TIMEOUT_SEC", 5.0),
        read_timeout=_float_env("YOLO_SYNC_READ_TIMEOUT_SEC", 15.0),
        verify=verify,
        allow_insecure_http=os.getenv("YOLO_ALLOW_INSECURE_HTTP", "0").strip() == "1",
    )
    return ModelSyncWorker(
        client,
        interval_seconds=_float_env("YOLO_MODEL_SYNC_INTERVAL_SEC", 60.0),
    )
