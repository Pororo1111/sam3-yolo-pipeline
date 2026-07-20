"""학습 모델을 중앙 서버에서 안전하게 게시·다운로드하는 레지스트리."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response


SCHEMA_VERSION = 1
API_PREFIX = "/model-registry/v1"
DEFAULT_MAX_MODEL_BYTES = 1024 * 1024 * 1024
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_slug(value: str, fallback: str = "model") -> str:
    """사용자 입력을 한 개의 안전한 폴더 이름 조각으로 바꾼다."""

    slug = _SLUG_RE.sub("-", (value or "").strip()).strip(".-_")
    return (slug[:80] or fallback).lower()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ModelRelease:
    schema_version: int
    release_id: str
    run_name: str
    sha256: str
    size_bytes: int
    published_at: str
    source: str
    download_path: str
    model_path: str

    @classmethod
    def from_dict(cls, value: dict) -> "ModelRelease":
        release = cls(
            schema_version=int(value["schema_version"]),
            release_id=str(value["release_id"]),
            run_name=str(value["run_name"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            published_at=str(value["published_at"]),
            source=str(value["source"]),
            download_path=str(value["download_path"]),
            model_path=str(value["model_path"]),
        )
        if release.schema_version != SCHEMA_VERSION:
            raise ValueError("지원하지 않는 모델 릴리스 스키마입니다.")
        if not release.release_id or not _HASH_RE.fullmatch(release.sha256):
            raise ValueError("잘못된 모델 릴리스 식별자 또는 해시입니다.")
        if release.size_bytes <= 0:
            raise ValueError("잘못된 모델 파일 크기입니다.")
        return release

    def public_dict(self) -> dict:
        value = asdict(self)
        value.pop("model_path", None)
        return value


class ModelRegistry:
    """불변 릴리스 메타데이터와 실제 가중치 경로를 관리한다."""

    def __init__(self, root: Path | None = None):
        self.root = (root or Path(__file__).resolve().parent.parent).resolve()
        self.runs_dir = (self.root / "runs" / "detect").resolve()
        self.blobs_dir = (self.runs_dir / ".registry-blobs").resolve()
        self.metadata_dir = (self.root / "runs" / "model_registry" / "releases").resolve()
        self.staging_dir = (self.root / "runs" / "model_registry" / ".staging").resolve()
        self._lock = threading.Lock()

    def _manifest_path(self, release_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", release_id):
            raise KeyError(release_id)
        return self.metadata_dir / f"{release_id}.json"

    def _relative_model_path(self, path: Path) -> str:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(self.runs_dir)
        except ValueError as exc:
            raise ValueError("게시할 모델은 runs/detect 아래에 있어야 합니다.") from exc
        return relative.as_posix()

    def resolve_model_path(self, release: ModelRelease) -> Path:
        path = (self.runs_dir / release.model_path).resolve()
        try:
            path.relative_to(self.runs_dir)
        except ValueError as exc:
            raise ValueError("릴리스 모델 경로가 runs/detect 밖을 가리킵니다.") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def publish_model(
        self,
        model_path: Path | str,
        run_name: str,
        source: str = "trained",
    ) -> ModelRelease:
        path = Path(model_path)
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise ValueError(f"게시할 .pt 모델 파일을 찾을 수 없습니다: {path}")

        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise ValueError("빈 모델 파일은 게시할 수 없습니다.")
        max_bytes = max_model_bytes_from_environment()
        if size_bytes > max_bytes:
            raise ValueError("모델 파일이 YOLO_MODEL_MAX_BYTES를 초과했습니다.")
        normalized_name = (run_name or path.parent.parent.name).strip()

        with self._lock:
            blob_path, size_bytes, digest = self._store_blob(path, max_bytes)
            relative_path = self._relative_model_path(blob_path)
            latest = self.latest_release()
            if (
                latest is not None
                and latest.sha256 == digest
                and latest.run_name == normalized_name
            ):
                return latest

            release_id = uuid.uuid4().hex
            release = ModelRelease(
                schema_version=SCHEMA_VERSION,
                release_id=release_id,
                run_name=normalized_name,
                sha256=digest,
                size_bytes=size_bytes,
                published_at=_utc_now(),
                source=source,
                download_path=f"{API_PREFIX}/releases/{release_id}/weights",
                model_path=relative_path,
            )
            _atomic_json_write(self._manifest_path(release_id), asdict(release))
            return release

    def _store_blob(self, source: Path, max_bytes: int) -> tuple[Path, int, str]:
        """원본과 독립적인 content-addressed 불변 가중치 복사본을 만든다."""

        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.blobs_dir / f".{uuid.uuid4().hex}.pt.part"
        digest = hashlib.sha256()
        written = 0
        try:
            with source.open("rb") as reader, temporary.open("wb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError("모델 파일이 허용 크기를 초과했습니다.")
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if written <= 0:
                raise ValueError("빈 모델 파일은 게시할 수 없습니다.")

            checksum = digest.hexdigest()
            target = self.blobs_dir / f"{checksum}.pt"
            if target.is_file():
                if target.stat().st_size != written or sha256_file(target) != checksum:
                    os.replace(temporary, target)
                else:
                    temporary.unlink(missing_ok=True)
            else:
                os.replace(temporary, target)
            return target, written, checksum
        finally:
            temporary.unlink(missing_ok=True)

    def get_release(self, release_id: str) -> ModelRelease | None:
        try:
            path = self._manifest_path(release_id)
        except KeyError:
            return None
        if not path.is_file():
            return None
        try:
            return ModelRelease.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def list_releases(self) -> list[ModelRelease]:
        releases: list[ModelRelease] = []
        for path in self.metadata_dir.glob("*.json"):
            release = self.get_release(path.stem)
            if release is None:
                continue
            try:
                self.resolve_model_path(release)
            except (OSError, ValueError):
                continue
            releases.append(release)
        releases.sort(key=lambda item: item.published_at, reverse=True)
        return releases

    def latest_release(self) -> ModelRelease | None:
        releases = self.list_releases()
        return releases[0] if releases else None

    def publish_latest_existing_if_empty(self) -> ModelRelease | None:
        """레지스트리 도입 전에 끝난 최신 학습 모델을 시작 시 한 번 게시한다."""

        if self.latest_release() is not None:
            return None
        candidates = sorted(
            self.runs_dir.glob("*/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        latest = candidates[0]
        return self.publish_model(latest, latest.parent.parent.name, source="existing")

    async def install_upload(
        self,
        upload: UploadFile,
        run_name: str,
        max_bytes: int,
    ) -> ModelRelease:
        filename = (upload.filename or "").strip()
        if Path(filename).suffix.lower() != ".pt":
            raise ValueError(".pt 모델 파일만 업로드할 수 있습니다.")

        release_id = uuid.uuid4().hex
        slug = safe_slug(run_name or Path(filename).stem, fallback="uploaded")
        staging = self.staging_dir / release_id
        final = self.runs_dir / f"uploaded-{slug}-{release_id[:8]}"
        part = staging / "weights" / "best.pt.part"
        completed = staging / "weights" / "best.pt"
        staging.mkdir(parents=True, exist_ok=False)
        part.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        try:
            with part.open("wb") as file:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError("업로드 모델이 허용 크기를 초과했습니다.")
                    file.write(chunk)
                file.flush()
                os.fsync(file.fileno())
            if written == 0:
                raise ValueError("빈 모델 파일은 업로드할 수 없습니다.")
            os.replace(part, completed)
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            model_path = final / "weights" / "best.pt"
            return self.publish_model(model_path, run_name or slug, source="uploaded")
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            if final.exists() and not (final / "release.json").exists():
                shutil.rmtree(final, ignore_errors=True)
            raise
        finally:
            await upload.close()


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def _require_token(authorization: str | None, expected: str, purpose: str) -> None:
    if not expected:
        raise HTTPException(status_code=503, detail=f"{purpose} 토큰이 설정되지 않았습니다.")
    received = _bearer_token(authorization)
    if not received or not secrets.compare_digest(received, expected):
        raise HTTPException(status_code=401, detail="인증 토큰이 올바르지 않습니다.")


def create_router(
    registry: ModelRegistry,
    read_token: str,
    publish_token: str = "",
    max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES,
) -> APIRouter:
    """중앙 서버에 장착할 인증된 모델 레지스트리 API 라우터."""

    router = APIRouter(prefix=API_PREFIX, tags=["model-registry"])

    @router.get("/releases/latest")
    def latest_release(
        authorization: str | None = Header(default=None),
        if_none_match: str | None = Header(default=None),
    ):
        _require_token(authorization, read_token, "읽기")
        release = registry.latest_release()
        if release is None:
            raise HTTPException(status_code=404, detail="게시된 모델이 없습니다.")
        etag = f'"{release.release_id}"'
        if if_none_match == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return JSONResponse(release.public_dict(), headers={"ETag": etag})

    @router.get("/releases/{release_id}/weights")
    def download_weights(
        release_id: str,
        authorization: str | None = Header(default=None),
    ):
        _require_token(authorization, read_token, "읽기")
        release = registry.get_release(release_id)
        if release is None:
            raise HTTPException(status_code=404, detail="모델 릴리스를 찾을 수 없습니다.")
        try:
            path = registry.resolve_model_path(release)
        except (OSError, ValueError):
            raise HTTPException(status_code=410, detail="모델 파일을 더 이상 사용할 수 없습니다.")
        if path.stat().st_size != release.size_bytes or sha256_file(path) != release.sha256:
            raise HTTPException(status_code=410, detail="게시 모델 무결성 검증에 실패했습니다.")
        return FileResponse(
            path,
            filename=f"{safe_slug(release.run_name)}.pt",
            media_type="application/octet-stream",
            headers={
                "ETag": f'"{release.sha256}"',
                "X-Model-SHA256": release.sha256,
            },
        )

    if publish_token:
        def require_publish_token(
            authorization: str | None = Header(default=None),
        ) -> None:
            _require_token(authorization, publish_token, "게시")

        @router.post("/releases", dependencies=[Depends(require_publish_token)])
        async def upload_release(
            file: UploadFile = File(...),
            run_name: str = Form(...),
        ):
            try:
                release = await registry.install_upload(file, run_name, max_model_bytes)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return JSONResponse(release.public_dict(), status_code=201)

    return router


class RegistryUploadGuardMiddleware:
    """multipart 파싱 전에 게시 토큰과 요청 크기를 검사한다."""

    def __init__(self, app, publish_token: str, max_model_bytes: int):
        self.app = app
        self.publish_token = publish_token
        self.max_body_bytes = int(max_model_bytes) + 1024 * 1024

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != f"{API_PREFIX}/releases"
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        try:
            _require_token(headers.get("authorization"), self.publish_token, "게시")
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            await response(scope, receive, send)
            return

        content_length = headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > self.max_body_bytes
            except ValueError:
                too_large = True
            if too_large:
                response = JSONResponse(
                    {"detail": "업로드 요청이 허용 크기를 초과했습니다."},
                    status_code=413,
                )
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="업로드 요청이 허용 크기를 초과했습니다.",
                    )
            return message

        await self.app(scope, limited_receive, send)


def registry_from_environment() -> ModelRegistry:
    return ModelRegistry()


def max_model_bytes_from_environment() -> int:
    try:
        return max(1, int(os.getenv("YOLO_MODEL_MAX_BYTES", DEFAULT_MAX_MODEL_BYTES)))
    except ValueError:
        return DEFAULT_MAX_MODEL_BYTES
