from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pipeline.model_registry import (
    ModelRegistry,
    RegistryUploadGuardMiddleware,
    create_router,
)
from pipeline.model_sync import ModelSyncClient


class _FakeResponse:
    def __init__(self, status_code=200, *, json_value=None, content=b"", headers=None):
        self.status_code = status_code
        self._json_value = json_value
        self._content = content
        self.headers = headers or {}

    def json(self):
        return self._json_value

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for offset in range(0, len(self._content), chunk_size):
            yield self._content[offset : offset + chunk_size]

    def close(self):
        pass


class _FakeSession:
    def __init__(self, manifest, model_bytes):
        self.manifest = manifest
        self.model_bytes = model_bytes
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/releases/latest"):
            if kwargs.get("headers", {}).get("If-None-Match") == '"release-1"':
                return _FakeResponse(304)
            return _FakeResponse(
                json_value=self.manifest,
                headers={"ETag": '"release-1"'},
            )
        return _FakeResponse(
            content=self.model_bytes,
            headers={"Content-Length": str(len(self.model_bytes))},
        )


class ModelRegistryTests(unittest.TestCase):
    def test_registry_auth_etag_and_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "runs" / "detect" / "train" / "weights" / "best.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"trusted-model")
            registry = ModelRegistry(root)
            release = registry.publish_model(model, "train")
            model.write_bytes(b"changed-after-publish")

            app = FastAPI()
            app.include_router(create_router(registry, "read-secret", "write-secret"))
            app.add_middleware(
                RegistryUploadGuardMiddleware,
                publish_token="write-secret",
                max_model_bytes=1024,
            )
            client = TestClient(app)

            self.assertEqual(client.get("/model-registry/v1/releases/latest").status_code, 401)
            response = client.get(
                "/model-registry/v1/releases/latest",
                headers={"Authorization": "Bearer read-secret"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("model_path", response.json())
            self.assertEqual(response.json()["release_id"], release.release_id)
            self.assertEqual(
                client.get(
                    "/model-registry/v1/releases/latest",
                    headers={
                        "Authorization": "Bearer read-secret",
                        "If-None-Match": response.headers["etag"],
                    },
                ).status_code,
                304,
            )
            download = client.get(
                response.json()["download_path"],
                headers={"Authorization": "Bearer read-secret"},
            )
            self.assertEqual(download.content, b"trusted-model")

            denied = client.post(
                "/model-registry/v1/releases",
                data={"run_name": "external"},
                files={"file": ("best.pt", b"external", "application/octet-stream")},
            )
            self.assertEqual(denied.status_code, 401)
            uploaded = client.post(
                "/model-registry/v1/releases",
                headers={"Authorization": "Bearer write-secret"},
                data={"run_name": "external"},
                files={"file": ("best.pt", b"external", "application/octet-stream")},
            )
            self.assertEqual(uploaded.status_code, 201)
            self.assertEqual(uploaded.json()["source"], "uploaded")

    def test_edge_sync_downloads_atomically_and_then_uses_etag(self):
        with tempfile.TemporaryDirectory() as directory:
            model_bytes = b"remote-model-weights"
            digest = hashlib.sha256(model_bytes).hexdigest()
            manifest = {
                "schema_version": 1,
                "release_id": "1" * 32,
                "run_name": "cone detector",
                "sha256": digest,
                "size_bytes": len(model_bytes),
                "published_at": "2026-07-20T10:00:00Z",
                "source": "trained",
                "download_path": f"/model-registry/v1/releases/{'1' * 32}/weights",
            }
            session = _FakeSession(manifest, model_bytes)
            runs_dir = Path(directory) / "runs" / "detect"
            client = ModelSyncClient(
                "https://central.example",
                "read-secret",
                runs_dir,
                session=session,
            )

            installed, _, release_id = client.sync_once()
            self.assertTrue(installed)
            self.assertEqual(release_id, "1" * 32)
            models = list(runs_dir.glob("remote-*/weights/best.pt"))
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].read_bytes(), model_bytes)
            self.assertFalse(list(runs_dir.glob(".sync-staging/*/weights/best.pt.part")))

            models[0].write_bytes(b"x" * len(model_bytes))
            installed, message, _ = client.sync_once()
            self.assertTrue(installed)
            self.assertIn("다운로드 완료", message)
            self.assertEqual(models[0].read_bytes(), model_bytes)

            installed, message, _ = client.sync_once()
            self.assertFalse(installed)
            self.assertIn("최신", message)
            state = json.loads((runs_dir / ".model-sync-state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["release_id"], "1" * 32)


if __name__ == "__main__":
    unittest.main()
