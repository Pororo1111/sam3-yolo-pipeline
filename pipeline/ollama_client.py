"""Ollama 모델 확인·다운로드용 경량 REST 클라이언트."""

from __future__ import annotations

import json
import os
import time
from urllib.parse import urljoin

import requests


DEFAULT_MODEL = "gemma4:e4b"


def base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")


def api_url(path: str) -> str:
    return urljoin(base_url() + "/", f"api/{path.lstrip('/')}")


def list_models(timeout: float = 10.0) -> set[str]:
    response = requests.get(api_url("tags"), timeout=(3.0, timeout))
    response.raise_for_status()
    names: set[str] = set()
    for item in response.json().get("models", []):
        for key in ("name", "model"):
            value = item.get(key)
            if value:
                names.add(str(value))
    return names


def is_model_installed(model: str) -> bool:
    name = (model or DEFAULT_MODEL).strip()
    installed = list_models()
    if name in installed:
        return True
    # Ollama가 ``:latest``를 생략해 반환하는 경우도 허용한다.
    return ":" not in name and f"{name}:latest" in installed


def pull_model(model: str):
    """Gradio에서 바로 사용할 수 있도록 다운로드 상태 문자열을 생성한다."""

    name = (model or DEFAULT_MODEL).strip()
    if not name:
        yield "Ollama 모델 이름을 입력하세요."
        return

    try:
        if is_model_installed(name):
            yield f"이미 설치된 모델입니다: {name}"
            return
    except requests.exceptions.ConnectionError:
        yield (
            f"Ollama 연결 실패 ({base_url()}). 먼저 `ollama serve`를 실행하세요."
        )
        return
    except Exception as exc:
        yield f"Ollama 모델 목록 확인 실패: {exc}"
        return

    yield f"{name} 다운로드 요청 중..."
    try:
        response = requests.post(
            api_url("pull"),
            json={"model": name, "stream": True},
            stream=True,
            timeout=(5.0, 3600.0),
        )
        response.raise_for_status()
        last_emit = 0.0
        last_status = ""
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise RuntimeError(str(event["error"]))

            status = str(event.get("status", "다운로드 중"))
            total = int(event.get("total") or 0)
            completed = int(event.get("completed") or 0)
            if total > 0:
                percent = min(100.0, completed / total * 100.0)
                message = (
                    f"{name} · {status} · {percent:.1f}% "
                    f"({completed / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MiB)"
                )
            else:
                message = f"{name} · {status}"

            now = time.monotonic()
            if message != last_status and (
                now - last_emit >= 0.2 or status.lower() == "success"
            ):
                last_emit = now
                last_status = message
                yield message

        yield f"Ollama 모델 다운로드 완료: {name}"
    except requests.exceptions.ConnectionError:
        yield (
            f"Ollama 연결 실패 ({base_url()}). 먼저 `ollama serve`를 실행하세요."
        )
    except Exception as exc:
        yield f"Ollama 모델 다운로드 실패: {exc}"
