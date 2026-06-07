#!/bin/bash
set -e  # 명령 실패 시 즉시 중단 → 가짜 성공 방지

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "📦 venv 생성 중..."
  python -m venv "$VENV_DIR"
else
  echo "✅ 기존 venv 사용"
fi

# OS별 파이썬 경로 분기
if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
  PY="$VENV_DIR/Scripts/python.exe"   # 윈도우
else
  PY="$VENV_DIR/bin/python"           # 리눅스/맥
fi

echo "🐍 사용 파이썬: $PY"

# pip 최신화
"$PY" -m pip install --upgrade pip

# macOS는 CUDA 미지원 → 기본 PyTorch(MPS/CPU)
# Windows/Linux는 cu128 인덱스 사용
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "🍎 macOS 감지 → CPU/MPS 빌드로 설치"
  "$PY" -m pip install torch torchaudio torchvision
  "$PY" -m pip install -r requirements.txt
else
  echo "🖥️ Windows/Linux 감지 → CUDA 128 빌드로 설치"
  "$PY" -m pip install torch torchvision --extra-index-url "https://download.pytorch.org/whl/cu126"
  "$PY" -m pip install -r requirements.txt --extra-index-url "https://download.pytorch.org/whl/cu126"
fi

echo "✅ venv에 설치 완료"