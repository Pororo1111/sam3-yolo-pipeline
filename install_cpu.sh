#!/bin/bash
set -e

VENV_DIR="venv"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv..."
  python -m venv "$VENV_DIR"
else
  echo "Using existing venv"
fi

if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
  PY="$VENV_DIR/Scripts/python.exe"
else
  PY="$VENV_DIR/bin/python"
fi

echo "Python: $PY"

"$PY" -m pip install --upgrade pip

echo "Installing PyTorch CPU build from $TORCH_INDEX_URL"
"$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"

echo "Installing project requirements"
REQ_WITHOUT_TORCH="$(mktemp)"
trap 'rm -f "$REQ_WITHOUT_TORCH"' EXIT
grep -Ev '^(torch|torchvision|torchaudio)([<>=!~ ]|$)' requirements.txt > "$REQ_WITHOUT_TORCH"
"$PY" -m pip install -r "$REQ_WITHOUT_TORCH"

echo "Install complete"
