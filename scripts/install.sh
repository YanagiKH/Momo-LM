#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VENV_PATH="$PROJECT_ROOT/.venv"

python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install -e "$PROJECT_ROOT"
"$VENV_PATH/bin/momo" init

printf '%s\n' "Momo-LM installed. Start it with:"
printf '  %s\n' "$VENV_PATH/bin/momo serve"
