#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON=""
for cmd in python3.11 python3.12 python3; do
    if command -v "$cmd" &>/dev/null; then PYTHON="$cmd"; break; fi
done
if [ -z "$PYTHON" ]; then echo "[ERROR] No se encontro Python."; exit 1; fi
exec "$PYTHON" "$SCRIPT_DIR/update.py" --root "$PROJECT_ROOT" "$@"
