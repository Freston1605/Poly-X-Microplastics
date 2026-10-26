#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Poly-X · Lanzador (Linux)
#
#  Equivalente de iniciar_polyx.bat / Poly-X.vbs. Arranca el launcher
#  sin ventana de consola adicional (el terminal desde el que se lanza
#  puede cerrarse; el proceso sigue).
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x ".venv/bin/python" ]; then
    echo "[ERROR] No se encuentra el entorno virtual."
    echo "Ejecuta primero: bash install/setup.sh"
    exit 1
fi

.venv/bin/python -m polyx.launcher "$@"