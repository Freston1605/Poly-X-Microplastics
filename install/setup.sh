#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Poly-X · Instalación en Linux / macOS
#
#  Thin wrapper: locates Python 3.11+ and delegates to
#  install/bootstrap.py, which holds the real pipeline.
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=""
for cmd in python3.11 python3.12 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        maj=$("$cmd" -c "import sys;print(sys.version_info[0])" 2>/dev/null)
        min=$("$cmd" -c "import sys;print(sys.version_info[1])" 2>/dev/null)
        if [ "$maj" = "3" ] && [ "$min" -ge 11 ] 2>/dev/null; then
            PYTHON="$cmd"; break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo "  [ERROR] Se necesita Python 3.11 o superior."
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/bootstrap.py" "$@"
