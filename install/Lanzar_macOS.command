#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Poly-X · Suite de detección de microplásticos (macOS)
#  Doble clic en Finder para ejecutar.
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

VENV_PY="$PROJECT_ROOT/.venv/bin/python"

# ── Si ya está instalado, arrancar y salir ──
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import PySide6" 2>/dev/null; then
    echo ""
    echo "  ═══════════════════════════════════════"
    echo "   Poly-X · Detección de microplásticos"
    echo "   © Cristofher Ferrada · PUCV"
    echo "  ═══════════════════════════════════════"
    echo ""
    "$VENV_PY" -m polyx.launcher
    exit 0
fi

echo ""
echo "  ═══════════════════════════════════════════════"
echo "   Poly-X — Primera instalación en este Mac"
echo "  ═══════════════════════════════════════════════"
echo ""

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        maj=$("$cmd" -c "import sys;print(sys.version_info[0])" 2>/dev/null)
        min=$("$cmd" -c "import sys;print(sys.version_info[1])" 2>/dev/null)
        if [ "$maj" = "3" ] && [ "$min" -ge 9 ] 2>/dev/null; then
            PYTHON="$cmd"; break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  [ERROR] Falta Python 3.9+. Descarga: https://www.python.org/downloads/"
    read -r -p "  Pulsa Enter para cerrar..."
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/bootstrap.py" --dest "$PROJECT_ROOT" --yes
if [ $? -ne 0 ]; then
    echo "  [ERROR] La instalación falló."
    read -r -p "  Pulsa Enter para cerrar..."
    exit 1
fi

# Acceso directo opcional
ESCRITORIO="$HOME/Desktop"
if [ -d "$ESCRITORIO" ] && [ ! -e "$ESCRITORIO/Poly-X.app" ]; then
    read -r -p "  ¿Crear un icono de Poly-X en el Escritorio? [S/n] " crear
    case "$crear" in
        n|N) echo "  De acuerdo, sin icono." ;;
        *)
            bash "$SCRIPT_DIR/crear_icono_macOS.command" --escritorio --silencioso 2>/dev/null || echo "  [AVISO] No se pudo crear el icono."
            ;;
    esac
fi

echo ""
echo "  ═══════════════════════════════════════"
echo "   Instalación completa. Abriendo Poly-X..."
echo "  ═══════════════════════════════════════"
echo ""
"$VENV_PY" -m polyx.launcher