#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Poly-X · Instalación en Linux
#
#  Crea el entorno virtual, instala las dependencias y deja el programa
#  listo para arrancar con install/run.sh (o el acceso directo del menu).
#
#  Equivalente de SETUP.bat. Ejecutar desde la carpeta del proyecto:
#      bash install/setup.sh
# ══════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "  ══════════════════════════════════════════"
echo "   Poly-X  ·  Instalación en Linux"
echo "  ══════════════════════════════════════════"
echo ""
echo "  Carpeta del proyecto: $PROJECT_ROOT"
echo ""

# ── 1. Localizar Python 3.11+ ──
PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3; do
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
    echo "          Instalalo con tu gestor de paquetes, por ejemplo:"
    echo "            sudo apt install python3.11 python3.11-venv   (Debian/Ubuntu)"
    echo "            sudo dnf install python3.11                     (Fedora)"
    exit 1
fi
echo "  [OK] Python: $($PYTHON --version)"

# ── 2. Entorno virtual ──
if [ ! -d ".venv" ]; then
    echo "  [INFO] Creando entorno virtual .venv ..."
    "$PYTHON" -m venv .venv || { echo "  [ERROR] No se pudo crear el venv."; exit 1; }
else
    echo "  [INFO] .venv ya existe; se reutiliza."
fi
VENV_PY="$PROJECT_ROOT/.venv/bin/python"

# ── 3. GPU NVIDIA (opcional) ──
TORCH_INDEX=""
if command -v nvidia-smi &>/dev/null && nvidia-smi >/dev/null 2>&1; then
    echo "  [INFO] GPU NVIDIA detectada."
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    CCMAJ=${CC%%.*}
    if [ "$CCMAJ" -ge 12 ] 2>/dev/null; then
        TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu128"
        echo "        -> CUDA 12.8 (Blackwell / RTX 50xx)."
    else
        TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu118"
        echo "        -> CUDA 11.8."
    fi
fi

# ── 4. PyTorch (antes que el resto, como en SETUP.bat) ──
echo "  [INFO] Instalando PyTorch..."
if [ -n "$TORCH_INDEX" ]; then
    "$VENV_PY" -m pip install --upgrade pip wheel setuptools
    "$VENV_PY" -m pip install "torch==2.7.1" "torchvision==0.22.1" $TORCH_INDEX
else
    "$VENV_PY" -m pip install --upgrade pip wheel setuptools
    "$VENV_PY" -m pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cpu
fi

# ── 5. Resto de dependencias ──
echo "  [INFO] Instalando dependencias del proyecto..."
"$VENV_PY" -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "  [ERROR] Hubo problemas instalando dependencias."
    exit 1
fi

# ── 6. Comprobacion ──
echo "  [INFO] Verificando la instalacion..."
"$VENV_PY" -c "import PySide6, ultralytics, cv2, numpy, matplotlib" 2>/dev/null \
    && echo "  [OK] Dependencias importadas correctamente." \
    || echo "  [AVISO] Alguna dependencia no se importo; revisa los mensajes de arriba."

# ── 7. Carpetas de trabajo ──
mkdir -p "$PROJECT_ROOT/models" "$PROJECT_ROOT/data"
echo "  [OK] Carpetas models/ y data/ listas."

# ── 8. Acceso directo en el menu de aplicaciones ──
DESKTOP_DIR="$HOME/.local/share/applications"
if [ -d "$DESKTOP_DIR" ] || mkdir -p "$DESKTOP_DIR" 2>/dev/null; then
    sed "s|@@PROJECT_ROOT@@|$PROJECT_ROOT|g" "$SCRIPT_DIR/polyx.desktop" \
        > "$DESKTOP_DIR/polyx.desktop" 2>/dev/null && \
        chmod +x "$DESKTOP_DIR/polyx.desktop" 2>/dev/null && \
        (update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true) && \
        echo "  [OK] Acceso directo 'Poly-X' creado en el menu de aplicaciones."
fi

echo ""
echo "  ══════════════════════════════════════════"
echo "   Instalacion completada"
echo "  ══════════════════════════════════════════"
echo ""
echo "  Poly-X quedo instalado en: $PROJECT_ROOT"
echo "  Inicialo con:  bash install/run.sh"
echo "  o desde el menu de aplicaciones (Poly-X)."
echo ""