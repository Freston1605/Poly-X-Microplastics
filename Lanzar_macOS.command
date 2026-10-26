#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Poly-X · Suite de detección de microplásticos
#  © Cristofher Ferrada · PUCV · macOS
#
#  Doble clic en Finder para ejecutar.
#  La primera vez instala todo solo (10-15 min); después arranca directo.
#
#  La ventana de Terminal que queda detrás es inevitable aquí — un .command
#  es, por definición, un script que Finder le entrega a la Terminal — y de
#  hecho conviene: es donde se lee el error si algo falla. Para el uso diario
#  sin esa ventana está crear_icono_macOS.command, que deja un Poly-X.app.
#
#  ⚠️  Si macOS dice «no se puede abrir porque es de un desarrollador no
#      identificado»: clic DERECHO sobre este archivo → Abrir → Abrir.
#      Solo hace falta la primera vez.
# ══════════════════════════════════════════════════════════════════

cd "$(dirname "$0")" || exit 1

# src/ layout: el paquete vive en src/polyx, asi que src debe estar en PYTHONPATH.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# Ruta absoluta: el enlace del Escritorio no puede apuntar a una ruta
# relativa -- quedaria roto en cuanto se abriera desde otro sitio.
INSTALL_DIR_ABS="$(pwd)"
VENV_PY=".venv/bin/python"

alerta() {
    osascript -e "display alert \"Poly-X\" message \"$1\"" 2>/dev/null || echo "[AVISO] $1"
}

# ── Si ya está instalado, arrancar y salir ──
# Se comprueba que PySide6 importe, no solo que exista la carpeta: una
# instalación interrumpida deja el .venv creado pero inservible, y sin esta
# comprobación el programa fallaba con un ImportError críptico.
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import PySide6" 2>/dev/null; then
    echo ""
    echo "  ═══════════════════════════════════════"
    echo "   Poly-X · Detección de microplásticos"
    echo "   © Cristofher Ferrada · PUCV"
    echo "   Puedes cerrar esta ventana al terminar."
    echo "  ═══════════════════════════════════════"
    echo ""
    "$VENV_PY" -m polyx.launcher
    exit 0
fi

echo ""
echo "  ═══════════════════════════════════════════════"
echo "   Poly-X — Primera instalación en este Mac"
echo "   Solo ocurre una vez. Tarda 10-15 minutos."
echo "  ═══════════════════════════════════════════════"
echo ""

# ── Arquitectura: decide qué PyTorch se puede instalar ──
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    echo "  [i] Apple Silicon detectado: la detección usará la GPU (MPS)."
    APPLE_SILICON=1
else
    echo "  [i] Mac Intel detectado: la detección usará el procesador (CPU)."
    echo "      Es más lento — cuenta ~1 minuto por foto en lotes grandes."
    APPLE_SILICON=0
fi
echo ""

# ── Buscar Python ──
# 3.9 es el mínimo real: ultralytics 8.3 no soporta menos. Se prueban las
# versiones concretas antes que 'python3' a secas porque en un Mac con
# Homebrew 'python3' puede apuntar a una versión que no sirva.
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
    alerta "No se encontró Python 3.9 o superior.\n\nDescárgalo de python.org/downloads y vuelve a hacer doble clic aquí."
    echo "  [ERROR] Falta Python 3.9+. Descarga: https://www.python.org/downloads/"
    read -r -p "  Pulsa Enter para cerrar..."
    exit 1
fi
echo "  [OK] Python: $($PYTHON --version)"

# ── Entorno virtual ──
echo "  [1/4] Creando entorno virtual..."
rm -rf .venv
"$PYTHON" -m venv .venv || {
    alerta "No se pudo crear el entorno virtual.\nRevisa los permisos de esta carpeta."
    read -r -p "  Pulsa Enter para cerrar..."; exit 1; }

echo "  [2/4] Actualizando pip..."
"$VENV_PY" -m pip install --upgrade pip --quiet

# ── PyTorch ──
# Se instala aparte y ANTES que el resto: ultralytics lo arrastraría como
# dependencia, pero en un Mac Intel se traería una versión sin ruedas
# disponibles y fallaría a mitad. Fijándolo antes, pip ya lo da por resuelto.
echo "  [3/4] Instalando PyTorch (~200 MB, varios minutos)..."
if [ "$APPLE_SILICON" = "1" ]; then
    "$VENV_PY" -m pip install torch torchvision --quiet
else
    # PyTorch dejó de publicar ruedas para macOS Intel a partir de la 2.3:
    # la 2.2.2 es la última que existe para x86_64. Sin este pin, pip no
    # encuentra candidato y la instalación muere sin explicar por qué.
    echo "        (Mac Intel: se fija torch 2.2.2, la última con soporte x86)"
    "$VENV_PY" -m pip install "torch==2.2.2" "torchvision==0.17.2" --quiet
fi
if [ $? -ne 0 ]; then
    alerta "Falló la instalación de PyTorch.\nRevisa tu conexión a internet e inténtalo de nuevo."
    echo "  [ERROR] Falló PyTorch."; read -r -p "  Pulsa Enter para cerrar..."; exit 1
fi

echo "  [4/4] Instalando el resto (PySide6, ultralytics, OpenCV...)..."
"$VENV_PY" -m pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    alerta "Falló la instalación de dependencias.\nRevisa tu conexión a internet e inténtalo de nuevo."
    echo "  [ERROR] Falló la instalación de requirements.txt"
    read -r -p "  Pulsa Enter para cerrar..."; exit 1
fi

# ── Comprobación real, no solo 'pip dijo que sí' ──
echo ""
echo "  Comprobando la instalación..."
"$VENV_PY" - <<'PY'
import sys
fallos = []
for mod in ("PySide6", "torch", "ultralytics", "cv2", "numpy", "matplotlib"):
    try:
        __import__(mod)
    except Exception as e:
        fallos.append(f"{mod}: {e}")
if fallos:
    print("  [ERROR] No se pudieron importar:")
    for f in fallos:
        print("     -", f)
    sys.exit(1)
import torch, platform
mps = getattr(torch.backends, "mps", None)
acel = "GPU del Mac (MPS)" if (mps and mps.is_available()) else "CPU"
print(f"  [OK] torch {torch.__version__} · aceleracion: {acel}")
PY
if [ $? -ne 0 ]; then
    alerta "La instalación terminó pero algo no se puede cargar.\nRevisa la ventana de Terminal para ver el detalle."
    read -r -p "  Pulsa Enter para cerrar..."; exit 1
fi

# ── Acceso directo en el Escritorio (opcional) ──
# Equivalente del Poly-X.lnk que ofrece SETUP.bat en Windows. Sin esto hay
# que recordar en que carpeta quedo el programa, que es justo lo que se
# olvida. Se pregunta en vez de imponerlo: no todo el mundo quiere cosas
# nuevas en su Escritorio.
ESCRITORIO="$HOME/Desktop"
if [ -d "$ESCRITORIO" ] && [ ! -e "$ESCRITORIO/Poly-X.app" ]; then
    echo ""
    read -r -p "  ¿Crear un icono de Poly-X en el Escritorio? [S/n] " crear
    case "$crear" in
        n|N) echo "  De acuerdo, sin icono. Puedes crearlo luego con crear_icono_macOS.command." ;;
        *)
            # Un .app y no un enlace a este mismo .command: el .command arrastra
            # siempre una ventana de Terminal detras, que se queda abierta todo
            # el rato que dure el programa. El .app arranca sin ella.
            bash "$INSTALL_DIR_ABS/crear_icono_macOS.command" --escritorio --silencioso || {
                echo "  [AVISO] No se pudo crear el icono. No es grave:"
                echo "          abre Poly-X desde esta misma carpeta."
            }
            ;;
    esac
fi

echo ""
echo "  ═══════════════════════════════════════"
echo "   Instalación completa. Abriendo Poly-X..."
echo "  ═══════════════════════════════════════"
echo ""
"$VENV_PY" -m polyx.launcher
