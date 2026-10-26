#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Poly-X · Icono de macOS sin ventana de Terminal
#  © Cristofher Ferrada · PUCV
#
#  Un .command SIEMPRE abre Terminal: eso es lo que es, un script que
#  Finder le entrega a la Terminal, y la ventana negra se queda ahi
#  mientras el programa corre. Para arrancar sin ella hace falta un
#  .app, y eso es lo unico que hace este script.
#
#  Es el equivalente de Poly-X.vbs en Windows (pythonw.exe, sin consola).
#
#  NO confundir con construir_app_macOS.command: aquel empaqueta PyTorch
#  entero con PyInstaller — 40 minutos y 3 GB. Este pesa unos KB y tarda
#  un segundo, porque no empaqueta nada: solo llama al .venv que ya esta
#  instalado en esta carpeta.
#
#  Doble clic en Finder para ejecutarlo.
# ══════════════════════════════════════════════════════════════════

# Sin 'set -u': macOS sigue trayendo bash 3.2, y ahi expandir "$@" sin
# argumentos —que es como lo abre Finder— aborta con 'unbound variable'.
cd "$(dirname "$0")" || exit 1
RAIZ="$(pwd)"
VENV_PY="$RAIZ/.venv/bin/python"

DESTINO=""
SILENCIOSO=0
for arg in "$@"; do
    case "$arg" in
        --escritorio)   DESTINO="$HOME/Desktop" ;;
        --aplicaciones) DESTINO="/Applications" ;;
        --aqui)         DESTINO="$RAIZ" ;;
        --silencioso)   SILENCIOSO=1 ;;
    esac
done

# Sin pausa cuando lo llama Lanzar_macOS.command: alli la ventana ya esta
# abierta y el mensaje se lee igual.
fin() {
    [ "$SILENCIOSO" = "1" ] || read -r -p "  Pulsa Enter para cerrar..."
    exit "${1:-0}"
}

if [ ! -x "$VENV_PY" ]; then
    echo "  [ERROR] No hay entorno instalado en esta carpeta."
    echo "          Ejecuta primero Lanzar_macOS.command."
    fin 1
fi

if [ -z "$DESTINO" ]; then
    echo ""
    echo "  ═══════════════════════════════════════════"
    echo "   Poly-X · icono sin ventana de Terminal"
    echo "  ═══════════════════════════════════════════"
    echo ""
    echo "   1) Escritorio"
    echo "   2) Aplicaciones — sale tambien en el Launchpad"
    echo "   3) Aqui mismo, junto al programa"
    echo ""
    read -r -p "  ¿Donde lo dejo? [1] " opcion
    case "$opcion" in
        2) DESTINO="/Applications" ;;
        3) DESTINO="$RAIZ" ;;
        *) DESTINO="$HOME/Desktop" ;;
    esac
fi

if [ ! -d "$DESTINO" ] || [ ! -w "$DESTINO" ]; then
    echo "  [ERROR] No se puede escribir en: $DESTINO"
    fin 1
fi

APP="$DESTINO/Poly-X.app"

# Las versiones viejas del instalador dejaban en el Escritorio un enlace
# llamado 'Poly-X' apuntando al .command. Se borra: si no, quedan dos iconos
# casi identicos y el de antes sigue abriendo la Terminal.
if [ -L "$DESTINO/Poly-X" ]; then
    rm -f "$DESTINO/Poly-X"
    echo "  [i] Retirado el enlace antiguo, que abria la Terminal."
fi

echo ""
echo "  Creando $APP ..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" || {
    echo "  [ERROR] No se pudo crear el .app"; fin 1; }

# ── Icono ──
# El unico icono del repo es assets/polyx.ico, formato Windows: macOS quiere
# .icns. Se convierte con la Pillow que ya trae el entorno. Si algo falla no
# se aborta nada — el .app sale con el icono generico y funciona igual.
ICONO_PLIST=""
ICO="$RAIZ/assets/polyx.ico"
if [ -f "$ICO" ] && command -v iconutil >/dev/null 2>&1; then
    TMP="$(mktemp -d)"
    ICONSET="$TMP/polyx.iconset"
    mkdir -p "$ICONSET"
    "$VENV_PY" - "$ICO" "$ICONSET" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

from PIL import Image

ico, destino = Path(sys.argv[1]), Path(sys.argv[2])
im = Image.open(ico)
try:
    # Un .ico guarda varias resoluciones y Pillow abre una cualquiera. Se toma
    # la mayor a proposito: partir de la miniatura de 16 px deja un icono
    # borroso en cuanto la pantalla es Retina.
    im = im.ico.getimage(max(im.ico.sizes()))
except Exception:
    pass
im = im.convert("RGBA")

for lado in (16, 32, 128, 256, 512):
    im.resize((lado, lado), Image.LANCZOS).save(destino / f"icon_{lado}x{lado}.png")
    im.resize((lado * 2, lado * 2), Image.LANCZOS).save(destino / f"icon_{lado}x{lado}@2x.png")
PY
    if [ $? -eq 0 ] && iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/polyx.icns" 2>/dev/null; then
        ICONO_PLIST="  <key>CFBundleIconFile</key><string>polyx</string>"
    fi
    rm -rf "$TMP"
fi
[ -n "$ICONO_PLIST" ] || echo "  [i] Sin icono propio: saldra el generico de macOS."

# ── Info.plist ──
# El identificador lleva '.lanzador' para no chocar con el 'cl.pucv.polyx' que
# usa construir_app_macOS.command: si coincidieran, LaunchServices mezclaria
# los dos bundles y abriria el que le diera la gana.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Poly-X</string>
  <key>CFBundleDisplayName</key><string>Poly-X</string>
  <key>CFBundleExecutable</key><string>Poly-X</string>
  <key>CFBundleIdentifier</key><string>cl.pucv.polyx.lanzador</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>10.13</string>
  <key>NSHighResolutionCapable</key><true/>
$ICONO_PLIST
</dict>
</plist>
PLIST

# ── Ejecutable del bundle ──
# La ruta va escrita dentro y no se calcula sola: asi el .app funciona tambien
# desde /Applications o el Dock, que es justo donde un camino relativo se
# rompe. Si se mueve la carpeta del programa, se vuelve a ejecutar esto.
{
    echo '#!/bin/bash'
    printf 'RAIZ=%q\n' "$RAIZ"
    cat <<'LANZADOR'
# Lanzador de Poly-X sin consola.
# Generado por crear_icono_macOS.command — no editar a mano: se reescribe
# entero cada vez que aquel se ejecuta.

VENV_PY="$RAIZ/.venv/bin/python"
LOG="$HOME/Library/Logs/Poly-X.log"

alerta() {
    osascript -e "display alert \"Poly-X\" message \"$1\" as critical" >/dev/null 2>&1
}

if [ ! -x "$VENV_PY" ]; then
    alerta "No encuentro la instalacion de Poly-X en:\n$RAIZ\n\nSi moviste o borraste esa carpeta, vuelve a ejecutar crear_icono_macOS.command desde donde este ahora."
    exit 1
fi

# Se comprueba que PySide6 importe de verdad, y no solo que el .venv exista:
# una instalacion interrumpida deja la carpeta creada e inservible, y aqui no
# hay ventana de Terminal donde leer el ImportError.
if ! "$VENV_PY" -c "import PySide6" 2>/dev/null; then
    alerta "La instalacion de Poly-X esta incompleta.\n\nAbre Lanzar_macOS.command en la carpeta del programa: reinstala lo que falte y muestra el detalle."
    exit 1
fi

mkdir -p "$(dirname "$LOG")"
cd "$RAIZ" || exit 1
export PYTHONPATH="$RAIZ/src${PYTHONPATH:+:$PYTHONPATH}"
printf '\n===== %s · Poly-X =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"

# exec, y no una llamada normal: asi Python REEMPLAZA a este script y conserva
# el PID con el que macOS abrio el .app. Lanzado como proceso hijo, el Dock
# mostraria dos iconos — el de Poly-X y uno generico de Python.
# Todo lo que el programa escriba va al registro: eso es lo que sustituye a la
# ventana de Terminal el dia que algo falle.
exec "$VENV_PY" -m polyx.launcher >> "$LOG" 2>&1
LANZADOR
} > "$APP/Contents/MacOS/Poly-X"

chmod +x "$APP/Contents/MacOS/Poly-X"
touch "$APP"   # Finder cachea el icono viejo si no se le toca la fecha

echo ""
echo "  ═══════════════════════════════════════════════════"
echo "   Listo: $APP"
echo ""
echo "   Doble clic y Poly-X abre sin ventana de Terminal."
echo "   Al estar creado en este Mac, Gatekeeper no protesta."
echo ""
echo "   Si algo falla, el detalle queda en:"
echo "     ~/Library/Logs/Poly-X.log"
echo "  ═══════════════════════════════════════════════════"
fin 0
