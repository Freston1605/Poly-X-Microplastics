#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Poly-X · Actualizador automático desde GitHub (Linux)
#
#  Consulta el último commit de la rama main; si difiere del instalado,
#  descarga el ZIP y sobrescribe SOLO los archivos del programa.
#  Conserva .venv, models, runs y tus datos.
#
#  Equivalente de actualizar.ps1 / actualizar_macOS.command.
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
INSTALL_DIR="$PROJECT_ROOT"

REPO="CrissFerrada/Poly-X-Microplastics"
RAMA="main"
VER_FILE="$INSTALL_DIR/.polyx_version"

alerta() { echo "[AVISO] $1"; }
fin() { read -r -p "  Pulsa Enter para cerrar..."; exit "${1:-0}"; }

echo ""
echo "  ══════════════════════════════════════════"
echo "   Poly-X  ·  Buscar actualizaciones"
echo "  ══════════════════════════════════════════"
echo ""
echo "  Carpeta: $INSTALL_DIR"
echo "  Consultando GitHub..."

REMOTO=$(curl -fsSL -H "User-Agent: PolyX-Updater" \
    "https://api.github.com/repos/$REPO/commits/$RAMA" 2>/dev/null \
    | grep -m1 '"sha"' | sed -E 's/.*"sha"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')

if [ -z "$REMOTO" ]; then
    echo "  [ERROR] No se pudo consultar GitHub. Revisa tu conexión."
    alerta "No se pudo consultar GitHub.\nRevisa tu conexión a internet."
    fin 1
fi

LOCAL=""
[ -f "$VER_FILE" ] && LOCAL=$(tr -d '[:space:]' < "$VER_FILE")

if [ -n "$LOCAL" ] && [ "$LOCAL" = "$REMOTO" ]; then
    echo ""
    echo "  [OK] Ya tienes la última versión. No hay nada que actualizar."
    fin 0
fi

echo "  Hay una versión nueva. Descargando..."

TMP=$(mktemp -d) || { echo "  [ERROR] No se pudo crear carpeta temporal."; fin 1; }
trap 'rm -rf "$TMP"' EXIT

if ! curl -fsSL -H "User-Agent: PolyX-Updater" \
        "https://github.com/$REPO/archive/refs/heads/$RAMA.zip" -o "$TMP/src.zip"; then
    echo "  [ERROR] Falló la descarga."
    alerta "Falló la descarga de la versión nueva."
    fin 1
fi

if ! unzip -q "$TMP/src.zip" -d "$TMP"; then
    echo "  [ERROR] Falló la descompresión."
    fin 1
fi

SRC=$(find "$TMP" -maxdepth 1 -type d -name "*-$RAMA" | head -1)
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
    echo "  [ERROR] El ZIP no tenía la estructura esperada."
    fin 1
fi

echo "  Aplicando (se conservan .venv, models, runs y tus datos)..."

# rsync SIN --delete, igual que robocopy /E: copia y sobrescribe lo que viene,
# pero no borra lo que solo existe en local. Por eso el entorno virtual, los
# modelos y los resultados sobreviven.
if ! rsync -a \
        --exclude '.git' \
        --exclude '.venv' \
        --exclude '.polyx_version' \
        --exclude 'actualizar_linux.sh' \
        "$SRC"/ "$INSTALL_DIR"/; then
    echo "  [ERROR] Falló la copia de archivos."
    alerta "Falló la copia de archivos de la actualización."
    fin 1
fi

chmod +x "$INSTALL_DIR"/*.sh 2>/dev/null

echo "$REMOTO" > "$VER_FILE"
echo "  [OK] Código actualizado."

# ── Dependencias, por si requirements.txt cambió ──
VENV_PY="$INSTALL_DIR/.venv/bin/python"
if [ -x "$VENV_PY" ]; then
    echo "  Revisando dependencias..."
    "$VENV_PY" -m pip install -r "$INSTALL_DIR/requirements.txt" --quiet \
        && echo "  [OK] Dependencias al día." \
        || echo "  [AVISO] Algo falló al revisar dependencias; el código sí se actualizó."
else
    echo "  [AVISO] No hay entorno .venv todavía. Abre install/setup.sh para instalarlo."
fi

echo ""
echo "  ══════════════════════════════════════════"
echo "   Actualizado a ${REMOTO:0:7}"
echo "   Abre install/run.sh para usar Poly-X."
echo "  ══════════════════════════════════════════"
fin 0