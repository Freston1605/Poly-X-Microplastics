#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Poly-X · Construir Poly-X.app (macOS)
#
#  ESTO NO ES NECESARIO PARA USAR POLY-X.
#  Para usarlo basta con Lanzar_macOS.command, que es la vía probada.
#  Este script empaqueta la aplicación en un Poly-X.app arrastrable a
#  /Applications, para quien prefiera un icono en el Dock.
#
#  LEE ESTO ANTES DE EJECUTARLO
#  ────────────────────────────
#   · Tarda 15-40 minutos y produce un .app de 2-5 GB: PyTorch y sus
#     librerías de cálculo pesan eso, y van dentro del bundle.
#   · Hay que ejecutarlo EN EL MAC. No se puede compilar desde Windows.
#   · Sin firma de Apple (cuenta de desarrollador, USD 99/año), Gatekeeper
#     lo bloquea igual en otro Mac: hay que abrirlo con clic derecho → Abrir.
#     Es decir, empaquetar NO evita el aviso de "desarrollador no
#     identificado"; solo cambia el icono por el que se hace doble clic.
#   · Empaquetar torch + ultralytics con PyInstaller es frágil. Si falla,
#     Lanzar_macOS.command sigue funcionando y es la opción recomendada.
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
VENV_PY="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "  [ERROR] No hay entorno instalado."
    echo "          Ejecuta primero Lanzar_macOS.command."
    read -r -p "  Pulsa Enter para cerrar..."; exit 1
fi

echo ""
echo "  ══════════════════════════════════════════"
echo "   Poly-X · Construir Poly-X.app"
echo "   Esto tarda entre 15 y 40 minutos."
echo "  ══════════════════════════════════════════"
echo ""
read -r -p "  ¿Continuar? [s/N] " resp
case "$resp" in
    s|S|y|Y) ;;
    *) echo "  Cancelado."; exit 0 ;;
esac

echo "  Instalando PyInstaller..."
"$VENV_PY" -m pip install --upgrade pyinstaller --quiet || {
    echo "  [ERROR] No se pudo instalar PyInstaller."
    read -r -p "  Pulsa Enter para cerrar..."; exit 1; }

rm -rf build dist Poly-X.spec

# --collect-all ultralytics: trae sus .yaml de configuracion, que carga por
#   ruta en tiempo de ejecucion. Sin esto el .app arranca y muere al primer
#   modelo con un "file not found" de un yaml interno.
# --collect-all torch: idem con sus librerias nativas.
# --windowed: sin ventana de terminal detras.
# --icon: se omite a proposito. El unico icono del repo es assets/polyx.ico,
#   que es formato Windows; macOS necesita .icns y PyInstaller falla si se le
#   pasa un .ico. El .app sale con el icono generico hasta que haya un .icns.
# No se pasa --add-data: polyx/assets solo lo crea ensure_dirs() en tiempo de
#   ejecucion, esta vacio y ningun modulo carga nada de ahi. Apuntar a una
#   carpeta inexistente aborta el empaquetado.
echo "  Empaquetando (paciencia)..."
"$VENV_PY" -m PyInstaller \
    --name "Poly-X" \
    --windowed \
    --noconfirm \
    --osx-bundle-identifier "cl.pucv.polyx" \
    --collect-all ultralytics \
    --collect-all torch \
    --collect-all torchvision \
    --collect-submodules polyx \
    install/lanzar_polyx.py

if [ $? -ne 0 ] || [ ! -d "dist/Poly-X.app" ]; then
    echo ""
    echo "  [ERROR] Falló el empaquetado."
    echo "          Usa Lanzar_macOS.command, que no depende de esto."
    read -r -p "  Pulsa Enter para cerrar..."; exit 1
fi

# El .pt no se mete DENTRO del bundle: pesa ~50 MB, cambia con cada
# reentrenamiento y meterlo obligaria a reempaquetar 3 GB para cambiar un
# modelo. Se deja al lado, que es donde paths.py lo busca.
echo ""
echo "  ══════════════════════════════════════════"
echo "   Listo: dist/Poly-X.app"
echo ""
echo "   Para instalarlo, arrástralo a /Applications."
echo "   La primera vez ábrelo con clic DERECHO → Abrir."
echo "  ══════════════════════════════════════════"
read -r -p "  Pulsa Enter para cerrar..."
