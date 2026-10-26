@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0..") do cd /d "%%~fI"

echo ============================================================
echo   Poly-X - Empaquetar dataset para Google Drive
echo ============================================================
echo.
echo   Crea un ZIP liviano con SOLO lo necesario para entrenar:
echo     - imagenes .jpg          (~53 MB)
echo     - etiquetas .txt (labels)
echo     - data.yaml (rutas relativas, portable)
echo   Excluye los .npy (copias crudas de 1.5 GB que NO se necesitan).
echo.

if not exist "data_microplastico\images" (
    echo [ERROR] No encuentro data_microplastico\images
    echo Ejecuta este .bat desde la carpeta del proyecto Poly-X.
    echo.
    pause
    exit /b 1
)

set "ZIP=%cd%\PolyX_dataset.zip"
if exist "%ZIP%" del "%ZIP%"

echo Empaquetando en la carpeta del proyecto:
echo   %ZIP%
echo (puede tardar unos segundos)...
echo.

tar --force-local -a -cf "%ZIP%" --exclude=*.npy --exclude=*.cache ^
  "data_microplastico/dataset.yaml" ^
  "data_microplastico/images" ^
  "data_microplastico/labels"

if not exist "%ZIP%" (
    echo [ERROR] No se pudo crear el ZIP. Revisa que 'tar' este disponible
    echo (Windows 10 1803+ lo trae).
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   LISTO. Dataset empaquetado en la carpeta del proyecto:
echo     PolyX_dataset.zip
echo ============================================================
echo.
echo   COMO LLEVARLO AL PC DEL PROFESOR:
echo.
echo   1) Sube PolyX_dataset.zip a tu Google Drive
echo      (arrastralo en drive.google.com).
echo.
echo   2) En el PC del profesor, baja el codigo con Git:
echo        git clone https://github.com/CrissFerrada/Poly-X-Microplastics
echo.
echo   3) Descarga PolyX_dataset.zip del Drive y extraelo DENTRO de la
echo      carpeta del proyecto, de modo que quede:
echo        Poly-X-Microplastics\data_microplastico\images
echo        Poly-X-Microplastics\data_microplastico\labels
echo.
echo   4) Ejecuta SETUP.bat y luego iniciar_polyx.bat.
echo      El data.yaml usa rutas relativas, asi que el Entrenador
echo      encuentra el dataset solo, sin editar nada.
echo.
pause
