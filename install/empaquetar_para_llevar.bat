@echo off
setlocal enabledelayedexpansion
title Poly-X - Empaquetar para llevar (Drive / USB)
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "SRC=%%~fI"
cd /d "%SRC%"

echo.
echo ============================================================
echo   Poly-X  -  Empaquetar para llevar a otro PC
echo ============================================================
echo.
echo Crea una carpeta LIMPIA y liviana con solo lo necesario para
echo correr Poly-X en otro PC (codigo + modelos + instalador).
echo NO incluye el .venv (no es portable: se recrea con SETUP.bat
echo en el otro PC) ni datasets/runs/archivos pesados.
echo.

REM ---- Donde dejar el paquete ----
set "OUT=%USERPROFILE%\Desktop\Poly-X_para_llevar"
echo Carpeta del paquete (ENTER para usar la sugerida):
echo   %OUT%
set "RESP="
set /p "RESP=Otra ruta (o ENTER): "
if defined RESP set "OUT=!RESP:"=!"

if exist "!OUT!" (
    echo [INFO] La carpeta ya existe; se limpiara: !OUT!
    rmdir /s /q "!OUT!"
)
mkdir "!OUT!"

echo.
echo [INFO] Copiando codigo del programa (src\polyx\)...
robocopy "%SRC%\src\polyx" "!OUT!\src\polyx" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP >nul

if exist "%SRC%\docs\assets\manual_screenshots" (
    echo [INFO] Copiando capturas del manual...
    robocopy "%SRC%\docs\assets\manual_screenshots" "!OUT!\docs\assets\manual_screenshots" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP >nul
)

if exist "%SRC%\docs\manual\Manual_PolyX.html" (
    echo [INFO] Copiando el manual compilado...
    robocopy "%SRC%\docs\manual" "!OUT!\docs\manual" /E /NFL /NDL /NJH /NJS /NP >nul
)

if exist "%SRC%\tools\manual" (
    echo [INFO] Copiando la cadena del manual (texto y motor)...
    robocopy "%SRC%\tools\manual" "!OUT!\tools\manual" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP >nul
)

if exist "%SRC%\models" (
    echo [INFO] Copiando carpeta models\...
    robocopy "%SRC%\models" "!OUT!\models" /E /NFL /NDL /NJH /NJS /NP >nul
)

echo [INFO] Copiando archivos sueltos esenciales...
for %%F in (SETUP.bat iniciar_polyx.bat actualizar.bat actualizar.ps1 migrar_instalacion.ps1 requirements.txt README.md LEEME.txt Poly-X.vbs CLAUDE.md) do (
    if exist "%SRC%\install\%%F" copy /y "%SRC%\install\%%F" "!OUT!\" >nul
)

REM ---- Sello de version ----
REM Sin este archivo el aviso de "hay version nueva" no puede funcionar: el
REM launcher compara este SHA contra el ultimo commit de main en GitHub. Se
REM sella con el commit del que sale ESTE paquete, no con el ultimo de GitHub,
REM para no declarar actualizado algo que quizas va atrasado.
set "SHA="
for /f "tokens=*" %%i in ('git -C "%SRC%" rev-parse HEAD 2^>nul') do set "SHA=%%i"
if defined SHA (
    >"!OUT!\.polyx_version" echo !SHA!
    echo [OK] Paquete sellado con el commit !SHA:~0,7!
) else (
    echo [AVISO] No se pudo leer el commit con git.
    echo         El paquete queda sin sellar y no avisara de versiones nuevas
    echo         hasta la primera actualizacion manual con actualizar.bat.
)

REM ---- Tamano del paquete ----
echo.
echo [OK] Paquete creado en:
echo   !OUT!
echo.

echo ============================================================
echo   COMO LLEVARLO AL OTRO PC
echo ============================================================
echo.
echo  1) Comprime la carpeta  "Poly-X_para_llevar"  en un .zip
echo     (clic derecho -^> Enviar a -^> Carpeta comprimida en zip).
echo  2) Sube ese .zip a Google Drive.
echo  3) En el PC del profe: descarga el .zip y DESCOMPRIMELO.
echo  4) Asegurate de tener Python 3.11 instalado en ese PC.
echo  5) Doble clic en  SETUP.bat  (recrea el entorno .venv ahi).
echo  6) Doble clic en  iniciar_polyx.bat  (o el acceso directo).
echo.
echo  NOTA: los modelos .pt SI van incluidos en este paquete, asi
echo        que el Detector funcionara apenas instales con SETUP.bat.
echo.
pause
