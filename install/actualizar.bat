@echo off
title Poly-X - Actualizar
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "RAIZ=%%~fI"

set "PYEXE="
where py >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
)
if "!PYEXE!"=="" (
    where python >nul 2>&1
    if %errorlevel%==0 (
        for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
    )
)
if "!PYEXE!"=="" (
    echo [ERROR] No se encontro Python.
    pause
    exit /b 1
)

"!PYEXE!" "%~dp0update.py" --root "!RAIZ!"
pause
