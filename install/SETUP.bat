@echo off
setlocal enabledelayedexpansion
title Poly-X - Instalacion
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "SRC=%%~fI"
cd /d "%SRC%"

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
    echo [ERROR] No se encontro Python 3.11 instalado.
    echo Descargalo desde https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

"!PYEXE!" "%SRC%\install\bootstrap.py" %*
exit /b %errorlevel%
