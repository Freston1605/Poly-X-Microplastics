@echo off
title Poly-X
cd /d "%~dp0.."

set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"

if not exist .venv\Scripts\python.exe (
    echo [ERROR] No se encuentra el entorno virtual.
    echo Ejecuta primero install\SETUP.bat
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m polyx.launcher
if errorlevel 1 (
    echo.
    echo Poly-X termino con errores. Revisa el mensaje de arriba.
    pause
)
