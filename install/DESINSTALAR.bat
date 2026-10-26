@echo off
REM Lanzador de doble clic del desinstalador. La logica esta en el .ps1:
REM buscar en un .bat carpetas por todo el disco y leer accesos directos es
REM mucho mas fragil que hacerlo en PowerShell.
title Poly-X - Desinstalador
cd /d "%~dp0"

set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"

where powershell >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se encontro PowerShell; no se puede desinstalar automaticamente.
    pause
    exit /b 1
)

echo.
echo  Este desinstalador busca TODO rastro de Poly-X en el equipo
echo  y lo retira, para poder instalar desde cero.
echo.
echo  Lo retirado va a la PAPELERA: si algo hacia falta, se restaura.
echo.
echo  Se te mostrara el inventario y se pedira confirmacion antes
echo  de tocar nada.
echo.
echo ------------------------------------------------------------
echo  1. Buscar y retirar  ^(recomendado^)
echo  2. Solo buscar, no tocar nada
echo  3. Buscar en TODO el disco  ^(lento, no se le escapa nada^)
echo  4. Retirar conservando modelos y resultados
echo  5. Salir
echo ------------------------------------------------------------
echo.
set "OPCION="
set /p "OPCION=Elige una opcion [1-5]: "

if "%OPCION%"=="2" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%\desinstalar.ps1" -SoloBuscar
    goto :fin
)
if "%OPCION%"=="3" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%\desinstalar.ps1" -Profundo
    goto :fin
)
if "%OPCION%"=="4" (
    set "RESPALDO=%USERPROFILE%\Desktop\PolyX_respaldo"
    echo.
    echo  Los datos se copiaran a: !RESPALDO!
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%\desinstalar.ps1" -ConservarDatos "%USERPROFILE%\Desktop\PolyX_respaldo"
    goto :fin
)
if "%OPCION%"=="5" goto :fin

powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%\desinstalar.ps1"

:fin
echo.
pause
