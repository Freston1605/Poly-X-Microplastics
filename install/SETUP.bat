@echo off
setlocal enabledelayedexpansion
title Poly-X - Instalacion
REM Trabajar siempre desde la carpeta donde esta este SETUP.bat
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "SRC=%%~fI"
cd /d "%SRC%"

echo.
echo ============================================================
echo   Poly-X  -  Instalacion del entorno
echo ============================================================
echo.
echo Carpeta de origen (lo que descargaste de GitHub):
echo   %SRC%
echo.

REM ============================================================
REM   0. Donde instalar
REM ============================================================
echo Donde quieres INSTALAR Poly-X?
echo   - Pulsa ENTER para instalarlo AQUI MISMO (recomendado)
echo   - O escribe/pega la ruta de otra carpeta y pulsa ENTER
echo.
set "DEST="
set /p "DEST=Carpeta de instalacion: "
REM Quitar comillas si las pego el usuario
if defined DEST set "DEST=!DEST:"=!"
if not defined DEST set "DEST=%SRC%"

if /i not "!DEST!"=="%SRC%" (
    echo.
    echo [INFO] Se instalara en: !DEST!
    if not exist "!DEST!" mkdir "!DEST!"
    echo [INFO] Copiando archivos del programa ^(esto puede tardar un momento^)...
    robocopy "%SRC%" "!DEST!" /E /XD .venv .git __pycache__ runs /XF *.log /NFL /NDL /NJH /NJS /NP >nul
    if errorlevel 8 (
        echo [ERROR] No se pudieron copiar los archivos a "!DEST!".
        pause & exit /b 1
    )
    cd /d "!DEST!"
)
set "INSTALL=%cd%"
echo.
echo [OK] Carpeta de instalacion: !INSTALL!
echo.

REM ============================================================
REM   1. Localizar Python 3.11
REM ============================================================
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
    echo Marca "Add Python to PATH" y "tcl/tk and IDLE" durante la instalacion.
    pause
    exit /b 1
)
echo [OK] Python encontrado en: !PYEXE!

REM ============================================================
REM   2. Limpiar venv viejo si esta roto
REM ============================================================
if exist .venv (
    set "VENV_ROTO="
    .venv\Scripts\python.exe --version >nul 2>&1
    if errorlevel 1 set "VENV_ROTO=1"

    REM No basta con que Python arranque. Una reinstalacion de torch hecha con el
    REM programa abierto deja DLLs a medio reemplazar: python funciona, pero
    REM "import torch" falla con WinError 1114 al cargar c10.dll. Reinstalar
    REM encima no lo arregla, porque pip ve el paquete como ya instalado. La
    REM unica salida fiable es rehacer el entorno.
    if not defined VENV_ROTO (
        .venv\Scripts\python.exe -c "import torch" >nul 2>&1
        if errorlevel 1 (
            .venv\Scripts\python.exe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') is None else 1)" >nul 2>&1
            if errorlevel 1 (
                echo [AVISO] PyTorch esta instalado pero no carga ^(DLL corrupta^).
                set "VENV_ROTO=1"
            )
        )
    )

    if defined VENV_ROTO (
        echo [INFO] El entorno .venv no sirve. Se rehace desde cero...
        rmdir /s /q .venv
        if exist .venv (
            echo [ERROR] No se pudo borrar .venv. Cierra Poly-X y vuelve a ejecutar.
            pause & exit /b 1
        )
    )
)

REM ============================================================
REM   3. Crear venv si no existe
REM ============================================================
if not exist .venv (
    echo [INFO] Creando entorno virtual .venv ...
    "!PYEXE!" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el venv.
        pause & exit /b 1
    )
)

REM ============================================================
REM   4. Actualizar pip
REM ============================================================
echo [INFO] Actualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools

REM ============================================================
REM   5. Detectar GPU NVIDIA
REM ============================================================
REM Version de torch fijada a proposito. Sin pin, pip trae la ultima (2.11 en
REM agosto 2026), que no es la probada con numpy 1.26.4 y ultralytics 8.3.40.
REM 2.7.1 tiene rueda cu128, asi que cubre tambien las tarjetas Blackwell.
set "TORCHVER=2.7.1"
set "TVISIONVER=0.22.1"

set "HAS_GPU=0"
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    nvidia-smi >nul 2>&1
    if !errorlevel!==0 set "HAS_GPU=1"
)

REM No basta con saber que hay GPU NVIDIA: hay que saber de que generacion es.
REM Las ruedas cu118 NO traen kernels sm_120 (Blackwell, RTX 50xx). Si se
REM instalan en una de esas tarjetas, torch.cuda.is_available() devuelve True y
REM todo parece bien hasta que el primer forward falla con
REM "no kernel image is available for execution on the device".
set "CC="
set "CCMAJ="
if "!HAS_GPU!"=="1" (
    REM Sin "noheader": la coma de "csv,noheader" la parte cmd como separador de
    REM argumentos y nvidia-smi falla. Con skip=1 se descarta el encabezado.
    for /f "skip=1 tokens=*" %%i in ('nvidia-smi --query-gpu^=compute_cap --format^=csv 2^>nul') do (
        if not defined CC set "CC=%%i"
    )
    if defined CC (
        for /f "tokens=1 delims=." %%a in ("!CC!") do set "CCMAJ=%%a"
    )
    REM Si nvidia-smi devolvio texto de error en vez de un numero, un "GEQ 12"
    REM lo compararia como cadena y daria verdadero por accidente, instalando la
    REM rueda equivocada. Solo se acepta un entero.
    if defined CCMAJ (
        echo !CCMAJ!| findstr /r /c:"^[0-9][0-9]*$" >nul || set "CCMAJ="
    )
    if not defined CCMAJ (
        REM Drivers viejos no soportan --query-gpu=compute_cap. Caemos al nombre.
        for /f "skip=1 tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv 2^>nul') do (
            echo %%i | findstr /i /c:"RTX 50" >nul && set "CCMAJ=12"
        )
    )
)

if "!HAS_GPU!"=="1" (
    set "CUDAWHL=cu118"
    set "CUDANOM=11.8"
    if defined CCMAJ (
        if !CCMAJ! GEQ 12 (
            set "CUDAWHL=cu128"
            set "CUDANOM=12.8"
        )
    )
    echo [OK] GPU NVIDIA detectada ^(compute capability: !CC!^).
    echo [INFO] Instalando PyTorch con CUDA !CUDANOM! ^(3-4 GB^)...
    .venv\Scripts\python.exe -m pip install torch==!TORCHVER! torchvision==!TVISIONVER! --index-url https://download.pytorch.org/whl/!CUDAWHL!
) else (
    echo [INFO] Sin GPU NVIDIA. Instalando PyTorch CPU...
    .venv\Scripts\python.exe -m pip install torch==!TORCHVER! torchvision==!TVISIONVER! --index-url https://download.pytorch.org/whl/cpu
)

REM ============================================================
REM   6. Resto de dependencias
REM ============================================================
echo [INFO] Instalando dependencias del proyecto...
.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Hubo problemas instalando dependencias.
    pause & exit /b 1
)

REM ============================================================
REM   7. Comprobacion rapida de que todo importa
REM ============================================================
echo [INFO] Verificando la instalacion...
.venv\Scripts\python.exe -c "import PySide6, ultralytics, cv2, numpy, matplotlib; import PySide6.QtWebEngineWidgets" 2>nul
if errorlevel 1 (
    echo [AVISO] Algunas dependencias no se importaron correctamente.
    echo         El programa podria fallar. Revisa los mensajes de arriba.
) else (
    echo [OK] Todas las dependencias se importan correctamente ^(incluida la exportacion a PDF^).
)

REM ============================================================
REM   7b. Comprobar que torch trae kernels para ESTA GPU
REM ============================================================
REM Que torch.cuda.is_available() diga True no garantiza nada: lo que importa
REM es que la arquitectura de la tarjeta este en get_arch_list(). Si no esta,
REM el fallo aparece recien al entrenar, y ahi cuesta mucho mas diagnosticarlo.
if "!HAS_GPU!"=="1" (
    echo [INFO] Verificando compatibilidad GPU/PyTorch...
    .venv\Scripts\python.exe -c "import sys, torch; ok=torch.cuda.is_available(); archs=torch.cuda.get_arch_list() if ok else []; cap='sm_%%d%%d'%%torch.cuda.get_device_capability(0) if ok else ''; print('   GPU  :', torch.cuda.get_device_name(0) if ok else 'no disponible'); print('   arch :', cap); print('   torch:', torch.__version__, '| soporta:', ' '.join(archs) if archs else 'nada'); sys.exit(0 if (ok and cap in archs) else 1)"
    if errorlevel 1 (
        echo.
        echo [ERROR] PyTorch NO trae kernels para esta GPU.
        echo         Entrenar o detectar fallaria con
        echo         "no kernel image is available for execution on the device".
        echo.
        REM Reinstalar de verdad, no solo avisar. El pip de arriba puede no haber
        REM hecho nada: si el .venv ya traia torch de un intento anterior o de una
        REM instalacion migrada, "pip install torch" lo da por satisfecho y no
        REM cambia la rueda, aunque sea la equivocada. Aqui hace falta forzarlo.
        echo [INFO] Reinstalando PyTorch con CUDA !CUDANOM! ^(~2.5 GB^)...
        .venv\Scripts\python.exe -m pip install --force-reinstall torch==!TORCHVER! torchvision==!TVISIONVER! --index-url https://download.pytorch.org/whl/!CUDAWHL!
        REM Reinstalar torch arrastra sus dependencias y se lleva por delante los
        REM pines de requirements.txt. En la practica subio numpy a 2.x, donde
        REM np.trapz ya no existe, y ultralytics 8.3.40 lo sigue usando: el
        REM entrenamiento moria al validar la primera epoca, con la GPU ya bien.
        echo [INFO] Restaurando las versiones fijadas en requirements.txt...
        .venv\Scripts\python.exe -m pip install -r requirements.txt
        echo.
        echo [INFO] Verificando de nuevo...
        .venv\Scripts\python.exe -c "import sys, torch; ok=torch.cuda.is_available(); archs=torch.cuda.get_arch_list() if ok else []; cap='sm_%%d%%d'%%torch.cuda.get_device_capability(0) if ok else ''; print('   torch:', torch.__version__, '| arch:', cap); sys.exit(0 if (ok and cap in archs) else 1)"
        if errorlevel 1 (
            echo.
            echo [ERROR] Sigue sin coincidir. Revisa que la rueda !CUDAWHL! cubra
            echo         esta tarjeta y reinstala a mano.
            echo.
            pause
        ) else (
            echo [OK] Corregido: PyTorch ya tiene kernels para esta GPU.
        )
    ) else (
        echo [OK] PyTorch tiene kernels para esta GPU.
    )
)

REM ============================================================
REM   7b-bis. Sello de version
REM ============================================================
REM Sin .polyx_version el launcher no puede saber si va atrasado y el aviso de
REM actualizacion nunca aparece. El paquete "para llevar" ya viene sellado; esto
REM cubre a quien descargo el ZIP directo desde GitHub, que llega sin sello.
if not exist "!INSTALL!\.polyx_version" (
    where powershell >nul 2>&1
    if !errorlevel!==0 (
        echo [INFO] Registrando la version instalada...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-RestMethod -Uri 'https://api.github.com/repos/CrissFerrada/Poly-X-Microplastics/commits/main' -Headers @{'User-Agent'='PolyX-Setup'} -TimeoutSec 15; Set-Content -LiteralPath '!INSTALL!\.polyx_version' -Value $r.sha -Encoding ascii -NoNewline } catch { }"
        if exist "!INSTALL!\.polyx_version" (
            echo [OK] Version registrada: el aviso de actualizaciones queda activo.
        ) else (
            echo [AVISO] No se pudo consultar GitHub. Poly-X funciona igual, pero
            echo         no avisara de versiones nuevas hasta la primera
            echo         actualizacion manual con actualizar.bat.
        )
    )
)

REM ============================================================
REM   7c. Instalaciones anteriores
REM ============================================================
REM Va al final a proposito: si algo de lo de arriba fallo, la copia vieja
REM sigue intacta y el equipo no queda sin un Poly-X que funcione.
if exist "%~dp0migrar_instalacion.ps1" (
    where powershell >nul 2>&1
    if !errorlevel!==0 (
        echo.
        echo [INFO] Buscando instalaciones anteriores de Poly-X...
        powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0migrar_instalacion.ps1" -InstallDir "!INSTALL!"
    )
)

echo.
echo ============================================================
echo   Instalacion completada
echo ============================================================
echo.
echo  Poly-X quedo instalado en:
echo     !INSTALL!
echo.

REM ============================================================
REM   8. Carpetas de trabajo
REM ============================================================
REM Se crean ahora para que el usuario tenga donde dejar el .pt y el dataset
REM sin tener que adivinar la estructura.
if not exist "!INSTALL!\models" mkdir "!INSTALL!\models"
if not exist "!INSTALL!\data"   mkdir "!INSTALL!\data"
echo [OK] Carpetas  models\  (pesos .pt)  y  data\  (datasets) listas.

REM ============================================================
REM   9. Acceso directo en el Escritorio
REM ============================================================
REM Apunta a Poly-X.vbs, no al .bat: el .vbs arranca con pythonw y no deja una
REM ventana negra abierta detras del programa.
REM (Se usa 'goto' para que los parentesis del comando PowerShell no
REM  interfieran con el parser de bloques de cmd.)
set "MKLINK="
set /p "MKLINK=Crear un acceso directo 'Poly-X' en el Escritorio? (ENTER = si): "
if not defined MKLINK set "MKLINK=S"
if /i not "!MKLINK!"=="S" goto :skip_shortcut
set "LAUNCH=!INSTALL!\Poly-X.vbs"
if not exist "!LAUNCH!" set "LAUNCH=!INSTALL!\iniciar_polyx.bat"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut((Join-Path $d 'Poly-X.lnk')); $s.TargetPath='!LAUNCH!'; $s.WorkingDirectory='!INSTALL!'; $s.Description='Poly-X - deteccion de microplasticos'; $ico=Join-Path '!INSTALL!' 'assets\polyx.ico'; if (Test-Path $ico) { $s.IconLocation=$ico }; $s.Save()"
if errorlevel 1 (echo [AVISO] No se pudo crear el acceso directo.) else (echo [OK] Acceso directo 'Poly-X' creado en el Escritorio.)
:skip_shortcut

echo.
echo  COMO INICIARLO:
echo     - Doble clic en el acceso directo 'Poly-X' del Escritorio
echo     - O doble clic en  Poly-X.vbs  dentro de !INSTALL!
echo.
echo  QUE FALTA (no viene en la descarga de GitHub, por tamano):
echo     1) Un modelo entrenado .pt  ->  dejalo en  !INSTALL!\models\
echo     2) El dataset de entrenamiento  ->  descomprimelo donde quieras
echo        y en el Entrenador elige su  dataset.yaml
echo.

echo.
set "RUNNOW="
set /p "RUNNOW=Quieres iniciar Poly-X ahora? (ENTER = si): "
if not defined RUNNOW set "RUNNOW=S"
if /i not "!RUNNOW!"=="S" goto :no_run
if exist "!INSTALL!\Poly-X.vbs" (
    start "" "!INSTALL!\Poly-X.vbs"
) else (
    start "" "!INSTALL!\iniciar_polyx.bat"
)
exit /b 0
:no_run

echo.
echo Listo. Cuando quieras, abre  Poly-X.vbs  en:
echo   !INSTALL!
echo.
pause
