# ============================================================
#  Poly-X - Actualizador automatico desde GitHub
#  Consulta el ultimo commit de la rama main; si difiere del
#  instalado, descarga el ZIP y sobrescribe SOLO los archivos
#  del programa. Conserva .venv, models, runs y datos locales.
#  No requiere tener Git instalado.
# ============================================================
param([string]$InstallDir)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # acelera Invoke-WebRequest en PS 5.1
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

if (-not $InstallDir) { $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$InstallDir = (Resolve-Path $InstallDir).Path

$repo    = 'CrissFerrada/Poly-X-Microplastics'
$branch  = 'main'
$ua      = @{ 'User-Agent' = 'PolyX-Updater' }
$verFile = Join-Path $InstallDir '.polyx_version'

Write-Host ''
Write-Host '============================================================'
Write-Host '  Poly-X  -  Buscar actualizaciones'
Write-Host '============================================================'
Write-Host ''
Write-Host "Carpeta de instalacion: $InstallDir"
Write-Host 'Comprobando en GitHub...'

try {
    $resp = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/commits/$branch" -Headers $ua
    $remote = $resp.sha
} catch {
    Write-Host '[ERROR] No se pudo consultar GitHub. Revisa tu conexion a internet.' -ForegroundColor Red
    Read-Host 'Pulsa ENTER para salir' | Out-Null
    exit 1
}

$local = ''
if (Test-Path $verFile) { $local = (Get-Content $verFile -Raw).Trim() }

if ($local -and ($local -eq $remote)) {
    Write-Host ''
    Write-Host '[OK] Ya tienes la ultima version. No hay nada que actualizar.' -ForegroundColor Green
    Read-Host 'Pulsa ENTER para salir' | Out-Null
    exit 0
}

Write-Host ''
Write-Host 'Hay una version nueva disponible. Descargando...'
$tmp = Join-Path $env:TEMP ('polyx_update_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $tmp | Out-Null
$zip = Join-Path $tmp 'src.zip'
try {
    Invoke-WebRequest -Uri "https://github.com/$repo/archive/refs/heads/$branch.zip" -OutFile $zip -Headers $ua
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
} catch {
    Write-Host "[ERROR] Fallo la descarga o la descompresion: $_" -ForegroundColor Red
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    Read-Host 'Pulsa ENTER para salir' | Out-Null
    exit 1
}

$srcDir = (Get-ChildItem -Directory $tmp | Select-Object -First 1).FullName
Write-Host 'Aplicando actualizacion (se conservan .venv, models, runs y tus datos)...'

# /E copia subcarpetas SIN borrar archivos locales que no esten en el ZIP
# (por eso .venv, models y runs quedan intactos). Se excluye el propio
# actualizador para no sobrescribirlo mientras se ejecuta.
& robocopy "$srcDir" "$InstallDir" /E /XF actualizar.bat actualizar.ps1 .polyx_version /XD .git /NFL /NDL /NJH /NJS /NP | Out-Null
$rc = $LASTEXITCODE
if ($rc -ge 8) {
    Write-Host "[ERROR] Fallo la copia de archivos (robocopy=$rc)." -ForegroundColor Red
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    Read-Host 'Pulsa ENTER para salir' | Out-Null
    exit 1
}

Set-Content -Path $verFile -Value $remote -Encoding ascii
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
Write-Host '[OK] Codigo actualizado a la ultima version.' -ForegroundColor Green

# Reinstalar dependencias por si requirements.txt cambio (rapido si ya estan)
$venvPy = Join-Path $InstallDir '.venv\Scripts\python.exe'
if (Test-Path $venvPy) {
    Write-Host 'Revisando dependencias (pip)...'
    & $venvPy -m pip install -r (Join-Path $InstallDir 'requirements.txt') -q
    Write-Host '[OK] Dependencias al dia.' -ForegroundColor Green
} else {
    Write-Host '[AVISO] No hay entorno .venv todavia. Ejecuta SETUP.bat para instalarlo.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Listo. Inicia Poly-X con iniciar_polyx.bat (o el acceso directo).'
Read-Host 'Pulsa ENTER para salir' | Out-Null
