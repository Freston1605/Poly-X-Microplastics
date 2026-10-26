param([string]$InstallDir)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = if ($InstallDir) { $InstallDir } else { Split-Path -Parent $here }
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) { Write-Host "No se encontro Python."; Read-Host "Pulsa ENTER"; exit 1 }
& $py.Source (Join-Path $here "update.py") --root $root
Read-Host "Pulsa ENTER para salir" | Out-Null
