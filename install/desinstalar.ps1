# ============================================================
#  Poly-X - Desinstalador (limpieza profunda)
#
#  Recorre el equipo buscando TODO rastro de Poly-X: carpetas de
#  instalacion, accesos directos, preferencias y entornos virtuales.
#  Muestra el inventario, pide confirmacion y recien entonces retira.
#
#  Lo retirado va a la PAPELERA, no se borra: si algo hacia falta,
#  se restaura. Con -Definitivo se borra de verdad.
#
#  Uso:
#     .\desinstalar.ps1                 busca y pregunta
#     .\desinstalar.ps1 -SoloBuscar     solo lista, no toca nada
#     .\desinstalar.ps1 -Profundo       recorre los discos enteros
#     .\desinstalar.ps1 -ConservarDatos C:\respaldo
#     .\desinstalar.ps1 -IncluirRepos   incluye copias de trabajo de git
# ============================================================
param(
    [switch]$SoloBuscar,
    [switch]$Profundo,
    [switch]$Definitivo,
    [switch]$IncluirRepos,
    [string]$ConservarDatos = ""
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Lo que se considera "datos" y puede rescatarse antes de retirar.
$CarpetasDatos = @('models', 'runs', 'runs_train', 'data_microplastico', 'data')

function Write-Titulo($t) {
    Write-Host ''
    Write-Host ('=' * 64)
    Write-Host "  $t"
    Write-Host ('=' * 64)
}

function Get-TamanoMB($dir) {
    try {
        $b = (Get-ChildItem -LiteralPath $dir -Recurse -File -ErrorAction SilentlyContinue |
              Measure-Object Length -Sum).Sum
        return [math]::Round(($b / 1MB), 1)
    } catch { return 0 }
}

function Test-EsPolyX($dir) {
    # Marca inequivoca del paquete. Un .bat suelto no basta: la gente los copia.
    return (Test-Path (Join-Path $dir 'polyx\launcher.py'))
}

function Find-Instalaciones {
    $encontradas = New-Object System.Collections.ArrayList

    if ($Profundo) {
        # Recorrido completo de los discos fijos. Lento pero no se le escapa nada.
        $discos = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
                  Select-Object -ExpandProperty DeviceID
        Write-Host "Modo profundo: recorriendo $($discos -join ', ') ..."
        Write-Host "(puede tardar varios minutos)"
        foreach ($d in $discos) {
            $marcas = @(Get-ChildItem -Path "$d\" -Filter 'launcher.py' -Recurse -File `
                        -ErrorAction SilentlyContinue)
            foreach ($m in $marcas) {
                # .../<raiz>/polyx/launcher.py  ->  <raiz>
                $padre = Split-Path (Split-Path $m.FullName -Parent) -Parent
                if ((Split-Path (Split-Path $m.FullName -Parent) -Leaf) -ne 'polyx') { continue }
                if (Test-EsPolyX $padre) { [void]$encontradas.Add($padre) }
            }
        }
    } else {
        $raices = @(
            $env:USERPROFILE,
            (Join-Path $env:USERPROFILE 'Desktop'),
            (Join-Path $env:USERPROFILE 'Documents'),
            (Join-Path $env:USERPROFILE 'Downloads'),
            (Join-Path $env:USERPROFILE 'OneDrive\Desktop'),
            (Join-Path $env:USERPROFILE 'OneDrive\Documentos'),
            'C:\', 'C:\Program Files', 'C:\Program Files (x86)'
        ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

        foreach ($raiz in $raices) {
            foreach ($d in @(Get-ChildItem -Path $raiz -Directory -Depth 4 -ErrorAction SilentlyContinue)) {
                if (Test-EsPolyX $d.FullName) { [void]$encontradas.Add($d.FullName) }
            }
        }
    }

    # Quitar duplicados y anidadas: retirar la de afuera se lleva la de adentro.
    $unicas = $encontradas | Select-Object -Unique
    $limpias = New-Object System.Collections.ArrayList
    foreach ($p in $unicas) {
        $dentro = $false
        foreach ($q in $unicas) {
            if ($p -ne $q -and $p.StartsWith($q + '\', [StringComparison]::OrdinalIgnoreCase)) {
                $dentro = $true; break
            }
        }
        if (-not $dentro) { [void]$limpias.Add($p) }
    }
    return $limpias
}

function Find-Rastros {
    <#
        Lo que queda fuera de la carpeta de instalacion: accesos directos y
        preferencias. Sin esto, una instalacion "limpia" hereda el idioma y
        deja iconos que apuntan a rutas que ya no existen.
    #>
    $rastros = New-Object System.Collections.ArrayList
    $candidatos = @(
        (Join-Path $env:USERPROFILE '.polyx_idioma.json'),
        (Join-Path $env:USERPROFILE 'Desktop\Poly-X.lnk'),
        (Join-Path $env:USERPROFILE 'OneDrive\Desktop\Poly-X.lnk'),
        (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Poly-X.lnk')
    )
    foreach ($c in $candidatos) {
        if (Test-Path $c) { [void]$rastros.Add($c) }
    }
    # Accesos directos con otro nombre pero que apuntan a Poly-X.
    foreach ($carpeta in @((Join-Path $env:USERPROFILE 'Desktop'),
                           (Join-Path $env:USERPROFILE 'OneDrive\Desktop'))) {
        if (-not (Test-Path $carpeta)) { continue }
        foreach ($lnk in @(Get-ChildItem -Path $carpeta -Filter '*.lnk' -File -ErrorAction SilentlyContinue)) {
            if ($rastros -contains $lnk.FullName) { continue }
            try {
                $sh = New-Object -ComObject WScript.Shell
                $destino = $sh.CreateShortcut($lnk.FullName).TargetPath
                if ($destino -match 'polyx|Poly-X') { [void]$rastros.Add($lnk.FullName) }
            } catch { }
        }
    }
    return $rastros
}

function Remove-Ruta($ruta) {
    if ($Definitivo) {
        Remove-Item -LiteralPath $ruta -Recurse -Force
        return 'borrado'
    }
    Add-Type -AssemblyName Microsoft.VisualBasic
    if (Test-Path -LiteralPath $ruta -PathType Container) {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
            $ruta,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
    } else {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
            $ruta,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
    }
    return 'a la papelera'
}

# ── Comprobar que no haya Poly-X corriendo ──────────────────
$vivos = @(Get-Process -Name 'python', 'pythonw' -ErrorAction SilentlyContinue |
           Where-Object { $_.Path -and $_.Path -match 'polyx|Poly-X' })
if ($vivos.Count -gt 0) {
    Write-Titulo 'Poly-X esta en ejecucion'
    Write-Host 'Cierra el programa antes de desinstalar. Procesos abiertos:' -ForegroundColor Yellow
    foreach ($v in $vivos) { Write-Host "  PID $($v.Id)  $($v.Path)" }
    Write-Host ''
    exit 1
}

# ── Busqueda ────────────────────────────────────────────────
Write-Titulo 'Poly-X - Desinstalador'
Write-Host 'Buscando instalaciones...'
$instalaciones = Find-Instalaciones
$rastros = Find-Rastros

$conRepo = @()
$aRetirar = @()
foreach ($i in $instalaciones) {
    if ((Test-Path (Join-Path $i '.git')) -and (-not $IncluirRepos)) {
        $conRepo += $i
    } else {
        $aRetirar += $i
    }
}

if ($aRetirar.Count -eq 0 -and $rastros.Count -eq 0) {
    Write-Host ''
    Write-Host '[OK] No se encontro ningun rastro de Poly-X.' -ForegroundColor Green
    if ($conRepo.Count -gt 0) {
        Write-Host ''
        Write-Host 'Se omitieron copias de trabajo de git (usa -IncluirRepos para incluirlas):' -ForegroundColor Yellow
        foreach ($r in $conRepo) { Write-Host "  $r" }
    }
    exit 0
}

Write-Titulo 'Encontrado'
$totalMB = 0
foreach ($i in $aRetirar) {
    $mb = Get-TamanoMB $i
    $totalMB += $mb
    Write-Host ''
    Write-Host "  CARPETA  $i"
    Write-Host "           $mb MB"
    $verFile = Join-Path $i '.polyx_version'
    if (Test-Path $verFile) {
        $sha = (Get-Content $verFile -Raw).Trim()
        if ($sha.Length -ge 7) { Write-Host "           version $($sha.Substring(0,7))" }
    }
    foreach ($c in $CarpetasDatos) {
        $ruta = Join-Path $i $c
        if (Test-Path $ruta) {
            $n = @(Get-ChildItem -LiteralPath $ruta -Recurse -File -ErrorAction SilentlyContinue).Count
            if ($n -gt 0) { Write-Host "           datos: $c ($n archivos)" -ForegroundColor Yellow }
        }
    }
}
foreach ($r in $rastros) { Write-Host ''; Write-Host "  RASTRO   $r" }

if ($conRepo.Count -gt 0) {
    Write-Host ''
    Write-Host '  OMITIDAS (copias de trabajo de git):' -ForegroundColor Yellow
    foreach ($r in $conRepo) { Write-Host "    $r" }
    Write-Host '    Usa -IncluirRepos si de verdad quieres retirarlas.'
}

Write-Host ''
Write-Host "  TOTAL a retirar: $([math]::Round($totalMB,1)) MB en $($aRetirar.Count) carpeta(s) y $($rastros.Count) rastro(s)."

if ($SoloBuscar) { Write-Host ''; Write-Host '(-SoloBuscar: no se toco nada)'; exit 0 }

# ── Confirmacion ────────────────────────────────────────────
Write-Titulo 'Que va a pasar'
if ($ConservarDatos) {
    Write-Host "  1. Se copian los datos (modelos, runs, datasets) a:"
    Write-Host "     $ConservarDatos"
    Write-Host '  2. Se retira todo lo listado arriba.'
} else {
    Write-Host '  Se retira todo lo listado arriba, INCLUIDOS los datos.'
    Write-Host '  Si quieres conservarlos, cancela y vuelve a correr con:'
    Write-Host '     .\desinstalar.ps1 -ConservarDatos C:\ruta\respaldo' -ForegroundColor Yellow
}
Write-Host ''
if ($Definitivo) {
    Write-Host '  BORRADO DEFINITIVO: no pasa por la papelera, no se puede deshacer.' -ForegroundColor Red
} else {
    Write-Host '  Todo va a la PAPELERA: reversible si algo hacia falta.' -ForegroundColor Green
}
Write-Host ''
$resp = Read-Host 'Escribe DESINSTALAR para continuar (cualquier otra cosa cancela)'
if ($resp -ne 'DESINSTALAR') {
    Write-Host ''
    Write-Host '[CANCELADO] No se toco nada.' -ForegroundColor Yellow
    exit 0
}

# ── Respaldo de datos ───────────────────────────────────────
if ($ConservarDatos) {
    Write-Titulo 'Respaldando datos'
    New-Item -ItemType Directory -Force -Path $ConservarDatos | Out-Null
    $n = 0
    foreach ($i in $aRetirar) {
        $nombre = Split-Path $i -Leaf
        foreach ($c in $CarpetasDatos) {
            $origen = Join-Path $i $c
            if (-not (Test-Path $origen)) { continue }
            $destino = Join-Path $ConservarDatos "$nombre\$c"
            Write-Host "  > $nombre\$c ..."
            $null = robocopy $origen $destino /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
            if ($LASTEXITCODE -ge 8) {
                Write-Host "  ! fallo copiando $c (codigo $LASTEXITCODE)" -ForegroundColor Red
                Write-Host '  NO se retira nada. Revisa y reintenta.' -ForegroundColor Red
                exit 1
            }
            $n++
        }
        foreach ($pt in @(Get-ChildItem -LiteralPath $i -Filter '*.pt' -File -ErrorAction SilentlyContinue)) {
            $destino = Join-Path $ConservarDatos $nombre
            New-Item -ItemType Directory -Force -Path $destino | Out-Null
            Copy-Item $pt.FullName $destino -Force
            $n++
        }
    }
    Write-Host ''
    Write-Host "  [OK] $n elemento(s) respaldados en $ConservarDatos" -ForegroundColor Green
}

# ── Retirar ─────────────────────────────────────────────────
Write-Titulo 'Retirando'
$fallos = 0
foreach ($ruta in ($aRetirar + $rastros)) {
    try {
        $como = Remove-Ruta $ruta
        Write-Host "  [OK] $como : $ruta" -ForegroundColor Green
    } catch {
        $fallos++
        Write-Host "  [ERROR] no se pudo retirar: $ruta" -ForegroundColor Red
        Write-Host "          $_"
    }
}

Write-Titulo 'Terminado'
if ($fallos -eq 0) {
    Write-Host '  El equipo quedo sin rastro de Poly-X.' -ForegroundColor Green
    Write-Host '  Ya puedes instalar desde cero con SETUP.bat.'
} else {
    Write-Host "  Quedaron $fallos elemento(s) sin retirar. Revisa los mensajes." -ForegroundColor Yellow
    Write-Host '  Causa mas comun: un archivo en uso. Cierra todo y reintenta.'
}
if (-not $Definitivo) {
    Write-Host ''
    Write-Host '  Todo esta en la papelera hasta que la vacies.'
}
Write-Host ''
