# ============================================================
#  Poly-X - Detectar y reemplazar una instalacion antigua
#
#  Busca otras copias de Poly-X en el equipo, rescata los datos
#  que tengan (modelos, entrenamientos, detecciones, datasets) y
#  recien entonces retira la copia vieja, a la PAPELERA, para que
#  siempre se pueda deshacer.
#
#  Orden deliberado: copiar -> verificar -> retirar. Nunca al reves.
# ============================================================
param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [switch]$SoloBuscar
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$InstallDir = (Resolve-Path $InstallDir).Path

# Carpetas cuyo contenido vale la pena rescatar de una instalacion vieja.
# El codigo no: ese lo trae la instalacion nueva.
$CarpetasDatos = @('models', 'runs', 'runs_train', 'data_microplastico', 'data')

function Write-Titulo($texto) {
    Write-Host ''
    Write-Host ('=' * 62)
    Write-Host "  $texto"
    Write-Host ('=' * 62)
}

function Test-EsPolyX($dir) {
    # Marca inequivoca: el paquete del programa. Con que exista launcher.py
    # dentro de polyx\ basta; los .bat sueltos pueden andar copiados por ahi.
    return (Test-Path (Join-Path $dir 'polyx\launcher.py'))
}

function Get-Inventario($dir) {
    <#
        Resume que datos tiene una instalacion. Se mide en archivos y MB
        porque es lo que el usuario necesita para decidir si le importa.
    #>
    $inv = [ordered]@{}
    $totalMB = 0.0

    # Pesos sueltos en la raiz (bestdetectormedium.pt y companía).
    $pt = @(Get-ChildItem -Path $dir -Filter '*.pt' -File -ErrorAction SilentlyContinue)
    if ($pt.Count -gt 0) {
        $mb = ($pt | Measure-Object Length -Sum).Sum / 1MB
        $inv['*.pt (raiz)'] = "$($pt.Count) archivos, $([math]::Round($mb,1)) MB"
        $totalMB += $mb
    }

    foreach ($c in $CarpetasDatos) {
        $ruta = Join-Path $dir $c
        if (-not (Test-Path $ruta)) { continue }
        $arch = @(Get-ChildItem -Path $ruta -Recurse -File -ErrorAction SilentlyContinue)
        if ($arch.Count -eq 0) { continue }
        $mb = ($arch | Measure-Object Length -Sum).Sum / 1MB
        $inv[$c] = "$($arch.Count) archivos, $([math]::Round($mb,1)) MB"
        $totalMB += $mb
    }

    return @{ Detalle = $inv; TotalMB = [math]::Round($totalMB, 1) }
}

function Find-Instalaciones($excluir) {
    <#
        Busca en las rutas donde realmente termina instalado esto, con
        profundidad acotada. Recorrer el disco entero tardaria minutos y
        no aporta: nadie instala Poly-X a diez niveles de profundidad.
    #>
    $raices = @(
        $env:USERPROFILE,
        (Join-Path $env:USERPROFILE 'Desktop'),
        (Join-Path $env:USERPROFILE 'Documents'),
        (Join-Path $env:USERPROFILE 'Downloads'),
        (Join-Path $env:USERPROFILE 'OneDrive\Desktop'),
        (Join-Path $env:USERPROFILE 'OneDrive\Documentos'),
        'C:\', 'C:\Program Files', 'C:\Program Files (x86)'
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    $encontradas = New-Object System.Collections.ArrayList

    foreach ($raiz in $raices) {
        # Depth 3 cubre C:\Users\PC\Desktop\Poly-X y equivalentes.
        $cands = @(Get-ChildItem -Path $raiz -Directory -Depth 3 -ErrorAction SilentlyContinue)
        foreach ($d in $cands) {
            $p = $d.FullName
            if ($p -eq $excluir) { continue }
            # Descartar subcarpetas de la instalacion nueva (p.ej. su .venv).
            if ($p.StartsWith($excluir + '\', [StringComparison]::OrdinalIgnoreCase)) { continue }
            if (-not (Test-EsPolyX $p)) { continue }
            # Una copia de trabajo de git NO es una instalacion vieja: es el
            # repositorio donde alguien desarrolla. Mandarla a la papelera seria
            # destruir trabajo sin versionar. Se ignora siempre.
            if (Test-Path (Join-Path $p '.git')) {
                Write-Host "  (omitida, es un repositorio git: $p)" -ForegroundColor DarkGray
                continue
            }
            if ($encontradas -contains $p) { continue }
            [void]$encontradas.Add($p)
        }
    }

    # Quitar anidadas: si una candidata vive dentro de otra, sobra. Retirar la
    # de afuera ya se lleva la de adentro, y listarlas por separado invita a
    # migrar dos veces lo mismo.
    $limpias = New-Object System.Collections.ArrayList
    foreach ($p in $encontradas) {
        $dentroDeOtra = $false
        foreach ($q in $encontradas) {
            if ($p -eq $q) { continue }
            if ($p.StartsWith($q + '\', [StringComparison]::OrdinalIgnoreCase)) {
                $dentroDeOtra = $true
                break
            }
        }
        if (-not $dentroDeOtra) { [void]$limpias.Add($p) }
    }

    return $limpias
}

# ── Busqueda ────────────────────────────────────────────────
Write-Titulo 'Poly-X - Buscando instalaciones anteriores'
Write-Host "Instalacion nueva : $InstallDir"
Write-Host 'Buscando otras copias en el equipo...'

$viejas = Find-Instalaciones -excluir $InstallDir

if ($viejas.Count -eq 0) {
    Write-Host ''
    Write-Host '[OK] No hay otra instalacion de Poly-X. Nada que migrar.' -ForegroundColor Green
    exit 0
}

Write-Host ''
Write-Host "Se encontraron $($viejas.Count) instalacion(es) anterior(es):" -ForegroundColor Yellow

$conDatos = @()
$i = 0
foreach ($v in $viejas) {
    $i++
    $inv = Get-Inventario $v
    Write-Host ''
    Write-Host "  [$i] $v"
    $verFile = Join-Path $v '.polyx_version'
    if (Test-Path $verFile) {
        $sha = (Get-Content $verFile -Raw).Trim()
        if ($sha.Length -ge 7) { Write-Host "      version: $($sha.Substring(0,7))" }
    }
    if ($inv.Detalle.Count -eq 0) {
        Write-Host '      (sin datos que rescatar: solo codigo)'
    } else {
        Write-Host "      datos: $($inv.TotalMB) MB en total"
        foreach ($k in $inv.Detalle.Keys) {
            Write-Host "        - $k : $($inv.Detalle[$k])"
        }
    }
    $conDatos += ,@{ Ruta = $v; Inv = $inv }
}

if ($SoloBuscar) { exit 0 }

# ── Confirmacion ────────────────────────────────────────────
Write-Titulo 'Que va a pasar'
Write-Host '  1. Se COPIAN a la instalacion nueva los datos que no tenga ya'
Write-Host '     (modelos, entrenamientos, detecciones, datasets).'
Write-Host '     No se sobrescribe ningun archivo existente.'
Write-Host '  2. Se verifica que la copia haya quedado bien.'
Write-Host '  3. Recien entonces la carpeta vieja se manda a la PAPELERA.'
Write-Host ''
Write-Host '  La papelera es reversible: si algo sale mal, se restaura.' -ForegroundColor Green
Write-Host ''
$resp = Read-Host 'Escribe SI para continuar (cualquier otra cosa cancela)'
if ($resp -ne 'SI') {
    Write-Host ''
    Write-Host '[CANCELADO] No se toco nada.' -ForegroundColor Yellow
    exit 0
}

# ── Migracion ───────────────────────────────────────────────
foreach ($item in $conDatos) {
    $vieja = $item.Ruta
    Write-Titulo "Migrando desde: $vieja"

    $copiados = 0
    $fallos = 0

    # Pesos sueltos de la raiz.
    $pt = @(Get-ChildItem -Path $vieja -Filter '*.pt' -File -ErrorAction SilentlyContinue)
    foreach ($f in $pt) {
        $destino = Join-Path $InstallDir $f.Name
        if (Test-Path $destino) {
            Write-Host "  = ya existe, se conserva el nuevo: $($f.Name)"
            continue
        }
        try {
            Copy-Item $f.FullName $destino -ErrorAction Stop
            Write-Host "  + $($f.Name)"
            $copiados++
        } catch {
            Write-Host "  ! no se pudo copiar $($f.Name): $_" -ForegroundColor Red
            $fallos++
        }
    }

    # Carpetas de datos. robocopy con /XC /XN /XO copia SOLO lo que no existe
    # en destino: ningun archivo de la instalacion nueva se pisa.
    foreach ($c in $CarpetasDatos) {
        $origen = Join-Path $vieja $c
        if (-not (Test-Path $origen)) { continue }
        $destino = Join-Path $InstallDir $c
        Write-Host "  > $c ..."
        $null = robocopy $origen $destino /E /XC /XN /XO /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
        # robocopy usa codigos de salida con bits; < 8 es exito.
        if ($LASTEXITCODE -ge 8) {
            Write-Host "  ! robocopy fallo en $c (codigo $LASTEXITCODE)" -ForegroundColor Red
            $fallos++
        } else {
            $copiados++
        }
    }

    # ── Verificacion antes de retirar nada ──────────────────
    if ($fallos -gt 0) {
        Write-Host ''
        Write-Host "[ATENCION] Hubo $fallos fallo(s) copiando datos." -ForegroundColor Red
        Write-Host '           NO se retira la carpeta vieja. Revisa y reintenta.'
        Write-Host "           Carpeta conservada: $vieja"
        continue
    }

    Write-Host ''
    Write-Host '  [OK] Datos migrados y verificados.' -ForegroundColor Green

    # ── Retirar a la papelera ───────────────────────────────
    try {
        Add-Type -AssemblyName Microsoft.VisualBasic
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
            $vieja,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
        Write-Host "  [OK] Instalacion antigua enviada a la papelera:" -ForegroundColor Green
        Write-Host "       $vieja"
    } catch {
        Write-Host "  [AVISO] No se pudo enviar a la papelera: $_" -ForegroundColor Yellow
        Write-Host '          Los datos YA se migraron; puedes borrarla a mano.'
        Write-Host "          $vieja"
    }
}

Write-Titulo 'Migracion terminada'
Write-Host "  Instalacion activa: $InstallDir"
Write-Host '  Si algo falta, revisa la papelera antes de vaciarla.'
Write-Host ''
