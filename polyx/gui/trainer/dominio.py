"""Elige el peso segun el subconjunto real de la validacion.

El dataset combina dos subconjuntos, marcados por el prefijo del nombre de
archivo que pone ``armar_dataset.py``: ``lab__`` y ``real__``. El subconjunto
``lab__`` aporta muchas mas cajas por imagen, asi que domina el mAP global y con
el se decide ``best.pt``.

Aqui se rehace esa eleccion mirando solo ``real__``, y el ganador se guarda
aparte. El entrenamiento deja tres pesos:

    best_sintetico.pt   el que elige Ultralytics por el mAP global
    best_real.pt        el mejor sobre el subconjunto real
    last.pt             ultima epoca, para reanudar

Si un dataset no trae los dos prefijos, el paso se salta sin ruido.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

PREFIJO_REAL = "real__"
PREFIJO_LAB = "lab__"

# Los tres pesos que quedan al terminar. best.pt de Ultralytics se conserva.
NOMBRE_MEJOR_REAL = "best_real.pt"
NOMBRE_MEJOR_LAB = "best_sintetico.pt"


def _imagenes_de_val(datos_yaml: Path) -> List[Path]:
    """Lista las imagenes del split de validacion declarado en el yaml."""
    cfg = yaml.safe_load(datos_yaml.read_text(encoding="utf-8")) or {}
    raiz = Path(cfg.get("path") or datos_yaml.parent)
    if not raiz.is_absolute():
        raiz = (datos_yaml.parent / raiz).resolve()

    val = cfg.get("val")
    if not val:
        return []
    ruta = Path(val)
    if not ruta.is_absolute():
        ruta = (raiz / ruta).resolve()

    if ruta.is_file():  # lista .txt de rutas
        return [Path(l.strip()) for l in ruta.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    if ruta.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted(p for p in ruta.iterdir()
                      if p.is_file() and p.suffix.lower() in exts)
    return []


def dominios_presentes(datos_yaml: Path) -> Dict[str, int]:
    """Cuenta cuantas imagenes de validacion hay de cada dominio."""
    conteo = {"real": 0, "lab": 0, "sin_marcar": 0}
    for p in _imagenes_de_val(Path(datos_yaml)):
        if p.name.startswith(PREFIJO_REAL):
            conteo["real"] += 1
        elif p.name.startswith(PREFIJO_LAB):
            conteo["lab"] += 1
        else:
            conteo["sin_marcar"] += 1
    return conteo


def se_puede_reelegir(datos_yaml: Path) -> bool:
    """True si la validacion mezcla dominios y por tanto reelegir tiene sentido."""
    c = dominios_presentes(Path(datos_yaml))
    return c["real"] > 0 and c["lab"] > 0


def _etiqueta_de(imagen: Path) -> Path:
    """Ruta del .txt de una imagen, con la convencion images/ -> labels/."""
    partes = list(imagen.parts)
    for i in range(len(partes) - 1, -1, -1):
        if partes[i] == "images":
            partes[i] = "labels"
            break
    return Path(*partes).with_suffix(".txt")


def _enlazar(origen: Path, destino: Path) -> None:
    """Enlace duro si se puede; copia si no (otro volumen, sin permisos)."""
    import os
    import shutil

    if destino.exists():
        return
    try:
        os.link(origen, destino)
    except (OSError, NotImplementedError):
        shutil.copy2(origen, destino)


def crear_yaml_solo_real(datos_yaml: Path, destino_dir: Path) -> Optional[Path]:
    """Escribe un dataset.yaml cuya validacion son solo las imagenes reales.

    El subconjunto vive en su **propio arbol** images/ + labels/, no en una lista
    que apunte al split original. Ultralytics deriva la ruta del cache de
    etiquetas de la carpeta que las contiene, asi que validar un subconjunto de
    ``images/val`` reescribe ``labels/val.cache`` con solo esas imagenes: el
    siguiente entrenamiento validaria en silencio sobre el subconjunto y su
    best.pt significaria otra cosa. Con arbol propio, el cache que se toca es
    ``_seleccion/labels/val.cache`` y el del dataset queda intacto.

    Se enlaza en duro en vez de copiar: mismo volumen, sin duplicar bytes.
    """
    datos_yaml = Path(datos_yaml)
    reales = [p for p in _imagenes_de_val(datos_yaml)
              if p.name.startswith(PREFIJO_REAL)]
    if not reales:
        return None

    img_dir = destino_dir / "images" / "val"
    lbl_dir = destino_dir / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img in reales:
        if not img.exists():
            continue
        _enlazar(img, img_dir / img.name)
        etq = _etiqueta_de(img)
        if etq.exists():
            _enlazar(etq, lbl_dir / etq.name)

    cfg = yaml.safe_load(datos_yaml.read_text(encoding="utf-8")) or {}
    nuevo = {
        "path": str(destino_dir.resolve()),
        # El train no se usa al validar, pero Ultralytics espera la clave.
        # Apunta al mismo subconjunto para no arrastrar el split grande.
        "train": "images/val",
        "val": "images/val",
        "names": cfg.get("names", {}),
    }
    salida = destino_dir / "dataset_solo_real.yaml"
    salida.write_text(yaml.safe_dump(nuevo, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    return salida


def candidatos(run_dir: Path) -> List[Path]:
    """Checkpoints a evaluar: best, last y los periodicos si los hay."""
    pesos = Path(run_dir) / "weights"
    if not pesos.is_dir():
        return []
    # Se excluyen los pesos que produce esta misma funcion: al reentrenar sobre
    # un run existente estarian ahi y se evaluarian como si fueran candidatos.
    propios = {NOMBRE_MEJOR_REAL, NOMBRE_MEJOR_LAB}
    vistos, salida = set(), []
    for p in [pesos / "best.pt", pesos / "last.pt"] + sorted(pesos.glob("epoch*.pt")):
        if p.exists() and p.name not in vistos and p.name not in propios:
            vistos.add(p.name)
            salida.append(p)
    return salida


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def elegir_mejor_en_real(run_dir: Path, datos_yaml: Path, imgsz: int, batch: int,
                         device: str,
                         log: Optional[Callable[[str], None]] = None,
                         ) -> Tuple[Optional[Path], List[Dict]]:
    """Evalua los checkpoints sobre sedimento real y copia el mejor.

    Devuelve (ruta de best_real.pt, tabla de resultados). Si no se puede evaluar
    nada, devuelve (None, []) y el entrenamiento sigue con su best.pt de siempre.
    """
    def _log(m: str) -> None:
        if log:
            log(m)

    run_dir = Path(run_dir)
    pesos = candidatos(run_dir)
    if not pesos:
        _log("[INFO] No hay checkpoints que evaluar; se conserva best.pt.")
        return None, []

    yaml_real = crear_yaml_solo_real(Path(datos_yaml), run_dir / "_seleccion")
    if yaml_real is None:
        _log("[INFO] La validacion no tiene subconjunto real; se conserva best.pt.")
        return None, []

    from ultralytics import YOLO

    _log("")
    _log("=" * 60)
    _log("  SELECCION DE PESOS")
    _log("=" * 60)
    _log(f"  Evaluando {len(pesos)} checkpoint(s) sobre el subconjunto real.")

    tabla: List[Dict] = []
    for p in pesos:
        try:
            res = YOLO(str(p)).val(data=str(yaml_real), imgsz=imgsz, batch=batch,
                                   device=device, plots=False, verbose=False)
            pr, rc = float(res.box.mp), float(res.box.mr)
            fila = {"checkpoint": p.name, "ruta": p,
                    "precision": pr, "recall": rc, "f1": _f1(pr, rc),
                    "map50": float(res.box.map50)}
        except Exception as exc:
            _log(f"  [WARN] no se pudo evaluar {p.name}: {exc}")
            continue
        tabla.append(fila)
        _log(f"    {fila['checkpoint']:<14} F1={fila['f1']:.4f}  "
             f"P={pr:.4f}  R={rc:.4f}  mAP50={fila['map50']:.4f}")

    if not tabla:
        _log("  No se pudo evaluar ningun checkpoint; se conserva best.pt.")
        return None, []

    # F1 como criterio, mAP50 como desempate: importa acertar cuantas
    # particulas hay, no afinar la caja.
    tabla.sort(key=lambda d: (d["f1"], d["map50"]), reverse=True)
    mejor = tabla[0]

    destino = run_dir / "weights" / NOMBRE_MEJOR_REAL
    try:
        shutil.copy2(mejor["ruta"], destino)
    except OSError as exc:
        _log(f"  [WARN] no se pudo copiar el ganador: {exc}")
        return None, tabla

    _log("")
    _log(f"  Mejor en el subconjunto real: {mejor['checkpoint']} (F1={mejor['f1']:.4f})")
    origen_best = next((d for d in tabla if d["checkpoint"] == "best.pt"), None)
    if origen_best and origen_best["checkpoint"] != mejor["checkpoint"]:
        _log(f"  best.pt daba F1={origen_best['f1']:.4f}.")
    elif origen_best:
        _log("  Coincide con best.pt.")

    # Copia con nombre propio del peso que elige Ultralytics, para que los tres
    # queden nombrados igual de claro en la carpeta del run.
    origen = run_dir / "weights" / "best.pt"
    if origen.exists():
        try:
            shutil.copy2(origen, run_dir / "weights" / NOMBRE_MEJOR_LAB)
        except OSError as exc:
            _log(f"  [WARN] no se pudo crear {NOMBRE_MEJOR_LAB}: {exc}")

    _log("")
    _log("  Pesos disponibles en weights/:")
    _log(f"    {NOMBRE_MEJOR_LAB:<20} mejor por mAP global")
    _log(f"    {NOMBRE_MEJOR_REAL:<20} mejor sobre el subconjunto real")
    _log(f"    {'last.pt':<20} ultima epoca, para reanudar")
    _log("=" * 60)
    return destino, tabla
