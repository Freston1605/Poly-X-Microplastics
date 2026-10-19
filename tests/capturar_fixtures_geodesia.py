"""Captura de fixtures reales para el harness de geodesia (S02-T03).

Serializa mascaras uint8 desde una corrida real del detector, para que
``tests/test_geodesia.py`` las recubra automaticamente en su comparacion
bit a bit de ``_propagar_geodesico`` pre/post reescritura.

La mascara NO se re-inventa: se reproduce con ``morfologia.segmentar()``
sobre la foto origen, exactamente como lo hace ``morfologia.medir_deteccion``
en el pipeline (mismo margen por defecto, misma caja), de modo que cada
fixture es literalmente lo que el pipeline midio.

Layout que lee (el que escribe ``polyx/gui/detector/runner.py``)::

    <runs>/<alias>/labels/NNNN_<imgstem>.txt   # YOLO: class cx cy w h conf
                                              # normalizado a la foto completa
    <runs>/<alias>/classes.txt

Cada caja se empareja con la foto original buscando bajo ``--fotos`` (busqueda
recursiva) un fichero cuyo stem contenga el stem del label: primero match
exacto y despues el mas corto que lo contenga (dos fotos de carpetas distintas
pueden llamarse ``1.1.jpg`` y el que mas cerca este del stem exacto es el de la
corrida). La foto se lee con ``np.fromfile`` + ``cv2.imdecode``: con
``cv2.imread`` las rutas con acentos fallan en silencio en Windows, y ahi es
donde viven estos datos.

Uso (en el equipo que tenga la corrida y sus fotos)::

    python tests/capturar_fixtures_geodesia.py \\
        --runs runs/detect_20260812_1030 \\
        --fotos F:/Poly-X/fotos_origen \\
        [--destino tests/fixtures/geodesia] \\
        [--reforzar]

Sin ``--reforzar`` nunca se sobreescribe un fixture existente cuyo contenido
difique del que se reproduciria ahora: la suite mide contra lo capturado, y
cambiarlo por debajo sin avisar invalidaria el harness. Vale si es NO-OP: si
la segunda pasada produce las mismas mascaras, no escribe nada.

Docstrings en ingles, comentarios en espanol (convencion del repo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyx.model import morfologia  # noqa: E402
from polyx.model.yolo_wrap import read_yolo_txt  # noqa: E402

# Sufijos de foto que cv2 puede abrir. El resto de ficheros bajo --fotos se
# ignoran sin riesgo: ahi pueden vivir excels, txts de conteo manual, informes.
SUFIJOS_FOTO = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Razones en las que una caja se queda sin fixture. Son claves del resumen y
# se asientan tal cual en el stdout del script, asi que se fijan como estan.
SK_FOTO_AUSENTE = "foto no encontrada"
SK_FOTO_ILEGIBLE = "foto ilegible (cv2.imdecode devolvio None)"
SK_TXT_VACIO = "txt sin cajas legibles"
SK_TXT_SIN_STEM = "label con stem vacio (empieza por '_')"
SK_SIN_MASCARA = "segmentar no devolvio mascara"
SK_MASCARA_VACIA = "mascara vacia (cero encendidos)"
SK_TILE_VACIO = "mascara sin encendidos tras recortar"
SK_ERROR = "error inesperado en la caja"


def _indizar_fotos(fotos: Path) -> list[tuple[str, Path]]:
    """Inventario recursivo de fotos bajo ``--fotos``, ordenado y estable.

    Se une lower() para que FOTO_0001 y foto_0001 casen igual: el stem del
    label sale del nombre original de la foto, pero nadie garantiza el caso
    dos corridas despues.
    """
    if not fotos.exists():
        return []
    return sorted(
        (f.stem.lower(), f)
        for f in fotos.rglob("*")
        if f.is_file() and f.suffix.lower() in SUFIJOS_FOTO
    )


def _buscar_foto(stem: str, inventario: list[tuple[str, Path]]) -> Path | None:
    """La foto del stem pedido. Exacto primero; si no, el mas cercano.

    El runner llama al label con el stem COMPLETO de la foto, asi que el match
    exacto es el caso normal; el substrain solo salva corridas donde la foto
    se renombro anadiendo prefijos (recortes de troceo, copias de placa).
    """
    objetivo = stem.lower()
    exactos = [(s, p) for s, p in inventario if s == objetivo]
    if exactos:
        return exactos[0][1]
    contentivos = [(s, p) for s, p in inventario if objetivo in s]
    if contentivos:
        # El stem mas corto: menos renombre metido, mas cerca de la original.
        return min(contentivos, key=lambda sp: (len(sp[0]), sp[1].name))[1]
    return None


def _cargar_bgr(ruta: Path) -> np.ndarray | None:
    """Foto como BGR via fromfile+imdecode (nunca imread: rutas acentuadas)."""
    try:
        buf = np.fromfile(ruta, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _limites_mascara(m: np.ndarray) -> tuple[int, int, int, int] | None:
    """Rectangulo de lo encendido. None si la mascara esta apagada."""
    m = m > 0
    if not m.any():
        return None
    filas = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    return (int(filas[0]), int(filas[-1]) + 1,
            int(cols[0]), int(cols[-1]) + 1)


def _stem_de_label(nombre: str) -> str:
    """El stem de la foto a partir del nombre del label ``NNNN_<stem>.txt``.

    El prefijo ``NNNN_`` (indice de imagen del run) se quita si esta; un
    label que empiece con ``_`` indicaria un stem vacio, y eso es un error
    del run que no tiene arreglo aqui.
    """
    return nombre[5:] if len(nombre) > 5 and nombre[:4].isdigit() and nombre[4] == "_" else nombre


def capturar(runs: Path, fotos: Path, destino: Path,
             reforzar: bool = False) -> dict:
    """Recorre ``runs`` y serializa una mascara .npy por caja anotada.

    Nunca lanza por caja particular: cada anomalía se apunta en ``resumen``
    y se sigue con la siguiente, de modo que una foto corrupta de media
    corrida no deje el lote a medias. Las carpetas sin ``labels/`` se
    ignoran (un run puede tener subecarpetas de servicios).
    """
    resumen: dict = {
        "fotos_encontradas": 0,
        "labels": 0,
        "cajas": 0,
        "serializadas": 0,
        "reutilizadas": 0,
        "desviadas_sin_reforzar": 0,
        "saltadas": {},
    }
    if not runs.exists():
        return resumen
    inventario = _indizar_fotos(fotos)
    destino.mkdir(parents=True, exist_ok=True)

    for alias in sorted(runs.iterdir()):
        if not alias.is_dir():
            continue
        carpeta_labels = alias / "labels"
        if not carpeta_labels.is_dir():
            continue
        for ruta_txt in sorted(carpeta_labels.glob("*.txt")):
            resumen["labels"] += 1
            # El runner escribe NNNN_<stem>.txt; el stem de la foto es lo que
            # sigue al primer guion bajo de 4 cifras.
            stem = ruta_txt.stem
            imgstem = _stem_de_label(stem)
            if not imgstem:
                resumen["saltadas"][SK_TXT_SIN_STEM] = (
                    resumen["saltadas"].get(SK_TXT_SIN_STEM, 0) + 1)
                continue
            ruta_foto = _buscar_foto(imgstem, inventario)
            if ruta_foto is None:
                resumen["saltadas"][SK_FOTO_AUSENTE] = (
                    resumen["saltadas"].get(SK_FOTO_AUSENTE, 0) + 1)
                continue
            img_bgr = _cargar_bgr(ruta_foto)
            if img_bgr is None or img_bgr.size == 0:
                resumen["saltadas"][SK_FOTO_ILEGIBLE] = (
                    resumen["saltadas"].get(SK_FOTO_ILEGIBLE, 0) + 1)
                continue
            resumen["fotos_encontradas"] += 1
            alto, ancho = img_bgr.shape[:2]
            try:
                clases = _cargar_classes(alias / "classes.txt")
                dets = read_yolo_txt(ruta_txt, ancho, alto, clases)
            except Exception as exc:  # noqa: BLE001 - todo lo parseable va por SK_*
                resumen["saltadas"][f"{SK_ERROR} (read_yolo_txt: {type(exc).__name__})"] = (
                    resumen["saltadas"].get(SK_ERROR, 0) + 1)
                continue
            if not dets:
                resumen["saltadas"][SK_TXT_VACIO] = (
                    resumen["saltadas"].get(SK_TXT_VACIO, 0) + 1)
                continue

            for idx, det in enumerate(dets):
                resumen["cajas"] += 1
                try:
                    mascara = morfologia.segment(
                        img_bgr, det.x1, det.y1, det.x2, det.y2)
                except Exception as exc:  # noqa: BLE001
                    motivo = f"{SK_ERROR} ({type(exc).__name__})"
                    resumen["saltadas"][motivo] = (
                        resumen["saltadas"].get(motivo, 0) + 1)
                    continue
                if mascara is None:
                    resumen["saltadas"][SK_SIN_MASCARA] = (
                        resumen["saltadas"].get(SK_SIN_MASCARA, 0) + 1)
                    continue
                limite = _limites_mascara(mascara)
                if limite is None:
                    resumen["saltadas"][SK_MASCARA_VACIA] = (
                        resumen["saltadas"].get(SK_MASCARA_VACIA, 0) + 1)
                    continue
                f0, f1, c0, c1 = limite
                recorte = np.ascontiguousarray(
                    (mascara > 0)[f0:f1, c0:c1].astype(np.uint8) * 255)
                if recorte.size == 0:
                    resumen["saltadas"][SK_TILE_VACIO] = (
                        resumen["saltadas"].get(SK_TILE_VACIO, 0) + 1)
                    continue

                nombre = f"{alias.name}__{imgstem}__{idx:03d}.npy"
                fichero = destino / nombre
                if fichero.exists() and not reforzar:
                    vieja = np.load(fichero, allow_pickle=False)
                    if (vieja.shape == recorte.shape
                            and vieja.dtype == recorte.dtype
                            and np.array_equal(vieja, recorte)):
                        resumen["reutilizadas"] += 1
                        continue
                    # Difiere y no hay permiso: se guarda la capturada por el
                    # run, no se pisa la del primer dia del harness.
                    resumen["desviadas_sin_reforzar"] += 1
                    continue
                np.save(fichero, recorte, allow_pickle=False)
                resumen["serializadas"] += 1
    return resumen


def _cargar_classes(ruta: Path) -> dict[int, str]:
    """classes.txt: una linea de nombre por id, en orden."""
    nombres: dict[int, str] = {}
    if ruta.exists():
        try:
            for i, linea in enumerate(
                    ruta.read_text(encoding="utf-8").splitlines()):
                if linea.strip():
                    nombres[i] = linea.strip()
        except UnicodeDecodeError:
            pass
    return nombres


def imprimir_resumen(resumen: dict) -> None:
    """Resumen legible al final: lo que va a correr -- y lo que no y por que."""
    print("== captura de fixtures geodesia ==")
    print(f"  labels procesados     : {resumen['labels']}")
    print(f"  fotos emparejadas     : {resumen['fotos_encontradas']}")
    print(f"  cajas anotadas        : {resumen['cajas']}")
    print(f"  mascaras serializadas : {resumen['serializadas']}")
    print(f"  ya existentes validas : {resumen['reutilizadas']}")
    print(f"  desviadas sin reforzar: {resumen['desviadas_sin_reforzar']}")
    if resumen["saltadas"]:
        print("  saltadas, por motivo:")
        for motivo, n in sorted(resumen["saltadas"].items()):
            print(f"    {n:4d}  {motivo}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serializa mascaras de particulas reales "
                    "(runs/detect_*/) para tests/test_geodesia.py.",
    )
    parser.add_argument("--runs", type=Path, required=True,
                        help="carpeta del run (p.ej. runs/detect_20260812_1030)")
    parser.add_argument("--fotos", type=Path, required=True,
                        help="carpeta con las fotos ORIGINALES del lote "
                             "(se busca recursivamente por stem)")
    parser.add_argument("--destino", type=Path, default=Path(__file__).resolve().parent
                        / "fixtures" / "geodesia",
                        help="donde escribir los .npy (por defecto "
                             "tests/fixtures/geodesia)")
    parser.add_argument("--reforzar", action="store_true",
                        help="sobreescribe fixtures existentes cuyo contenido "
                             "difera del recien reproducido")
    args = parser.parse_args(argv)
    resumen = capturar(args.runs, args.fotos, args.destino, args.reforzar)
    imprimir_resumen(resumen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
