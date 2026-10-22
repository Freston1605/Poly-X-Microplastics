"""Pruebas de la logica de captura de ``tests/capturar_fixtures_geodesia.py``.

El harness de ``test_geodesia.py`` recubre automaticamente cada ``.npy`` que
este en ``tests/fixtures/geodesia/``; su valor depende de que la captura sea
fiel (la mascara guardada TIENE que ser la misma que midio el pipeline) y de
que el script nunca pise por debajo lo ya capturado.

Aqui queda probado, contra un layout ``runs/detect_*`` en miniatura montado en
``tmp_path`` (la forma exacta que escribe ``polyx/gui/detector/runner.py``):

  * encuentra la foto origen por stem (busqueda recursiva, sin importar el
    caso del nombre),
  * la mascara serializada es bit-identica a la que ``morfologia.segmentar``
    devuelve sobre la misma foto y caja,
  * segunda corrida sin ``--reforzar``: no-op (idempotente),
  * fixture preexistente distinto: desviado sin ``--reforzar``, pisado con
    ella,
  * fotos sin match se cuentan como saltadas, no tumban el lote,
  * ``segmentar`` devolviendo None se cuenta como saltada, sin crash.

Docstrings en ingles, comentarios en espanol (convencion del repo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from polyx.ml import morfologia as M  # noqa: E402

import capturar_fixtures_geodesia as CFG  # noqa: E402
import generar_fixtures_geodesia as GFG  # noqa: E402


# ── helpers: foto sintetica y layout de run en miniatura ─────────────────
def _foto_particula(centro: tuple[int, int], radio: int = 6,
                    lado: int = 64, valor: int = 240) -> np.ndarray:
    """Foto BGR en miniatura con UNA particula brillante nitida.

    Fondo oscuro, disco claro: cruza de sobra el CONTRASTE_MINIMO=12 de
    ``segmentar`` para que su umbral anillo/percentil90 encienda el disco y
    no encienda fondo. Con eso el test puede comparar lo que el script
    captura contra lo que ``segmentar`` devuelve sobre la misma foto, sin
    depender de una foto real.
    """
    img = np.full((lado, lado, 3), 10, dtype=np.uint8)
    cv2.circle(img, centro, radio, (valor, valor, valor), -1)
    return img


def _guardar_foto(tmp: Path, foto: np.ndarray, stem: str) -> Path:
    """Escribe la foto de mediante cv2.imencode + write_bytes, como el pipeline."""
    carpeta = tmp / "fotos"
    carpeta.mkdir(parents=True, exist_ok=True)
    is_ok, buf = cv2.imencode(".png", foto)
    assert is_ok
    ruta = carpeta / f"{stem}.png"
    ruta.write_bytes(bytes(buf))
    return ruta


def _escribir_labels(carpeta_labels: Path, stem: str, idx_img: int,
                     W: int, H: int, cajas: list[tuple[int, float, float,
                                                     float, float, float]],
                     ) -> Path:
    """Escribe el .txt en el layout EXACTO de runner._guardar_labels.

    Se reimplementan a mano las 6 columnas (class cx cy w h conf, en
    coordenadas normalizadas) en lugar de importar detector/runner.py: ese
    modulo arrastra PySide6/Qt y este test solo puede depender de
    numpy/cv2/pytest. El formato es el que ``read_yolo_txt`` espera y el que
    ``_guardar_labels`` escribe (Dominio compartido).
    """
    carpeta_labels.mkdir(parents=True, exist_ok=True)
    lineas = []
    for cls, cx, cy, w, h, conf in cajas:
        lineas.append(f"{cls} {cx / W:.6f} {cy / H:.6f} "
                      f"{w / W:.6f} {h / H:.6f} {conf:.4f}")
    ruta = carpeta_labels / f"{idx_img:04d}_{stem}.txt"
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    return ruta


def _montar_run(tmp: Path, stem: str, idx_img: int, alias: str,
                cajas: list[tuple[float, float, float, float]],
                conf: float = 0.87) -> Path:
    """Layout runs/detect_* en miniatura: run/<alias>/{labels,classes.txt}."""
    run = tmp / "runs" / "detect_20260812_1030"
    alias_dir = run / alias
    alias_dir.mkdir(parents=True, exist_ok=True)
    # caja como (x1, y1, x2, y2) -> cx, cy, w, h en px absolutos: misma
    # conversion que hace detector.runner._guardar_labels antes de normalizar.
    dets = [(1, x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, x2 - x1, y2 - y1, conf)
            for (x1, y1, x2, y2) in cajas]
    # dimensiones de la foto de referencia del runner (el test usa siempre
    # una de lado 64: ver _foto_particula por defecto).
    _escribir_labels(alias_dir / "labels", stem, idx_img, 64, 64, dets)
    (alias_dir / "classes.txt").write_text("no_clasificado\nparticula\n",
                                           encoding="utf-8")
    return alias_dir


def _leer_npy(ruta: Path) -> np.ndarray:
    return np.load(ruta, allow_pickle=False)


def _recorte_rect(mascara: np.ndarray) -> np.ndarray:
    """La misma operacion que hace el capturador al serializar (uint8 0/255)."""
    encendidos = mascara > 0
    filas = np.flatnonzero(encendidos.any(axis=1))
    cols = np.flatnonzero(encendidos.any(axis=0))
    return (encendidos[filas[0]:filas[-1] + 1, cols[0]:cols[-1] + 1]
            ).astype(np.uint8) * 255


# ── capturas ─────────────────────────────────────────────────────────────
def test_captura_encuentra_foto_y_serializa_segmentar(tmp_path):
    """La mascara capturada es bit-identica a segmentar sobre foto+caja."""
    foto = _foto_particula(centro=(30, 22))
    stem = "placa_101_1"
    _guardar_foto(tmp_path, foto, stem)

    _montar_run(tmp_path, stem, idx_img=7, alias="y11n",
                cajas=[(20, 16, 40, 28)])
    destino = tmp_path / "fixtures" / "geodesia"

    resumen = CFG.capturar(runs=tmp_path / "runs" / "detect_20260812_1030",
                           fotos=tmp_path / "fotos", destino=destino)
    assert resumen["cajas"] == 1
    # La mascara se serializo, con el nombre <alias>__<stem>__<idx>.npy
    ficheros = sorted(destino.glob("*.npy"))
    assert len(ficheros) == 1
    assert ficheros[0].name == f"y11n__{stem}__000.npy"

    capturada = _leer_npy(ficheros[0])
    # La referencia del pipeline sobre la MISMA foto y la MISMA caja: es lo
    # que el script tenia que haber reproducido.
    mascara_referencia = M.segment(foto, 20, 16, 40, 28)
    assert mascara_referencia is not None and mascara_referencia.any()
    recorte_ref = _recorte_rect(mascara_referencia)
    # BIT-identico: forma, dtype y 0/255 identicos.
    assert capturada.shape == recorte_ref.shape, (
        "el fixture debe ser el recorte al rectangulo de la mascara")
    assert capturada.dtype == np.uint8
    assert np.array_equal(capturada, recorte_ref), (
        "la mascara capturada debe ser la misma que segmentar devuelve")


def test_captura_idempotente_sin_reforzar(tmp_path):
    """Segunda corrida sin --reforzar: no-op si el fixture ya vale."""
    foto = _foto_particula(centro=(30, 22))
    stem = "f2_batch"
    _guardar_foto(tmp_path, foto, stem)
    _montar_run(tmp_path, stem, idx_img=0, alias="y11n",
                cajas=[(20, 16, 40, 28)])
    destino = tmp_path / "fixtures" / "geodesia"
    runs = tmp_path / "runs" / "detect_20260812_1030"

    primera = CFG.capturar(runs=runs, fotos=tmp_path / "fotos",
                           destino=destino)
    assert primera["serializadas"] == 1

    fichero = sorted(destino.glob("*.npy"))[0]
    mtime1 = fichero.stat().st_mtime_ns
    segunda = CFG.capturar(runs=runs, fotos=tmp_path / "fotos",
                           destino=destino)
    # sin --reforzar: la misma mascara, ya capturada, NO se reescribe
    assert segunda["serializadas"] == 0
    assert segunda["reutilizadas"] == 1
    assert fichero.stat().st_mtime_ns == mtime1


def test_captura_fixture_distinto_se_protege_y_reforzar_pisa(tmp_path):
    """Fixture preexistente distinto: desviado sin --reforzar, pisado con ella."""
    foto = _foto_particula(centro=(30, 22))
    stem = "f3_batch"
    _guardar_foto(tmp_path, foto, stem)
    _montar_run(tmp_path, stem, idx_img=0, alias="y11n",
                cajas=[(20, 16, 40, 28)])
    runs = tmp_path / "runs" / "detect_20260812_1030"
    destino = tmp_path / "fixtures" / "geodesia"

    CFG.capturar(runs=runs, fotos=tmp_path / "fotos", destino=destino)
    fichero = sorted(destino.glob("*.npy"))[0]
    original = _leer_npy(fichero)
    mtime1 = fichero.stat().st_mtime_ns

    # Un fixture manualmente poeta (otra forma): representa una captura del
    # pasado que pudiera no cuadrar con lo que hoy reproduciria segmentar.
    otro = np.zeros_like(original)
    otro[0, 0] = 255  # cambia su contenido, mismas dimensiones
    np.save(fichero, otro, allow_pickle=False)

    sin_ = CFG.capturar(runs=runs, fotos=tmp_path / "fotos", destino=destino)
    assert sin_["serializadas"] == 0
    assert sin_["desviadas_sin_reforzar"] == 1
    # El fixture queda intacto (por debajo, sin avisar, no se pisa).
    assert _leer_npy(fichero)[0, 0] == 255

    con_ = CFG.capturar(runs=runs, fotos=tmp_path / "fotos", destino=destino,
                        reforzar=True)
    assert con_["serializadas"] == 1
    # Ahora SI se pisa y vuelve a cuadrar con lo que hoy produce segmentar.
    assert np.array_equal(_leer_npy(fichero), original)
    assert fichero.stat().st_mtime_ns > mtime1


def test_captura_foto_ausente_reporta_salto(tmp_path):
    """Foto sin match bajo --fotos: se salta esa caja, no se tumba el lote."""
    foto = _foto_particula(centro=(30, 22))
    _guardar_foto(tmp_path, foto, "placa_B")
    # El label apunta a un stem que NO existe entre las fotos.
    _montar_run(tmp_path, "foto_que_no_existe", idx_img=3, alias="y11n",
                cajas=[(20, 16, 40, 28)])
    destino = tmp_path / "fixtures" / "geodesia"

    resumen = CFG.capturar(runs=tmp_path / "runs" / "detect_20260812_1030",
                           fotos=tmp_path / "fotos", destino=destino)
    assert resumen["labels"] == 1
    assert resumen["fotos_encontradas"] == 0
    assert resumen["cajas"] == 0
    assert destino.glob("*.npy") is not None and not list(destino.glob("*.npy"))
    assert resumen["saltadas"].get(CFG.SK_FOTO_AUSENTE) == 1


def test_captura_sin_mascara_y_metodos(tmp_path):
    """``segmentar`` devolviendo None se cuenta como saltada, sin crash.

    Se revisa que una caja SIN particula (solo fondo, sin contraste) counts
    sans plomear ni escribir fixture: la particula detectada no debe de
    poder inventar mascaras sobre fondo puro.
    """
    foto = _foto_particula(centro=(30, 22))
    stem = "f4_batch"
    _guardar_foto(tmp_path, foto, stem)
    # Caja completamente fuera de la particula: solo fondo oscuro.
    caixa_fondo = (2, 2, 10, 10)
    _montar_run(tmp_path, stem, idx_img=1, alias="y11n", cajas=[caixa_fondo])
    destino = tmp_path / "fixtures" / "geodesia"

    resumen = CFG.capturar(runs=tmp_path / "runs" / "detect_20260812_1030",
                           fotos=tmp_path / "fotos", destino=destino)
    assert not list(destino.glob("*.npy"))
    # El salto se attribution a uno de los dos motivos esperados cuando
    # segmentar no saca una particula de fondo: None devuelto (SK_SIN_MASCARA)
    # o mascara vacia (SK_MASCARA_VACIA). La decision exacta vive dentro de
    # segmentar; el contrato del capturador es contarla e ignorar el ruido.
    total = sum(resumen["saltadas"].values())
    assert total >= 1
    assert resumen["serializadas"] == 0


def test_sufijo_recursivo_y_stem_de_label(tmp_path):
    """El prefijo NNNN_ del label se quita para buscar, la foto se busca
    recursivamente y sin hacer frecuencia con mayusculas."""
    assert CFG._stem_de_label("0012_placa_1_1") == "placa_1_1"
    assert CFG._stem_de_label("placa_sin_prefijo") == "placa_sin_prefijo"
    foto = _foto_particula(centro=(30, 22))
    # Mismo stem, mayuscula distinta y a un nivel mas profundo: el capturador
    # debe encontrarla de todos modos.
    sub = tmp_path / "fotos" / "subdir" / "mas"
    sub.mkdir(parents=True)
    archivo = sub / "PLACA_101_1.PNG"
    is_ok, buf = cv2.imencode(".png", foto)
    assert is_ok
    archivo.write_bytes(bytes(buf))
    stem = "placa_101_1"
    _montar_run(tmp_path, stem, idx_img=1, alias="y11n",
                cajas=[(20, 16, 40, 28)])
    destino = tmp_path / "fixtures" / "geodesia"
    resumen = CFG.capturar(runs=tmp_path / "runs" / "detect_20260812_1030",
                           fotos=tmp_path / "fotos", destino=destino)
    assert resumen["serializadas"] == 1, (
        "debe encontrar la foto aun con mayusculas y subcarpetas")


def test_generador_fixtures_reproducible(tmp_path):
    """El generador sintetico crea las 8 clases canonicas y es idempotente."""
    destino = tmp_path / "fixtures"
    resumen1 = GFG.generar_fixtures(destino=destino, reforzar=False)
    assert resumen1["serializadas"] == 8
    assert resumen1["reutilizadas"] == 0

    esperados = [
        "y11n__fibra_curva__000.npy",
        "y11n__fibra_recta__000.npy",
        "y11n__fibra_sinuosa__000.npy",
        "y11n__fragmento_circular__000.npy",
        "y11n__fragmento_irregular__000.npy",
        "y11n__fragmento_rugoso__000.npy",
        "y11n__particula_alargada_diagonal__000.npy",
        "y11n__particula_bifurcada__000.npy",
    ]
    generados = sorted(f.name for f in destino.glob("*.npy"))
    assert generados == sorted(esperados)

    for nombre in esperados:
        arr = np.load(destino / nombre, allow_pickle=False)
        assert arr.ndim == 2 and arr.dtype == np.uint8
        assert np.isin(arr, [0, 255]).all()
        assert (arr > 0).any()

    # Segunda corrida sin reforzar: idempotente
    resumen2 = GFG.generar_fixtures(destino=destino, reforzar=False)
    assert resumen2["serializadas"] == 0
    assert resumen2["reutilizadas"] == 8

