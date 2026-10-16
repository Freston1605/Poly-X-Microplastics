"""Generador reproducible de fixtures canonicos de geodesia (S02-T01).

Crea detecciones sinteticas representativas para las clases morfologicas clave
de particulas:
  - fragmento_circular: fragmento compacto circular
  - fragmento_irregular: fragmento irregular concavo
  - fragmento_rugoso: fragmento con borde rugoso / aserrado
  - fibra_recta: fibra alargada recta
  - fibra_curva: fibra curva / arco
  - fibra_sinuosa: fibra delgada sinuosa
  - particula_alargada_diagonal: particula alargada orientada a 45 grados
  - particula_bifurcada: particula con ramificacion en Y

Construye una estructura de corrida mock compatible con ``polyx/detector/runner.py``
(layout ``runs/<alias>/labels/NNNN_<stem>.txt`` y ``<alias>/classes.txt``) y fotos
sinteticas BGR con fondo oscuro y particula brillante con contraste superior a
``CONTRASTE_MINIMO``.

Invoca ``tests/capturar_fixtures_geodesia.capturar()`` para que cada particula pase
por ``morfologia.segmentar()`` y se recorte al rectangulo envolvente como mascara
uint8 (0/255) canonica en ``tests/fixtures/geodesia/y11n__<stem>__000.npy``.

Uso::

    python tests/generar_fixtures_geodesia.py [--destino tests/fixtures/geodesia]

Docstrings en ingles, comentarios en espanol (convencion del repo).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import capturar_fixtures_geodesia as CFG  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "geodesia"


def _crear_lienzo(lado: int = 200, fondo: int = 15) -> np.ndarray:
    """Lienzo BGR cuadrado con fondo oscuro constante."""
    return np.full((lado, lado, 3), fondo, dtype=np.uint8)


def _obtener_caja_del_primer_plano(
    img: np.ndarray, umbral_brillo: int = 100
) -> tuple[float, float, float, float]:
    """Calcula la caja envolvente (x1, y1, x2, y2) del primer plano brillante."""
    fg = img.max(axis=2) > umbral_brillo
    ys, xs = np.where(fg)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def construir_morfologias(
    lado: int = 200, fondo: int = 15, brillo: int = 240
) -> list[tuple[str, np.ndarray, tuple[float, float, float, float]]]:
    """Genera las 8 morfologias canonicas con sus imagenes BGR y cajas envolventes."""
    color_brillo = (brillo, brillo, brillo)
    formas: list[tuple[str, np.ndarray, tuple[float, float, float, float]]] = []

    # 1. fragmento_circular: disco compacto
    img_circ = _crear_lienzo(lado, fondo)
    cv2.circle(img_circ, (100, 100), 20, color_brillo, -1)
    formas.append(("fragmento_circular", img_circ, _obtener_caja_del_primer_plano(img_circ)))

    # 2. fragmento_irregular: poligono concavo
    img_irreg = _crear_lienzo(lado, fondo)
    pts_irreg = np.array(
        [[75, 75], [130, 75], [130, 130], [105, 108], [75, 130]], dtype=np.int32
    )
    cv2.fillPoly(img_irreg, [pts_irreg], color_brillo)
    formas.append(("fragmento_irregular", img_irreg, _obtener_caja_del_primer_plano(img_irreg)))

    # 3. fragmento_rugoso: borde con perturbaciones sinusoidales
    img_rugoso = _crear_lienzo(lado, fondo)
    ang = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    r = 22 + 4 * np.sin(7 * ang) + 3 * np.cos(11 * ang)
    pts_rugoso = np.c_[100 + r * np.cos(ang), 100 + r * np.sin(ang)].astype(np.int32)
    cv2.fillPoly(img_rugoso, [pts_rugoso], color_brillo)
    formas.append(("fragmento_rugoso", img_rugoso, _obtener_caja_del_primer_plano(img_rugoso)))

    # 4. fibra_recta: linea horizontal delgada
    img_recta = _crear_lienzo(lado, fondo)
    cv2.line(img_recta, (55, 100), (145, 100), color_brillo, 4)
    formas.append(("fibra_recta", img_recta, _obtener_caja_del_primer_plano(img_recta)))

    # 5. fibra_curva: arco de circunferencia
    img_curva = _crear_lienzo(lado, fondo)
    t_curva = np.linspace(0.2, 0.8 * np.pi, 60)
    pts_curva = np.c_[100 + 40 * np.cos(t_curva), 110 - 40 * np.sin(t_curva)].astype(np.int32)
    cv2.polylines(img_curva, [pts_curva], False, color_brillo, 4)
    formas.append(("fibra_curva", img_curva, _obtener_caja_del_primer_plano(img_curva)))

    # 6. fibra_sinuosa: trayectoria sinusoidal
    img_sinuosa = _crear_lienzo(lado, fondo)
    xs = np.linspace(50, 150, 80)
    ys = 100 + 18 * np.sin(2 * np.pi * (xs - 50) / 50)
    pts_sin = np.c_[xs, ys].astype(np.int32)
    cv2.polylines(img_sinuosa, [pts_sin], False, color_brillo, 4)
    formas.append(("fibra_sinuosa", img_sinuosa, _obtener_caja_del_primer_plano(img_sinuosa)))

    # 7. particula_alargada_diagonal: fibra recta orientada a 45 grados
    img_diag = _crear_lienzo(lado, fondo)
    cv2.line(img_diag, (60, 60), (140, 140), color_brillo, 4)
    formas.append(
        ("particula_alargada_diagonal", img_diag, _obtener_caja_del_primer_plano(img_diag))
    )

    # 8. particula_bifurcada: estructura ramificada en Y
    img_bif = _crear_lienzo(lado, fondo)
    cv2.line(img_bif, (100, 140), (100, 100), color_brillo, 4)
    cv2.line(img_bif, (100, 100), (68, 68), color_brillo, 4)
    cv2.line(img_bif, (100, 100), (132, 68), color_brillo, 4)
    formas.append(("particula_bifurcada", img_bif, _obtener_caja_del_primer_plano(img_bif)))

    return formas


def montar_mock_run(
    base_dir: Path,
    alias: str = "y11n",
    lado: int = 200,
) -> tuple[Path, Path]:
    """Crea la estructura de corrida mock con fotos y labels YOLO."""
    runs_dir = base_dir / "runs"
    fotos_dir = base_dir / "fotos"
    fotos_dir.mkdir(parents=True, exist_ok=True)

    alias_dir = runs_dir / alias
    labels_dir = alias_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    (alias_dir / "classes.txt").write_text("no_clasificado\nparticula\n", encoding="utf-8")

    formas = construir_morfologias(lado=lado)
    W, H = lado, lado

    for idx_img, (stem, img_bgr, (x1, y1, x2, y2)) in enumerate(formas):
        # Guardar foto con cv2.imencode para consistencia con el pipeline
        is_ok, buf = cv2.imencode(".png", img_bgr)
        assert is_ok, f"Fallo al codificar foto {stem}"
        (fotos_dir / f"{stem}.png").write_bytes(bytes(buf))

        # Escribir label YOLO: class cx cy w h conf (normalizados)
        cx = (x1 + x2) / (2.0 * W)
        cy = (y1 + y2) / (2.0 * H)
        w = (x2 - x1) / float(W)
        h = (y2 - y1) / float(H)
        conf = 0.9500
        linea = f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n"

        label_path = labels_dir / f"{idx_img:04d}_{stem}.txt"
        label_path.write_text(linea, encoding="utf-8")

    return runs_dir, fotos_dir


def generar_fixtures(
    destino: Path = FIXTURES_DIR,
    reforzar: bool = False,
    workdir: Path | None = None,
) -> dict:
    """Genera las mascaras .npy capturadas a traves del pipeline real."""
    if workdir is not None:
        runs_dir, fotos_dir = montar_mock_run(workdir)
        resumen = CFG.capturar(runs=runs_dir, fotos=fotos_dir, destino=destino, reforzar=reforzar)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_dir, fotos_dir = montar_mock_run(tmp_path)
            resumen = CFG.capturar(
                runs=runs_dir, fotos=fotos_dir, destino=destino, reforzar=reforzar
            )
    return resumen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera fixtures canonicos .npy para tests/fixtures/geodesia "
        "a traves de capturar_fixtures_geodesia.capturar()."
    )
    parser.add_argument(
        "--destino",
        type=Path,
        default=FIXTURES_DIR,
        help="Directorio destino de los archivos .npy",
    )
    parser.add_argument(
        "--reforzar",
        action="store_true",
        default=False,
        help="Sobreescribe fixtures si difieren del generado (por defecto False)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directorio de trabajo persistente opcional para inspeccionar fotos/labels",
    )
    args = parser.parse_args(argv)
    resumen = generar_fixtures(destino=args.destino, reforzar=args.reforzar, workdir=args.workdir)
    CFG.imprimir_resumen(resumen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
