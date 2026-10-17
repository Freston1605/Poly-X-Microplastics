"""Relee el ground truth del disco y recalcula sin volver a pasar el modelo.

La inferencia es lo caro; el ground truth no. Cuando se corrige una anotacion
mal puesta no hay ninguna razon para volver a inferir sobre las 69 placas: las
predicciones ya estan en memoria y solo hay que reemparejar y redibujar la
imagen de control.

Hasta ahora la unica salida era reejecutar el lote completo, y quien no lo hacia
se llevaba un informe con la anotacion vieja sin ningun aviso, porque la corrida
congela el GT que leyo. Este modulo cierra ese hueco.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import cv2

from polyx.model.yolo_wrap import (
    find_gt_for_image, read_yolo_txt, compute_box_size_um, tamano_imagen,
)
from polyx.model.metrics import match_image
from .runner import _draw_annotated, _color_bgr_for_class


def _mismas_cajas(a, b, tol: float = 0.5) -> bool:
    """True si dos listas de cajas coinciden en clase y coordenadas."""
    if len(a) != len(b):
        return False
    for d1, d2 in zip(a, b):
        if d1.class_id != d2.class_id:
            return False
        if (abs(d1.x1 - d2.x1) > tol or abs(d1.y1 - d2.y1) > tol
                or abs(d1.x2 - d2.x2) > tol or abs(d1.y2 - d2.y2) > tol):
            return False
    return True


def _um_px_de(state, ruta, params):
    """Escala de esa imagen segun la corrida, con el valor manual de respaldo.

    Se lee de ``state.calibraciones`` y no de ``params.um_per_px`` porque la
    escala es por foto: releer el GT no puede recalcular los tamanos con un
    factor distinto del que uso la corrida, o el informe diria una cosa y las
    cajas otra.
    """
    cal = (getattr(state, "calibraciones", None) or {}).get(Path(ruta).name)
    if cal is not None and getattr(cal, "valida", False):
        return cal.um_por_px
    return params.um_per_px if params.um_per_px > 0 else None


def recargar_gt(state, progreso=None) -> dict:
    """Relee el GT de cada imagen y recalcula metricas e imagenes anotadas.

    Solo toca las imagenes cuyo .txt cambio de verdad: comparar es mucho mas
    barato que redibujar una placa de 3260 px, y en el caso normal -- se corrigio
    una placa de sesenta y nueve -- el trabajo se reduce a esa.

    ``progreso`` se llama como ``progreso(hechas, total, nombre)`` si se pasa; si
    devuelve ``False`` se aborta y se conserva lo ya recalculado. Devuelve un
    resumen con las imagenes revisadas, las que cambiaron y las que siguen sin
    .txt.
    """
    resumen = {"revisadas": 0, "cambiadas": [], "sin_gt": [], "recalculadas": 0}
    if not state.results or state.run_dir is None:
        return resumen

    params = state.params
    # Los nombres de clase salen del primer modelo cargado: el .txt guarda solo
    # el indice, y sin el mapa las cajas quedarian sin clase legible.
    nombres = {}
    for slot in state.active_models():
        if slot.loaded is not None and slot.loaded.names:
            nombres = slot.loaded.names
            break

    por_imagen: dict = {}
    for mi in sorted(state.results):
        for r in state.results[mi]:
            por_imagen.setdefault(r.image_path, []).append((mi, r))

    total = len(por_imagen)
    for hechas, (ruta, grupo) in enumerate(por_imagen.items(), start=1):
        if progreso is not None and progreso(hechas, total, ruta.name) is False:
            break
        resumen["revisadas"] += 1

        gt_txt = find_gt_for_image(ruta, state.gt_folder)
        has_gt = gt_txt is not None
        if not has_gt:
            resumen["sin_gt"].append(ruta.name)

        # El tamano se lee de la cabecera, sin decodificar la foto entera: hace
        # falta para desnormalizar el .txt y en la mayoria de las imagenes es lo
        # unico que se va a necesitar.
        wh = tamano_imagen(ruta)
        if wh is None:
            continue
        W, H = wh

        gts = []
        if has_gt:
            gts = read_yolo_txt(gt_txt, W, H, nombres)
            for d in gts:
                compute_box_size_um(
                    d, _um_px_de(state, r.image_path, params))

        primero = grupo[0][1]
        if has_gt == primero.has_gt and _mismas_cajas(gts, primero.gt):
            continue
        resumen["cambiadas"].append(ruta.name)

        img_bgr = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        try:
            idx_img = state.images.index(ruta)
        except ValueError:
            idx_img = hechas - 1

        for mi, r in grupo:
            r.gt = list(gts)
            r.has_gt = has_gt
            if has_gt:
                m = match_image(r.predictions, gts, iou_thr=params.iou_tp)
                r.tp, r.fp, r.fn, r.miscls = m.tp, m.fp, m.fn, m.miscls
            else:
                r.tp = r.fp = r.fn = r.miscls = 0

            # La imagen de predicciones no cambia -- las predicciones son las
            # mismas -- pero la de control y la combinada si.
            sub = state.run_dir / state.model_slots[mi].alias
            sub.mkdir(parents=True, exist_ok=True)
            base = f"{idx_img:04d}_{ruta.stem}"

            pred_img = _draw_annotated(
                img_bgr, r.predictions,
                color_for=lambda d: _color_bgr_for_class(d.class_name),
                label_for=lambda d: f"{d.class_name} {d.conf:.2f}",
            )
            if has_gt:
                combinada = _draw_annotated(
                    pred_img, gts,
                    color_for=lambda d: (255, 200, 80),
                    label_for=lambda d: f"GT {d.class_name}",
                )
                gt_img = _draw_annotated(
                    img_bgr, gts,
                    color_for=lambda d: _color_bgr_for_class(d.class_name),
                    label_for=lambda d: f"{d.class_name}",
                )
            else:
                combinada, gt_img = pred_img, None

            r.annotated_path = _guardar(combinada, sub, base, "_annot")
            r.gt_path = _guardar(gt_img, sub, base, "_gt")
            resumen["recalculadas"] += 1

    return resumen


def _guardar(image, sub: Path, base: str, sufijo: str):
    """Escribe el PNG en la carpeta del run y devuelve su ruta."""
    if image is None:
        return None
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return None
    destino = sub / f"{base}{sufijo}.png"
    destino.write_bytes(bytes(buf))
    return destino
