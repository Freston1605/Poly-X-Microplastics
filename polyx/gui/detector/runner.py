"""Hilo de inferencia en background. Procesa todas las imágenes con todos los modelos."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import io
import json

import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal

from polyx.model.yolo_wrap import (
    YoloModel, Detection, find_gt_for_image, read_yolo_txt, compute_box_size_um,
)
from polyx.model.metrics import match_image
from polyx.model.asignacion import marcar_asignables
from polyx.model.procedencia import procedencia, sha256_archivo
from polyx.model.calibracion import (resolver, cargar_indice, buscar_indice,
                                resumen_lote)
from polyx.model.morfologia import aplicar_a_deteccion
from polyx.gui import theme as T
from .state import DetectorState, ImageResult
from polyx.model import observabilidad as obs


def _draw_annotated(img_bgr: np.ndarray, dets, color_for, label_for) -> np.ndarray:
    """Dibuja las cajas con un grosor proporcional al tamano de la imagen.

    Con grosor fijo de 2 px las anotaciones se volvian invisibles en las placas
    completas: una foto de 4096 px se reescala a 1100 para incrustarla en el
    informe, asi que esa linea acababa midiendo medio pixel. Escalando con el
    lado mayor, una placa se dibuja con ~6 px (lo mismo que usa
    paper/recomponer_placas.py) y un recorte de 1630 px conserva sus 2.
    """
    out = img_bgr.copy()
    lado = max(out.shape[:2])
    grosor = max(2, int(round(lado / 700)))
    escala_txt = max(0.5, lado / 3000)
    grosor_txt = max(1, int(round(escala_txt * 2)))

    for d in dets:
        c = color_for(d)
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, grosor)
        label = label_for(d)
        # El numero delante de todo lo demas: es lo que se busca cuando se viene
        # desde la tabla del informe.
        if getattr(d, "numero", None):
            label = f"{d.numero}. {label}" if label else str(d.numero)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          escala_txt, grosor_txt)
            margen = max(4, grosor * 2)
            cv2.rectangle(out, (x1, y1 - th - margen), (x1 + tw + margen, y1), c, -1)
            cv2.putText(out, label, (x1 + margen // 2, y1 - margen // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, escala_txt, (255, 255, 255),
                        grosor_txt, cv2.LINE_AA)
    return out


def _guardar_labels(dets, ruta_txt: Path, W: int, H: int) -> None:
    """Escribe las predicciones en formato YOLO dentro de la carpeta del run.

    Nunca junto a la foto: ahi vive el .txt del conteo manual, y volcar las
    predicciones encima destruiria el ground truth de forma irrecuperable.

    La confianza va como sexta columna. ``read_yolo_txt`` ignora las columnas de
    mas, asi que el archivo se puede reabrir en el Etiquetador como
    pre-anotacion -- contar a mano 500 particulas partiendo de las cajas del
    modelo es otra tarea que hacerlo desde cero -- y a la vez queda el registro
    de con que confianza salio cada caja.
    """
    lineas = []
    for d in dets:
        cx = ((d.x1 + d.x2) / 2) / W
        cy = ((d.y1 + d.y2) / 2) / H
        w = (d.x2 - d.x1) / W
        h = (d.y2 - d.y1) / H
        lineas.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {d.conf:.4f}")
    ruta_txt.parent.mkdir(parents=True, exist_ok=True)
    ruta_txt.write_text("\n".join(lineas), encoding="utf-8")


def _color_bgr_for_class(name: str):
    hex_ = T.CLASS_COLOR_HEX.get(name, "#33aaff").lstrip("#")
    r, g, b = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
    return (b, g, r)


class DetectorRunner(QThread):
    progress = Signal(int, int, str)
    image_done = Signal(int, object)     # model_idx, ImageResult
    finished_ok = Signal()
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: DetectorState, parent=None):
        super().__init__(parent)
        self.state = state

    def run(self):
        # Logica de observabilidad: el run_id arranca None y se fija cuando la
        # carpeta del run existe; evento_error deja el traceback que antes
        # se perdia bajo pythonw.
        _log = obs.logger_para("detector")
        obs.set_run_id(None)
        try:
            state = self.state
            params = state.params
            slots = state.active_models()
            inicio = datetime.now()
            obs.evento(_log, "run.config", placas=len(state.images),
                       modelos=[s.alias for s in slots],
                       conf=params.conf, iou=params.iou_nms,
                       imgsz=params.imgsz, device=params.device,
                       medir_forma=params.medir_forma,
                       um_por_px=params.um_per_px)
            # Lo que de verdad se uso, imagen por imagen, frente a lo que se
            # pidio en Parametros: el auto-fallback puede haber bajado el imgsz
            # por falta de VRAM y el troceado depende del tamano de cada foto.
            # Sin registrarlo, el informe declara una resolucion que quiza no
            # fue la que produjo esas cajas.
            ejecucion: dict[str, dict] = {}

            # La escala se resuelve por imagen, no una vez para el lote. El
            # indice se lee una sola vez: son 552 filas y releerlo por foto seria
            # tirar tiempo.
            indice = cargar_indice(params.indice_calibracion) if params.indice_calibracion else {}
            if not indice and state.images:
                indice = buscar_indice(state.images[0])
            state.calibraciones = {}
            # Particulas que no se pudieron segmentar, por modelo. Va al
            # resumen: una talla medida sobre la caja no es comparable con
            # una medida sobre la mascara, y hay que poder decir cuantas son.
            sin_forma: dict[str, int] = {}
            if not slots:
                self.failed.emit("No hay modelos cargados.")
                return
            if not state.images:
                self.failed.emit("No hay imágenes seleccionadas.")
                return

            # Crear carpeta de run
            run_dir = Path(__file__).resolve().parents[3] / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state.run_dir = run_dir / f"detect_{stamp}"
            state.run_dir.mkdir(parents=True, exist_ok=True)
            obs.set_run_id(stamp)

            # Cargar modelos
            for s in slots:
                if s.loaded is None:
                    s.loaded = YoloModel(str(s.path), alias=s.alias)
                try:
                    s.loaded.load()
                    obs.evento(_log, "model.loaded", alias=s.alias,
                               archivo=str(s.path))
                except Exception as exc:
                    obs.evento_error(_log, "model.error", exc,
                                     alias=s.alias, archivo=str(s.path))
                    raise

            # Resetear resultados
            state.results = {i: [] for i in range(len(state.model_slots))}
            total = len(state.images) * len(slots)
            done = 0

            for idx_img, img_path in enumerate(state.images):
                if state.consume_abort():
                    self.aborted.emit()
                    return

                # Cargar imagen una vez para GT/anotación
                img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    obs.evento(_log, "image.undecodable", level=logging.WARNING,
                               archivo=str(img_path))
                    done += len(slots)
                    self.progress.emit(done, total, img_path.name)
                    continue
                H, W = img_bgr.shape[:2]

                # Escala de ESTA foto. Se le pasa la imagen ya leida para no
                # volver a decodificar 12 MP solo para medir el anillo.
                cal = resolver(img_path, indice=indice,
                               um_por_px_manual=params.um_per_px,
                               diametro_mm=params.diametro_placa_mm,
                               medir_placa=params.medir_placa,
                               bgr=img_bgr,
                               altura_placa_mm=params.altura_placa_mm,
                               distancia_camara_mm=params.distancia_camara_mm)
                state.calibraciones[img_path.name] = cal
                um_px = cal.um_por_px if cal.valida else None
                if not cal.valida:
                    obs.evento(_log, "calibration.invalid", level=logging.WARNING,
                               archivo=str(img_path),
                               razon=getattr(cal, "razon", None))

                # Cargar GT si existe (usa nombres del primer modelo si está disponible)
                names_first = slots[0].loaded.names if slots[0].loaded else {}
                gt_txt = find_gt_for_image(img_path, state.gt_folder)
                gts: list[Detection] = []
                if gt_txt is not None:
                    gts = read_yolo_txt(gt_txt, W, H, names_first)
                    # El tamaño del GT no depende del modelo → se calcula una vez
                    for d in gts:
                        compute_box_size_um(d, um_px)
                        if params.medir_forma:
                            aplicar_a_deteccion(d, img_bgr, um_px)

                for mi_idx, slot in enumerate(slots):
                    # mi_idx es el orden dentro de active_models()
                    # necesitamos el índice real del slot dentro de model_slots
                    real_idx = state.model_slots.index(slot)
                    plan_troceo: dict = {}
                    preds = slot.loaded.predict_auto(
                        str(img_path),
                        conf=params.conf,
                        iou=params.iou_nms,
                        imgsz=params.imgsz,
                        device=params.device,
                        troceo=params.troceo,
                        umbral_px=params.troceo_umbral_px,
                        tile=params.troceo_tile,
                        overlap=params.troceo_overlap,
                        registro=plan_troceo,
                    )
                    # Se guarda la decision, no se registra de paso: el informe
                    # tiene que poder declarar si esa foto se conto troceada.
                    if plan_troceo.get("troceado"):
                        p = plan_troceo["plan"]
                        troceo_desc = (
                            f"{plan_troceo['ancho']}×{plan_troceo['alto']} px → "
                            f"{p['n_tiles']} tiles de {p['tile']} px, "
                            f"{int(p['overlap'] * 100)}% solape"
                        )
                    else:
                        troceo_desc = ""

                    reg = ejecucion.setdefault(
                        slot.alias,
                        {"imagenes": 0, "troceadas": 0, "imgsz_usados": set(),
                         "fallback_en": []})
                    reg["imagenes"] += 1
                    if plan_troceo.get("troceado"):
                        reg["troceadas"] += 1
                        # El tile entra a la red a su resolucion nativa; el
                        # imgsz pedido no se le aplica a la foto completa.
                        reg["imgsz_usados"].add(int(plan_troceo["plan"]["tile"]))
                    else:
                        # last_imgsz/last_fallback solo los escribe predict(),
                        # que es justo el camino sin trocear.
                        usado = getattr(slot.loaded, "last_imgsz", 0)
                        if usado:
                            reg["imgsz_usados"].add(int(usado))
                        if getattr(slot.loaded, "last_fallback", False):
                            reg["fallback_en"].append(img_path.name)

                    # Calcular tamaño de las predicciones (el GT ya se calculó arriba)
                    for d in preds:
                        # La caja primero, como respaldo: si la segmentación no
                        # logra aislar la partícula es preferible una talla
                        # aproximada -- y anotada como tal -- a perderla.
                        compute_box_size_um(d, um_px)
                        if params.medir_forma and not aplicar_a_deteccion(d, img_bgr, um_px):
                            sin_forma[slot.alias] = sin_forma.get(slot.alias, 0) + 1

                    # Filtro por tamaño
                    if um_px and (params.size_min_um > 0 or params.size_max_um > 0):
                        keep = []
                        for d in preds:
                            if d.diam_um is None:
                                keep.append(d); continue
                            if params.size_min_um > 0 and d.diam_um < params.size_min_um: continue
                            if params.size_max_um > 0 and d.diam_um > params.size_max_um: continue
                            keep.append(d)
                        preds = keep

                    # Abstencion: marcar cuales sostienen su polimero. Va DESPUES
                    # del filtro de tamano y ANTES del match con el GT, porque no
                    # descarta particulas -- solo decide si se les atribuye un
                    # polimero o se reportan como «no asignable».
                    marcar_asignables(preds, params.conf_asignacion)

                    # Numerar de arriba a abajo y de izquierda a derecha, no en
                    # el orden en que el modelo las devolvio: asi el numero 1
                    # esta arriba del todo y buscar el 37 en la foto es viable.
                    for _i, _d in enumerate(
                            sorted(preds, key=lambda z: (round(z.y1 / 40), z.x1)), 1):
                        _d.numero = _i

                    # Match con GT
                    # El .txt existe = hay ground truth, aunque no tenga
                    # ninguna caja: una placa revisada con cero particulas es un
                    # dato, no una placa sin revisar. Con bool(gts) esas placas
                    # quedaban fuera de las metricas y las detecciones que el
                    # modelo hacia sobre ellas no contaban como falso positivo.
                    has_gt = gt_txt is not None
                    if has_gt:
                        m = match_image(preds, gts, iou_thr=params.iou_tp)
                        tp, fp, fn, mc = m.tp, m.fp, m.fn, m.miscls
                    else:
                        tp = fp = fn = mc = 0

                    # ── Imágenes anotadas ──────────────────────────────
                    # 1) Predicciones del modelo (cajas de YOLO en color de clase).
                    pred_img = _draw_annotated(
                        img_bgr, preds,
                        color_for=lambda d: _color_bgr_for_class(d.class_name),
                        label_for=lambda d: f"{d.class_name} {d.conf:.2f}",
                    )
                    # 2) Combinada (predicción + GT en cian) para preview e
                    #    inspección de errores. Si no hay GT, coincide con pred.
                    combined = pred_img
                    gt_img = None
                    if has_gt:
                        combined = _draw_annotated(
                            pred_img, gts,
                            color_for=lambda d: (255, 200, 80),
                            label_for=lambda d: f"GT {d.class_name}",
                        )
                        # 3) Ground Truth solo (cajas reales en color de clase)
                        #    sobre la imagen original → comparación lado a lado.
                        gt_img = _draw_annotated(
                            img_bgr, gts,
                            color_for=lambda d: _color_bgr_for_class(d.class_name),
                            label_for=lambda d: f"{d.class_name}",
                        )

                    # ── Guardar PNGs en run_dir ────────────────────────
                    sub = state.run_dir / slot.alias
                    sub.mkdir(parents=True, exist_ok=True)

                    def _encode_and_save(image, suffix: str):
                        """Codifica a PNG, lo guarda en disco y devuelve su ruta.

                        Devuelve la ruta y no los bytes a proposito: en un lote
                        grande retener cada PNG en memoria llegaba a varios GB, y
                        el archivo ya queda escrito aqui mismo.
                        """
                        if image is None:
                            return None
                        is_ok, buf = cv2.imencode(".png", image)
                        if not is_ok:
                            return None
                        # El nombre lleva el indice de la imagen porque dos fotos
                        # de carpetas distintas pueden compartir nombre (1.1.jpg
                        # existe en varios testigos) y se pisarian entre si.
                        destino = sub / f"{idx_img:04d}_{img_path.stem}{suffix}.png"
                        destino.write_bytes(bytes(buf))
                        return destino

                    annotated_path = _encode_and_save(combined, "_annot")
                    pred_path = _encode_and_save(pred_img, "_pred")
                    gt_path = _encode_and_save(gt_img, "_gt") if has_gt else None

                    # Las cajas en texto, no solo dibujadas: un PNG no se puede
                    # reanalizar ni reabrir para corregir.
                    _guardar_labels(
                        preds,
                        sub / "labels" / f"{idx_img:04d}_{img_path.stem}.txt",
                        W, H)

                    res = ImageResult(
                        image_path=img_path, model_idx=real_idx,
                        predictions=preds, gt=gts, has_gt=has_gt,
                        tp=tp, fp=fp, fn=fn, miscls=mc,
                        annotated_path=annotated_path,
                        pred_path=pred_path,
                        gt_path=gt_path,
                        troceo=troceo_desc,
                    )
                    state.results.setdefault(real_idx, []).append(res)
                    self.image_done.emit(real_idx, res)

                    done += 1
                    self.progress.emit(done, total, img_path.name)

            ms = int((datetime.now() - inicio).total_seconds() * 1000)
            obs.evento(_log, "run.finish", ms=ms,
                       placas=len(state.images),
                       imgsz_usados=sorted({int(s) for reg in ejecucion.values()
                                            for s in reg.get("imgsz_usados", [])}),
                       sin_forma=dict(sin_forma))
            # Un lote con muchas particulas sin forma merece mirarse: puede ser
            # el modelo, el umbral o fotos fuera de foco, y no se ve en la UI.
            peor = max(sin_forma.items(), key=lambda kv: kv[1], default=())
            if peor and peor[1] > 0:
                obs.evento(_log, "run.shapeless_high", level=logging.WARNING,
                           modelo=peor[0], sin_forma=peor[1])
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            obs.evento_error(_log, "run.error", e)

    def _escribir_resumen(self, state, slots, inicio=None, ejecucion=None,
                          sin_forma=None) -> None:
        """Deja el run autodescrito: procedencia, parametros y totales por modelo.

        Sin esto una carpeta de run son PNGs sueltos y no se puede saber con que
        confianza ni con que peso salieron. No se falla la corrida si el volcado
        no se puede escribir: los resultados ya estan en memoria y en disco.
        """
        p = state.params
        ejecucion = ejecucion or {}
        fin = datetime.now()
        datos = {
            # Quien produjo esto: version, commit y librerias. Va primero porque
            # es lo que se mira al intentar rehacer una tabla del paper.
            "procedencia": procedencia(),
            "cuando": {
                "inicio": inicio.isoformat(timespec="seconds") if inicio else "",
                "fin": fin.isoformat(timespec="seconds"),
                "duracion_s": round((fin - inicio).total_seconds(), 1) if inicio else 0.0,
            },
            "parametros": {
                "confianza": p.conf, "iou_nms": p.iou_nms, "iou_tp": p.iou_tp,
                "imgsz": p.imgsz, "device": p.device, "um_por_px": p.um_per_px,
                "troceo": p.troceo, "troceo_umbral_px": p.troceo_umbral_px,
            },
            # Contra que patron se midieron los tamanos. Un valor en um sin esto
            # no se puede verificar ni reproducir.
            "calibracion": resumen_lote(list(getattr(state, "calibraciones", {}).values())),
            "imagenes": len(state.images),
            "modelos": [],
        }
        for slot in slots:
            mi = state.model_slots.index(slot)
            rs = state.results.get(mi, [])
            con_gt = [r for r in rs if r.has_gt]
            tp = sum(r.tp for r in con_gt); fp = sum(r.fp for r in con_gt)
            fn = sum(r.fn for r in con_gt); mc = sum(r.miscls for r in con_gt)
            reg = ejecucion.get(slot.alias, {})
            datos["modelos"].append({
                "alias": slot.alias,
                "peso": str(slot.path) if slot.path else "",
                # Dos best.pt de dos entrenamientos distintos se llaman igual;
                # la huella es lo unico que los distingue de verdad.
                "sha256": sha256_archivo(slot.path) if slot.path else "",
                "detecciones": sum(len(r.predictions) for r in rs),
                "imagenes_con_gt": len(con_gt),
                "tp": tp, "fp": fp, "fn": fn, "mal_clasificados": mc,
                "imgsz_usados": sorted(reg.get("imgsz_usados", ())),
                "imagenes_troceadas": reg.get("troceadas", 0),
                # Si esta lista no viene vacia, esas fotos se infirieron a menos
                # resolucion que la pedida porque la GPU se quedo sin memoria.
                "fallback_por_memoria": reg.get("fallback_en", []),
                "particulas_sin_forma": (sin_forma or {}).get(slot.alias, 0),
            })
            # classes.txt junto a los labels, para poder reabrirlos como
            # pre-anotacion en el Etiquetador sin adivinar el orden de clases.
            nombres = getattr(slot.loaded, "names", None) or {}
            if nombres:
                destino = state.run_dir / slot.alias / "classes.txt"
                try:
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    destino.write_text(
                        "\n".join(nombres[k] for k in sorted(nombres)),
                        encoding="utf-8")
                except OSError:
                    pass
        try:
            (state.run_dir / "resumen.json").write_text(
                json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
