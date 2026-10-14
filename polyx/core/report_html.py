"""Generador del informe de detección en HTML (autocontenido, imágenes en base64).

Contenido:
  1. Abstract (resumen ejecutivo)
  2. Métodos (modelo, params, calibración, dispositivo)
  3. Resultados generales (clase, conf, tamaño, P/R/F1)
  4. Resumen por modelo
  5. Análisis de errores (matriz de confusión + galería)
  6. Comparación entre modelos
  7. Galería completa por imagen anotada
  8. Conteo por muestra y tipo de plástico
  9. Referencias bibliográficas
"""
from __future__ import annotations
import base64
import html
import io
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# matplotlib en modo no-interactivo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import theme as T
from .i18n import tr, idioma
from .metrics import (
    aggregate_per_class, confusion_matrix,
    LABEL_TP, LABEL_FP, LABEL_FN, LABEL_MISCLS,
)
from .yolo_wrap import Detection
from .asignacion import NO_ASIGNABLE, clase_reportada, resumir


# ────────────────────────────────────────────────────────────────────
# Estilos del reporte (idénticos al manual)
# ────────────────────────────────────────────────────────────────────
REPORT_CSS = """
:root{--ink:#1f2328;--ink2:#424a53;--ink3:#656d76;--muted:#8c959f;
--rule:#d0d7de;--rule_soft:#eaeef2;--bg:#ffffff;--bg_soft:#f6f8fa;
--accent:#0969da;--accent_d:#0550ae;
--ok:#1f6b5e;--warn:#9a6700;--err:#cf222e;--vio:#6639ba;}
*{box-sizing:border-box;}
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:var(--ink);
background:var(--bg);margin:0;padding:0;line-height:1.55;}
.container{max-width:1100px;margin:0 auto;padding:40px 48px 80px;}
header.cover{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:28px;}
.kicker{font-size:10pt;letter-spacing:.14em;text-transform:uppercase;
color:var(--accent_d);font-weight:600;margin-bottom:8px;}
h1{font-size:28pt;margin:4px 0 8px;line-height:1.18;}
h2{font-size:18pt;margin-top:38px;padding-bottom:8px;border-bottom:1px solid var(--rule);}
h3{font-size:13pt;margin-top:22px;color:var(--ink2);}
h4{font-size:11pt;margin-top:14px;color:var(--ink2);}
.meta{font-size:10pt;color:var(--ink3);}
.toc{background:var(--bg_soft);border-left:3px solid var(--accent);
padding:14px 20px;margin:18px 0 28px;}
.toc h3{margin:0 0 8px;color:var(--ink);}
.toc ol{margin:0;padding-left:22px;font-size:10.5pt;}
.toc a{color:var(--ink2);text-decoration:none;}
.toc a:hover{color:var(--accent);}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0;}
.kpi{background:var(--bg);border:1px solid var(--rule);border-radius:8px;
padding:14px 16px;}
.kpi .v{font-size:22pt;font-weight:700;color:var(--ink);margin:4px 0 6px;}
.kpi .l{font-size:9pt;font-weight:600;color:var(--ink3);letter-spacing:1.4px;
text-transform:uppercase;}
.kpi .b{height:3px;background:var(--accent);border-radius:2px;}
table.data{width:100%;border-collapse:collapse;font-size:10pt;margin:12px 0;}
table.data th{background:var(--bg_soft);text-align:left;padding:8px 10px;
border-bottom:1px solid var(--rule);font-weight:600;}
table.data td{padding:7px 10px;border-bottom:1px solid var(--rule_soft);}
table.data tr:last-child td{border-bottom:none;}
table.data td.r{text-align:right;font-variant-numeric:tabular-nums;}
.fig{margin:18px 0;}
.fig img{width:100%;border:1px solid var(--rule);border-radius:6px;display:block;}
.caption{font-size:9pt;color:var(--ink3);margin-top:6px;text-align:center;}
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0;}
.gallery .item{border:1px solid var(--rule);border-radius:6px;overflow:hidden;background:var(--bg_soft);}
.gallery .item img{width:100%;display:block;}
.gallery .item .cap{font-size:8.5pt;color:var(--ink3);padding:5px 8px;}
/* Comparación lado a lado: Predicción vs Ground Truth */
.compare{border:1px solid var(--rule);border-radius:8px;overflow:hidden;
margin:18px 0;background:var(--bg_soft);}
.compare-pair{display:grid;grid-template-columns:1fr 1fr;gap:0;}
.compare-pair figure{margin:0;border-right:1px solid var(--rule);background:var(--bg);}
.compare-pair figure:last-child{border-right:none;}
.compare-pair img{width:100%;display:block;}
.compare-pair figcaption{font-size:9pt;color:var(--ink2);font-weight:600;
padding:7px 10px;text-align:center;border-top:1px solid var(--rule_soft);}
.compare-pair figcaption .sub{display:block;font-weight:400;color:var(--ink3);font-size:8.5pt;margin-top:2px;}
.compare-meta{font-size:9pt;color:var(--ink2);padding:9px 12px;
border-top:1px solid var(--rule);font-variant-numeric:tabular-nums;}
.compare-meta .tag{display:inline-block;margin-right:10px;}
/* Conteo por polimero de una sola foto, bajo el par de imagenes. */
table.conteo{width:100%;border-collapse:collapse;font-size:9.5pt;
border-top:1px solid var(--rule);background:var(--bg);}
table.conteo th{background:var(--bg_soft);text-align:right;padding:6px 12px;
font-weight:600;color:var(--ink2);border-bottom:1px solid var(--rule_soft);}
table.conteo th:first-child{text-align:left;}
table.conteo td{padding:5px 12px;text-align:right;
font-variant-numeric:tabular-nums;border-bottom:1px solid var(--rule_soft);}
table.conteo td:first-child{text-align:left;font-weight:600;}
table.conteo tr.tot td{border-bottom:none;background:var(--bg_soft);font-weight:700;}
table.conteo td.dif-pos{color:var(--err);}
table.conteo td.dif-neg{color:var(--accent_d);}
.nogt{display:flex;align-items:center;justify-content:center;min-height:180px;
color:var(--muted);font-size:10pt;text-align:center;padding:16px;
background:repeating-linear-gradient(45deg,#fbfcfd,#fbfcfd 12px,#f0f2f4 12px,#f0f2f4 24px);}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:8.5pt;
font-weight:600;letter-spacing:.04em;}
.b-blue{background:#ddf4ff;color:var(--accent_d);}
.b-green{background:#d8f5ea;color:var(--ok);}
.b-amber{background:#fff8c5;color:var(--warn);}
.b-red{background:#ffebe9;color:var(--err);}
.b-violet{background:#ece0fd;color:var(--vio);}
footer{border-top:1px solid var(--rule);margin-top:60px;padding-top:18px;
font-size:9.5pt;color:var(--ink3);}
@media print{.container{max-width:none;padding:24px;}
h2{page-break-after:avoid;}.fig,.gallery .item{page-break-inside:avoid;}
/* Que la foto y su tabla no queden en paginas distintas del PDF. */
.compare{page-break-inside:avoid;}}
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ────────────────────────────────────────────────────────────────────
# Conteo por polímero y por muestra
# ────────────────────────────────────────────────────────────────────
# Orden de presentación de los polímeros. Cualquier clase que aparezca en los
# datos y no esté aquí se agrega al final, para que un dataset con otras clases
# no pierda columnas en silencio.
_ORDEN_CLASES = ["PET", "PP", "LDPE"]

# Las placas del río Loa se nombran `<tramo>.<testigo>` con sufijo `x`/`xx` para
# la segunda y tercera placa del mismo tramo. Los recortes agregan `__f<fila>c<col>`.
_RE_PLACA = re.compile(r"^(\d+)\.(\d+)(x*)$")
_ESTACION = {"1": "Chiu Chiu", "2": "Taira", "3": "Desembocadura"}
_ORDEN_ESTACION = ["Chiu Chiu", "Taira", "Desembocadura"]
_PLACA = {"": "a", "x": "b", "xx": "c"}


def _conteo_por_clase(detecciones) -> Counter:
    """Partículas por polímero reportado.

    Usa la clase *reportada* y no ``class_name`` a secas: con la abstención
    activa, una partícula cuya confianza no sostiene el polímero cuenta bajo
    «no asignable». Sigue siendo una partícula detectada — solo no se le
    atribuye un polímero.
    """
    c = Counter()
    for d in detecciones:
        c[clase_reportada(d)] += 1
    return c


def _clases_vistas(resultados_planos) -> List[str]:
    """Clases presentes en el lote, en orden de presentación.

    Se toman de las detecciones **y** del Ground Truth: un polímero que el
    modelo nunca detecta seguiría siendo una fila de la tabla, con cero, que es
    justo el dato que interesa mirar.
    """
    vistas = {clase_reportada(d) for r in resultados_planos
              for d in list(r.predictions) + list(r.gt)}
    orden = [c for c in _ORDEN_CLASES if c in vistas]
    # «no asignable» va al final y no alfabéticamente entre los polímeros: no es
    # un polímero, es la ausencia de uno.
    resto = sorted(vistas - set(orden) - {NO_ASIGNABLE})
    return orden + resto + ([NO_ASIGNABLE] if NO_ASIGNABLE in vistas else [])


def _muestra_de(ruta) -> Optional[tuple]:
    """(estación, tramo, placa) a partir del nombre, o None si no sigue la pauta.

    El informe no puede exigir esta nomenclatura: si las imágenes vienen de otro
    estudio, la tabla agrupada simplemente no se emite y queda la tabla por
    imagen, que siempre se puede construir.
    """
    stem = Path(ruta).stem.split("__")[0]
    m = _RE_PLACA.match(stem)
    if not m:
        return None
    tramo, testigo, sufijo = m.group(1), m.group(2), m.group(3)
    return (_ESTACION.get(testigo, f"Testigo {testigo}"),
            int(tramo), _PLACA.get(sufijo, sufijo))


def _clave_orden(ruta) -> tuple:
    """Ordena por estación/tramo/placa cuando el nombre lo permite."""
    m = _muestra_de(ruta)
    if m is None:
        return (1, 0, 0, "", Path(ruta).name)
    estacion, tramo, placa = m
    idx = _ORDEN_ESTACION.index(estacion) if estacion in _ORDEN_ESTACION else 99
    return (0, idx, tramo, placa, Path(ruta).name)


def _esc(v) -> str:
    """Escapa un VALOR interpolado al HTML: <, > y &; las comillas quedan.

    ``quote=False`` es deliberado: los valores del informe no van entre atribu-
    tos, y escapar tambien las comillas conservaria intacta la comilla simple
    de un nombre de archivo real con apostrofes (la identidad de bytes del ren-
    der benigno depende de eso). Los literales de plantilla (td, tr, span, es-
    tilos) jamas pasan por aqui: solo el dato que el lote aporta.
    """
    return html.escape(str(v), quote=False)


def _tabla_conteo(manual: Optional[Counter], detectado: Counter,
                  clases: List[str]) -> str:
    """Tabla de una foto: partículas por polímero, contadas a mano y por el modelo.

    Es conteo, no evaluación: la columna del modelo son **todas** sus
    detecciones de ese polímero, sin descontar falsos positivos ni emparejar
    caja a caja. Para eso está el análisis de errores.
    """
    hay_manual = manual is not None
    filas = ""
    for c in clases:
        det = detectado.get(c, 0)
        if hay_manual:
            man = manual.get(c, 0)
            dif = det - man
            css = "dif-pos" if dif > 0 else ("dif-neg" if dif < 0 else "")
            filas += (f"<tr><td>{_esc(tr(c))}</td><td>{man}</td><td>{det}</td>"
                      f"<td class='{css}'>{dif:+d}</td></tr>")
        else:
            filas += f"<tr><td>{_esc(tr(c))}</td><td>{det}</td></tr>"

    tot_det = sum(detectado.get(c, 0) for c in clases)
    if hay_manual:
        tot_man = sum(manual.get(c, 0) for c in clases)
        dif = tot_det - tot_man
        css = "dif-pos" if dif > 0 else ("dif-neg" if dif < 0 else "")
        filas += (f"<tr class='tot'><td>Total</td><td>{tot_man}</td>"
                  f"<td>{tot_det}</td><td class='{css}'>{dif:+d}</td></tr>")
        cabecera = tr("<tr><th>Polímero</th><th>Conteo manual</th>"
                      "<th>Detectadas por el modelo</th><th>Diferencia</th></tr>")
    else:
        filas += f"<tr class='tot'><td>Total</td><td>{tot_det}</td></tr>"
        cabecera = tr("<tr><th>Polímero</th><th>Detectadas por el modelo</th></tr>")

    return f"<table class='conteo'>{cabecera}{filas}</table>"


# Ancho máximo de las imágenes de galería, en píxeles. Se incrustan en base64
# dentro del HTML, así que su tamaño es el del reporte: sin reducir, cuatro
# imágenes de 4096 px pesaban 14.5 MB y un lote de varios cientos generaba un
# archivo de gigabytes que ningún navegador abre.
_ANCHO_GALERIA = 1100


def _uri_galeria(data: bytes, ancho_max: int = _ANCHO_GALERIA) -> str:
    """Data URI de una imagen de galería: siempre JPEG, reescalada si es ancha.

    Se recodifica **siempre**, no solo al reducir. Las anotadas se generan en
    PNG (sin pérdida), formato pésimo para fotografía: una sola placa ocupaba
    ~3.6 MB y cuatro imágenes producían un reporte de 14.5 MB. En JPEG la misma
    imagen baja un orden de magnitud sin diferencia visible.

    Devuelve el URI completo con su tipo MIME, no solo el base64: declararlo
    como PNG tras recodificar impediría que el navegador lo mostrara.
    """
    try:
        import cv2
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            if arr.shape[1] > ancho_max:
                alto = int(round(arr.shape[0] * ancho_max / arr.shape[1]))
                arr = cv2.resize(arr, (ancho_max, alto), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                return ("data:image/jpeg;base64,"
                        + base64.b64encode(buf.tobytes()).decode("ascii"))
    except Exception:
        pass
    return "data:image/png;base64," + _img_b64(data)


def _equipo() -> List[tuple]:
    """Componentes de la maquina, para poder declarar donde se ejecuto."""
    import platform

    filas = [(tr('Sistema operativo'), f"{platform.system()} {platform.release()}"),
             (tr('Procesador'), platform.processor() or "—")]
    try:
        import psutil
        filas.append((tr('Memoria RAM'),
                      f"{psutil.virtual_memory().total / 1024**3:.1f} GB"))
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            filas += [
                ("GPU", props.name),
                ("VRAM", f"{props.total_memory / 1024**3:.1f} GB"),
                ("Compute capability", f"{props.major}.{props.minor}"),
                ("PyTorch", f"{torch.__version__} (CUDA {torch.version.cuda})"),
            ]
        else:
            filas += [("GPU", tr('no disponible')), ("PyTorch", torch.__version__)]
    except Exception:
        pass
    return filas


def _entrenamiento_de(peso: Path) -> Optional[Dict]:
    """Lee del checkpoint con que configuracion y resultado se entreno.

    Ultralytics guarda ``train_args`` y ``train_metrics`` dentro del propio .pt,
    asi que el informe puede declarar imgsz, batch, epocas y metricas sin
    depender de que la carpeta del run siga existiendo.
    """
    try:
        import torch
        ck = torch.load(str(peso), map_location="cpu", weights_only=False)
    except Exception:
        return None
    args = ck.get("train_args") or {}
    met = ck.get("train_metrics") or {}
    if not args and not met:
        return None
    return {
        "base": args.get("model", "—"),
        "imgsz": args.get("imgsz", "—"),
        "batch": args.get("batch", "—"),
        "epocas": args.get("epochs", "—"),
        "optimizador": args.get("optimizer", "—"),
        "lr0": args.get("lr0", "—"),
        "amp": args.get("amp", "—"),
        "precision": met.get("metrics/precision(B)"),
        "recall": met.get("metrics/recall(B)"),
        "map50": met.get("metrics/mAP50(B)"),
        "map": met.get("metrics/mAP50-95(B)"),
    }


def _seccion_equipo(active) -> str:
    """Tabla de componentes y de como se entreno cada modelo cargado."""
    filas_eq = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in _equipo())
    html = (f"<h3>@@N@@.1 {tr('Equipo de cómputo')}</h3>"
            f"<table class='data'><tr>{tr('<th>Componente</th><th>Detalle</th>')}</tr>"
            f"{filas_eq}</table>")

    entrenos = []
    for s in active:
        if s.path is None:
            continue
        info = _entrenamiento_de(Path(s.path))
        if info:
            entrenos.append((s.alias, info))
    if not entrenos:
        return html

    def _n(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    filas = ""
    for alias, i in entrenos:
        filas += (
            f"<tr><td>{_esc(alias)}</td><td>{_esc(i['base'])}</td>"
            f"<td class='r'>{_esc(i['imgsz'])}</td><td class='r'>{_esc(i['batch'])}</td>"
            f"<td class='r'>{_esc(i['epocas'])}</td><td>{_esc(i['optimizador'])}</td>"
            f"<td class='r'>{_n(i['precision'])}</td><td class='r'>{_n(i['recall'])}</td>"
            f"<td class='r'>{_n(i['map50'])}</td><td class='r'>{_n(i['map'])}</td></tr>")

    html += (
        f"<h3>@@N@@.2 {tr('Entrenamiento de cada modelo')}</h3>"
        + tr("<p>Configuración y métricas de validación con que se entrenó cada peso, "
             "leídas del propio archivo <code>.pt</code>.</p>")
        + "<table class='data'>"
        + tr("<tr><th>Modelo</th><th>Arquitectura base</th><th>imgsz</th><th>batch</th>"
             "<th>épocas</th><th>optimizador</th><th>Precisión</th><th>Recall</th>"
             "<th>mAP@50</th><th>mAP@50-95</th></tr>")
        + f"{filas}</table>")
    return html


def _barrido_confianza(resultados, state, iou_thr: float, conf_actual: float):
    """Recalcula P/R/F1 a distintos umbrales de confianza, sin re-inferir.

    Las predicciones guardadas ya pasaron el umbral con que se ejecuto, asi que
    subirlo es solo filtrar y volver a emparejar. Bajarlo NO se puede: esas
    detecciones nunca se calcularon. Por eso el barrido arranca en el umbral
    usado y avisa de esa limitacion.

    Devuelve {alias: [(conf, precision, recall, f1, n_det), ...]}.
    """
    from .metrics import match_image

    umbrales = [round(u, 2) for u in np.arange(conf_actual, 0.96, 0.05)]
    salida = {}
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None:
            continue
        rs = [r for r in resultados.get(mi, []) if r.has_gt]
        if not rs:
            continue
        filas = []
        for u in umbrales:
            tp = fp = fn = mc = nd = 0
            for r in rs:
                preds = [p for p in r.predictions if p.conf >= u]
                nd += len(preds)
                m = match_image(preds, r.gt, iou_thr=iou_thr)
                tp += m.tp; fp += m.fp; fn += m.fn; mc += m.miscls
            # Estricto: la mala clasificacion penaliza en ambos lados.
            p = tp / (tp + fp + mc) if (tp + fp + mc) else 0.0
            rec = tp / (tp + fn + mc) if (tp + fn + mc) else 0.0
            f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
            filas.append((u, p, rec, f1, nd))
        salida[slot.alias] = filas
    return salida


def _fig_barrido(barrido: Dict[str, list]) -> str:
    """Curva de F1 frente al umbral de confianza, una linea por modelo."""
    if not barrido:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for alias, filas in barrido.items():
        ax.plot([f[0] for f in filas], [f[3] for f in filas],
                marker="o", markersize=3, label=alias, linewidth=1.6)
        mejor = max(filas, key=lambda f: f[3])
        ax.plot([mejor[0]], [mejor[3]], marker="*", markersize=13, zorder=5,
                color=ax.lines[-1].get_color())
    ax.set_xlabel(tr("Umbral de confianza"))
    ax.set_ylabel(tr("F1 (con clase)"))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


_PALETA = (T.DOC.ACCENT, T.DOC.VIO, T.DOC.WARN)


def _fig_comparacion_metricas(datos: List[tuple]) -> str:
    """Barras agrupadas de P/R/F1 por modelo, en los dos criterios.

    ``datos``: ``[(alias, p_loc, r_loc, f1_loc, p_cls, r_cls, f1_cls), ...]``.

    Se dibujan los dos paneles juntos y no solo el permisivo: la distancia entre
    ambos es la confusion entre polimeros, y verla al lado evita leer el numero
    bueno como si fuera el desempeno completo.
    """
    if not datos:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    etiquetas = [tr("Precisión"), "Recall", "F1"]
    x = np.arange(len(etiquetas))
    ancho = 0.8 / max(1, len(datos))
    for panel, (ax, desplazamiento, titulo) in enumerate((
            (axes[0], 1, tr("Localización")), (axes[1], 4, tr("Con clase")))):
        for i, fila in enumerate(datos):
            vals = fila[desplazamiento:desplazamiento + 3]
            pos = x - 0.4 + ancho * (i + 0.5)
            barras = ax.bar(pos, vals, ancho, label=fila[0],
                            color=_PALETA[i % len(_PALETA)])
            if panel == 1 or len(datos) <= 2:
                ax.bar_label(barras, fmt="%.3f", fontsize=7, padding=1)
        ax.set_title(titulo, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(tr("Valor"))
    # La leyenda va bajo la figura: dentro de los ejes se montaba sobre las
    # barras, que con estos valores llegan casi al techo en las dos mitades.
    manejadores, nombres = axes[0].get_legend_handles_labels()
    fig.legend(manejadores, nombres, frameon=False, fontsize=9,
               loc="lower center", ncol=len(datos),
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_acuerdo_por_imagen(alias_a: str, alias_b: str, pares: List[tuple]) -> str:
    """Detecciones por imagen de un modelo frente al otro, con la diagonal.

    El total de un lote esta dominado por unas pocas placas densas, asi que dos
    modelos pueden empatar en la suma y discrepar foto a foto. Aqui cada punto es
    una imagen: sobre la diagonal cuentan igual, y la distancia a ella dice
    cuanto se separan y en que placas.

    Escala simetrica-logaritmica porque la mayoria de las placas tiene 0-10
    particulas y unas pocas pasan de 500; en escala lineal el grueso de los
    puntos se apelotona en el origen.
    """
    if not pares:
        return ""
    a = [p[0] for p in pares]
    b = [p[1] for p in pares]
    tope = max(max(a), max(b), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, tope], [0, tope], "--", color="#999", linewidth=1,
            label="acuerdo exacto")
    ax.scatter(a, b, s=26, alpha=0.75, color=T.DOC.ACCENT, edgecolor="white",
               linewidth=0.5)
    # Se rotulan las placas que mas se separan: son las que hay que ir a mirar.
    for na, nb, nombre in sorted(pares, key=lambda p: -abs(p[0] - p[1]))[:3]:
        if abs(na - nb) > 0:
            ax.annotate(nombre, (na, nb), fontsize=7, color=T.DOC.INK2,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(-1, tope * 1.6)
    ax.set_ylim(-1, tope * 1.6)
    ax.set_xlabel(f"Detecciones · {alias_a}")
    ax.set_ylabel(f"Detecciones · {alias_b}")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_metricas_por_clase(por_modelo: Dict[str, Dict[str, tuple]]) -> str:
    """F1 por clase y por modelo: donde gana cada uno.

    El F1 global promedia polimeros con desempeno muy distinto -- PET separa
    bien, PP y LDPE comparten tono y se confunden -- de modo que un modelo puede
    ganar en el agregado y perder en la clase que interesa.
    """
    if not por_modelo:
        return ""
    clases = []
    for m in por_modelo.values():
        for c in m:
            if c not in clases:
                clases.append(c)
    if not clases:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(clases))
    ancho = 0.8 / max(1, len(por_modelo))
    for i, (alias, metricas) in enumerate(por_modelo.items()):
        vals = [metricas.get(c, (0, 0, 0))[2] for c in clases]
        barras = ax.bar(x - 0.4 + ancho * (i + 0.5), vals, ancho, label=alias,
                        color=_PALETA[i % len(_PALETA)])
        ax.bar_label(barras, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(clases)
    ax.set_ylabel(tr("F1 por clase"))
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_class_distribution(per_class_counts: Dict[str, int]) -> str:
    if not per_class_counts:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = list(per_class_counts.keys())
    vals = list(per_class_counts.values())
    colors = [T.CLASS_COLOR_HEX.get(n, "#888") for n in names]
    ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel(tr("Detecciones"))
    ax.set_title(tr("Distribución por clase"))
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_confidence_hist(confs: List[float]) -> str:
    if not confs:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(confs, bins=20, color=T.DOC.ACCENT, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(tr("Confianza"))
    ax.set_ylabel(tr("Frecuencia"))
    ax.set_title(tr("Histograma de confianza"))
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_size_hist(sizes_um: List[float]) -> str:
    if not sizes_um:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(sizes_um, bins=25, color=T.DOC.WARN, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(tr("Diámetro equivalente (μm)"))
    ax.set_ylabel(tr("Frecuencia"))
    ax.set_title(tr("Distribución de tamaños"))
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_tallas_por_tramo(largos_por_clase: Dict[str, List[float]]) -> str:
    """Histograma apilado de tallas, en tramos, con una barra por clase.

    Apilado y no superpuesto: la pregunta que responde es "cuantas particulas
    hay en cada tramo de talla y de que polimero son", y con barras superpuestas
    la de delante tapa a la de atras.
    """
    todos = [v for vals in largos_por_clase.values() for v in vals]
    if not todos:
        return ""
    a = np.array(todos, dtype=float)
    # Tramos en escala logaritmica: las tallas se reparten entre decenas y
    # miles de micrometros, y en escala lineal todo se amontona en la primera
    # barra y la cola queda ilegible.
    lo, hi = max(a.min(), 1.0), a.max()
    bordes = np.logspace(np.log10(lo), np.log10(hi * 1.02), 13)

    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=140)
    abajo = np.zeros(len(bordes) - 1)
    centros = np.sqrt(bordes[:-1] * bordes[1:])
    anchos = np.diff(bordes) * 0.86
    for cls in _ORDEN_CLASES + [c for c in largos_por_clase if c not in _ORDEN_CLASES]:
        vals = largos_por_clase.get(cls)
        if not vals:
            continue
        cuenta, _ = np.histogram(np.array(vals, dtype=float), bins=bordes)
        ax.bar(centros, cuenta, width=anchos, bottom=abajo, label=cls,
               color=T.CLASS_COLOR_HEX.get(cls, "#8b949e"), edgecolor="white",
               linewidth=0.6)
        abajo += cuenta
    ax.set_xscale("log")
    ax.set_xlabel(tr("talla (µm, dimensión mayor siguiendo la curva)"))
    ax.set_ylabel(tr("partículas"))
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Comparación de talla entre grupos (carpeta, o foto dentro de una carpeta) ──
#
# El pipeline del paper comparaba "por estación" parseando un patrón de
# nombre de archivo (N.T de un estudio concreto). Eso no sirve para un lote
# cualquiera que alguien cargue en el Detector. Lo único que la app sabe de
# CUALQUIER lote es de qué carpeta vino cada foto, así que la comparación se
# generaliza a eso: por carpeta, y por foto dentro de cada carpeta.
#
# La prueba (Kruskal-Wallis) se implementa aquí a mano, sin scipy: scipy no es
# una dependencia declarada del proyecto -- llega de rebote con ultralytics, y
# apoyarse en él rompería la instalación el día que ultralytics deje de
# traerlo (mismo motivo por el que morfologia.py evita scipy). El estadístico H
# y el valor p se verificaron contra scipy.stats.kruskal en un centenar de
# casos aleatorios, con y sin empates, y coinciden a precisión de máquina.

_MAX_GRUPOS_BOXPLOT = 12
_MAX_CARPETAS_POR_FOTO = 6


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Rango de cada valor, promediando el rango entre los empates."""
    orden = np.argsort(a, kind="mergesort")
    ordenado = a[orden]
    rangos_ord = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and ordenado[j + 1] == ordenado[i]:
            j += 1
        rangos_ord[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rangos = np.empty(len(a), dtype=float)
    rangos[orden] = rangos_ord
    return rangos


def _chi2_sf(h: float, df: int) -> float:
    """P(X > h) de una chi-cuadrado con ``df`` grados de libertad, sin scipy.

    Se construye con la recurrencia de la función gamma incompleta superior
    Q(a+1,x) = Q(a,x) + x^a e^-x / Gamma(a+1), subiendo de a en a=0.5 desde el
    caso cerrado (erfc para df impar, exp para df par). df aquí es siempre un
    entero chico (número de grupos - 1), así que la recurrencia converge en
    pocos pasos.
    """
    if h <= 0 or df < 1:
        return 1.0
    xa = h / 2.0
    if df % 2 == 1:
        q = math.erfc(math.sqrt(xa))
        a0, n = 0.5, (df - 1) // 2
    else:
        q = math.exp(-xa)
        a0, n = 1.0, (df - 2) // 2
    if n <= 0:
        return min(1.0, max(0.0, q))
    termino = (xa ** a0) * math.exp(-xa) / math.gamma(a0 + 1.0)
    q += termino
    for k in range(1, n):
        termino *= xa / (a0 + k)
        q += termino
    return min(1.0, max(0.0, q))


# Tabla t de Student a dos colas, 95% (alfa=0.05), para el intervalo de
# confianza de la escala media del lote. Mismo motivo que _chi2_sf para no
# usar scipy.stats.t. Valores exactos de tabla estandar para gl 1-30, mas
# los puntos de referencia habituales 40/60/120 para lotes grandes.
_TABLA_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}


def _t_critico_95(df: int) -> float:
    """t crítico a dos colas (95%) para ``df`` grados de libertad.

    Más allá de gl=30 no hay una entrada por cada gl: se usa el escalón
    tabulado más chico que sea >= el real. Como t decrece con los grados de
    libertad, ese escalón da un intervalo algo más ancho que el exacto, nunca
    más angosto -- conservador en vez de optimista.
    """
    if df <= 0:
        return float("nan")
    if df in _TABLA_T95:
        return _TABLA_T95[df]
    for umbral in (40, 60, 120):
        if df <= umbral:
            return _TABLA_T95[umbral]
    return 1.960


def _media_ic95(valores: List[float]) -> Optional[Tuple[float, float, int]]:
    """(media, margen del IC 95%, n), o None si no hay con qué estimarlo.

    El margen es t-critico * desvio_muestral / sqrt(n); el IC se reporta
    como media ± margen. Con un solo valor no hay dispersión que estimar.
    """
    a = np.asarray(valores, dtype=float)
    n = len(a)
    if n < 2:
        return None
    media = float(a.mean())
    std = float(a.std(ddof=1))
    margen = _t_critico_95(n - 1) * std / math.sqrt(n)
    return media, margen, n


def _kruskal_wallis(grupos: List[np.ndarray]) -> Optional[Tuple[float, float, int]]:
    """(H, p, grados de libertad), o None si no hay datos suficientes.

    Incluye la corrección estándar por empates (frecuente aquí: las tallas se
    redondean a un decimal). Sin corrección, un lote con muchos valores
    repetidos infla H artificialmente.
    """
    grupos = [np.asarray(g, dtype=float) for g in grupos if len(g) > 0]
    if len(grupos) < 2:
        return None
    todos = np.concatenate(grupos)
    n_total = len(todos)
    if n_total < 3:
        return None
    rangos = _rankdata(todos)
    h = 0.0
    idx = 0
    for g in grupos:
        n_i = len(g)
        r_i = rangos[idx:idx + n_i].sum()
        h += (r_i * r_i) / n_i
        idx += n_i
    h = (12.0 / (n_total * (n_total + 1))) * h - 3 * (n_total + 1)
    _, cuentas = np.unique(todos, return_counts=True)
    empate = np.sum(cuentas ** 3 - cuentas)
    denom = n_total ** 3 - n_total
    if denom > 0 and empate > 0:
        c = 1.0 - empate / denom
        if c > 0:
            h = h / c
    df = len(grupos) - 1
    p = _chi2_sf(h, df)
    return h, p, df


def _texto_kruskal(resultado: Optional[Tuple[float, float, int]], n_grupos: int) -> str:
    if resultado is None:
        return tr("Sin datos suficientes para una prueba estadística (hace falta "
                  "más de una observación por grupo).")
    h, p, df = resultado
    frase = (tr("<strong>diferencia significativa</strong>") if p < 0.05
             else tr("sin diferencia significativa"))
    return tr("Kruskal-Wallis entre los {n} grupos: {frase} (H={h}, gl={gl}, "
              "p={p}). No dice CUÁL grupo difiere de cuál, solo que no todos "
              "comparten la misma distribución de talla.").format(
                  n=n_grupos, frase=frase, h=f"{h:.2f}", gl=df, p=f"{p:.3g}")


def _fig_boxplot_grupos(grupos: Dict[str, List[float]], titulo: str, xlabel: str,
                        max_grupos: int = _MAX_GRUPOS_BOXPLOT) -> str:
    """Boxplot de talla por grupo (carpeta, o foto dentro de una carpeta).

    Se trunca a los ``max_grupos`` con más partículas: un boxplot con más de
    una docena de cajas se vuelve ilegible antes de aportar nada, y los grupos
    con pocas partículas son los que menos dicen de todas formas.
    """
    items = sorted(grupos.items(), key=lambda kv: -len(kv[1]))
    truncado = len(items) > max_grupos
    items = items[:max_grupos]
    items.sort(key=lambda kv: kv[0])

    datos = [np.asarray(v, dtype=float) for _, v in items]
    etiquetas = [f"{k}\n(n={len(v)})" for k, v in items]

    fig, ax = plt.subplots(figsize=(max(5.5, 1.15 * len(items) + 2), 4.2), dpi=140)
    bp = ax.boxplot(datos, tick_labels=etiquetas, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.35, markeredgewidth=0))
    for patch in bp["boxes"]:
        patch.set_facecolor(T.DOC.ACCENT)
        patch.set_alpha(0.30)
        patch.set_edgecolor(T.DOC.ACCENT)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)
    ax.set_yscale("log")
    ax.set_ylabel(tr("talla (µm, escala log)"))
    ax.set_xlabel(xlabel)
    titulo_completo = titulo + (tr(" (las {n} con más partículas)").format(n=max_grupos)
                                if truncado else "")
    ax.set_title(titulo_completo, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="both", alpha=0.2)
    rota = len(items) > 5
    plt.setp(ax.get_xticklabels(), rotation=30 if rota else 0, ha="right" if rota else "center")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _spearman(x, y) -> Optional[float]:
    """Correlacion de rangos entre dos series. None si no se puede calcular.

    De rangos y no de Pearson porque aqui no hay motivo para esperar una
    relacion LINEAL: lo que se pregunta es si la talla (o el conteo) crece o
    decrece con la profundidad, no a que ritmo. Ademas el conteo de particulas
    tiene una cola larga -- unas pocas placas con cientos -- que dominaria una
    correlacion de Pearson.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _p_permutacion(x, y, n_perm: int = 20000, semilla: int = 0) -> Optional[float]:
    """p bilateral de la correlacion, barajando una de las dos series.

    Por permutacion y no por la aproximacion t habitual: con 7 u 8 tramos esa
    aproximacion no vale, y aqui n es el numero de TRAMOS, no de particulas.
    Barajar no supone nada sobre la distribucion y responde exactamente la
    pregunta que toca -- con que frecuencia el azar da una correlacion asi de
    grande.

    La semilla es fija para que el informe sea reproducible: el mismo lote
    tiene que dar el mismo numero cada vez que se genere.
    """
    rho = _spearman(x, y)
    if rho is None:
        return None
    rng = np.random.default_rng(semilla)
    y = np.asarray(y, dtype=float)
    extremos = 0
    for _ in range(n_perm):
        r = _spearman(x, rng.permutation(y))
        if r is not None and abs(r) >= abs(rho) - 1e-12:
            extremos += 1
    # +1 en numerador y denominador: sin eso, p podria salir 0, que no existe
    # en una prueba por permutacion.
    return (extremos + 1) / (n_perm + 1)


def _texto_tendencia(x, y, que: str) -> str:
    """Frase con la correlacion, su p y el n. Vacia si no se puede calcular."""
    rho = _spearman(x, y)
    if rho is None:
        return ""
    p = _p_permutacion(x, y)
    sentido = tr("crece") if rho > 0 else tr("decrece")
    if p is not None and p < 0.05:
        # Mayuscula inicial: la frase va detras del punto que cierra el nombre
        # de la estacion, y en minuscula se lee como una linea cortada.
        veredicto = tr("<strong>{que} {sentido} con la profundidad</strong>").format(
            que=que[0].upper() + que[1:], sentido=sentido)
    else:
        veredicto = tr("No se detecta tendencia en {que} con la profundidad").format(
            que=que)
    # El operador viaja con el valor: con "p = {p}" y un p pequeño salia
    # "p = < 0.001", con dos signos seguidos.
    #
    # Punto decimal y no coma: el resto de las cifras del informe salen de
    # formatos de Python y llevan punto; mezclarlos se lee como una errata.
    p_txt = "< 0.001" if p is not None and p < 0.001 else f"= {p:.3f}"
    return tr("{veredicto} (Spearman ρ = {rho}, p {p}, n = {n} tramos).").format(
        veredicto=veredicto, rho=f"{rho:+.2f}", p=p_txt, n=len(x))


def _fig_perfil_profundidad(perfiles: dict) -> str:
    """Perfil del testigo: profundidad en el eje vertical, hacia abajo.

    Es la convencion de cualquier perfil de sedimento, y no es cosmetica: con
    la profundidad en el eje X hay que girar la cabeza para leer un testigo.
    """
    if not perfiles:
        return ""
    # Una linea por estacion, con los colores del informe.
    colores = [T.DOC.ACCENT, T.DOC.OK, T.DOC.VIO, T.DOC.WARN, T.DOC.ERR]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.6), dpi=140, sharey=True)
    for i, (estacion, filas) in enumerate(sorted(perfiles.items())):
        tramos = [f[0] for f in filas]
        color = colores[i % len(colores)]
        ax1.plot([f[1] for f in filas], tramos, marker="o", markersize=4,
                 linewidth=1.4, label=estacion, color=color)
        medianas = [f[2] for f in filas]
        ax2.plot(medianas, tramos, marker="o", markersize=4, linewidth=1.4,
                 label=estacion, color=color)
        # Rango intercuartilico como banda: la mediana sola esconde si el
        # reparto de tallas se ensancha o se estrecha con la profundidad.
        ax2.fill_betweenx(tramos, [f[3] for f in filas], [f[4] for f in filas],
                          alpha=0.15, color=color, linewidth=0)
    ax1.set_xlabel(tr("partículas por tramo"))
    ax2.set_xlabel(tr("talla mediana (µm)  ·  banda = rango intercuartílico"))
    ax1.set_ylabel(tr("tramo (mayor = más profundo)"))
    ax1.invert_yaxis()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if len(perfiles) > 1:
        ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Como se llama cada morfotipo de cara al usuario ──
# El codigo distingue "fibra" y "fragmento" (morfologia.FIBRA / FRAGMENTO) y
# eso NO cambia: es la clasificacion con la que se mide y se decide si el largo
# sale por Feret o por geodesico. Lo que cambia es la palabra que se imprime.
#
# Va en una funcion y no repetido en cada tabla porque ya paso una vez: se
# renombro solo en las fichas y el informe quedo llamando "particula" y
# "fragmento" a la misma cosa en paginas distintas.
#
# NOTA PARA EL PAPER: en la literatura de microplasticos el termino estandar es
# "fragment" (fibra / fragmento / film / pellet). Para el manuscrito en ingles
# conviene volver a "fragment"; aqui se usa "particula" por decision del autor,
# porque en este material casi todo es fragmento y repetir la palabra en cada
# ficha no aportaba nada.
# Se traduce al LLAMAR, no aqui: un tr() en el cuerpo del modulo se resuelve al
# importar y se quedaria con el idioma de ese momento.
_NOMBRE_MORFOTIPO = {"fibra": "Fibra", "fragmento": "Partícula"}


def _morfotipo(valor: Optional[str], minuscula: bool = False) -> str:
    """Nombre presentable de un morfotipo. '—' si no se pudo determinar."""
    if not valor:
        return "—"
    nombre = tr(_NOMBRE_MORFOTIPO.get(valor, valor))
    return nombre.lower() if minuscula else nombre


def elegir_fichas(formas: List[tuple], max_fichas: int) -> List[tuple]:
    """Que particulas se enseñan con su medicion dibujada, y en que orden.

    ``formas`` es una lista de ``(Detection, ruta)``. Devuelve como mucho
    ``max_fichas`` de ellas.

    Antes entraban las PRIMERAS que aparecieran, por orden de nombre de
    archivo, y eso tenia dos consecuencias malas:

      * La particula MAYOR -- la que sostiene cualquier afirmacion sobre talla
        maxima -- podia quedarse fuera por estar en una foto de nombre alto.
      * Las fibras no salian NUNCA en la practica. Son ~1% del material, asi
        que el corte llegaba mucho antes de encontrar una. Y la fibra es
        justamente el caso donde actua el diametro geodesico: si no se ve
        ninguna medida, no hay como juzgar si ese metodo funciona.

    Ahora la muestra se reparte entre los dos morfotipos, a mitades, y dentro
    de cada uno manda el tamaño. Asi la mayor de cada tipo entra siempre.

    El reparto no es rigido: si no hay fibras suficientes -- lo normal -- el
    hueco lo ocupan las particulas mas grandes, en vez de devolver una muestra
    corta. Y al reves tambien.

    Es una muestra SESGADA a proposito, no representativa del lote; el informe
    lo declara donde la enseña.
    """
    if max_fichas <= 0:
        return []

    def talla(par) -> float:
        return par[0].largo_um or 0.0

    fibras = sorted([t for t in formas if t[0].morfotipo == "fibra"],
                    key=talla, reverse=True)
    otras = sorted([t for t in formas if t[0].morfotipo != "fibra"],
                   key=talla, reverse=True)

    mitad = max(1, max_fichas // 2)
    elegidas = fibras[:mitad]
    elegidas += otras[:max_fichas - len(elegidas)]
    # Huecos que dejo el grupo escaso los cubre el otro, para no entregar una
    # muestra mas corta de lo pedido habiendo material.
    if len(elegidas) < max_fichas:
        ya = {id(d) for d, _ in elegidas}
        sobrantes = [t for t in fibras + otras if id(t[0]) not in ya]
        elegidas += sobrantes[:max_fichas - len(elegidas)]
    return elegidas


def seccion_profundidad(formas, resultados) -> str:
    """Cuantas particulas y de que talla hay a cada profundidad del testigo.

    El tramo no es una carpeta cualquiera: es una variable ORDENADA. Eso
    permite preguntar por TENDENCIA -- si hay mas particulas abajo que arriba,
    o si las de abajo son mas grandes -- y no solo si los grupos difieren, que
    es todo lo que puede contestar la comparacion generica por carpeta.

    ``formas`` es [(Detection, ruta)] con talla medida; ``resultados`` son los
    ImageResult, que hacen falta para contar TODAS las particulas y no solo las
    que se pudieron medir en micrometros.

    Devuelve "" si los nombres no siguen la pauta ``tramo.testigo``: sin tramo
    no hay profundidad, y el informe no puede exigir esa nomenclatura.
    """
    # -- Conteo por tramo, sumando las placas del mismo tramo --
    # Se suman porque son submuestras de la misma masa de sedimento, no
    # repeticiones fotograficas. Es la misma regla que la seccion de conteo.
    conteo: Dict[tuple, int] = {}
    placas: Dict[tuple, set] = {}
    for r in resultados:
        m = _muestra_de(r.image_path)
        if m is None:
            continue
        estacion, tramo, placa = m
        conteo[(estacion, tramo)] = conteo.get((estacion, tramo), 0) + len(r.predictions)
        placas.setdefault((estacion, tramo), set()).add(placa)

    # -- Tallas por tramo --
    tallas: Dict[tuple, List[float]] = {}
    for d, ruta in formas:
        m = _muestra_de(ruta)
        if m is None or not d.largo_um:
            continue
        tallas.setdefault((m[0], m[1]), []).append(float(d.largo_um))

    if len(conteo) < 3:
        # Con uno o dos tramos no hay perfil que dibujar ni tendencia que
        # medir; decirlo es mas util que una figura de dos puntos.
        return ""

    perfiles: Dict[str, list] = {}
    for (estacion, tramo), n in sorted(conteo.items()):
        vals = np.asarray(tallas.get((estacion, tramo), []), dtype=float)
        if len(vals):
            mediana = float(np.median(vals))
            q1, q3 = (float(np.percentile(vals, 25)), float(np.percentile(vals, 75)))
        else:
            mediana = q1 = q3 = float("nan")
        perfiles.setdefault(estacion, []).append((tramo, n, mediana, q1, q3))

    fig = _fig_perfil_profundidad(perfiles)

    # -- Tendencia, por estacion --
    # n es el numero de TRAMOS, no de particulas: dos particulas de la misma
    # placa no son observaciones independientes de la profundidad, y contarlas
    # como tales inflaria cualquier significacion. Con pocos tramos la
    # correlacion es debil por construccion, y por eso se declara el n.
    parrafos = []
    for estacion, filas in sorted(perfiles.items()):
        filas = sorted(filas)
        x = [f[0] for f in filas]
        n_part = [f[1] for f in filas]
        med = [f[2] for f in filas]
        con_talla = [(a, b) for a, b in zip(x, med) if b == b]   # descarta NaN
        piezas = [_texto_tendencia(x, n_part, tr("el número de partículas"))]
        if len(con_talla) >= 3:
            piezas.append(_texto_tendencia([a for a, _ in con_talla],
                                           [b for _, b in con_talla],
                                           tr("la talla mediana")))
        piezas = [p for p in piezas if p]
        if piezas:
            parrafos.append(f"<p><strong>{_esc(estacion)}.</strong> " + " ".join(piezas) + "</p>")

    # -- Tabla --
    filas_tabla = ""
    for estacion, filas in sorted(perfiles.items()):
        for tramo, n, mediana, _q1, _q3 in sorted(filas):
            vals = tallas.get((estacion, tramo), [])
            # Un tramo puede tener particulas contadas y ninguna con talla, si
            # esa foto no llego a calibrarse. Se enseña el conteo y una raya en
            # las columnas de talla, que es lo que de verdad ocurrio.
            celdas = [
                _esc(estacion),
                str(tramo),
                str(len(placas.get((estacion, tramo), ()))),
                str(n),
                str(len(vals)),
                f"{mediana:.0f}" if vals else "—",
                f"{max(vals):.0f}" if vals else "—",
            ]
            filas_tabla += ("<tr><td>" + celdas[0] + "</td>"
                            + "".join(f"<td class='r'>{c}</td>" for c in celdas[1:])
                            + "</tr>")

    return (
        tr("<p>El tramo es la <strong>profundidad</strong> a la que se tomó el "
           "sedimento, y a diferencia de una carpeta cualquiera está "
           "<em>ordenado</em>. Eso permite preguntar por una tendencia —si hay "
           "más partículas abajo que arriba, o si las de abajo son más "
           "grandes— y no solo si los grupos difieren entre sí.</p>")
        + (f"<img src='data:image/png;base64,{fig}' style='max-width:100%'>" if fig else "")
        + "".join(parrafos)
        + tr("<p class='caption' style='text-align:left'>La correlación se calcula "
             "sobre los <strong>tramos</strong>, no sobre las partículas: dos "
             "partículas de la misma placa no son observaciones independientes de la "
             "profundidad, y contarlas como tales inflaría cualquier significación. "
             "El valor de p sale de barajar la serie 20 000 veces, no de una "
             "aproximación que con tan pocos tramos no valdría.</p>")
        + f"<table class='data'><tr>{tr('<th>Estación</th><th>Tramo</th><th>Placas</th><th>Partículas</th><th>Con talla</th><th>Mediana (µm)</th><th>Mayor (µm)</th>')}</tr>{filas_tabla}</table>")


def _figura_calibracion(state) -> str:
    """Foto de una placa con el circulo ajustado dibujado encima.

    El informe declaraba la escala -- "32.86 µm/px, placa de 100 mm" -- pero no
    la ENSEÑABA. Toda la cadena de tallas del estudio cuelga de ese numero, y
    hasta aqui habia que creerselo: si el ajuste hubiera agarrado el borde
    equivocado (la sombra de la placa, el reflejo del anillo, el borde del
    filtro de dentro) nada en el documento lo delataria.

    Con la circunferencia dibujada sobre la foto se ve de un vistazo si el
    ajuste cayo donde debia. Es el mismo criterio que ya se aplica a cada
    particula en las fichas: una medida que no se puede ver medida no se puede
    verificar.

    Se elige la foto de MEDIANA escala, no la primera: la primera puede ser
    justo la atipica del lote, y entonces la figura ilustraria el caso raro.

    Devuelve "" si no hay ninguna placa medida sobre la que dibujar.
    """
    from .calibracion import ORIGEN_PLACA, DIAMETRO_PLACA_MM

    cals = getattr(state, "calibraciones", None) or {}
    medidas = [(nombre, c) for nombre, c in cals.items()
               if getattr(c, "valida", False)
               and getattr(c, "origen", "") == ORIGEN_PLACA
               and getattr(c, "radio_px", 0) > 0]
    if not medidas:
        return ""

    # La de escala mediana representa al lote; la primera por nombre no.
    medidas.sort(key=lambda t: t[1].um_por_px)
    nombre, cal = medidas[len(medidas) // 2]

    # Hay que volver a la foto original: la calibracion guarda cx/cy/radio pero
    # no los pixeles. Se busca entre las imagenes analizadas por nombre.
    ruta = None
    for r in [x for rs in state.results.values() for x in rs]:
        if Path(r.image_path).name == nombre:
            ruta = Path(r.image_path)
            break
    if ruta is None or not ruta.exists():
        return ""

    try:
        import cv2
        from .calibracion import leer_imagen
        bgr = leer_imagen(ruta)
        if bgr is None:
            return ""

        vis = bgr.copy()
        cx, cy, r_px = float(cal.cx), float(cal.cy), float(cal.radio_px)
        # Grosores proporcionales al tamano de la foto: un trazo de 2 px es
        # invisible en una imagen de 4096, y uno de 20 tapa el borde que
        # justamente se quiere enseñar.
        grosor = max(2, int(round(min(vis.shape[:2]) / 500)))

        # Circunferencia ajustada + centro.
        cv2.circle(vis, (int(cx), int(cy)), int(r_px), (0, 255, 0), grosor)
        cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 0),
                       cv2.MARKER_CROSS, grosor * 14, grosor)
        # El diametro, que es la magnitud conocida (100 mm nominales).
        cv2.line(vis, (int(cx - r_px), int(cy)), (int(cx + r_px), int(cy)),
                 (0, 210, 255), grosor)

        escala = 900.0 / max(vis.shape[:2])
        if escala < 1:
            vis = cv2.resize(vis, None, fx=escala, fy=escala,
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return ""
        img64 = base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        # Una figura ilustrativa no puede tumbar el informe entero.
        return ""

    diam_px = 2.0 * r_px
    diam_mm = getattr(cal, "diametro_mm", 0) or DIAMETRO_PLACA_MM
    return f"""
<h3>{NUM}.1 {tr('Cómo se obtiene la escala, sobre una placa real')}</h3>
<p>{tr('En verde, la circunferencia que el ajuste encontró para el borde externo del anillo; en amarillo, su diámetro. Ese diámetro es la magnitud conocida')}
—<strong>{diam_mm:g} mm</strong>—{tr(' y de ahí sale todo lo demás.')}</p>
<img src='data:image/jpeg;base64,{img64}' style='max-width:100%'>
<p style='font-family:monospace;font-size:10pt'>
{diam_mm:g} mm ÷ {diam_px:.1f} px = <strong>{cal.um_por_px:.4f} µm/px</strong>
&nbsp;&nbsp;<span style='font-family:inherit;color:#656d76'>({_esc(nombre)},
{tr('{n} puntos de borde').format(n=cal.n_puntos)})</span></p>
<p>{tr('Conviene mirar si el círculo verde sigue el borde del anillo y no otra cosa —la sombra de la placa, un reflejo, o el borde del filtro de dentro—. Si cayera mal, <strong>todas</strong> las tallas de esa foto saldrían escaladas por el mismo factor equivocado, y ninguna otra cifra del informe lo delataría.')}</p>"""


def _ficha_particula(det, ruta_imagen, um_por_px: Optional[float],
                     titulo: str) -> str:
    """Recorte de una particula con su mascara y la cuenta de px a µm.

    Se muestra la conversion completa -- pixeles medidos, escala de ESA foto y
    resultado -- porque una talla en µm sin la cuenta que la produjo no se puede
    comprobar, y esta es la particula que mas peso tiene en cualquier conclusion
    sobre tamano maximo.
    """
    if det is None or not ruta_imagen:
        return ""
    try:
        # cv2 se importa aqui y no arriba: este modulo lo hace asi en el resto de
        # funciones para no cargar OpenCV cuando solo se pide texto.
        import cv2
        from .calibracion import leer_imagen
        from .morfologia import margen_de_caja, segmentar
        bgr = leer_imagen(ruta_imagen)
        if bgr is None:
            return ""
        # EL MISMO MARGEN QUE USA segmentar(), no uno parecido. La mascara viene
        # recortada con margen_de_caja(); si aqui se recorta con otro numero, el
        # contorno se dibuja desde un origen distinto y sale desplazado sobre la
        # foto. Ya paso con un 0.35 copiado a mano frente al 0.50 real, y se vio
        # a ojo en el informe: "la mascara quedo corrida".
        mg = margen_de_caja(det.x2 - det.x1, det.y2 - det.y1)
        x1 = max(0, int(det.x1) - mg); y1 = max(0, int(det.y1) - mg)
        x2 = min(bgr.shape[1], int(det.x2) + mg); y2 = min(bgr.shape[0], int(det.y2) + mg)
        sub = bgr[y1:y2, x1:x2].copy()
        if sub.size == 0:
            return ""
        mascara = segmentar(bgr, det.x1, det.y1, det.x2, det.y2)
        ov = sub.copy()
        if mascara is not None:
            h = min(ov.shape[0], mascara.shape[0]); w = min(ov.shape[1], mascara.shape[1])
            # Contorno y no relleno: tapar la particula con verde impide juzgar
            # si la mascara la sigue de verdad, que es justo lo que se enseña.
            cont, _ = cv2.findContours(mascara[:h, :w], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov[:h, :w], cont, -1, (0, 255, 0),
                             max(1, min(ov.shape[:2]) // 90))
        par = np.hstack([sub, ov])
        escala = min(4.0, 620 / max(1, par.shape[1]))
        if escala > 1:
            par = cv2.resize(par, None, fx=escala, fy=escala,
                             interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", par)
        if not ok:
            return ""
        img = f"<img src='data:image/png;base64,{_img_b64(buf.tobytes())}' style='max-width:100%'>"
    except Exception:
        # Una particula que no se deja recortar no puede tumbar el informe
        # entero: la ficha es ilustrativa y las tablas ya estan calculadas.
        return ""

    largo_px = (det.largo_um / um_por_px) if (det.largo_um and um_por_px) else None
    cuenta = ""
    if largo_px and um_por_px:
        cuenta = (f"<p style='font-family:monospace;font-size:10pt'>"
                  f"{largo_px:.1f} px &times; {um_por_px:.4f} µm/px = "
                  f"<strong>{det.largo_um:.0f} µm</strong></p>")
        # Las tres medidas juntas: si el rectangulo equivalente se dispara
        # respecto de las otras dos, el borde de la particula esta dentado y la
        # talla merece un vistazo.
        otras = []
        if det.feret_um:
            otras.append(f"<tr><td>{tr('Feret máximo (cuerda)')}</td><td>{det.feret_um:.0f}</td></tr>")
        if det.geodesico_um:
            otras.append(f"<tr><td>{tr('Geodésico (sigue la curva)')}</td><td>{det.geodesico_um:.0f}</td></tr>")
        if otras:
            cuenta += (f"<table class='data'><tr>{tr('<th>Medida</th><th>µm</th>')}</tr>"
                       f"{''.join(otras)}</table>"
                       f"<p>{tr('Se reporta la mayor de las dos primeras')}"
                       + (tr(", en este caso por <strong>{metodo}</strong>").format(
                           metodo=tr(det.metodo_largo)) if det.metodo_largo else "") + ".</p>")
    return (f"<h3>{NUM}.{titulo}</h3>"
            + tr("<p>A la izquierda la partícula; a la derecha, en verde, el "
                 "contorno que se midió sobre ella.</p>") + f"{img}{cuenta}")


def _fig_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> str:
    labels = list(class_names) + ["background"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(tr("Predicción")); ax.set_ylabel("Ground Truth")
    ax.set_title(tr("Matriz de confusión"))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_b64(fig)


def _tabla_ancha(filas: List[tuple], clases: List[str], titulo_col: str,
                hay_manual: bool, totales: bool = True) -> str:
    """Tabla Muestra x polimero con las columnas manual y modelo lado a lado.

    ``filas`` es una lista de (etiqueta, Counter manual, Counter modelo).
    """
    if hay_manual:
        cab = (f"<tr><th rowspan='2'>{_esc(titulo_col)}</th>"
               + "".join(f"<th colspan='2' style='text-align:center'>{_esc(tr(c))}</th>"
                         for c in clases)
               + "<th colspan='2' style='text-align:center'>Total</th></tr><tr>"
               + "".join("<th class='r'>manual</th><th class='r'>modelo</th>"
                         for _ in clases + ["total"]) + "</tr>")
    else:
        cab = (f"<tr><th>{_esc(titulo_col)}</th>"
               + "".join(f"<th class='r'>{_esc(tr(c))}</th>" for c in clases)
               + "<th class='r'>Total</th></tr>")

    cuerpo = ""
    acum_man, acum_det = Counter(), Counter()
    for etiqueta, man, det in filas:
        acum_man.update(man)
        acum_det.update(det)
        celdas = ""
        for c in clases:
            if hay_manual:
                celdas += f"<td class='r'>{man.get(c, 0)}</td><td class='r'>{det.get(c, 0)}</td>"
            else:
                celdas += f"<td class='r'>{det.get(c, 0)}</td>"
        t_det = sum(det.get(c, 0) for c in clases)
        if hay_manual:
            t_man = sum(man.get(c, 0) for c in clases)
            celdas += f"<td class='r'><strong>{t_man}</strong></td><td class='r'><strong>{t_det}</strong></td>"
        else:
            celdas += f"<td class='r'><strong>{t_det}</strong></td>"
        cuerpo += f"<tr><td>{etiqueta}</td>{celdas}</tr>"

    if totales and filas:
        celdas = ""
        for c in clases:
            if hay_manual:
                celdas += (f"<td class='r'>{acum_man.get(c, 0)}</td>"
                           f"<td class='r'>{acum_det.get(c, 0)}</td>")
            else:
                celdas += f"<td class='r'>{acum_det.get(c, 0)}</td>"
        t_det = sum(acum_det.get(c, 0) for c in clases)
        if hay_manual:
            t_man = sum(acum_man.get(c, 0) for c in clases)
            celdas += f"<td class='r'>{t_man}</td><td class='r'>{t_det}</td>"
        else:
            celdas += f"<td class='r'>{t_det}</td>"
        cuerpo += (f"<tr style='font-weight:700;background:var(--bg_soft)'>"
                   f"<td>TOTAL</td>{celdas}</tr>")

    return f"<table class='data'>{cab}{cuerpo}</table>"


def _seccion_conteo(resultados: Dict[int, list], state, active,
                    clases: List[str]) -> str:
    """Seccion 8: cuantas particulas de cada polimero hay en cada muestra.

    Es la tabla que se pide para el manuscrito: conteo por tipo de plastico y
    muestra, con el conteo manual y el del modelo uno junto al otro. **No** es
    evaluacion del detector: la columna del modelo son todas sus detecciones,
    sin descontar falsos positivos. Esa lectura esta en la seccion de errores.

    Con varios modelos cargados se usa el primero activo. Poner dos modelos en
    la misma celda haria la tabla ilegible, y el informe ya compara modelos en
    su propia seccion.
    """
    if not active:
        return ""
    main_mi = state.model_slots.index(active[0])
    rs = resultados.get(main_mi, [])
    if not rs:
        return ""

    rs = sorted(rs, key=lambda r: _clave_orden(r.image_path))
    hay_manual = any(r.has_gt for r in rs)

    # -- Por imagen --
    filas_img = []
    for r in rs:
        muestra = _muestra_de(r.image_path)
        etiqueta = _esc(r.image_path.name)
        if muestra:
            estacion, tramo, placa = muestra
            etiqueta = (f"{_esc(r.image_path.name)}<br>"
                        f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                        f"{_esc(estacion)} &middot; tramo {_esc(tramo)} &middot; placa {_esc(placa)}</span>")
        filas_img.append((etiqueta,
                          _conteo_por_clase(r.gt) if r.has_gt else Counter(),
                          _conteo_por_clase(r.predictions)))
    tabla_img = _tabla_ancha(filas_img, clases, "Muestra (imagen)", hay_manual)

    # -- Por tramo y por estacion, si los nombres siguen la pauta --
    agrupadas = [(r, _muestra_de(r.image_path)) for r in rs]
    parseables = [(r, m) for r, m in agrupadas if m is not None]
    tabla_tramo = tabla_estacion = ""
    nota_agrupacion = ""
    if parseables:
        por_tramo: Dict[tuple, list] = {}
        por_estacion: Dict[str, list] = {}
        for r, (estacion, tramo, _placa) in parseables:
            por_tramo.setdefault((estacion, tramo), []).append(r)
            por_estacion.setdefault(estacion, []).append(r)

        def _sumar(lista):
            man, det = Counter(), Counter()
            for r in lista:
                if r.has_gt:
                    man.update(_conteo_por_clase(r.gt))
                det.update(_conteo_por_clase(r.predictions))
            return man, det

        def _orden_est(e):
            return _ORDEN_ESTACION.index(e) if e in _ORDEN_ESTACION else 99

        filas_t = []
        for clave in sorted(por_tramo, key=lambda k: (_orden_est(k[0]), k[1])):
            estacion, tramo = clave
            lista = por_tramo[clave]
            man, det = _sumar(lista)
            palabra = tr("imágenes") if len(lista) != 1 else tr("imagen")
            filas_t.append((f"{_esc(estacion)} &middot; tramo {_esc(tramo)} "
                            f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                            f"({len(lista)} {palabra})</span>",
                            man, det))
        tabla_tramo = ("<h3>@@N@@.2 " + tr("Por tramo de profundidad") + "</h3>"
                       + tr("<p>Las placas del mismo tramo se <strong>suman</strong>: son "
                            "submuestras de la misma masa de sedimento, no repeticiones "
                            "fotográficas. El tramo es la unidad de análisis.</p>")
                       + _tabla_ancha(filas_t, clases, tr("Tramo"), hay_manual))

        filas_e = []
        for estacion in sorted(por_estacion, key=_orden_est):
            man, det = _sumar(por_estacion[estacion])
            filas_e.append((_esc(estacion), man, det))
        tabla_estacion = ("<h3>@@N@@.3 " + tr("Por estación") + "</h3>"
                          + _tabla_ancha(filas_e, clases, tr("Estación"), hay_manual,
                                         totales=False))

        fuera = len(agrupadas) - len(parseables)
        if fuera:
            nota_agrupacion = ("<p class='caption'>" + tr(
                "{n} imagen(es) no siguen la nomenclatura <code>tramo.testigo</code> y "
                "quedan fuera de las tablas agrupadas; sí están en la tabla por "
                "imagen.").format(n=fuera) + "</p>")

    coletilla = tr(", el primer modelo activo.") if len(active) > 1 else "."
    intro = (
        tr("<p>Partículas contadas en cada muestra, desglosadas por polímero. "
           "La columna <em>manual</em> es la anotación humana (Ground Truth) y la "
           "columna <em>modelo</em> son todas las detecciones de "
           "<strong>{alias}</strong>{coletilla}</p>").format(
               alias=_esc(active[0].alias), coletilla=coletilla)
        + tr("<p><strong>Léase como conteo, no como evaluación.</strong> La "
             "columna del modelo no descuenta falsos positivos ni empareja caja a caja: "
             "es cuántas partículas de cada polímero reportó. Coincidir en el total no "
             "implica haber acertado partícula por partícula; para eso está el análisis "
             "de errores.</p>")
        + "<h3>@@N@@.1 " + tr("Por imagen") + "</h3>"
    )
    return intro + tabla_img + nota_agrupacion + tabla_tramo + tabla_estacion


# ────────────────────────────────────────────────────────────────────
# ── Secciones del informe ────────────────────────────────────────────────────
# El orden de esta lista es el orden del informe, y la numeracion sale de aqui:
# si se desmarca una seccion las siguientes se renumeran solas, en vez de dejar
# un hueco ("4, 6, 7") que en un documento entregable se lee como un error.
#
# Anadir una seccion nueva es anadir una tupla: la interfaz dibuja sus casillas
# leyendo esta lista, de modo que no hay una segunda lista que mantener en
# sincronia.
SECCIONES = [
    ("resumen",     "Resumen"),
    ("metodos",     "Métodos"),
    ("calibracion", "Calibración de escala"),
    ("forma",       "Forma y talla de las partículas"),
    ("talla_carpetas", "Talla por carpeta y por foto (comparación entre sitios)"),
    ("profundidad", "Perfil en profundidad del testigo"),
    ("fichas",      "Ficha de partículas medidas (muestra)"),
    ("resultados",  "Resultados generales"),
    ("modelos",     "Resumen por modelo"),
    ("errores",     "Análisis de errores"),
    ("comparacion", "Comparación entre modelos"),
    ("galeria",     "Galería por imagen"),
    ("conteo",      "Conteo por muestra y tipo de plástico"),
    ("referencias", "Referencias bibliográficas"),
]

IDS_SECCIONES = [s[0] for s in SECCIONES]

# Combinaciones de uso frecuente. Existen porque el problema real no era que no
# se pudiera elegir, sino que elegir diez casillas una por una es tedioso y
# nadie lo hace: con un preset el informe corto queda a un clic.
PRESETS = {
    "completo": IDS_SECCIONES,
    "resumen": ["resumen", "calibracion", "forma", "conteo", "galeria"],
    "metodologico": ["resumen", "metodos", "calibracion", "forma", "fichas",
                     "modelos", "errores", "comparacion", "referencias"],
}

# Marca que se sustituye por el numero definitivo de la seccion al ensamblar.
# Los numeros no se pueden escribir a mano en los titulos porque dependen de que
# otras secciones se hayan pedido.
NUM = "@@N@@"


def _generate_report_impl(state, output_path: Path,
                    include_refs: bool = True,
                    include_gallery: bool = True,
                    max_gallery: int = 60,
                    max_fichas: int = 12,
                    solo_imagenes=None,
                    secciones=None) -> Path:
    """Genera el reporte HTML. `state` es un DetectorState con resultados.

    ``max_gallery`` limita cuántas imágenes se incrustan en la galería. Las
    imágenes van en base64 dentro del propio HTML, de modo que sin tope un lote
    grande generaba un archivo inabrible (14.5 MB con solo 4 imágenes).

    ``solo_imagenes`` acota el informe a un conjunto de rutas. Con ``None`` entra
    el trabajo completo. El filtro se aplica antes de calcular nada, de forma que
    los totales, los gráficos y la matriz de confusión describan exactamente el
    mismo subconjunto que se muestra; si se filtrara solo la galería, las cifras
    de arriba hablarían de un lote y las imágenes de otro.

    ``secciones`` es el conjunto de identificadores de :data:`SECCIONES` que se
    quieren incluir; ``None`` las incluye todas. Una sección pedida que no tenga
    contenido -- errores sin ground truth, por ejemplo -- se omite igualmente:
    marcarla no inventa datos que no existen.
    """
    pedidas = set(IDS_SECCIONES if secciones is None else secciones)
    # Las dos casillas antiguas siguen mandando si se pasan en False, para no
    # cambiar el significado de las llamadas que ya existian.
    if not include_refs:
        pedidas.discard("referencias")
    if not include_gallery:
        pedidas.discard("galeria")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    active = state.active_models()

    if solo_imagenes is None:
        resultados = dict(state.results)
    else:
        elegidas = {Path(p) for p in solo_imagenes}
        resultados = {mi: [r for r in rs if Path(r.image_path) in elegidas]
                      for mi, rs in state.results.items()}

    all_results = [r for rs in resultados.values() for r in rs]
    total_imgs = len({r.image_path for r in all_results})
    total_dets = sum(len(r.predictions) for r in all_results)
    confs = [p.conf for r in all_results for p in r.predictions]
    sizes = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
    avg_conf = sum(confs)/len(confs) if confs else 0
    avg_size = sum(sizes)/len(sizes) if sizes else 0
    any_gt = any(r.has_gt for r in all_results)

    clases_lote = _clases_vistas(all_results)

    # ── Distribución por clase ──
    per_class = Counter()
    for r in all_results:
        for p in r.predictions:
            per_class[p.class_name] += 1

    # ── Análisis de errores: matriz + métricas por clase (modelo principal) ──
    err_section = ""
    if any_gt and active:
        main_mi = state.model_slots.index(active[0])
        rs = resultados.get(main_mi, [])
        gts_only = [r.gt for r in rs if r.has_gt]
        preds_only = [r.predictions for r in rs if r.has_gt]
        cls_ids = sorted({d.class_id for lst in gts_only + preds_only for d in lst})
        cls_names = []
        for cid in cls_ids:
            name = next((d.class_name for lst in gts_only + preds_only for d in lst if d.class_id == cid), str(cid))
            cls_names.append(name)
        cm = confusion_matrix(preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp)
        per_class_metrics = aggregate_per_class(preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp)

        cm_img = _fig_confusion_matrix(cm, cls_names)
        rows = ""
        for cid, name in zip(cls_ids, cls_names):
            cm_ = per_class_metrics[cid]
            rows += (
                f"<tr><td>{name}</td>"
                f"<td class='r'>{cm_.tp}</td><td class='r'>{cm_.fp}</td><td class='r'>{cm_.fn}</td>"
                f"<td class='r'>{cm_.precision:.3f}</td><td class='r'>{cm_.recall:.3f}</td>"
                f"<td class='r'>{cm_.f1:.3f}</td></tr>"
            )
        # Las clases que el modelo conoce pero que no aparecen ni en el GT ni en
        # las predicciones se quedan fuera de la tabla, porque no tienen metrica
        # que calcular. Se declaran igual: un informe que anuncia PET/PP/LDPE y
        # luego muestra dos filas parece roto, y el lector no puede distinguir
        # "ausente del lote" de "se perdio por el camino".
        conocidas = dict(getattr(getattr(active[0], "loaded", None), "names", {}) or {})
        ausentes = [n for cid, n in sorted(conocidas.items()) if cid not in set(cls_ids)]
        nota_ausentes = ""
        if ausentes:
            nota_ausentes = (
                "<p class='caption' style='text-align:left'>"
                + tr("Sin fila para {clases}: no aparece ni en la anotación manual ni "
                     "entre las predicciones de este lote, de modo que no hay métrica "
                     "que informar. El modelo sí está entrenado para esa clase.").format(
                         clases=_esc(", ".join(ausentes)))
                + "</p>")

        err_section = f"""
        
        <h3>@@N@@.1 {tr('Matriz de confusión')}</h3>
        <div class='fig'><img src='data:image/png;base64,{cm_img}' />
            <div class='caption'>{tr('Figura. Matriz de confusión (modelo principal: {modelo}, IoU = {iou}).').format(modelo=_esc(active[0].alias), iou=state.params.iou_tp)}</div></div>
        <h3>@@N@@.2 {tr('Precisión / Recall / F1 por clase')}</h3>
        <table class='data'><tr>{tr('<th>Clase</th>')}<th>{LABEL_TP}</th><th>{LABEL_FP}</th><th>{LABEL_FN}</th>
        {tr('<th>Precisión</th><th>Recall</th><th>F1</th>')}</tr>{rows}</table>
        {nota_ausentes}
        """

    # ── Resumen por modelo (tabla comparativa) ──
    #
    # Se reportan DOS F1, y la diferencia entre ambos no es un detalle:
    #
    #   localizacion  tp/(tp+fp) y tp/(tp+fn). Las cajas bien situadas pero con
    #                 la clase equivocada quedan fuera de ambos denominadores,
    #                 asi que mide solo si el detector encuentra la particula.
    #   con clase     esas mismas cajas cuentan como falso positivo de la clase
    #                 predicha y falso negativo de la real, que es lo estricto.
    #
    # Publicar solo el primero sobreestima el desempeno, y ademas contradice la
    # tabla por clase, que si penaliza la confusion. Se declaran los dos.
    def _pr_f1(tp: int, fp: int, fn: int):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)

    rows_models = ""
    total_mc = 0
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None: continue
        rs = resultados.get(mi, [])
        n_img = len({r.image_path for r in rs})
        n_det = sum(len(r.predictions) for r in rs)
        cf = [p.conf for r in rs for p in r.predictions]
        tp = sum(r.tp for r in rs); fp = sum(r.fp for r in rs)
        fn = sum(r.fn for r in rs); mc = sum(r.miscls for r in rs)
        total_mc += mc
        _, _, f1_loc = _pr_f1(tp, fp, fn)
        _, _, f1_cls = _pr_f1(tp, fp + mc, fn + mc)
        avg_cf = (sum(cf)/len(cf)) if cf else 0
        hay_gt = any(r.has_gt for r in rs)
        c_loc = f"{f1_loc:.3f}" if hay_gt else "—"
        c_cls = f"{f1_cls:.3f}" if hay_gt else "—"
        rows_models += (
            f"<tr><td>{_esc(slot.alias)}</td><td class='r'>{n_img}</td><td class='r'>{n_det}</td>"
            f"<td class='r'>{avg_cf:.3f}</td>"
            f"<td class='r'>{tp}</td><td class='r'>{fp}</td><td class='r'>{fn}</td>"
            f"<td class='r'>{mc}</td>"
            f"<td class='r'>{c_loc}</td><td class='r'>{c_cls}</td></tr>"
        )

    # Nota que acompana la tabla, con las cifras del propio lote.
    nota_miscls = ""
    if any_gt:
        tp_g = sum(r.tp for r in all_results)
        fp_g = sum(r.fp for r in all_results)
        fn_g = sum(r.fn for r in all_results)
        p_loc, r_loc, f_loc = _pr_f1(tp_g, fp_g, fn_g)
        p_cls, r_cls, f_cls = _pr_f1(tp_g, fp_g + total_mc, fn_g + total_mc)
        nota_miscls = tr(
            "<p><strong>Los dos F1 miden cosas distintas.</strong> "
            "<em>Localización</em> responde si el detector encuentra la partícula "
            "(P {pl} · R {rl} · <strong>F1 {fl}</strong>). "
            "<em>Con clase</em> exige además acertar el polímero, contando cada "
            "caja mal clasificada como falso positivo de la clase predicha y "
            "falso negativo de la real "
            "(P {pc} · R {rc} · <strong>F1 {fc}</strong>).</p>"
            "<p>La diferencia corresponde a <strong>{mc}</strong> "
            "partícula(s) bien localizada(s) pero asignada(s) a la clase "
            "incorrecta. Es la cifra que concilia esta tabla con la de "
            "precisión por clase de la sección de errores.</p>").format(
                pl=f"{p_loc:.3f}", rl=f"{r_loc:.3f}", fl=f"{f_loc:.3f}",
                pc=f"{p_cls:.3f}", rc=f"{r_cls:.3f}", fc=f"{f_cls:.3f}",
                mc=total_mc)

    # ── Veredicto y barrido de confianza ──
    veredicto_html = ""
    if any_gt:
        barrido = _barrido_confianza(resultados, state, state.params.iou_tp,
                                     state.params.conf)
        if barrido:
            resumen = []
            for alias, filas in barrido.items():
                mejor = max(filas, key=lambda f: f[3])
                actual = filas[0]
                resumen.append((alias, actual, mejor))

            # Veredicto: gana quien tenga mejor F1 con clase en su mejor umbral.
            resumen.sort(key=lambda x: x[2][3], reverse=True)
            g_alias, g_act, g_mej = resumen[0]
            texto = tr("<p><strong>Mejor desempeño: {alias}</strong>, con F1 {f1} "
                       "al umbral {umbral} (P {p} · R {r}).</p>").format(
                           alias=_esc(g_alias), f1=f"{g_mej[3]:.3f}", umbral=f"{g_mej[0]:.2f}",
                           p=f"{g_mej[1]:.3f}", r=f"{g_mej[2]:.3f}")
            if len(resumen) > 1:
                p_alias, _, p_mej = resumen[-1]
                d = g_mej[3] - p_mej[3]
                texto += tr(
                    "<p>La diferencia con {otro} es de <strong>{d}</strong> de F1. Con "
                    "un solo entrenamiento por arquitectura, una diferencia pequeña no "
                    "distingue el diseño de la red del azar de inicialización: haría "
                    "falta repetir con distintas semillas para afirmar que una es "
                    "superior.</p>").format(otro=_esc(p_alias), d=f"{d:.3f}")

            veredicto_html = texto

    # ── Comparación entre modelos, foto por foto ──
    # La tabla global de arriba dice cual modelo gana en total; esta dice en
    # cuales fotos gana. Con lotes desiguales como los del Loa, donde un tramo
    # concentra la mayoria de las particulas, el total lo decide ese tramo y
    # puede esconder que en el resto el otro modelo va mejor.
    compare_html = ""
    activos = [mi for mi, s in enumerate(state.model_slots) if s.path is not None]
    if len(activos) > 1:
        por_foto = {}
        for mi in activos:
            for r in resultados.get(mi, []):
                por_foto.setdefault(r.image_path, {})[mi] = r

        hay_gt = any(r.has_gt for rs in resultados.values() for r in rs)
        cab = "<tr><th>Imagen</th>"
        for mi in activos:
            alias = state.model_slots[mi].alias
            cab += (f"<th>{_esc(alias)}<br><span style='font-weight:400;font-size:8.5pt'>dets</span></th>")
            if hay_gt:
                cab += (f"<th><span style='font-weight:400;font-size:8.5pt'>"
                        f"{LABEL_TP}/{LABEL_FP}/{LABEL_FN}</span></th>")
        cab += "</tr>"

        filas = ""
        for ruta in sorted(por_foto):
            filas += f"<tr><td>{_esc(Path(ruta).name)}</td>"
            for mi in activos:
                r = por_foto[ruta].get(mi)
                if r is None:
                    filas += "<td class='r'>—</td>"
                    if hay_gt:
                        filas += "<td class='r'>—</td>"
                    continue
                filas += f"<td class='r'>{len(r.predictions)}</td>"
                if hay_gt:
                    celda = (f"{r.tp}/{r.fp}/{r.fn}" if r.has_gt else "—")
                    filas += f"<td class='r'>{celda}</td>"
            filas += "</tr>"

        # Veredicto: solo se declara si hay ground truth con que juzgar. Sin el,
        # mas detecciones no es mejor, y decir lo contrario seria enganoso.
        veredicto = ""
        if hay_gt:
            marcador = []
            for mi in activos:
                rs = resultados.get(mi, [])
                tp = sum(r.tp for r in rs); fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs)
                pr = tp/(tp+fp) if (tp+fp) else 0
                rc = tp/(tp+fn) if (tp+fn) else 0
                marcador.append((2*pr*rc/(pr+rc) if (pr+rc) else 0,
                                 state.model_slots[mi].alias))
            marcador.sort(reverse=True)
            mejor_f1, mejor = marcador[0]
            # Este F1 es el permisivo (localizacion). Decia solo "F1 global" y
            # convivia con el veredicto estricto de arriba: dos numeros distintos
            # llamados igual en la misma seccion.
            veredicto = tr(
                "<p><b>Mejor F1 de localización:</b> {alias} ({f1}) — encontrar la "
                "partícula, sin exigir que acierte el polímero. El veredicto con clase "
                "está más arriba.</p>").format(alias=_esc(mejor), f1=f"{mejor_f1:.3f}")
        else:
            veredicto = ("<p class='caption' style='text-align:left'>"
                         + tr("Sin ground truth no se puede declarar un ganador: un "
                              "modelo con más detecciones puede estar acertando o "
                              "inventando. Carga anotaciones para obtener F1 por "
                              "modelo.") + "</p>")

        # ── Figuras de comparación ──
        figuras_cmp = ""
        if hay_gt:
            datos_barras = []
            por_clase_modelo: Dict[str, Dict[str, tuple]] = {}
            for mi in activos:
                alias = state.model_slots[mi].alias
                rs = [r for r in resultados.get(mi, []) if r.has_gt]
                tp = sum(r.tp for r in rs); fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs); mc = sum(r.miscls for r in rs)
                p_l, r_l, f_l = _pr_f1(tp, fp, fn)
                p_c, r_c, f_c = _pr_f1(tp, fp + mc, fn + mc)
                datos_barras.append((alias, p_l, r_l, f_l, p_c, r_c, f_c))

                gts_m = [r.gt for r in rs]
                preds_m = [r.predictions for r in rs]
                ids = sorted({d.class_id for lst in gts_m + preds_m for d in lst})
                if ids:
                    metricas = aggregate_per_class(preds_m, gts_m, ids,
                                                   iou_thr=state.params.iou_tp)
                    nombres_cls = {
                        cid: next((d.class_name for lst in gts_m + preds_m
                                   for d in lst if d.class_id == cid), str(cid))
                        for cid in ids}
                    por_clase_modelo[alias] = {
                        nombres_cls[cid]: (metricas[cid].precision,
                                           metricas[cid].recall,
                                           metricas[cid].f1) for cid in ids}

            f_barras = _fig_comparacion_metricas(datos_barras)
            if f_barras:
                figuras_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_barras}' />"
                    "<div class='caption'>"
                    + tr("Figura. Precisión, Recall y F1 de cada modelo al umbral "
                         "{umbral}, en los dos criterios. La distancia entre paneles es "
                         "la confusión entre polímeros.").format(
                             umbral=f"{state.params.conf:g}")
                    + "</div></div>")

            f_clase = _fig_metricas_por_clase(por_clase_modelo)
            if f_clase:
                figuras_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_clase}' />"
                    "<div class='caption'>"
                    + tr("Figura. F1 por clase. Un modelo puede ganar en el agregado y "
                         "perder en el polímero que interesa.")
                    + "</div></div>")

            # Acuerdo foto a foto entre los dos primeros modelos.
            if len(activos) >= 2:
                a, b = activos[0], activos[1]
                pares = []
                for ruta in sorted(por_foto):
                    ra, rb = por_foto[ruta].get(a), por_foto[ruta].get(b)
                    if ra is not None and rb is not None:
                        pares.append((len(ra.predictions), len(rb.predictions),
                                      Path(ruta).name))
                f_acuerdo = _fig_acuerdo_por_imagen(
                    state.model_slots[a].alias, state.model_slots[b].alias, pares)
                if f_acuerdo:
                    figuras_cmp += (
                        f"<div class='fig'><img src='data:image/png;base64,{f_acuerdo}' />"
                        "<div class='caption'>"
                        + tr("Figura. Detecciones por imagen de un modelo frente al "
                             "otro; la diagonal es el acuerdo exacto. Escala "
                             "simétrica-logarítmica, porque casi todas las placas "
                             "tienen pocas partículas y unas pocas cientos.")
                        + "</div></div>")

            # Curva de F1 frente al umbral: sostiene la eleccion del punto de
            # operacion con evidencia, en vez de afirmarla.
            barrido = _barrido_confianza(resultados, state, state.params.iou_tp,
                                         state.params.conf)
            f_barrido = _fig_barrido(barrido)
            if f_barrido:
                figuras_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_barrido}' />"
                    "<div class='caption'>"
                    + tr("Figura. F1 (con clase) frente al umbral de confianza; la "
                         "estrella marca el máximo de cada modelo. La curva arranca en "
                         "{umbral}, el umbral con que se ejecutó: por debajo las "
                         "detecciones no se calcularon.").format(
                             umbral=f"{state.params.conf:g}")
                    + "</div></div>")

        compare_html = (
            tr("<p>Detecciones de cada modelo imagen por imagen. El total puede "
               "estar dominado por unas pocas fotos densas, así que conviene mirar "
               "el detalle antes de elegir modelo.</p>")
            + veredicto
            + figuras_cmp
            + f"<table class='data'>{cab}{filas}</table>")
    elif len(activos) == 1:
        compare_html = ("<p class='caption' style='text-align:left'>"
                        + tr("Solo se ejecutó un modelo, así que no hay comparación. "
                             "Carga un segundo modelo en la pestaña Modelos para "
                             "compararlos sobre las mismas imágenes.") + "</p>")

    # ── Galería comparativa: Predicción vs Ground Truth (lado a lado) ──
    gallery_html = ""
    omitidas = 0
    if include_gallery:
        blocks = []
        # Tope de imágenes en galería: cada una se incrusta en base64, así que
        # sin límite un lote de varios cientos produce un HTML de gigabytes.
        # Se agrupa por imagen y no por modelo. Aplanando modelo a modelo, el
        # tope se agotaba dentro del primero y el segundo no aparecia nunca en
        # la galeria: con dos modelos cargados solo se veian las fotos de uno,
        # que es justo la comparacion visual que se quiere. Ahora el tope cuenta
        # imagenes, y de cada una entran todos los modelos, uno bajo el otro.
        por_imagen: dict = {}
        for mi in sorted(resultados):
            for r in resultados[mi]:
                por_imagen.setdefault(r.image_path, []).append((mi, r))
        # Orden por estación/tramo/placa cuando el nombre de la placa lo
        # permite: el informe se lee siguiendo el testigo hacia abajo, no en el
        # orden arbitrario en que se cargaron los archivos.
        orden_imgs = sorted(por_imagen, key=_clave_orden)
        candidatas = []
        for ruta in orden_imgs[:max_gallery]:
            candidatas.extend(por_imagen[ruta])
        omitidas = max(0, len(por_imagen) - max_gallery)
        for mi, r in candidatas:
            slot = state.model_slots[mi]
            if True:
                # Imagen de predicción (preferimos pred_png; si no, la combinada)
                pred_bytes = getattr(r, "pred_png", None) or r.annotated_png
                if not pred_bytes:
                    continue
                left = (
                    f"<figure><img src='{_uri_galeria(pred_bytes)}' />"
                    f"<figcaption>{tr('Predicción del modelo')} · {_esc(slot.alias)}"
                    f"<span class='sub'>"
                    + tr("{n} detección(es) por YOLO").format(n=len(r.predictions))
                    + "</span>"
                    f"</figcaption></figure>"
                )

                # Imagen de Ground Truth (control)
                gt_bytes = getattr(r, "gt_png", None)
                if r.has_gt and gt_bytes:
                    right = (
                        f"<figure><img src='{_uri_galeria(gt_bytes)}' />"
                        f"<figcaption>Ground Truth (control)"
                        f"<span class='sub'>{len(r.gt)} etiqueta(s) reales</span>"
                        f"</figcaption></figure>"
                    )
                else:
                    right = (
                        f"<figure><div class='nogt'>"
                        f"{tr('Sin Ground Truth para esta imagen')}</div>"
                        f"<figcaption>Ground Truth (control)"
                        f"<span class='sub'>{tr('no disponible')}</span>"
                        f"</figcaption></figure>"
                    )

                # Encabezado de la foto: nombre y, si el nombre lo permite,
                # de qué tramo y placa viene.
                muestra = _muestra_de(r.image_path)
                donde = ""
                if muestra:
                    estacion, tramo, placa = muestra
                    donde = (f" &nbsp;<span class='tag'>{_esc(estacion)} · tramo {_esc(tramo)} "
                             f"· placa {_esc(placa)}</span>")
                meta = (f"<strong>{_esc(r.image_path.name)}</strong>{donde}"
                        f"&nbsp;<span class='tag'>modelo: {_esc(slot.alias)}</span>")

                # Tabla de la foto: partículas por polímero. Los aciertos y
                # fallos emparejados caja a caja son otra pregunta y viven en
                # la sección de análisis de errores.
                tabla = _tabla_conteo(_conteo_por_clase(r.gt) if r.has_gt else None,
                                      _conteo_por_clase(r.predictions),
                                      clases_lote)

                blocks.append(
                    f"<div class='compare'><div class='compare-pair'>{left}{right}</div>"
                    f"<div class='compare-meta'>{meta}</div>{tabla}</div>"
                )

        if blocks:
            nota = ""
            if omitidas:
                n_mostradas = len({r.image_path for _, r in candidatas})
                nota = ("<p class='caption'>" + tr(
                    "Se muestran las primeras {n} imágenes, cada una con todos los "
                    "modelos; {fuera} quedaron fuera de la galería para que el archivo "
                    "siga siendo manejable. Las métricas de las secciones anteriores "
                    "sí incluyen todas.").format(n=n_mostradas, fuera=omitidas)
                        + "</p>")
            gallery_html = (
                tr("<p>Cada bloque muestra, a la izquierda, las detecciones del "
                   "modelo (<em>bounding boxes</em> dibujadas por YOLO con su clase y "
                   "confianza) y, a la derecha, las etiquetas reales de control "
                   "(<em>Ground Truth</em>). Esta vista lado a lado permite evaluar "
                   "visualmente dónde acertó o falló el modelo.</p>")
                + nota + "".join(blocks)
            )

    # ── Figuras agregadas ──
    fig_classes = _fig_class_distribution(dict(per_class))
    fig_conf = _fig_confidence_hist(confs)
    fig_size = _fig_size_hist(sizes) if sizes else ""

    figures_html = ""
    if fig_classes:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_classes}' /><div class='caption'>{tr('Figura. Distribución de detecciones por clase.')}</div></div>"
    if fig_conf:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_conf}' /><div class='caption'>{tr('Figura. Histograma de confianza.')}</div></div>"
    if fig_size:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_size}' /><div class='caption'>{tr('Figura. Distribución de tamaños (diámetro equivalente en μm).')}</div></div>"

    # ── Métodos ──
    p = state.params
    # Cuando la calibracion es por foto (medir_placa) no hay UN valor de
    # um_per_px que reportar aqui -- p.um_per_px se queda en 0 a proposito.
    # En vez de un "—" que no dice nada, se da la media del lote con su
    # intervalo de confianza del 95%: un numero que sirve para citar en un
    # metodo aunque la calibracion real sea foto a foto (eso se explica en
    # detalle en la seccion "Calibracion de escala", con el minimo y el
    # maximo que muestran cuanto varia).
    if p.um_per_px > 0:
        um_px_txt = f"{p.um_per_px:g}"
    else:
        _ums_metodos = [c.um_por_px for c in (getattr(state, "calibraciones", None) or {}).values()
                        if getattr(c, "valida", False)]
        _mic = _media_ic95(_ums_metodos)
        if _mic:
            _media_m, _margen_m, _n_m = _mic
            um_px_txt = f"{_media_m:.4f} ± {_margen_m:.4f} (IC 95%, n={_n_m} fotos)"
        elif _ums_metodos:
            um_px_txt = (f"{_ums_metodos[0]:.4f} "
                         + tr("(una sola foto calibrada, sin IC)"))
        else:
            um_px_txt = "—"
    methods_rows = [
        (tr("Modelos cargados"), _esc(", ".join(s.alias for s in active)) or "—"),
        (tr("Confianza mínima"), f"{p.conf:g}"),
        ("IoU NMS", f"{p.iou_nms:g}"),
        (tr("IoU para emparejar Verdaderos Positivos"), f"{p.iou_tp:g}"),
        (tr("Tamaño de imagen (imgsz)"), f"{p.imgsz}"),
        (tr("Dispositivo"), p.device),
        (tr("μm por píxel"), um_px_txt),
        (tr("Filtro tamaño (μm)"), f"{p.size_min_um} – {p.size_max_um}" if (p.size_min_um or p.size_max_um) else tr("sin filtro")),
        (tr("Imágenes procesadas"), str(total_imgs)),
        (tr("Total de detecciones"), str(total_dets)),
    ]
    # Abstención: cuántas partículas se contaron sin atribuirles polímero. Va en
    # Métodos y no enterrado en una tabla de resultados porque es una decisión
    # del análisis, y quien lea el informe necesita saber que se tomó.
    _res_asig = resumir([d for r in all_results for d in r.predictions],
                        getattr(p, "conf_asignacion", 0.0))
    if _res_asig.activa:
        methods_rows.append((
            tr("Confianza mínima para asignar polímero"),
            f"{p.conf_asignacion:g}"))
        methods_rows.append((
            tr("Partículas sin polímero asignable"),
            f"{_res_asig.no_asignables} / {_res_asig.total} "
            f"({_res_asig.porcentaje_no_asignable:.1f} %)"))
    methods_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in methods_rows)
    equipo_html = _seccion_equipo(active)

    # Párrafo de métodos redactado, listo para copiar al manuscrito
    # (reproducibilidad: modelo, versión y parámetros exactos).
    try:
        import ultralytics
        _ul_ver = ultralytics.__version__
    except Exception:
        _ul_ver = "—"
    _model_names = _esc(", ".join(s.alias for s in active)) if active else "—"
    methods_para = (
        "<p><strong>" + tr("Resumen de la configuración empleada:") + "</strong></p>"
        "<blockquote style='border-left:3px solid var(--accent); margin:8px 0; "
        "padding:6px 14px; color:#424a53; background:#f6f8fa; border-radius:0 6px 6px 0;'>"
        + tr("La detección automatizada se realizó con el modelo YOLO «{modelos}» "
             "(Ultralytics {ul}) a una resolución de entrada de {imgsz} px, umbral de "
             "confianza {conf} y supresión de no-máximos con IoU {iou}. ").format(
                 modelos=_model_names, ul=_ul_ver, imgsz=p.imgsz,
                 conf=f"{p.conf:g}", iou=f"{p.iou_nms:g}")
        + (tr("Las métricas de error se calcularon contra anotación manual "
              "independiente, emparejando predicciones y etiquetas con IoU ≥ {iou}. "
              ).format(iou=f"{p.iou_tp:g}") if any_gt else "")
        + (tr("La calibración óptica fue de {um} μm/píxel. ").format(
            um=f"{p.um_per_px:g}") if p.um_per_px > 0 else "")
        + tr("Se procesaron {imgs} imágenes con un total de {dets} detecciones."
             ).format(imgs=total_imgs, dets=total_dets)
        + "</blockquote>"
    )

    conteo_html = _seccion_conteo(resultados, state, active, clases_lote)

    refs_html = ""
    if include_refs:
        refs_html = """
        
        <ol>
          <li>Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).
              Development of a new methodology for the determination of PET microplastics in sediment,
              based on microwave-assisted acid digestion.
              <em>PLoS ONE</em> 19(12): e0314520.
              <a href='https://doi.org/10.1371/journal.pone.0314520'>https://doi.org/10.1371/journal.pone.0314520</a></li>
          <li>Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).
              Evaluation of microwave-assisted acid/oxidant digestion method for the detection of
              polyethylene microplastics in <em>Merluccius gayi</em> fish by Nile Red fluorescent
              staining and image analysis. <em>J. Chil. Chem. Soc.</em> 69(1): 6082-6085.
              <a href='https://doi.org/10.4067/s0717-97072024000106082'>https://doi.org/10.4067/s0717-97072024000106082</a></li>
        </ol>
        """

    # ── Calibración de escala ──
    # Se reporta la procedencia, no solo el número: un tamaño en µm sin decir
    # contra qué patrón se midió no es verificable. Si no hay calibración la
    # sección no se emite, en vez de mostrar un "—" que parece un dato.
    calib_html = ""
    cals = getattr(state, "calibraciones", None) or {}
    if cals:
        from .calibracion import (resumen_lote, ORIGEN_PLACA,
                                  ESPESOR_PARED_MM, SESGO_MAXIMO_PARED_PCT)
        res = resumen_lote(list(cals.values()))
        if res["n"]:
            etiqueta = {"placa": tr("medida sobre la placa Petri de esta foto"),
                        "indice": tr("heredada del recorte (índice de calibración)"),
                        "manual": tr("introducida a mano en Parámetros")}
            filas = "".join(
                f"<tr><td>{etiqueta.get(o, o)}</td>"
                f"<td>{k} {tr('imágenes') if k != 1 else tr('imagen')}</td></tr>"
                for o, k in sorted(res["origenes"].items()))
            aviso = ""
            if res["variacion"] > 1.05:
                aviso = tr(
                    "<p><strong>La escala no es común a todo el lote:</strong> varía "
                    "{var}× entre la foto más cercana y la más lejana. Por eso cada "
                    "imagen se convierte con su propio factor; usar uno solo para todas "
                    "daría tamaños con hasta un {pct}% de error.</p>").format(
                        var=f"{res['variacion']:.2f}",
                        pct=f"{100 * (res['variacion'] - 1):.0f}")
            _mic_calib = _media_ic95(
                [c.um_por_px for c in cals.values() if getattr(c, "valida", False)])
            _fila_media = ""
            if _mic_calib:
                _media_c, _margen_c, _n_c = _mic_calib
                _fila_media = (f"<tr>{tr('<td>media ± IC 95%</td>')}"
                               f"<td>{_media_c:.4f} ± {_margen_c:.4f} (n={_n_c})</td></tr>")
            calib_html = f"""
<table class='data'><tr>{tr('<th>Procedencia de la escala</th><th>Imágenes</th>')}</tr>{filas}</table>
<table class='data'>
<tr>{tr('<th>µm por píxel</th><th>Valor</th>')}</tr>
<tr>{tr('<td>mínimo</td>')}<td>{res['min']:.4f}</td></tr>
<tr>{tr('<td>mediana</td>')}<td>{res['mediana']:.4f}</td></tr>
<tr>{tr('<td>máximo</td>')}<td>{res['max']:.4f}</td></tr>
{_fila_media}
</table>
{aviso}
<p>{tr('El patrón de longitud es el diámetro externo nominal de la placa Petri. El radio en píxeles se obtiene ajustando un círculo por mínimos cuadrados al borde del anillo muestreado en 720 direcciones, con rechazo de atípicos; la transformada de Hough solo aporta el centro aproximado, porque su radio llega a errar un 12&nbsp;% y ese error entraría entero en todos los tamaños reportados.')}</p>

<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;background:var(--bg_soft)'>
{tr('<strong>Incertidumbre de la escala, y en qué dirección.</strong> El anillo de la placa tiene una pared de unos <strong>{pared:g}&nbsp;mm</strong> (medido sobre las fotos de este estudio: el borde interno cae en 0,960 del radio ajustado y el externo en 1,000). El diámetro nominal de una placa Petri es ambiguo a ese nivel: puede referirse al diámetro <em>externo</em> o al <em>útil interior</em>. Aquí se toma el <strong>externo</strong>, que es el borde al que ajusta el círculo. Si el nominal se refiriera al interior, la escala correcta sería un <strong>{sesgo:g}&nbsp;% mayor</strong> y todas las tallas de este informe estarían <strong>subestimadas</strong> en esa cifra. El sesgo solo puede ir en ese sentido, porque el externo es el mayor de los dos bordes posibles.').format(pared=ESPESOR_PARED_MM, sesgo=SESGO_MAXIMO_PARED_PCT)}</p>
{_figura_calibracion(state)}"""
            if res["avisos"]:
                items = "".join(f"<li>{a}</li>" for a in res["avisos"][:12] if a)
                if items:
                    calib_html += (f"<p>{tr('Incidencias durante la calibración:')}"
                                   f"</p><ul>{items}</ul>")
    elif p.um_per_px > 0:
        calib_html = tr(
            "<p>Escala única para todo el lote, introducida a mano: "
            "<strong>{um} µm/píxel</strong>. No se midió ninguna placa, "
            "de modo que este factor no está trazado a un patrón de longitud y "
            "cualquier variación en la distancia de disparo entre fotos queda sin "
            "corregir.</p>").format(um=f"{p.um_per_px:g}")

    # ── Forma y talla ──
    # Se separa de "Resultados" porque responde otra pregunta: no cuantas
    # particulas hay, sino que forma tienen. El morfotipo y la curvatura son
    # variables que la literatura de microplasticos reporta y que la caja del
    # detector no puede dar.
    forma_html = ""

    # ── Lo que cae sobre el anillo de la placa no es muestra ──
    # El aro es una banda brillante y continua, y el detector la confunde con
    # material. Como ademas sale enorme, entraba entre las mayores: en un
    # informe ocupo una ficha de ejemplo como "PET fibra" de 5978 um, que es
    # mas que el limite de 5 mm con que se define un microplastico.
    #
    # El circulo ajustado ya se conocia -- de el sale la escala -- y solo hacia
    # falta preguntarle. Se aparta de la TALLA y de las fichas, no del conteo:
    # quitar detecciones del recuento es una decision del estudio y no del
    # generador del informe. Lo que si se hace es declarar cuantas se apartaron.
    from .calibracion import sobre_el_anillo, FRACCION_AREA_UTIL

    _cals = getattr(state, "calibraciones", None) or {}

    def _en_el_aro(det, ruta) -> bool:
        return sobre_el_anillo(_cals.get(Path(ruta).name),
                               det.x1, det.y1, det.x2, det.y2)

    # Cada particula con su imagen de origen, para poder ir a buscarla luego.
    _con_forma = [(p_, r.image_path) for r in all_results for p_ in r.predictions
                  if getattr(p_, "aspecto", None)]
    formas = [(d, ruta) for d, ruta in _con_forma if not _en_el_aro(d, ruta)]
    n_en_el_aro = len(_con_forma) - len(formas)
    n_sin_forma = sum(1 for r in all_results for p_ in r.predictions
                      if getattr(p_, "aspecto", None) is None)
    ficha_mayor = None
    ficha_menor = None
    tabla_morfotipo = tablas = recuento = metodo = ejemplo = ""
    # Independiente de "forma": es su propia seccion opcional en el informe
    # (ver SECCIONES mas abajo), asi que no entra en forma_html.
    por_carpeta_html = ""
    # El perfil por profundidad solo existe si los nombres llevan el tramo.
    profundidad_html = seccion_profundidad(formas, all_results) if formas else ""
    if formas:
        con_talla = [(d, ruta) for d, ruta in formas if d.largo_um]
        fibras = sum(1 for d, _ in formas if d.morfotipo == "fibra")
        curvas = sum(1 for d, _ in formas if d.curvatura and d.curvatura >= 1.15)
        pc_f = 100.0 * fibras / len(formas)

        # ── Particulas cuya talla pide un vistazo ──
        # Se declara en la seccion, no solo en cada ficha: quien lee las cifras
        # de talla tiene que saber cuantas de ellas se apoyaron en una mascara
        # que hubo que recortar. Afecta sobre todo al extremo grande de la
        # distribucion, que es justo el que se reporta como "la particula
        # mayor".
        #
        # OJO CON EL SENTIDO DEL SESGO: mientras la mascara no se recortaba, la
        # fuga inflaba la talla y el sesgo iba hacia ARRIBA. Ahora el recorte la
        # acota, asi que lo reportado es el trozo de dentro de la caja y el
        # sesgo va hacia ABAJO. El texto de abajo lo dice en ese sentido.
        n_revisar = sum(1 for d, _ in formas if getattr(d, "revisar", False))
        aviso_revisar = ""
        if n_revisar:
            # Un decimal por debajo del 1%: con ":.0f" una proporcion de 4 sobre
            # 909 salia como "(0 %)" junto a "4 particulas", que se lee como si
            # la cifra estuviera rota.
            _pc = 100.0 * n_revisar / len(formas)
            _pc_txt = f"{_pc:.1f}" if _pc < 1 else f"{_pc:.0f}"
            aviso_revisar = (
                f"<p style='border-left:3px solid {T.DOC.WARN};padding:6px 12px;"
                f"background:#fff8e6'>"
                + tr("<strong>{n} de {total} partículas ({pct}&nbsp;%) tienen una talla "
                     "que pide comprobación</strong>: al segmentarlas, la máscara se "
                     "salía bastante más allá de la caja del detector, casi siempre "
                     "porque el umbral enganchó una partícula vecina o una zona "
                     "brillante del fondo. En esos casos la máscara se <em>recorta a la "
                     "caja</em>, de modo que la talla reportada no está inflada; lo que "
                     "puede es quedarse <em>corta</em>, si la partícula seguía de verdad "
                     "más allá de la caja. Van marcadas con ⚠ en sus fichas.").format(
                         n=n_revisar, total=len(formas), pct=_pc_txt)
                + "</p>")

        # ── Lo apartado por caer sobre el aro ──
        if n_en_el_aro:
            aviso_revisar += (
                f"<p style='border-left:3px solid {T.DOC.INK3};padding:6px 12px;"
                f"background:var(--bg_soft)'>"
                + tr("<strong>{n} detección(es) caen sobre el anillo de la placa</strong> "
                     "—más allá del {pct} % del radio ajustado— y quedan fuera de esta "
                     "sección: el aro es una banda brillante que el detector confunde "
                     "con material, y como sale enorme desplazaría a las partículas de "
                     "verdad del extremo grande. Siguen contadas en la sección de "
                     "conteo.").format(n=n_en_el_aro, pct=f"{100 * FRACCION_AREA_UTIL:.0f}")
                + "</p>")

        # ── Reparto por morfotipo ──
        tabla_morfotipo = aviso_revisar + f"""
<table class='data'><tr>{tr('<th>Morfotipo</th><th>Cuántas</th><th>%</th>')}</tr>
<tr>{tr('<td>Fibra (relación de aspecto ≥ 3)</td>')}<td>{fibras}</td><td>{pc_f:.1f} %</td></tr>
<tr>{tr('<td>Partícula (no fibrosa)</td>')}<td>{len(formas) - fibras}</td><td>{100 - pc_f:.1f} %</td></tr>
</table>"""

        if con_talla:
            largos = np.array([d.largo_um for d, _ in con_talla], dtype=float)
            mayor, ruta_mayor = max(con_talla, key=lambda t: t[0].largo_um)
            menor, ruta_menor = min(con_talla, key=lambda t: t[0].largo_um)

            def _num(v, dec=0):
                """Celda con el valor, o una raya si esa magnitud no se midio."""
                return f"{v:.{dec}f}" if v else "—"

            def _fila(rotulo, d, ruta):
                return (f"<tr><td><strong>{rotulo}</strong></td><td>{_esc(d.class_name)}</td>"
                        f"<td>{_num(d.largo_um)}</td><td>{_num(d.ancho_um)}</td>"
                        f"<td>{_num(d.area_um2)}</td><td>{_num(d.aspecto, 1)}</td>"
                        f"<td>{_num(d.curvatura, 2)}</td><td>{_morfotipo(d.morfotipo)}</td>"
                        f"<td style='font-size:9pt'>{_esc(Path(ruta).name)}</td></tr>")

            tablas = f"""
<h3>{NUM}.4 {tr('Partícula mayor y menor')}</h3>
<table class='data'>
<tr>{tr('<th></th><th>Clase</th><th>Largo</th><th>Ancho</th><th>Área</th><th>Aspecto</th><th>Curvatura</th><th>Morfotipo</th><th>Imagen</th>')}</tr>
{_fila(tr('Mayor'), mayor, ruta_mayor)}
{_fila(tr('Menor'), menor, ruta_menor)}
</table>

<h3>{NUM}.5 {tr('Distribución de tallas')}</h3>
<table class='data'>
<tr>{tr('<th>Estadístico</th><th>Largo (µm)</th>')}</tr>
<tr>{tr('<td>mínimo</td>')}<td>{largos.min():.0f}</td></tr>
<tr>{tr('<td>percentil 10</td>')}<td>{np.percentile(largos, 10):.0f}</td></tr>
<tr>{tr('<td>mediana</td>')}<td>{np.median(largos):.0f}</td></tr>
<tr>{tr('<td>percentil 90</td>')}<td>{np.percentile(largos, 90):.0f}</td></tr>
<tr>{tr('<td>máximo</td>')}<td>{largos.max():.0f}</td></tr>
</table>"""

            # ── Tallas por tramo, apiladas por clase ──
            por_clase: Dict[str, List[float]] = {}
            for d, _ruta in con_talla:
                por_clase.setdefault(d.class_name, []).append(d.largo_um)
            fig = _fig_tallas_por_tramo(por_clase)
            if fig:
                tablas += f"<img src='data:image/png;base64,{fig}' style='max-width:100%'>"

            # ── Tabla por clase, con su mayor ──
            filas_cls = ""
            for cls in _ORDEN_CLASES + [c for c in por_clase if c not in _ORDEN_CLASES]:
                v = sorted(por_clase.get(cls, []))
                if not v:
                    continue
                nf = sum(1 for d, _r in con_talla
                         if d.class_name == cls and d.morfotipo == "fibra")
                filas_cls += (f"<tr><td>{cls}</td><td>{len(v)}</td>"
                              f"<td>{v[0]:.0f}</td><td>{v[len(v) // 2]:.0f}</td>"
                              f"<td>{v[-1]:.0f}</td><td>{nf}</td></tr>")
            if filas_cls:
                tablas += (
                    f"<h3>{NUM}.6 {tr('Talla por tipo de plástico')}</h3>"
                    f"<table class='data'><tr><th>Clase</th><th>n</th>"
                    f"<th>menor<br>(µm)</th><th>mediana<br>(µm)</th>"
                    f"<th>mayor<br>(µm)</th><th>Fibras</th></tr>{filas_cls}</table>")

            # La ficha de ejemplo se pinta mas abajo, DESPUES de explicar el
            # metodo: enseñar el recorte con una recta amarilla encima antes de
            # decir que esa recta es el diametro de Feret no explica nada.
            cal_mayor = (getattr(state, "calibraciones", {}) or {}).get(
                Path(ruta_mayor).name)
            um_mayor = cal_mayor.um_por_px if cal_mayor and cal_mayor.valida else None
            ficha_mayor = (mayor, ruta_mayor, um_mayor)

            # La menor tambien se enseña, no solo se tabula: una talla minima
            # sin foto no se puede distinguir de ruido o ficha de forma quede
            # y un lector no tiene como comprobar que de verdad hay una
            # particula ahi y no una mota del fondo mal segmentada.
            cal_menor = (getattr(state, "calibraciones", {}) or {}).get(
                Path(ruta_menor).name)
            um_menor = cal_menor.um_por_px if cal_menor and cal_menor.valida else None
            ficha_menor = (menor, ruta_menor, um_menor)

            # ── {tr('Talla por carpeta')}, y por foto dentro de cada carpeta ──
            # Generaliza la comparacion "por estacion/tramo" del pipeline del
            # paper (que dependia de un patron de nombre de archivo especifico
            # de un estudio) a lo unico que CUALQUIER lote tiene: de que
            # carpeta vino cada foto. Sirve para comparar sitios de muestreo,
            # tratamientos, o cualquier otra agrupacion que el usuario haya
            # expresado como carpetas al elegir las imagenes.
            grupos_carpeta: Dict[str, List[float]] = {}
            for d, ruta in con_talla:
                carpeta = Path(ruta).parent.name or str(Path(ruta).parent)
                grupos_carpeta.setdefault(carpeta, []).append(d.largo_um)

            if len(grupos_carpeta) >= 2:
                fig_carpeta = _fig_boxplot_grupos(
                    grupos_carpeta, tr("Talla por carpeta"), tr("carpeta de origen"))
                veredicto_carpeta = _texto_kruskal(
                    _kruskal_wallis(list(grupos_carpeta.values())), len(grupos_carpeta))
                por_carpeta_html = (
                    f"<h3>{NUM}.1 {tr('Talla por carpeta')}</h3>"
                    + tr("<p>Compara la talla entre las carpetas del lote analizado: "
                         "cada carpeta como un grupo distinto. Útil cuando cada carpeta "
                         "es un sitio de muestreo, una estación o una condición.</p>")
                    + f"<img src='data:image/png;base64,{fig_carpeta}' style='max-width:100%'>"
                    f"<p>{veredicto_carpeta}</p>")

                # -- Por foto, dentro de las carpetas con mas de una imagen --
                piezas_foto = []
                for carpeta in sorted(grupos_carpeta, key=lambda c: -len(grupos_carpeta[c])):
                    if len(piezas_foto) >= _MAX_CARPETAS_POR_FOTO:
                        break
                    fotos_de_carpeta: Dict[str, List[float]] = {}
                    for d, ruta in con_talla:
                        if Path(ruta).parent.name == carpeta:
                            fotos_de_carpeta.setdefault(Path(ruta).name, []).append(d.largo_um)
                    if len(fotos_de_carpeta) < 2:
                        continue
                    fig_foto = _fig_boxplot_grupos(
                        fotos_de_carpeta,
                        tr("Talla por foto — {carpeta}").format(carpeta=carpeta),
                        tr("foto"),
                        max_grupos=20)
                    veredicto_foto = _texto_kruskal(
                        _kruskal_wallis(list(fotos_de_carpeta.values())), len(fotos_de_carpeta))
                    piezas_foto.append(
                        f"<p><strong>{_esc(carpeta)}</strong> "
                        + tr("({n} fotos)").format(n=len(fotos_de_carpeta)) + "</p>"
                        f"<img src='data:image/png;base64,{fig_foto}' style='max-width:100%'>"
                        f"<p>{veredicto_foto}</p>")
                if piezas_foto:
                    n_carpetas_con_fotos = sum(
                        1 for c in grupos_carpeta
                        if len({Path(r).name for d, r in con_talla if Path(r).parent.name == c}) >= 2)
                    nota_tope = (tr(" Se muestran las {n} carpetas con más partículas "
                                    "de las {total} que tienen más de una foto.").format(
                                        n=_MAX_CARPETAS_POR_FOTO,
                                        total=n_carpetas_con_fotos)
                                 if n_carpetas_con_fotos > _MAX_CARPETAS_POR_FOTO else "")
                    por_carpeta_html += (
                        f"<h3>{NUM}.2 {tr('Talla por foto, dentro de cada carpeta')}</h3>"
                        + tr("<p>Compara la talla entre las fotos individuales de una "
                             "misma carpeta: por ejemplo, si el tamaño cambia con la "
                             "profundidad o el momento de muestreo dentro de un mismo "
                             "sitio.{nota}</p>").format(nota=nota_tope)
                        + "".join(piezas_foto))

        # ── Recuento fibra/fragmento por foto ──
    # Va dentro de la seccion de forma: responde "cuantas fibras y cuantos
    # fragmentos hay en cada placa", que es la pregunta que se le hace a la
    # tabla cuando se revisa foto por foto.
    if formas:
        por_img: Dict[str, list] = {}
        for d, ruta in formas:
            por_img.setdefault(Path(ruta).name, []).append(d)
        filas_img = ""
        for nombre in sorted(por_img):
            ds = por_img[nombre]
            nf = sum(1 for d in ds if d.morfotipo == "fibra")
            largos_i = [d.largo_um for d in ds if d.largo_um]
            filas_img += (
                f"<tr><td style='font-size:9pt'>{_esc(nombre)}</td><td>{len(ds)}</td>"
                f"<td>{nf}</td><td>{len(ds) - nf}</td>"
                f"<td>{(f'{np.median(largos_i):.0f}' if largos_i else '—')}</td>"
                f"<td>{(f'{max(largos_i):.0f}' if largos_i else '—')}</td></tr>")
        # La primera columna pasa a llamarse "Total" y no "Partículas": con el
        # cambio de nomenclatura habria dos columnas con el mismo nombre, una
        # como suma y otra como morfotipo. Total = Fibras + Partículas.
        recuento = (
            f"<h3>{NUM}.7 {tr('Recuento por imagen')}</h3>"
            f"<table class='data'><tr>"
            f"{tr('<th>Imagen</th><th>Total</th><th>Fibras</th><th>Partículas</th>')}"
            f"{tr('<th>Largo mediano<br>(µm)</th><th>Mayor<br>(µm)</th>')}"
            f"</tr>{filas_img}</table>")

        # Un "—" en las columnas de talla no es un fallo del analisis: la
        # particula SI se midio, en pixeles, pero sin µm/px no hay forma de
        # convertirlo a micrometros. Antes el guion aparecia sin explicacion y
        # parecia que la medicion habia fallado. Se dice que falta y como se
        # arregla, en vez de inventar un numero o callarse.
        sin_escala = [n for n, ds in por_img.items()
                      if not any(d.largo_um for d in ds)]
        if sin_escala:
            todas = len(sin_escala) == len(por_img)
            cuales = (tr("<strong>Ninguna imagen tiene escala</strong>")
                      if todas else
                      tr("<strong>{n} de {total} imágenes no tienen escala</strong> "
                         "({cuales})").format(
                             n=len(sin_escala), total=len(por_img),
                             cuales=", ".join(sorted(sin_escala)[:6])
                             + (", …" if len(sin_escala) > 6 else "")))
            recuento += (
                "<p>" + cuales
                + tr(", y por eso su largo aparece como «—». Las partículas <em>sí</em> "
                     "se midieron —el conteo y la forma de la tabla son correctos—, pero "
                     "una medida en píxeles no se puede pasar a micrómetros sin saber "
                     "cuántos µm mide cada píxel.</p>")
                + tr("<p>Para obtenerlo, en <em>Parámetros</em>: activa <strong>«Medir "
                     "la placa Petri»</strong> —cada foto obtiene su propia escala del "
                     "anillo de la placa, que es lo más fiable— o escribe un valor de "
                     "<strong>µm/píxel</strong>")
                + ("." if todas else
                   tr(". Si ya está activo, en esas fotos concretas no se encontró el "
                      "anillo de la placa: revisa que se vea entero en el encuadre."))
                + "</p>")

        # Fuera de la f-string: dentro habria que anidar comillas iguales.
        pct_curvas = f"{100.0 * curvas / len(formas):.1f}"
        metodo = f"""
<h3>{NUM}.1 {tr('Cómo se mide el largo')}</h3>
{tr('<p>Las magnitudes se miden sobre la <strong>máscara de cada partícula</strong>, no sobre la caja del detector. La caja de una partícula alargada está casi vacía y depende de cómo haya caído: una fibra tumbada en diagonal tiene caja cuadrada, de modo que medir sobre la caja la reportaría como si no fuera fibra, y con una talla equivocada. La máscara se obtiene umbralizando cada recorte a medio camino entre el nivel del fondo —la mediana del anillo que rodea la caja— y el de la partícula —el percentil 90 dentro de ella—.</p>')}

{tr('<p><strong>El criterio general es la línea recta más larga que cabe en la partícula</strong>, es decir la mayor distancia entre dos puntos de su contorno: el <em>diámetro de Feret máximo</em>. Para una partícula de forma irregular pero no doblada, esa recta es su talla, y tiene dos propiedades que la hacen fiable: no depende de la orientación con que la partícula haya caído, y un borde dentado no la altera.</p>')}

{tr('<p><strong>Esa recta deja de servir cuando la partícula está contorsionada.</strong> En una fibra doblada la distancia entre sus extremos es la cuerda, no su longitud: en un arco de media circunferencia se queda un 35&nbsp;% corta. Para esos casos se mide el <em>diámetro geodésico</em>, que es el camino más largo que cabe <strong>dentro</strong> de la partícula. Al no poder salirse de la máscara, ese camino rodea la curva y devuelve la longitud recorrida.</p>')}

<table class='data'>
<tr>{tr('<th>Forma de la partícula</th><th>Qué se reporta como largo</th>')}</tr>
<tr>{tr('<td>Compacta o irregular, pero no doblada</td><td>Feret máximo (la recta más larga)</td>')}</tr>
<tr>{tr('<td>Alargada y contorsionada (fibra)</td><td>Diámetro geodésico (sigue la curva)</td>')}</tr>
</table>

{tr('<p>La distinción se hace por el <strong>grosor</strong>: solo se usa el geodésico si el largo supera al menos cuatro veces el grosor máximo inscrito de la partícula. La razón es que en una partícula gruesa cualquier concavidad obliga al camino geodésico a bordearla en vez de atravesarla, y entonces el número se infla; en una fibra delgada eso no puede ocurrir.</p>')}

{tr('<p>Contrastado con formas de talla conocida —rectas, rectas giradas 45&nbsp;°, arcos de 60, 120 y 180&nbsp;°, un círculo y una recta de borde dentado— el largo así medido da un <strong>error mediano del 0,8&nbsp;% y del 4,7&nbsp;% en el peor caso</strong>.</p>')}

{tr('<p><strong>{n} partículas ({pct} %) están contorsionadas</strong>, entendiendo por tal que su largo supera en más de un 15&nbsp;% su extensión en línea recta.</p>').format(n=curvas, pct=pct_curvas)}

{tr('<p><em>Partículas en contacto.</em> Dos partículas que se tocan forman una sola mancha, y medirlas juntas sumaría sus tallas. Se separan por <em>watershed</em> sobre la transformada de distancia: el centro de cada una queda lejos del fondo y el cuello que las une queda cerca, de modo que el corte cae por el cuello. Sobre círculos de talla conocida las separa hasta un 27&nbsp;% de solapamiento del diámetro, sin partir ninguna partícula de una sola pieza.</p>')}

{tr('<p><em>Limitaciones conocidas.</em> Dos partículas solapadas más allá de un 40&nbsp;% de su diámetro se siguen midiendo como una sola: a esa altura ya no hay un cuello por el que cortar. Y en una fibra muy enroscada el camino geodésico ataja por el interior de cada codo, subestimando la longitud hasta un 19&nbsp;% en el caso más cerrado que se ensayó.</p>')}"""
        if n_sin_forma:
            metodo += tr(
                "<p>En {n} partículas no se pudo separar la partícula del fondo; su "
                "talla proviene de la caja y no es comparable con el resto.</p>").format(
                    n=n_sin_forma)

        # ── Los ejemplos, ya explicado el método ──
        # Las dos, no solo la mayor: una talla minima sin foto no se puede
        # distinguir a ojo de ruido mal segmentado, y es la que mas facil se
        # cuestiona en una revision.
        if ficha_mayor is not None:
            ejemplo = _ficha_particula(
                *ficha_mayor, f"2 {tr('Un ejemplo: la partícula mayor, medida')}")
        if ficha_menor is not None:
            ejemplo += _ficha_particula(
                *ficha_menor, f"3 {tr('Un ejemplo: la partícula menor, medida')}")

    # Orden de la sección: primero cómo se mide, luego un caso medido, y al
    # final las cifras. Al revés, las tablas llegan antes de que nada haya
    # explicado qué significa aquí "largo".
    if formas:
        forma_html = metodo + ejemplo + tabla_morfotipo + tablas + recuento

    # ── Ficha de cada partícula ──
    # Una entrada por partícula, con su número, su recorte y la medida dibujada
    # encima. Es lo que permite comprobar partícula a partícula que la talla que
    # se reporta se corresponde con lo que hay en la foto, en vez de tener que
    # fiarse de la tabla.
    fichas_html = ""
    if formas and max_fichas:
        import cv2
        from .morfologia import dibujar_medicion

        elegidas = elegir_fichas(formas, max_fichas)
        n_fibras_muestra = sum(1 for d, _ in elegidas if d.morfotipo == "fibra")
        por_img_f: Dict[str, list] = {}
        for d, ruta in elegidas:
            por_img_f.setdefault(str(ruta), []).append(d)
        piezas, hechas = [], 0
        for ruta in sorted(por_img_f):
            bgr = None
            try:
                from .calibracion import leer_imagen
                bgr = leer_imagen(ruta)
            except Exception:
                bgr = None
            if bgr is None:
                continue
            cal = (getattr(state, "calibraciones", {}) or {}).get(Path(ruta).name)
            um = cal.um_por_px if cal and cal.valida else None
            tarjetas = ""
            for d in sorted(por_img_f[ruta], key=lambda z: (z.numero or 0)):
                if hechas >= max_fichas:
                    break
                img, m = dibujar_medicion(bgr, d)
                if img is None or m is None or not m.ok:
                    continue
                ok, buf = cv2.imencode(".png", img)
                if not ok:
                    continue
                hechas += 1
                cuenta = ""
                if d.largo_um and um:
                    cuenta = (f"{m.largo_px:.1f} px &times; {um:.4f} = "
                              f"<strong>{d.largo_um:.0f} µm</strong>")
                # d.aspecto/morfotipo se llenan con solo segmentar (no necesitan
                # calibracion), pero d.ancho_um solo si la foto tenia um_por_px
                # valido cuando se midio -- una foto sin placa detectada deja
                # aspecto con dato y ancho_um en None. Sin este guard, esa
                # combinacion revienta el formato ":.0f" sobre None.
                detalle = [f"ancho {d.ancho_um:.0f} µm"] if d.ancho_um else []
                if d.aspecto:
                    detalle.append(f"aspecto {d.aspecto:.1f}")
                detalle.append(tr("medido por {metodo}").format(metodo=tr(m.metodo)))
                # Si una parte apreciable de la recta cae fuera, se dice con un
                # numero. Por debajo del 5% es discretizacion del contorno y
                # decirlo solo añadiria ruido a la ficha.
                if getattr(m, "feret_fuera", 0) > 0.05:
                    detalle.append(tr("{pct} % de la recta fuera del contorno").format(
                        pct=f"{100 * m.feret_fuera:.0f}"))
                color = "#1f6b5e" if d.morfotipo == "fibra" else "#656d76"
                etiqueta_morfo = _morfotipo(d.morfotipo, minuscula=True)
                # El aviso se ENSEÑA. Antes se calculaba y se tiraba: una talla
                # inflada por varias particulas en contacto llegaba al informe
                # como un numero limpio, y quien lo leia no tenia como saberlo.
                # Se lee de la Detection, NO de la Morfologia que devuelve
                # dibujar_medicion: esa viene de medir(), y el aviso lo pone
                # medir_deteccion() al comparar con la diagonal de la caja. En
                # la Morfologia de aqui el campo esta siempre en False, asi que
                # mirarlo hacia que el ⚠ no apareciera nunca.
                alerta = ""
                if getattr(d, "revisar", False):
                    alerta = (
                        f"<div style='font-size:8.5pt;color:{T.DOC.WARN};margin-top:3px'>"
                        f"⚠ {d.aviso_forma or tr('la máscara se salía de la caja y se recortó a ella')}</div>")
                tarjetas += (
                    f"<div style='display:inline-block;vertical-align:top;margin:0 14px 18px 0;"
                    f"max-width:290px'>"
                    f"<div style='font-weight:700'>#{d.numero} · {_esc(d.class_name)} · "
                    f"<span style='color:{color}'>{etiqueta_morfo}</span></div>"
                    f"<img src='data:image/png;base64,{_img_b64(buf.tobytes())}' style='max-width:100%'>"
                    f"<div style='font-size:9pt;color:#656d76'>{cuenta}<br>"
                    f"{' · '.join(detalle)}</div>{alerta}</div>")
            if tarjetas:
                piezas.append(f"<h3>{_esc(Path(ruta).name)}</h3>{tarjetas}")
        if piezas:
            # Se declara COMO se eligio la muestra. Sin esto, un lector podria
            # tomarla por representativa del lote y no lo es: esta sesgada a
            # proposito hacia lo grande y hacia las fibras.
            n_otras_muestra = hechas - n_fibras_muestra
            if n_fibras_muestra:
                criterio = tr("Se muestran las <strong>{fibras} fibra(s) y las {otras} "
                              "partículas más grandes</strong>").format(
                                  fibras=n_fibras_muestra, otras=n_otras_muestra)
            else:
                criterio = tr("Se muestran las <strong>{otras} partículas más "
                              "grandes</strong>. En este lote no se detectó ninguna "
                              "fibra, así que la muestra no incluye ninguna").format(
                                  otras=n_otras_muestra)
            fichas_html = (
                tr("<p>Cada partícula con el número que lleva en la imagen anotada, su "
                   "recorte y la medida dibujada <strong>sobre ella</strong>: en "
                   "amarillo la recta de Feret, en magenta el camino geodésico, según "
                   "cuál de las dos haya decidido su talla. El contorno verde es la "
                   "máscara que se midió. A la izquierda va la partícula sin marcas, "
                   "para poder juzgar si el contorno la sigue.</p>")
                + tr("<p>En una partícula <strong>cóncava</strong> la recta amarilla "
                     "cruza por fuera del contorno, y eso es correcto: Feret es la "
                     "separación de dos mordazas de calibre, no un camino por dentro "
                     "de la partícula. Para que se vea, el tramo que cae fuera va "
                     "<strong>a trazos</strong>. El camino que sí va por dentro es el "
                     "geodésico, en magenta, y se usa cuando la partícula es delgada "
                     "y está doblada.</p>")
                + "<p>" + criterio
                + tr(". El reparto es deliberado: las fibras son el caso donde actúa el "
                     "método geodésico —y son minoría, de modo que una muestra tomada al "
                     "azar podría no enseñar ninguna—, y la mayor de cada tipo es la que "
                     "sostiene cualquier afirmación sobre talla máxima. <strong>No es "
                     "una muestra representativa del lote</strong>, sino la selección "
                     "que permite comprobar el método donde más puede fallar.</p>")
                + "".join(piezas))

    # ── Ensamblar HTML ──
    # Cada bloque es (id, cuerpo). El cuerpo NO lleva su <h2>: el titulo y el
    # numero los pone el ensamblador, que es el unico que sabe que secciones
    # sobrevivieron al filtro y por tanto que numero le toca a cada una.
    # Fuera de la f-string del resumen: lleva un plural y un formato numerico,
    # y anidarlos dentro de {tr(...)} pediria comillas del mismo tipo.
    _resumen_parrafo = tr(
        "Se analizaron <strong>{imgs} imágenes</strong> con <strong>{n} {modelos}</strong> "
        "YOLO entrenado para detectar microplásticos de PET, PP y LDPE bajo "
        "fluorescencia Nile Red (254 nm). El total de detecciones fue "
        "<strong>{dets}</strong> con una confianza media de <strong>{conf}</strong>."
    ).format(imgs=total_imgs, n=len(active),
             modelos=tr("modelos") if len(active) != 1 else tr("modelo"),
             dets=total_dets, conf=f"{avg_conf:.3f}")

    cuerpos = {
        "resumen": f"""
<div class='kpis'>
  <div class='kpi'><div class='l'>{tr('Imágenes')}</div><div class='v'>{total_imgs}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{tr('Detecciones')}</div><div class='v'>{total_dets}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>{tr('Conf. media')}</div><div class='v'>{avg_conf:.3f}</div><div class='b' style='background:var(--vio)'></div></div>
  <div class='kpi'><div class='l'>{tr('Tamaño medio (μm)')}</div><div class='v'>{(f"{avg_size:.1f}" if avg_size else "—")}</div><div class='b' style='background:var(--warn)'></div></div>
</div>
<p>{_resumen_parrafo}
{tr("Se incluyó análisis de errores con Ground Truth (Verdaderos Positivos, Falsos Positivos, Falsos Negativos y Mal Clasificados).") if any_gt else tr("No se aportó Ground Truth, por lo que no se reportan métricas de error.")}
</p>""",
        "metodos": f"""
<table class='data'><tr>{tr('<th>Parámetro</th><th>Valor</th>')}</tr>{methods_html}</table>
{equipo_html}
{methods_para}""",
        "calibracion": calib_html,
        "forma": forma_html,
        "talla_carpetas": por_carpeta_html,
        "profundidad": profundidad_html,
        "fichas": fichas_html,
        "resultados": figures_html,
        "modelos": f"""
<table class='data'><tr>{tr('<th>Modelo</th><th>Imágenes</th><th>Detecciones</th><th>Conf. media</th>')}
<th>{LABEL_TP}</th><th>{LABEL_FP}</th><th>{LABEL_FN}</th>
<th>{LABEL_MISCLS}</th><th>F1<br><span style='font-weight:400;font-size:8.5pt'>{tr('localización')}</span></th>
<th>F1<br><span style='font-weight:400;font-size:8.5pt'>{tr('con clase')}</span></th></tr>{rows_models}</table>
{nota_miscls}""",
        "errores": err_section,
        "comparacion": f"{veredicto_html}\n{compare_html}",
        "galeria": gallery_html,
        "conteo": conteo_html,
        "referencias": refs_html,
    }

    # Un id en ``pedidas`` con cuerpo vacio se cae aqui: se marco la casilla pero
    # no habia con que llenarla (errores sin ground truth, conteo sin muestras).
    # Nombres de ancla heredados, para no romper enlaces a informes ya
    # repartidos. Una seccion nueva NO tiene que aparecer aqui: su propio id ya
    # sirve de ancla. Este diccionario era una segunda lista que mantener en
    # sincronia con SECCIONES, y al añadir "profundidad" reventaba con
    # KeyError -- justo lo que SECCIONES evita al ser lista unica.
    _ANCLAS = {"resumen": "abstract", "metodos": "methods", "calibracion": "calib",
               "resultados": "results", "modelos": "models", "errores": "errors",
               "comparacion": "compare", "galeria": "gallery", "referencias": "refs"}

    def ancla_de(sid: str) -> str:
        return _ANCLAS.get(sid, sid)
    # tr() sobre el titulo: SECCIONES guarda la version en espanol, que es la
    # clave del diccionario. La interfaz ya lo traducia para sus casillas; el
    # documento no, y salia con los titulos en espanol aunque el resto no.
    vivas = [(sid, tr(titulo), cuerpos.get(sid, ""))
             for sid, titulo in SECCIONES
             if sid in pedidas and (cuerpos.get(sid) or "").strip()]

    toc_html = "".join(
        f"<li><a href='#{ancla_de(sid)}'>{titulo}</a></li>" for sid, titulo, _ in vivas)
    secciones_html = "".join(
        f"<h2 id='{ancla_de(sid)}'>{i}. {titulo}</h2>\n{cuerpo.replace(NUM, str(i))}\n"
        for i, (sid, titulo, cuerpo) in enumerate(vivas, start=1))

    html = f"""<!doctype html>
<html lang='{idioma()}'><head><meta charset='utf-8'>
<title>{tr('Informe de detección')} · Poly-X</title>
<style>{REPORT_CSS}</style></head><body>
<div class='container'>

<header class='cover'>
  <div class='kicker'>Poly-X · {tr('Informe de detección')}</div>
  <h1>{tr('Informe de detección de microplásticos')}<br>{tr('por fluorescencia Nile Red')}</h1>
  <p class='meta'><strong>{tr('Autor:')}</strong> Cristofher Ferrada &middot;
    <strong>{tr('Modelos:')}</strong> {_esc(', '.join(s.alias for s in active)) or '—'}</p>
</header>

<div class='toc'>
  <h3>📑 {tr('Índice')}</h3>
  <ol>{toc_html}</ol>
</div>

{secciones_html}
<footer>
  © Cristofher Ferrada · {tr('Generado por Poly-X')}<br>
  {tr('Suite de detección de microplásticos por fluorescencia Nile Red (254 nm) e IA (YOLO v8/v11).')}
</footer>
</div></body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_report(state, output_path: Path,
                    include_refs: bool = True,
                    include_gallery: bool = True,
                    max_gallery: int = 60,
                    max_fichas: int = 12,
                    solo_imagenes=None,
                    secciones=None) -> Path:
    """Generates the HTML report with an observability wrapper.

    Generates the HTML report, measuring it and recording report.generate /
    report.finish / report.error in the detector JSONL. Re-throws failures:
    the caller decides what to do, the log keeps the diagnosis.
    """
    from .observabilidad import cronometro, logger_para
    n_resultados = sum(len(rs) for rs in (state.results or {}).values())
    with cronometro(logger_para("detector"), "report",
                    resultados=n_resultados, salida=str(output_path),
                    secciones=sorted(secciones) if secciones else None,
                    solo_imagenes=(len(solo_imagenes)
                                   if solo_imagenes is not None else None)):
        return _generate_report_impl(state, output_path,
                                     include_refs=include_refs,
                                     include_gallery=include_gallery,
                                     max_gallery=max_gallery,
                                     max_fichas=max_fichas,
                                     solo_imagenes=solo_imagenes,
                                     secciones=secciones)
