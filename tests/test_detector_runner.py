"""Characterization tests for ``DetectorRunner.run``.

The batch loop used to be one 325-line method and the suite never called it:
the refactor into ``_prepare_image`` / ``_process_model`` / ``_score_predictions``
etc. therefore had no safety net. These tests pin the observable contract of
the orchestration itself -- signals, result bookkeeping, files on disk and
abort behaviour -- against a fake model, so the structure can keep moving.

Nothing here touches ultralytics or the real ``runs/`` folder: the model is a
stub and the run folder is redirected to ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_RAIZ = Path(__file__).resolve().parents[1]
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.yolo import Detection  # noqa: E402
from polyx.gui.detector.runner import DetectorRunner  # noqa: E402
from polyx.gui.detector.state import DetectorState  # noqa: E402

# Two bright squares on a black plate: high contrast, so the Otsu/ring
# segmentation inside each box succeeds and the shape measurement runs.
_CAJA_ARRIBA = (20, 20, 60, 60)
_CAJA_ABAJO = (90, 90, 130, 130)


def _placa(ruta: Path) -> Path:
    """Write a 160x160 plate with the two white squares the boxes point at."""
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in (_CAJA_ARRIBA, _CAJA_ABAJO):
        img[y1 + 5 : y2 - 5, x1 + 5 : x2 - 5] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(bytes(buf))
    return ruta


class _ModeloFalso:
    """Stands in for ``YoloModel``: loads instantly, returns two boxes."""

    names = {0: "PET", 1: "PP"}

    def __init__(self):
        self.cargado = False

    def load(self):
        self.cargado = True
        return self

    def predict_auto(self, _ruta, **_kw) -> list[Detection]:
        return [
            Detection(
                class_id=0,
                class_name="PET",
                conf=0.9,
                x1=_CAJA_ARRIBA[0],
                y1=_CAJA_ARRIBA[1],
                x2=_CAJA_ARRIBA[2],
                y2=_CAJA_ARRIBA[3],
            ),
            Detection(
                class_id=1,
                class_name="PP",
                conf=0.4,
                x1=_CAJA_ABAJO[0],
                y1=_CAJA_ABAJO[1],
                x2=_CAJA_ABAJO[2],
                y2=_CAJA_ABAJO[3],
            ),
        ]


def _estado(tmp_path: Path, imagenes: list[Path]) -> DetectorState:
    st = DetectorState()
    slot = st.model_slots[0]
    slot.alias = "modelo_a"
    slot.path = tmp_path / "best.pt"
    slot.loaded = _ModeloFalso()
    st.images = imagenes
    st.params.measure_plate = False
    st.params.um_per_px = 2.0  # manual scale, so the sizes come out in um
    st.params.tiling = "nunca"
    return st


def _runner_con_carpeta(state: DetectorState, tmp_path: Path, monkeypatch) -> DetectorRunner:
    """Runner whose run folder lands in tmp_path instead of <project>/runs."""

    def _create(_self, st) -> str:
        stamp = "20260101_000000"
        st.run_dir = tmp_path / "runs" / f"detect_{stamp}"
        st.run_dir.mkdir(parents=True, exist_ok=True)
        return stamp

    monkeypatch.setattr(DetectorRunner, "_create_run_dir", _create)
    return DetectorRunner(state)


def _ejecutar(runner: DetectorRunner) -> dict:
    """Call ``run()`` synchronously and record every signal it emits."""
    eventos: dict[str, list] = {"progress": [], "image": [], "ok": [], "failed": [], "aborted": []}
    runner.progress.connect(lambda d, t, n: eventos["progress"].append((d, t, n)))
    runner.image_done.connect(lambda i, r: eventos["image"].append((i, r)))
    runner.finished_ok.connect(lambda: eventos["ok"].append(True))
    runner.failed.connect(eventos["failed"].append)
    runner.aborted.connect(lambda: eventos["aborted"].append(True))
    runner.run()  # QThread.run() called directly: synchronous, no event loop
    return eventos


def test_run_una_imagen_produce_resultado_archivos_y_progreso(qapp, tmp_path, monkeypatch):
    ruta = _placa(tmp_path / "fotos" / "placa1.png")
    estado = _estado(tmp_path, [ruta])
    runner = _runner_con_carpeta(estado, tmp_path, monkeypatch)

    eventos = _ejecutar(runner)

    assert eventos["failed"] == [], "el run no debe fallar"
    assert eventos["ok"] == [True], "debe emitirse finished_ok una vez"

    # One result for the one active model, with both detections.
    assert set(estado.results) == {0, 1, 2}
    assert len(estado.results[0]) == 1
    res = estado.results[0][0]
    assert res.model_idx == 0
    assert [d.number for d in res.predictions] == [1, 2], "numeracion de arriba hacia abajo"
    assert res.tiling == "", "sin troceo el descriptor queda vacio"

    # Files: annotated PNGs plus the YOLO labels with the confidence column.
    assert res.annotated_path and res.annotated_path.exists()
    assert res.pred_path and res.pred_path.exists()
    assert res.gt_path is None, "no hay GT, no debe dibujarse su PNG"
    etiquetas = estado.run_dir / "modelo_a" / "labels" / "0000_placa1.txt"
    lineas = etiquetas.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 2
    assert all(len(l.split()) == 6 for l in lineas)  # noqa: E741

    # Progress counts models, and the batch closes on the last one.
    assert eventos["progress"] == [(1, 1, "placa1.png")]
    assert eventos["image"][-1][0] == 0

    # The scale of the photo is registered for the report.
    assert estado.calibrations["placa1.png"].um_por_px == 2.0


def test_run_con_ground_truth_cuenta_tp_y_fp(qapp, tmp_path, monkeypatch):
    ruta = _placa(tmp_path / "fotos" / "placa2.png")
    # GT only for the upper box: the lower prediction must be a false positive.
    x1, y1, x2, y2 = _CAJA_ARRIBA
    cx, cy = (x1 + x2) / 2 / 160, (y1 + y2) / 2 / 160
    (ruta.parent / "placa2.txt").write_text(
        f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / 160:.6f} {(y2 - y1) / 160:.6f}\n",
        encoding="utf-8",
    )
    estado = _estado(tmp_path, [ruta])
    runner = _runner_con_carpeta(estado, tmp_path, monkeypatch)

    eventos = _ejecutar(runner)

    assert eventos["failed"] == []
    res = estado.results[0][0]
    assert res.has_gt is True
    assert (res.tp, res.fp, res.fn, res.miscls) == (1, 1, 0, 0)
    assert res.gt_path and res.gt_path.exists(), "con GT debe guardarse su PNG"
    assert res.gt_path != res.pred_path


def test_run_aborta_en_el_borde_de_la_imagen(qapp, tmp_path, monkeypatch):
    rutas = [
        _placa(tmp_path / "fotos" / "a.png"),
        _placa(tmp_path / "fotos" / "b.png"),
    ]
    estado = _estado(tmp_path, rutas)
    runner = _runner_con_carpeta(estado, tmp_path, monkeypatch)

    # Ask to stop as soon as the first image is done: the abort is honoured at
    # the image boundary, never mid-image, so no half-written result remains.
    def _pedir(_i, _r):
        estado.request_abort()

    runner.image_done.connect(_pedir)
    eventos = _ejecutar(runner)

    assert eventos["aborted"] == [True], "debe emitirse aborted"
    assert eventos["ok"] == [], "un run abortado no puede emitir finished_ok"
    assert [r.image_path.name for r in estado.results[0]] == ["a.png"]


def test_run_imagen_indescodificable_avanza_el_progreso(qapp, tmp_path, monkeypatch):
    ruta = tmp_path / "fotos" / "rota.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(b"esto no es un png")
    estado = _estado(tmp_path, [ruta])
    runner = _runner_con_carpeta(estado, tmp_path, monkeypatch)

    eventos = _ejecutar(runner)

    assert eventos["failed"] == [], "una imagen mala no tumbe el lote"
    assert eventos["ok"] == [True]
    assert estado.results[0] == []
    # The progress still advances by the whole model count of that image.
    assert eventos["progress"] == [(1, 1, "rota.png")]


def test_run_sin_modelos_falla_sin_carpeta(qapp, tmp_path, monkeypatch):
    ruta = _placa(tmp_path / "fotos" / "placa3.png")
    estado = _estado(tmp_path, [ruta])
    estado.model_slots[0].path = None  # nothing active
    runner = _runner_con_carpeta(estado, tmp_path, monkeypatch)

    eventos = _ejecutar(runner)

    assert eventos["failed"] == ["No hay modelos cargados."]
    assert eventos["ok"] == []
    assert not (tmp_path / "runs").exists(), "sin modelos no debe crearse la carpeta del run"
