"""El ciclo de vida del TrainerRunner con torch y ultralytics falsos.

``TrainerRunner`` (``polyx/gui/trainer/runner.py``) es el unico punto donde el
Entrenador arranca de verdad un entrenamiento Ultralytics: emite los eventos
``train.config`` / ``train.progress`` / ``train.complete`` / ``train.aborted`` /
``train.error`` a ``trainer.jsonl`` y traduce una solicitud de detencion del
usuario (``TrainerState.request_abort``) en una parada limpia en la frontera de
epoca. Probarlo de verdad exigiria GPU, el paquete ultralytics y un dataset
entero; el entorno de CI no tiene ninguno de los tres.

Aqui se ejercita el codigo REAL del runner --sus callbacks, sus senales, sus
eventos de observabilidad-- sustituyendo solo los dos limites externos: como
``run()`` hace ``import torch`` / ``from ultralytics import YOLO`` dentro del
metodo, alcanza con dejar modulos falsos en ``sys.modules``. El doble de YOLO
registra los callbacks, simula epocas en memoria, honra ``trainer.stop`` para
cortar el loop y escribe ``weights/best.pt`` donde el runner lo busca.

Todo corre sincrono y offscreen: se llama ``runner.run()`` directo (sin
``start()``, sin hilo aparte), las senales se conectan a listas en memoria y
los eventos se leen de ``tmp_path / "_polyx_logs" / "trainer.jsonl``, que el
fixture autouse de ``conftest.py`` aisla por prueba.

Nota lateral: el runner crea sus corridas bajo ``<repo>/runs_train``, asi que
estas pruebas dejan ahi ``weights/best.pt`` falsos. Es el comportamiento real
del codigo bajo prueba, no un efecto del doble.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from polyx.gui.trainer import runner as runner_mod
from polyx.gui.trainer.runner import TrainerRunner
from polyx.gui.trainer.state import TrainerState

# Las páginas son QWidgets y TrainerRunner es un QThread: sin display real las
# pruebas corren «offscreen».
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# El runner deriva el directorio de runs de su propia ubicación:
# Path(__file__).resolve().parents[3] / "runs_train".
RAIZ_PROYECTO = Path(runner_mod.__file__).resolve().parents[4]
RUNS_ENTRENAMIENTO = RAIZ_PROYECTO / "runs_train"


# ── Dobles de torch / ultralytics ────────────────────────────────────
class _TrainerFalso:
    """El ``trainer`` interno de Ultralytics: epocas, metricas y optimizer."""

    def __init__(self) -> None:
        self.epoch = 0
        self.stop = False
        self.metrics: dict[str, float] = {}
        self.loss_items = [0.10, 0.20, 0.30]
        self.optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.01}])
        # _on_train_start hace next(trainer.model.parameters()).device.
        self.model = types.SimpleNamespace(
            parameters=lambda: iter([types.SimpleNamespace(device="cpu")])
        )


class FakeYOLO:
    """Doble de ``ultralytics.YOLO``: callbacks registrados y epocas simuladas.

    ``al_entrenar`` es un gancho de clase con firma ``(modelo, kwargs) -> None``
    que puede levantar (RuntimeError, KeyboardInterrupt) para simular un
    entrenamiento que revienta a mitad. Por defecto corre ``epochs`` epocas en
    memoria, dispara los callbacks registrados, honra ``trainer.stop`` cortando
    el loop y escribe ``weights/best.pt`` bajo ``Path(kwargs['project'])/name``,
    que es exactamente donde el runner resuelve ``best_path``.
    """

    al_entrenar = None  # type: ignore[assignment]
    instancias: list["FakeYOLO"] = []

    def __init__(self, weights) -> None:
        self.weights = str(weights)
        self.callbacks: dict[str, object] = {}
        self.trainer = _TrainerFalso()
        FakeYOLO.instances.append(self)

    def add_callback(self, nombre: str, fn) -> None:
        self.callbacks[nombre] = fn

    def _disparar(self, nombre: str) -> None:
        fn = self.callbacks.get(nombre)
        if fn is not None:
            fn(self.trainer)

    def train(self, name: str | None = None, **kwargs) -> None:
        self._disparar("on_train_start")
        if type(self).al_entrenar is not None:
            type(self).al_entrenar(self, kwargs)
        epocas = int(kwargs.get("epochs", 1))
        for ep in range(epocas):
            self.trainer.epoch = ep
            self.trainer.metrics = {
                "metrics/precision(B)": 0.50 + 0.10 * ep,
                "metrics/recall(B)": 0.40 + 0.10 * ep,
                "metrics/mAP50(B)": 0.60 + 0.05 * ep,
                "metrics/mAP50-95(B)": 0.30 + 0.05 * ep,
            }
            self._disparar("on_train_epoch_end")
            if self.trainer.stop:
                break
        self._disparar("on_train_end")
        pesos = Path(kwargs["project"]) / str(name) / "weights"
        pesos.mkdir(parents=True, exist_ok=True)
        (pesos / "best.pt").write_bytes(b"pesos falsos")


@pytest.fixture(scope="module")
def qapp():
    """Una sola QApplication por módulo: PySide6 no permite recrearla."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def deps_falsos(monkeypatch):
    """Instala torch y ultralytics falsos en sys.modules y limpia el doble.

    ``runner.run()`` importa ambos dentro del metodo, asi que las entradas de
    sys.modules bastan: nunca se toca el torch/ultralytics real (que en este
    entorno ni siquiera esta instalado).
    """
    torch_falso = types.ModuleType("torch")
    torch_falso.__version__ = "2.5.1+fake"
    torch_falso.version = types.SimpleNamespace(cuda="12.4")
    torch_falso.cuda = types.SimpleNamespace(is_available=lambda: False)

    ultralytics_falso = types.ModuleType("ultralytics")
    ultralytics_falso.YOLO = FakeYOLO

    monkeypatch.setitem(sys.modules, "torch", torch_falso)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_falso)
    FakeYOLO.al_entrenar = None
    FakeYOLO.instances = []
    return types.SimpleNamespace(torch=torch_falso, ultralytics=ultralytics_falso)


# ── Constructores de dataset y estado ────────────────────────────────
def _dataset_minimo(root: Path) -> Path:
    """data.yaml minimo (3 train + 3 val, una clase) como el runner lo espera.

    ``elegir_mejor_en_real`` lee el yaml y lista ``images/val``; sin imagenes
    con prefijo ``real__`` devuelve ``(None, [])`` y el run conserva best.pt.
    """
    for split in ("train", "val"):
        imgs = root / "images" / split
        lbls = root / "labels" / split
        imgs.mkdir(parents=True, exist_ok=True)
        lbls.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (imgs / f"{split}_{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (lbls / f"{split}_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    yaml = root / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "names:\n  0: particula\n",
        encoding="utf-8",
    )
    return yaml


def _estado(yaml_path: Path | None = None, **params) -> TrainerState:
    """TrainerState con el dataset indicado y los parámetros sobreescritos."""
    st = TrainerState()
    st.dataset.yaml_path = yaml_path
    for clave, valor in params.items():
        setattr(st.params, clave, valor)
    return st


# ── Recolección de señales y eventos ─────────────────────────────────
class _Recolector:
    """Junta en listas todo lo que el runner emite por senales."""

    def __init__(self) -> None:
        self.metricas: list = []
        self.lineas: list[str] = []
        self.progreso: list[tuple[int, int]] = []
        self.finales: list[str] = []
        self.abortados: int = 0
        self.fallos: list[str] = []

    def conectar(self, runner: TrainerRunner) -> None:
        runner.epoch_metrics.connect(self.metricas.append)
        runner.log_line.connect(self.lineas.append)
        runner.progress.connect(lambda ep, total: self.progreso.append((ep, total)))
        runner.finished_ok.connect(self.finales.append)
        runner.aborted.connect(lambda: setattr(self, "abortados", self.abortados + 1))
        runner.failed.connect(self.fallos.append)


def _correr(st: TrainerState) -> _Recolector:
    """Llama ``runner.run()`` directo: sincrono, sin QThread ni event loop."""
    runner = TrainerRunner(st)
    recolector = _Recolector()
    recolector.conectar(runner)
    runner.run()
    return recolector


def _eventos(tmp_path: Path) -> list[dict]:
    """Los registros JSON de trainer.jsonl (lista vacia si no existe)."""
    ruta = tmp_path / "_polyx_logs" / "trainer.jsonl"
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Camino feliz: secuencia completa del ciclo de vida ───────────────
class TestCicloDeVidaFeliz:
    def test_secuencia_completa_y_esquema_ingles(self, qapp, tmp_path, deps_falsos):
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="feliz")
        _correr(st)

        eventos = _eventos(tmp_path)
        assert [e["event"] for e in eventos] == [
            "train.config",
            "train.progress", "train.progress", "train.progress",
            "train.complete",
        ]
        # Ingles core schema: ts/level/module/event/run_id en TODAS las lineas.
        for e in eventos:
            assert isinstance(e["ts"], str) and e["ts"].endswith("Z")
            assert e["level"] == "INFO"
            assert e["module"] == "polyx.trainer"
            assert e["run_id"] == "yolov8m.pt"   # el base_weights_name del modelo
        # El nombre viejo no puede quedar en ninguna parte del archivo.
        assert "train.finish" not in (
            tmp_path / "_polyx_logs" / "trainer.jsonl"
        ).read_text(encoding="utf-8")

    def test_config_declara_modelo_e_hiperparametros(self, qapp, tmp_path, deps_falsos):
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="feliz2")
        _correr(st)

        config = _eventos(tmp_path)[0]
        assert config["event"] == "train.config"
        assert config["data"]["modelo"] == "yolov8m"
        assert config["data"]["epochs"] == 3
        assert config["data"]["batch"] == 8
        assert config["data"]["imgsz"] == 1280

    def test_progress_una_por_epoca_con_senales(self, qapp, tmp_path, deps_falsos):
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="feliz3")
        recolector = _correr(st)

        eventos = _eventos(tmp_path)
        progreso = [e["data"] for e in eventos if e["event"] == "train.progress"]
        assert [p["epoca"] for p in progreso] == [1, 2, 3]
        assert all(p["total"] == 3 for p in progreso)
        assert all({"precision", "recall", "caja"} <= set(p) for p in progreso)
        # Las metricas suben epoca a epoca (el doble las hace crecientes).
        assert progreso[-1]["map50"] > progreso[0]["map50"]

        # Las senales de Qt llegaron a la misma corrida, en el mismo orden.
        assert recolector.progreso == [(1, 3), (2, 3), (3, 3)]
        assert [m.epoch for m in recolector.metricas] == [1, 2, 3]
        assert recolector.metricas[-1].map50 > recolector.metricas[0].map50
        assert recolector.abortados == 0
        assert recolector.fallos == []
        assert any("Entrenamiento iniciado" in l for l in recolector.lineas)
        assert any("PyTorch 2.5.1+fake" in l for l in recolector.lineas)

    def test_complete_apunta_a_best_pt_real(self, qapp, tmp_path, deps_falsos):
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="feliz4")
        recolector = _correr(st)

        eventos = _eventos(tmp_path)
        completa = next(e for e in eventos if e["event"] == "train.complete")
        peso = completa["data"]["weights"]
        esperado = RUNS_ENTRENAMIENTO / "feliz4" / "weights" / "best.pt"
        assert peso == str(esperado)
        assert esperado.exists()            # el doble escribio donde el runner busca
        assert completa["data"]["familias"] == 1
        assert isinstance(completa["ms"], int)
        assert recolector.finales == [peso]

    def test_mejor_epoca_queda_en_el_estado(self, qapp, tmp_path, deps_falsos):
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="feliz5")
        _correr(st)

        assert st.best_map50 == pytest.approx(0.70)   # 0.60 + 0.05 * 2 epocas
        assert st.best_epoch == 3


# ── Aborto: la solicitud del usuario se ejecuta en frontera de época ──
class TestAborto:
    def test_solicitud_de_detencion_para_en_frontera_de_epoca(self, qapp, tmp_path, deps_falsos):
        """request_abort() a mitad de corrida: las epocas que faltaban nunca corren.

        La solicitud llega dentro del propio ``emit`` de progreso (slot sincrono,
        sin segundo plano en la prueba), asi que ``consume_abort`` la ve en
        la MISMA frontera de epoca 1. En Qt real el click del usuario cae
        entre fronteras (senal encolada cross-thread) y el stop cae en la
        siguiente; el contrato es el mismo: se para en una frontera, nunca a
        mitad, y el run NO llega a train.complete.
        """
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="aborto")
        runner = TrainerRunner(st)
        recolector = _Recolector()
        estado = {"pedido": False}

        def _al_progreso(ep: int, total: int) -> None:
            recolector.progreso.append((ep, total))
            if ep >= 1 and not estado["pedido"]:
                estado["pedido"] = True
                st.request_abort()   # el usuario pulso «Detener» tras la epoca 1

        runner.epoch_metrics.connect(recolector.metricas.append)
        runner.progress.connect(_al_progreso)
        runner.aborted.connect(lambda: setattr(recolector, "abortados", recolector.abortados + 1))
        runner.log_line.connect(recolector.lineas.append)
        runner.run()

        eventos = _eventos(tmp_path)
        assert [e["event"] for e in eventos] == [
            "train.config",
            "train.progress",   # epoca 1 completa: ahi llego la solicitud
            "train.aborted",    # y ahi mismo se honro: epocas 2 y 3 no corren
        ]
        assert recolector.abortados == 1
        assert recolector.finales == []          # un abort NO es un train.complete
        assert recolector.progreso == [(1, 3)]
        assert [m.epoch for m in recolector.metricas] == [1]
        assert estado["pedido"]                  # la solicitud se hizo en mitad del run
        assert st._abort is False, "la bandera quedo consumida exacta vez, no clavada"
        assert any("Solicitud de detención" in l for l in recolector.lineas)

        abortado = next(e for e in eventos if e["event"] == "train.aborted")
        assert abortado["data"]["familias_completadas"] == 1
        assert isinstance(abortado["ms"], int)
        assert abortado["run_id"] == "yolov8m.pt"

    def test_keyboard_interrupt_emite_train_aborted_sin_completar(self, qapp, tmp_path, deps_falsos):
        def _ctrl_c(_modelo, _kwargs) -> None:
            raise KeyboardInterrupt

        FakeYOLO.al_entrenar = _ctrl_c
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="teclado")
        recolector = _correr(st)

        eventos = _eventos(tmp_path)
        assert [e["event"] for e in eventos] == ["train.config", "train.aborted"]
        assert recolector.abortados == 1
        assert recolector.finales == []
        assert recolector.fallos == []
        # El aborto por interrupcion no trae ms ni familias_completadas.
        assert "ms" not in eventos[-1]


# ── Errores: cada fallo tiene su evento y su senal ───────────────────
class TestErrores:
    def test_fallo_de_train_emite_train_error_con_traceback(self, qapp, tmp_path, deps_falsos):
        def _boom(_modelo, _kwargs) -> None:
            raise RuntimeError("boom simulado")

        FakeYOLO.al_entrenar = _boom
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="error")
        recolector = _correr(st)

        eventos = _eventos(tmp_path)
        assert [e["event"] for e in eventos] == ["train.config", "train.error"]
        error = eventos[-1]
        assert error["level"] == "ERROR"
        assert error["module"] == "polyx.trainer"
        assert error["run_id"] == "yolov8m.pt"
        assert "RuntimeError: boom simulado" in "".join(error["exc"])

        assert len(recolector.fallos) == 1
        assert "RuntimeError" in recolector.fallos[0]
        assert recolector.finales == []
        assert recolector.abortados == 0
        assert recolector.progreso == []        # ni una epoca completa

    def test_ultralytics_inexistente_falla_antes_de_train_config(self, qapp, tmp_path, deps_falsos, monkeypatch):
        """sys.modules['ultralytics'] = None: el import revienta y no se escribe nada."""
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        st = _estado(_dataset_minimo(tmp_path / "dataset"), epochs=3, run_name="sinultra")
        recolector = _correr(st)

        assert len(recolector.fallos) == 1
        assert "Ultralytics" in recolector.fallos[0]
        assert _eventos(tmp_path) == [], "sin ultralytics no se escribio ni train.config"
        assert not (tmp_path / "_polyx_logs" / "trainer.jsonl").exists()

    def test_sin_data_yaml_falla_sin_escribir_eventos(self, qapp, tmp_path, deps_falsos):
        st = _estado(None, run_name="sinyaml")   # TrainerState sin dataset
        recolector = _correr(st)

        assert len(recolector.fallos) == 1
        assert "data.yaml" in recolector.fallos[0]
        assert _eventos(tmp_path) == []


# ── Comparación de familias: v8 + v11 en la misma corrida ────────────
class TestComparacionDeFamilias:
    def test_comparar_familias_entrena_las_dos_y_reporta_la_ultima(self, qapp, tmp_path, deps_falsos):
        st = _dataset_minimo(tmp_path / "dataset") and _estado(
            _dataset_minimo(tmp_path / "dataset"), epochs=2, run_name="corrida"
        )
        st.model.compare_families = True
        recolector = _correr(st)

        # Cada familia arranco con sus propios pesos base.
        assert [m.weights for m in FakeYOLO.instances] == ["yolov8m.pt", "yolo11m.pt"]
        # Y cada una corrio sus 2 epocas, con el progreso reiniciado por familia.
        assert recolector.progreso == [(1, 2), (2, 2), (1, 2), (2, 2)]
        assert [m.epoch for m in recolector.metricas] == [1, 2, 1, 2]

        eventos = _eventos(tmp_path)
        assert [e["event"] for e in eventos] == [
            "train.config",
            "train.progress", "train.progress",
            "train.progress", "train.progress",
            "train.complete",
        ]
        completa = next(e for e in eventos if e["event"] == "train.complete")
        esperado = RUNS_ENTRENAMIENTO / "corrida_v11" / "weights" / "best.pt"
        assert completa["data"]["weights"] == str(esperado)   # la ultima familia manda
        assert completa["data"]["familias"] == 2
        assert recolector.finales == [str(esperado)]
        assert esperado.exists()
        assert (RUNS_ENTRENAMIENTO / "corrida_v8" / "weights" / "best.pt").exists()
        assert any("COMPARACION DE ARQUITECTURAS" in l for l in recolector.lineas)
