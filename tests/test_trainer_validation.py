"""La validación pre-vuelo del entrenador, contra datasets YOLO sintéticos.

Cada prueba arma un dataset mínimo en un directorio temporal y comprueba qué
hace ``validate_training_config`` con él: rechazar lo que rompería o envilecería
un entrenamiento (data.yaml corrupto o ausente, splits o labels faltantes,
hiperparámetros fuera de rango) y aceptar una configuración sana.

Los datasets son deliberadamente diminutos (un puñado de imágenes y labels de
dos cajas) para que la suite corra en milisegundos, pero el formato es el real
que produce el etiquetador: ``images/<split>/*.jpg`` + ``labels/<split>/*.txt``
y un ``data.yaml`` con ``path``/``train``/``val``/``names``.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from PySide6.QtWidgets import QApplication, QMessageBox

from polyx.gui.trainer.state import ModelConfig, TrainerState, validate_training_config
from polyx.gui.trainer.pages.dataset import DatasetPage
from polyx.gui.trainer.pages.entrenar import EntrenarPage

# Las páginas son QWidgets: sin display real las pruebas corren «offscreen».
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── Constructores de datasets sintéticos ────────────────────────────
# Dos cajas por imagen. Las áreas se eligen MUY distintas a propósito
# (0.10×0.10 = 0.01 vs 0.30×0.30 = 0.09, ratio 9): si fueran parecidas la
# auditoría emitiría el aviso «tamaño casi idéntico» y el dataset que
# llamamos «limpio» dejaría de serlo.
_BOX_CLASE_0 = "0 0.5 0.5 0.10 0.10"
_BOX_CLASE_1 = "1 0.5 0.5 0.30 0.30"


def crear_dataset(root: Path, *, n_train: int = 6, n_val: int = 6,
                  splits: tuple[str, ...] = ("train", "val"),
                  con_labels: bool = True, labels_vacios: bool = False,
                  clases: int = 2) -> Path:
    """Crea un dataset YOLO mínimo y devuelve la ruta de su ``data.yaml``.

    ``con_labels=False`` omite el directorio ``labels/`` entero.
    ``labels_vacios=True`` deja los ``.txt`` presentes pero sin cajas válidas.
    """
    for split in splits:
        n = n_train if split == "train" else n_val
        carpeta_img = root / "images" / split
        carpeta_img.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (carpeta_img / f"{split}_{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        if con_labels:
            carpeta_lbl = root / "labels" / split
            carpeta_lbl.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                contenido = "" if labels_vacios else f"{_BOX_CLASE_0}\n{_BOX_CLASE_1}\n"
                (carpeta_lbl / f"{split}_{i}.txt").write_text(contenido, encoding="utf-8")

    yaml_path = root / "data.yaml"
    lineas = ["path: ."]
    if "train" in splits:
        lineas.append("train: images/train")
    if "val" in splits:
        lineas.append("val: images/val")
    lineas.append("names:")
    for c in range(clases):
        lineas.append(f"  {c}: clase{c}")
    yaml_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return yaml_path


def estado(yaml_path: Path | None = None, **params) -> TrainerState:
    """TrainerState con el dataset indicado y los parámetros sobreescritos."""
    st = TrainerState()
    st.dataset.yaml_path = yaml_path
    for clave, valor in params.items():
        setattr(st.params, clave, valor)
    return st


def validar(yaml_path: Path | None = None, *, aug: dict | None = None, **params):
    """Atajo: valida un estado recién construido y devuelve (ok, mensajes)."""
    st = estado(yaml_path, **params)
    for clave, valor in (aug or {}).items():
        setattr(st.aug, clave, valor)
    return validate_training_config(st)


@pytest.fixture
def dataset_valido(tmp_path: Path) -> Path:
    """Dataset limpio (6 train + 6 val, 2 clases) que debe aceptarse tal cual."""
    return crear_dataset(tmp_path / "valido")


# ── Contrato y dataset sano ─────────────────────────────────────────
def test_contrato_devuelve_tupla_bool_y_lista():
    resultado = validar()
    assert isinstance(resultado, tuple) and len(resultado) == 2
    ok, msgs = resultado
    assert isinstance(ok, bool)
    assert isinstance(msgs, list)


def test_sin_dataset_se_rechaza():
    ok, msgs = validar()
    assert not ok
    assert any("data.yaml" in m for m in msgs), msgs


def test_dataset_valido_se_acepta(dataset_valido):
    ok, msgs = validar(dataset_valido)
    assert ok, msgs
    assert any("Dataset válido" in m for m in msgs), msgs
    # Un dataset limpio no debe arrastrar avisos: si aparece uno, el fixture
    # dejó de ser limpio y las demás pruebas pierden valor de referencia.
    assert not any(m.startswith("[aviso]") for m in msgs), msgs


def test_aviso_no_bloquea(tmp_path):
    """Val con solo 3 imágenes: la auditoría avisa pero no bloquea el run."""
    yaml_path = crear_dataset(tmp_path / "val_pequeno", n_val=3)
    ok, msgs = validar(yaml_path)
    assert ok, msgs
    assert any(m.startswith("[aviso]") for m in msgs), msgs


def test_mensajes_de_error_son_texto_no_vacio(tmp_path):
    ok, msgs = validar(tmp_path / "no" / "data.yaml")
    assert not ok
    assert msgs
    assert all(isinstance(m, str) and m.strip() for m in msgs)


# ── Modelo: nombres de pesos y familias/tamaños desconocidos ────────
def test_nombres_de_pesos_por_familia():
    """La nomenclatura de Ultralytics no es uniforme: v8 lleva «v», la 11 no."""
    v8 = ModelConfig(family="v8", size="m")
    v11 = ModelConfig(family="v11", size="n")
    assert v8.name == "yolov8m"
    assert v8.base_weights_name == "yolov8m.pt"
    assert v11.name == "yolo11n"
    assert v11.base_weights_name == "yolo11n.pt"


@pytest.mark.parametrize("familia,tamano", [
    ("v9", "m"), ("yolov8", "m"), ("v8", "z"), ("v11", "g"),
])
def test_modelo_desconocido_se_rechaza(dataset_valido, familia, tamano):
    st = estado(dataset_valido)
    st.model.family = familia
    st.model.size = tamano
    ok, msgs = validate_training_config(st)
    assert not ok, f"familia={familia} tamaño={tamano} debió rechazarse"
    assert any("Familia" in m or "Tamaño" in m for m in msgs), msgs


# ── data.yaml: ausencia, corrupción y dependencias externas ─────────
def test_yaml_inexistente_se_rechaza(tmp_path):
    ok, msgs = validar(tmp_path / "no" / "data.yaml")
    assert not ok
    assert any("no encontrado" in m for m in msgs), msgs


def test_yaml_corrupto_se_rechaza(tmp_path):
    malo = tmp_path / "malo.yaml"
    malo.write_text("names: [a, b\n", encoding="utf-8")  # corchete sin cerrar
    ok, msgs = validar(malo)
    assert not ok
    assert msgs


def test_yaml_que_es_directorio_no_tumba_el_preflight(tmp_path):
    """Dependencia externa hostil: la «ruta» existe pero es un directorio."""
    carpeta = tmp_path / "data.yaml"
    carpeta.mkdir()
    ok, msgs = validar(carpeta)
    assert not ok
    assert msgs  # mensaje descriptivo, no una excepción sin capturar


def test_yaml_no_utf8_se_rechaza(tmp_path):
    binario = tmp_path / "data.yaml"
    binario.write_bytes(b"\xff\xfe\x00\x01names:\x80")
    ok, msgs = validar(binario)
    assert not ok
    assert msgs


def test_fallo_inesperado_de_auditoria_se_reporta_sin_excepcion(monkeypatch, dataset_valido):
    """Si audit_dataset explota (p. ej. disco desconectado), el pre-flight
    devuelve ok=False con un mensaje, en vez de propagar la excepción."""
    import polyx.gui.trainer.state as state_mod

    def boom(_yaml):
        raise RuntimeError("disco desconectado")

    monkeypatch.setattr(state_mod, "audit_dataset", boom)
    ok, msgs = validar(dataset_valido)
    assert not ok
    assert any("No se pudo analizar el dataset" in m for m in msgs), msgs


# ── Splits: claves ausentes y rutas que no existen ──────────────────
def test_split_train_ausente_se_rechaza(tmp_path):
    yaml_path = crear_dataset(tmp_path / "sin_train", splits=("val",))
    ok, msgs = validar(yaml_path)
    assert not ok
    assert any("train" in m for m in msgs), msgs


def test_split_val_ausente_se_rechaza(tmp_path):
    yaml_path = crear_dataset(tmp_path / "sin_val", splits=("train",))
    ok, msgs = validar(yaml_path)
    assert not ok
    assert any("val" in m for m in msgs), msgs


def test_ruta_de_split_inexistente_se_rechaza(dataset_valido):
    """El data.yaml se movió de carpeta: la clave está pero la ruta no."""
    shutil.rmtree(dataset_valido.parent / "images" / "train")
    ok, msgs = validar(dataset_valido)
    assert not ok
    assert any("Ruta de train" in m for m in msgs), msgs


def test_sin_clases_definidas_se_rechaza(tmp_path):
    yaml_path = crear_dataset(tmp_path / "sin_clases")
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: {}\n", encoding="utf-8"
    )
    ok, msgs = validar(yaml_path)
    assert not ok
    assert any("Sin clases" in m for m in msgs), msgs


def test_names_sin_valor_se_rechaza_sin_typeerror(tmp_path):
    """``names:`` sin valor parsea a None: debe dar el hallazgo limpio, no un TypeError."""
    yaml_path = crear_dataset(tmp_path / "names_nulo")
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n", encoding="utf-8"
    )
    ok, msgs = validar(yaml_path)
    assert not ok
    assert any("Sin clases" in m for m in msgs), msgs
    assert not any("No se pudo analizar" in m for m in msgs), msgs


# ── Labels: ausentes por completo o sin cajas válidas ───────────────
def test_imagenes_sin_directorio_de_labels_se_rechazan(tmp_path):
    yaml_path = crear_dataset(tmp_path / "sin_labels", con_labels=False)
    ok, msgs = validar(yaml_path)
    assert not ok
    assert any("fondo" in m or "etiquetas" in m for m in msgs), msgs


def test_train_con_labels_vacios_se_rechaza(tmp_path):
    """Labels presentes pero sin cajas: el modelo aprendería solo fondo."""
    yaml_path = crear_dataset(tmp_path / "labels_vacios", labels_vacios=True)
    ok, msgs = validar(yaml_path)
    assert not ok
    assert any("fondo" in m for m in msgs), msgs


# ── Hiperparámetros: fuera de rango ─────────────────────────────────
# El tercer elemento es el fragmento que debe aparecer en el mensaje: los
# nombres visibles no siempre coinciden con el atributo («epochs» → «Épocas»).
@pytest.mark.parametrize("campo,valor,fragmento", [
    ("epochs", 0, "Épocas"), ("epochs", -5, "Épocas"), ("epochs", 10001, "Épocas"),
    ("batch", 0, "Batch"), ("batch", 513, "Batch"),
    ("imgsz", 100, "imgsz"), ("imgsz", 0, "imgsz"), ("imgsz", 16, "imgsz"),
    ("imgsz", 16384, "imgsz"), ("imgsz", "abc", "imgsz"),
    ("lr0", 0.0, "lr0"), ("lr0", 1.5, "lr0"),
    ("weight_decay", 1.5, "weight_decay"), ("weight_decay", -0.1, "weight_decay"),
    ("momentum", -0.1, "momentum"), ("momentum", 1.5, "momentum"),
    ("label_smoothing", 1.0, "label_smoothing"), ("label_smoothing", -0.01, "label_smoothing"),
    ("workers", 65, "workers"), ("workers", -1, "workers"),
    ("save_period", 0, "save_period"),
])
def test_hiperparametro_fuera_de_rango_se_rechaza(dataset_valido, campo, valor, fragmento):
    ok, msgs = validar(dataset_valido, **{campo: valor})
    assert not ok, f"{campo}={valor!r} debió rechazarse"
    assert any(fragmento in m for m in msgs), msgs


@pytest.mark.parametrize("imgsz", [32, 64, 1280, 1920, 8192, "1280"])
def test_imgsz_valido_se_acepta(dataset_valido, imgsz):
    ok, msgs = validar(dataset_valido, imgsz=imgsz)
    assert ok, msgs


@pytest.mark.parametrize("epochs,batch", [(1, 1), (150, 8), (10000, 512)])
def test_epochs_y_batch_en_los_bordes_se_aceptan(dataset_valido, epochs, batch):
    ok, msgs = validar(dataset_valido, epochs=epochs, batch=batch)
    assert ok, msgs


# ── Device y cache: los dos campos que Ultralytics valida tarde ─────
@pytest.mark.parametrize("device", ["gpu", "", "cuda:x", "0,", "tpu", "  "])
def test_device_invalido_se_rechaza(dataset_valido, device):
    ok, msgs = validar(dataset_valido, device=device)
    assert not ok, f"device={device!r} debió rechazarse"
    assert any("Device" in m for m in msgs), msgs


@pytest.mark.parametrize("device", ["cpu", "mps", "0", "cuda:0", "0,1"])
def test_device_valido_se_acepta(dataset_valido, device):
    ok, msgs = validar(dataset_valido, device=device)
    assert ok, msgs


@pytest.mark.parametrize("cache", ["nope", "disk2", 123, 1.5])
def test_cache_invalido_se_rechaza(dataset_valido, cache):
    ok, msgs = validar(dataset_valido, cache=cache)
    assert not ok, f"cache={cache!r} debió rechazarse"
    assert any("cache" in m for m in msgs), msgs


@pytest.mark.parametrize("cache", ["disk", "ram", "true", "false", True, False])
def test_cache_valido_se_acepta(dataset_valido, cache):
    ok, msgs = validar(dataset_valido, cache=cache)
    assert ok, msgs


# ── Aumentación: rangos [0, 1] ──────────────────────────────────────
@pytest.mark.parametrize("campo,valor", [
    ("hsv_h", 1.5), ("hsv_s", -0.1), ("hsv_v", 2.0),
    ("fliplr", 1.1), ("mosaic", -0.5), ("mixup", 1.01), ("copy_paste", 1.5),
])
def test_augmentacion_fuera_de_rango_se_rechaza(dataset_valido, campo, valor):
    ok, msgs = validar(dataset_valido, aug={campo: valor})
    assert not ok, f"{campo}={valor!r} debió rechazarse"
    assert any(campo in m for m in msgs), msgs


# ── Q5: entradas malformadas no deben lanzar excepción ──────────────
def test_preflight_nunca_lanza_excepcion_con_tipos_raros(dataset_valido):
    """Tipos inesperados en los parámetros (None, texto) se reportan como
    error de validación, no como un TypeError que tumbe la UI."""
    casos = [
        {"epochs": None}, {"batch": "muchas"}, {"imgsz": None},
        {"lr0": "alto"}, {"device": None}, {"cache": None},
        {"workers": None}, {"label_smoothing": None},
    ]
    for caso in casos:
        ok, msgs = validar(dataset_valido, **caso)
        assert ok is False, f"{caso!r} debió rechazarse"
        assert isinstance(msgs, list) and msgs, f"{caso!r} sin mensajes"


# ── Q6: 10× el fixture base (12 → 120 imágenes) ─────────────────────
def test_dataset_10x_valida_bajo_cota_de_tiempo(tmp_path):
    """A 10× el tamaño base el escaneo de labels sigue siendo lineal y rápido.

    El recurso que satura primero es el I/O de leer un ``.txt`` por imagen
    (``audit_dataset`` hace ``rglob`` y luego un read por label). A 120
    imágenes debe resolverse en bastante menos que la cota, que deja margen
    para máquinas lentas de CI.
    """
    yaml_path = crear_dataset(tmp_path / "grande", n_train=60, n_val=60)
    inicio = time.monotonic()
    ok, msgs = validar(yaml_path)
    duracion = time.monotonic() - inicio
    assert ok, msgs
    assert duracion < 15.0, f"validación 10× demasiado lenta: {duracion:.2f}s"


# ── T03: cableado de UI ────────────────────────────────────────────────────
# El gate de arranque de EntrenarPage y la reactividad de DatasetPage. Se
# construyen las páginas REALES (no dobles) con una QApplication offscreen; se
# sustituyen solo los límites externos que no se pueden ejercitar en CI: el
# diálogo modal y el TrainerRunner, que lanzaría un entrenamiento de verdad.


@pytest.fixture(scope="module")
def qapp():
    """Una sola QApplication por módulo: PySide6 no permite recrearla."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def estado_valido(tmp_path: Path) -> TrainerState:
    """Estado con dataset limpio en disco y parámetros por defecto (válidos)."""
    return estado(crear_dataset(tmp_path / "valido"))


class _SenalFalsa:
    """Señal mínima: ``connect`` no hace nada (no hay hilo real en los tests)."""

    def connect(self, *args, **kwargs):
        pass


def _parchear_runner(monkeypatch) -> list:
    """Sustituye TrainerRunner por un doble que registra creación y arranque."""
    llamadas: list[tuple] = []

    class RunnerFalso:
        epoch_metrics = _SenalFalsa()
        log_line = _SenalFalsa()
        progress = _SenalFalsa()
        finished_ok = _SenalFalsa()
        aborted = _SenalFalsa()
        failed = _SenalFalsa()

        def __init__(self, state):
            llamadas.append(("creado", state))

        def start(self):
            llamadas.append(("arrancado", None))

    monkeypatch.setattr("polyx.gui.trainer.pages.entrenar.TrainerRunner", RunnerFalso)
    return llamadas


def _parchear_avisos(monkeypatch) -> list[str]:
    """Captura los diálogos QMessageBox.warning sin bloquear la prueba."""
    avisos: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda parent, titulo, texto: avisos.append(texto)
    )
    return avisos


class TestGateDeArranque:
    """``EntrenarPage._start`` no crea el runner si la validación pre-vuelo falla."""

    def test_bloquea_con_epochs_fuera_de_rango(self, qapp, tmp_path, monkeypatch):
        st = estado_valido(tmp_path)
        st.params.epochs = 0  # fuera de rango
        page = EntrenarPage(st)
        avisos = _parchear_avisos(monkeypatch)
        corridas = _parchear_runner(monkeypatch)

        page._start()

        assert avisos, "el gate debe mostrar un diálogo descriptivo"
        assert any("Épocas" in a for a in avisos), avisos
        assert corridas == [], "no debe construirse el runner con la config inválida"

    def test_sin_dataset_bloquea_mencionando_data_yaml(self, qapp, monkeypatch):
        st = TrainerState()  # sin data.yaml
        page = EntrenarPage(st)
        avisos = _parchear_avisos(monkeypatch)
        corridas = _parchear_runner(monkeypatch)

        page._start()

        assert any("data.yaml" in a for a in avisos), avisos
        assert corridas == []

    def test_config_valida_arranca_sin_dialogos(self, qapp, tmp_path, monkeypatch):
        st = estado_valido(tmp_path)
        page = EntrenarPage(st)
        avisos = _parchear_avisos(monkeypatch)
        corridas = _parchear_runner(monkeypatch)

        page._start()

        assert avisos == [], avisos
        assert corridas == [("creado", st), ("arrancado", None)], corridas
        assert not page.btn_start.isEnabled()
        assert page.btn_stop.isEnabled()

    def test_avisos_no_bloquean_el_arranque(self, qapp, tmp_path, monkeypatch):
        """Val chico avisa pero no es error: el run debe poder arrancar."""
        st = estado(crear_dataset(tmp_path / "val_pequeno", n_val=3))
        page = EntrenarPage(st)
        avisos = _parchear_avisos(monkeypatch)
        corridas = _parchear_runner(monkeypatch)

        page._start()

        assert avisos == [], avisos
        assert len(corridas) == 2, corridas


class TestReactividadValidacion:
    """DatasetPage revalida en caliente: parámetros al instante, dataset completo."""

    def test_cambio_de_parametros_actualiza_la_fila(self, qapp, tmp_path):
        st = estado_valido(tmp_path)
        page = DatasetPage(st)
        page._validate_dataset()
        assert page._val_checks[5][0].text() == "✓", "la config por defecto es válida"

        st.params.epochs = 0
        st.params_changed.emit()

        assert page._val_checks[5][0].text() == "✗"
        assert "Épocas" in page._val_checks[5][1].toolTip()

        st.params.epochs = 100
        st.params_changed.emit()

        assert page._val_checks[5][0].text() == "✓"

    def test_modelo_invalido_actualiza_la_fila(self, qapp, tmp_path):
        st = estado_valido(tmp_path)
        page = DatasetPage(st)
        page._validate_dataset()

        st.model.size = "z"
        st.model_changed.emit()

        assert page._val_checks[5][0].text() == "✗"
        assert "Tamaño" in page._val_checks[5][1].toolTip()

    def test_augmentacion_fuera_de_rango_actualiza_la_fila(self, qapp, tmp_path):
        st = estado_valido(tmp_path)
        page = DatasetPage(st)
        page._validate_dataset()

        st.aug.mosaic = 5.0
        st.aug_changed.emit()

        assert page._val_checks[5][0].text() == "✗"
        assert "mosaic" in page._val_checks[5][1].toolTip()

    def test_cambio_de_dataset_dispara_la_auditoria_completa(self, qapp, tmp_path):
        st = TrainerState()
        page = DatasetPage(st)
        page._validate_dataset()
        assert page._val_checks[0][0].text() == "✗", "sin data.yaml debe fallar"

        st.dataset.yaml_path = crear_dataset(tmp_path / "valido")
        st.dataset.train_count = 6
        st.dataset.val_count = 6
        st.dataset_changed.emit()

        assert page._val_checks[0][0].text() == "✓"
        assert page._val_checks[1][0].text() == "✓"
        assert page._val_checks[2][0].text() == "✓"
        assert page._val_checks[3][0].text() == "✓"
        assert page._val_checks[4][0].text() == "✓"
        assert page._val_checks[5][0].text() == "✓"
        assert page.audit_list.count() > 0, "los hallazgos deben volcarse al panel"

    def test_recargar_yaml_valida_por_senal(self, qapp, tmp_path):
        """``_reload_yaml`` ya no valida a mano: lo hace la señal ``dataset_changed``."""
        st = TrainerState()
        page = DatasetPage(st)
        page.ed_yaml.setText(str(crear_dataset(tmp_path / "valido")))

        page._reload_yaml()

        assert "✓" in page.lbl_yaml_status.text()
        assert page._val_checks[0][0].text() == "✓"
        assert page._val_checks[5][0].text() == "✓"
