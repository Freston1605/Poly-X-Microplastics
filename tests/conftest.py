"""Configuracion compartida de la suite.

El idioma del programa se resuelve una vez a partir del entorno (ver
``polyx.core.i18n``) y se cachea en el modulo, asi que varios tests que no
fuercen idioma propio --p. ej. ``test_assignment`` lee texto del informe--
acaban dependiendo del locale de la maquina donde corra pytest. Con un
locale en ingles el informe sale en ingles y la prueba falla; en CI el
locale suele estar vacio y pasa por casualidad.

The suite pins Spanish explicitly and reproducibly. Tests that exercise the
language itself -- ``test_bilingual_report``, the CLI output and the
widget-tree switch -- select a language through ``i18n.set_language`` (the
``selected_language`` fixture below) and restore it afterwards. The old
environment variable is gone: ``POLYX_LOCALE`` is the only override.

Aislamiento de observabilidad y logs (Decision D007):
Varios modulos de prueba (como ``test_assignment`` o ``test_bilingual_report``)
invocan el pipeline real de generacion de reportes (``report_html.generate_report``),
el cual instrumenta eventos a traves de ``logger_para("detector")``. Sin una variable
de entorno ``POLYX_LOG_DIR`` explicita, estos loggers escriben en el directorio
real del usuario (``~/.polyx/logs/``). Para garantizar que la ejecucion de la suite
de pruebas sea completamente hermetica y nunca contamine ``~/.polyx/logs/``,
definimos un fixture autouse que aisla ``POLYX_LOG_DIR`` en un directorio temporal
por test (``tmp_path / "_polyx_logs"``), limpia variables de entorno ambientales
(``POLYX_LOG``, ``POLYX_LOG_TOTAL_CAP_MB``) y cierra y reinicia los loggers y el
run_id en setup y teardown.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_RAIZ = Path(__file__).resolve().parents[1]
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("POLYX_LOCALE", "es")


def split_dataset(
    root: Path,
    splits: dict[str, dict[str, str | None]],
    *,
    names: tuple[str, ...] = ("PET", "PP"),
    suffix: str = ".png",
    classes: bool = False,
    data_yaml: bool = False,
) -> Path:
    """Build a conventional dataset (``images/<split>`` beside ``labels/<split>``).

    ``splits`` maps a split name to its images as ``{stem: label text}``; a
    ``None`` value writes the image without a label. The folders are created
    for every declared split even when it holds no image, so a test can add
    files later. ``classes`` writes the dataset-root ``classes.txt``;
    ``data_yaml`` writes a ``data.yaml`` declaring the splits and ``names``.
    Returns the dataset root.
    """
    root.mkdir(parents=True, exist_ok=True)
    for split, images in splits.items():
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for stem, label_text in images.items():
            ok, buf = cv2.imencode(suffix, np.zeros((40, 60, 3), dtype=np.uint8))
            assert ok
            (images_dir / f"{stem}{suffix}").write_bytes(bytes(buf))
            if label_text is not None:
                (labels_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    if classes:
        (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    if data_yaml:
        lines = ["path: .", *(f"{split}: images/{split}" for split in splits), "names:"]
        lines += [f"  {index}: {name}" for index, name in enumerate(names)]
        (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _aislar_logs_suite(tmp_path, monkeypatch):
    """Aisla POLYX_LOG_DIR a tmp_path / '_polyx_logs' para toda la suite.

    Evita que generate_report y modulos de prueba escriban en ~/.polyx/logs/.
    No crea el directorio con .mkdir() para no colisionar con tests que
    crean tmp_path / 'logs'. Cierra loggers y limpia run_id antes y despues
    de cada prueba.
    """
    monkeypatch.setenv("POLYX_LOG_DIR", str(tmp_path / "_polyx_logs"))
    monkeypatch.delenv("POLYX_LOG", raising=False)
    monkeypatch.delenv("POLYX_LOG_TOTAL_CAP_MB", raising=False)

    import polyx.core.observability as ob

    ob._close_loggers()
    ob.clear_run_id()
    yield
    ob._close_loggers()
    ob.clear_run_id()


@pytest.fixture(autouse=True)
def _isolate_theme_saves(tmp_path, monkeypatch):
    """Keep saved preferences out of the developer's home directory.

    ``theme.set_theme`` and ``i18n.set_language`` persist through
    ``polyx.core.preferences``. Point that file at the test's ``tmp_path``
    and restore the active palette afterwards, so the suite is hermetic. The
    theme part is lazy: a test module that never imported the theme does not
    acquire a PySide6 dependency, so headless suites such as the install
    smoke tests stay headless.
    """
    from polyx.core import preferences

    monkeypatch.setattr(preferences, "FILE", tmp_path / "polyx_preferences.json")

    theme = sys.modules.get("polyx.gui.theme")
    if theme is None:
        yield
        return

    original = theme.theme()
    yield
    theme._apply(original)


@pytest.fixture
def qapp():
    """Una sola QApplication por módulo: PySide6 no permite recrearla."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def selected_language():
    """Select the active language through the public API and restore it.

    The one seam every surface test uses: set the language, render the
    surface, assert what the user sees. Restoring keeps the suite in the
    language conftest pins, so a test never depends on which one ran before.
    """
    from polyx.core import i18n

    previous = i18n.language()

    def choose(tag: str) -> None:
        assert i18n.set_language(tag) is True

    yield choose
    i18n.set_language(previous)
