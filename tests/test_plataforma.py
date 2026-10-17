"""Que Poly-X siga arrancando en macOS y en Linux, no solo en Windows.

Estas pruebas existen porque el fallo que cubren no se ve desarrollando: en
Windows todo pasa aunque el codigo tenga ``os.startfile`` o ``device="0"``
grabados a fuego, y el problema solo aparece en el Mac de otra persona, que
no lo puede depurar.

Se simula el sistema operativo recargando el modulo con ``sys.platform``
falseado. No sustituye a probarlo en un Mac de verdad -- no valida que Qt
pinte bien ni que MPS funcione -- pero si fija que la logica de despacho no
se rompa al editar el resto.
"""
from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest

import polyx.model.plataforma as P


def _recargar_como(plataforma: str, maquina: str = "x86_64"):
    """Devuelve el modulo recargado como si corriera en ese sistema."""
    with mock.patch.object(sys, "platform", plataforma), \
         mock.patch("platform.machine", return_value=maquina), \
         mock.patch("platform.mac_ver", return_value=("14.5", ("", "", ""), "")):
        importlib.reload(P)
        # Se devuelven los datos ya extraidos: fuera del contexto el modulo
        # se recarga al estado real y las funciones dejarian de mentir.
        return {
            "es_mac": P.ES_MAC,
            "es_windows": P.ES_WINDOWS,
            "es_linux": P.ES_LINUX,
            "arquitectura": P.arquitectura_mac(),
            "sistema": P.nombre_sistema(),
            "actualizador": P.nombre_actualizador(),
            "rotulo": P.etiqueta_dispositivos(),
        }


@pytest.fixture(autouse=True)
def _restaurar_modulo():
    """Deja el modulo como estaba: si no, contamina las pruebas siguientes."""
    yield
    importlib.reload(P)


# ── Identificacion del sistema ──
def test_reconoce_los_tres_sistemas():
    assert _recargar_como("darwin", "arm64")["es_mac"]
    assert _recargar_como("win32")["es_windows"]
    assert _recargar_como("linux")["es_linux"]


def test_distingue_apple_silicon_de_intel():
    """MPS solo existe en Apple Silicon: confundirlos ofreceria al usuario de
    un Mac Intel una aceleracion que su equipo no puede dar."""
    assert _recargar_como("darwin", "arm64")["arquitectura"] == "Apple Silicon"
    assert _recargar_como("darwin", "x86_64")["arquitectura"] == "Intel"
    # Fuera de un Mac la pregunta no aplica y debe quedar vacia, no "Intel".
    assert _recargar_como("win32")["arquitectura"] == ""


def test_el_nombre_del_sistema_en_mac_no_usa_la_version_de_darwin():
    """platform.release() en un Mac da el kernel (23.5.0), que no le dice nada
    a nadie. Tiene que salir la version de macOS."""
    sistema = _recargar_como("darwin", "arm64")["sistema"]
    assert "macOS 14.5" in sistema
    assert "Apple Silicon" in sistema


# ── Actualizador ──
def test_cada_sistema_pide_su_actualizador():
    """Un .bat no se ejecuta en un Mac. Si esto se rompe, el boton
    'Actualizar' apunta a un archivo que no existe y no hace nada."""
    assert _recargar_como("win32")["actualizador"] == "actualizar.bat"
    assert _recargar_como("darwin", "arm64")["actualizador"].endswith(".command")
    assert _recargar_como("linux")["actualizador"].endswith(".sh")


# ── Dispositivo de computo ──
def test_el_rotulo_no_ofrece_gpu_donde_no_la_hay():
    """En un Mac no hay una GPU '0' que elegir; ofrecerla lleva a que el
    analisis falle con un error de CUDA que confunde."""
    assert "0" in _recargar_como("win32")["rotulo"]
    assert "mps" in _recargar_como("darwin", "arm64")["rotulo"]
    intel = _recargar_como("darwin", "x86_64")["rotulo"]
    assert "mps" not in intel and "cpu" in intel


def test_dispositivo_disponible_devuelve_algo_usable():
    """Sea cual sea la maquina, tiene que salir un valor que YOLO acepte.
    Nunca cadena vacia ni None: eso reventaria la inferencia."""
    d = P.dispositivo_disponible()
    assert d in (P.DISPOSITIVO_CUDA, P.DISPOSITIVO_MPS, P.DISPOSITIVO_CPU)


def test_sin_torch_cae_a_cpu_en_vez_de_reventar():
    """Un entorno a medio instalar no debe tumbar el arranque: sin torch la
    respuesta correcta es 'cpu', no una excepcion."""
    with mock.patch.dict(sys.modules, {"torch": None}):
        assert P.dispositivo_disponible() == P.DISPOSITIVO_CPU


def test_toda_descripcion_de_dispositivo_es_legible():
    for d in ("0", "mps", "cpu"):
        assert P.descripcion_dispositivo(d)
        assert d != P.descripcion_dispositivo(d), "debe explicar, no repetir"


# ── Abrir archivos ──
def test_abrir_algo_que_no_existe_devuelve_False_sin_lanzar():
    """Se usa tras generar un informe. Que no se abra el visor es molesto;
    que tumbe el analisis recien terminado, no es aceptable."""
    assert P.abrir_en_el_sistema("no_existe_este_archivo_12345.txt") is False


def test_abrir_usa_el_comando_de_cada_sistema(tmp_path):
    archivo = tmp_path / "x.txt"
    archivo.write_text("hola", encoding="utf-8")

    with mock.patch.object(P, "ES_WINDOWS", False), \
         mock.patch.object(P, "ES_MAC", True), \
         mock.patch("subprocess.Popen") as popen:
        assert P.abrir_en_el_sistema(archivo) is True
        assert popen.call_args[0][0][0] == "open"

    with mock.patch.object(P, "ES_WINDOWS", False), \
         mock.patch.object(P, "ES_MAC", False), \
         mock.patch("subprocess.Popen") as popen:
        assert P.abrir_en_el_sistema(archivo) is True
        assert popen.call_args[0][0][0] == "xdg-open"


def test_si_el_comando_falla_devuelve_False_sin_lanzar(tmp_path):
    archivo = tmp_path / "x.txt"
    archivo.write_text("hola", encoding="utf-8")
    with mock.patch.object(P, "ES_WINDOWS", False), \
         mock.patch("subprocess.Popen", side_effect=OSError("no hay xdg-open")):
        assert P.abrir_en_el_sistema(archivo) is False
