"""La abstención: cuándo se reporta el polímero y cuándo «no asignable».

Lo que se fija aquí es que abstenerse NO pierde partículas. El estudio informa
por número de partículas, así que una que no se puede atribuir a un polímero
tiene que seguir contando en el total; lo único que cambia es bajo qué etiqueta.
"""

from polyx.core.detect.yolo import Detection
from polyx.core.quality.assignment import (
    NO_ASSIGNABLE,
    class_id_reported,
    count_reported,
    mark_assignables,
    resume,
)


def _det(clase: str, conf: float, cid: int = 0) -> Detection:
    return Detection(class_id=cid, class_name=clase, conf=conf, x1=0, y1=0, x2=10, y2=10)


def test_por_defecto_todo_se_asigna():
    """Sin tocar nada, una Detection recién creada sostiene su polímero."""
    assert _det("PET", 0.31).assignable is True
    assert class_id_reported(_det("PET", 0.31)) == "PET"


def test_umbral_cero_desactiva_la_abstencion():
    dets = [_det("PP", 0.05), _det("LDPE", 0.99)]
    assert mark_assignables(dets, 0.0) == 0
    assert [class_id_reported(d) for d in dets] == ["PP", "LDPE"]


def test_por_debajo_del_umbral_no_se_asigna():
    dets = [_det("PP", 0.30), _det("LDPE", 0.80)]
    assert mark_assignables(dets, 0.50) == 1
    assert class_id_reported(dets[0]) == NO_ASSIGNABLE
    assert class_id_reported(dets[1]) == "LDPE"


def test_el_umbral_es_inclusivo():
    """Justo en el umbral se asigna: el criterio es 'conf >= umbral'."""
    d = _det("PET", 0.50)
    mark_assignables([d], 0.50)
    assert class_id_reported(d) == "PET"


def test_abstenerse_no_pierde_particulas():
    """El total no cambia; solo se mueve de un polímero a «no asignable»."""
    dets = [_det("PET", 0.90), _det("PP", 0.20), _det("LDPE", 0.25)]
    antes = count_reported(dets)
    mark_assignables(dets, 0.50)
    despues = count_reported(dets)
    assert sum(antes.values()) == sum(despues.values()) == 3
    assert despues["PET"] == 1
    assert despues[NO_ASSIGNABLE] == 2
    assert "PP" not in despues and "LDPE" not in despues


def test_resumen_cuenta_y_porcentaje():
    dets = [_det("PET", 0.90), _det("PP", 0.20), _det("LDPE", 0.25), _det("PP", 0.10)]
    mark_assignables(dets, 0.50)
    r = resume(dets, 0.50)
    assert (r.total, r.assigned, r.no_assignables) == (4, 1, 3)
    assert r.percentage_no_assignable == 75.0
    assert r.active is True


def test_resumen_sin_abstencion_lo_declara():
    dets = [_det("PET", 0.10)]
    mark_assignables(dets, 0.0)
    r = resume(dets, 0.0)
    assert r.active is False
    assert r.no_assignables == 0
    assert "desactivada" in str(r)


def test_resumen_vacio_no_divide_por_cero():
    r = resume([], 0.5)
    assert r.total == 0
    assert r.percentage_no_assignable == 0.0


def test_el_informe_declara_lo_que_no_pudo_asignar(tmp_path):
    """De extremo a extremo: la etiqueta y el porcentaje llegan al HTML.

    Que la lógica funcione no basta: el fallo real sería que la clase reportada
    se calculara bien y el informe siguiera imprimiendo `class_name` a secas.
    """
    import cv2
    import numpy as np

    from polyx.core.detection_run import ImageResult, ModelInfo, RunParams, RunResult
    from polyx.gui import report_html as rh

    img = np.full((260, 320, 3), 18, np.uint8)
    cv2.circle(img, (110, 130), 20, (60, 210, 240), -1)
    cv2.circle(img, (210, 130), 20, (60, 210, 240), -1)
    image_path = tmp_path / "placa.png"
    ok, buffer = cv2.imencode(".png", img)
    assert ok
    image_path.write_bytes(buffer.tobytes())

    confident = Detection(class_id=0, class_name="PET", conf=0.90, x1=90, y1=110, x2=130, y2=150)
    uncertain = Detection(class_id=2, class_name="LDPE", conf=0.28, x1=190, y1=110, x2=230, y2=150)
    mark_assignables([confident, uncertain], 0.50)

    st = RunResult(
        run_dir=tmp_path / "run",
        images=[image_path],
        models=[ModelInfo(alias="m1", path=tmp_path / "fake.pt")],
        params=RunParams(conf_assignment=0.50),
        results={
            0: [
                ImageResult(
                    image_path=image_path,
                    model_idx=0,
                    predictions=[confident, uncertain],
                    gt=[],
                    has_gt=False,
                )
            ]
        },
    )

    output = tmp_path / "informe.html"
    rh.generate_report(st, output, sections=list(rh.IDS_SECTIONS))
    html = output.read_text(encoding="utf-8")

    assert NO_ASSIGNABLE in html
    # La partícula dudosa cuenta igual: dos detecciones, no una.
    assert "Total de detecciones" in html
    assert "1 / 2 (50.0 %)" in html


def test_detecciones_antiguas_sin_el_campo_se_asignan():
    """Un run guardado antes de que existiera la abstención sigue leyéndose.

    Las detecciones del ground truth y las de runs viejos no traen 'asignable';
    ahí la respuesta correcta es la clase de siempre, no «no asignable».
    """

    class DeteccionVieja:
        class_name = "PET"

    assert class_id_reported(DeteccionVieja()) == "PET"
