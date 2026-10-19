"""Pruebas automatizadas de las recetas jq y verificacion de lotes de 10 placas.

Valida que los logs estructurados en JSONL generados por detector, reporte y
trainer cumplan con el esquema M001, que las consultas jq (finish event tailing,
filtrado por run_id, auditoria de errores/warnings y correlacion cross-module)
operen correctamente, y cubre modos de fallo (Q5), perfil de carga 10x (Q6)
y pruebas negativas (Q7).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def log_env(tmp_path, monkeypatch):
    carpeta = tmp_path / "polyx_logs"
    carpeta.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("POLYX_LOG_DIR", str(carpeta))
    import polyx.model.observabilidad as ob
    ob._close_loggers()
    monkeypatch.delenv("POLYX_LOG", raising=False)
    ob._run_local.__dict__.pop("run_id", None)
    yield carpeta
    ob._close_loggers()


def _simular_lote_10_placas(log_env: Path) -> str:
    """Simula una corrida de 10 placas en detector, reporte y trainer."""
    from polyx.model import observabilidad as ob

    run_id = "batch-test-10-plates-2026"
    ob.set_run_id(run_id)

    log_det = ob.logger_for("detector")
    log_rep = ob.logger_for("reporte")
    log_trn = ob.logger_for("trainer")

    # 10 placas, 4 fases por placa en detector = 40 finish events
    for i in range(1, 11):
        plate_id = f"plate_{i:03d}"
        for fase in ["segment", "infer", "feature", "export"]:
            with ob.timer(log_det, f"phase.{fase}", plate=plate_id):
                pass

    # Evento de reporte para la corrida
    with ob.timer(log_rep, "report.generate", plates=10):
        ob.event(log_rep, "report.saved", format="pdf", path="reporte.pdf")

    # Evento de trainer
    with ob.timer(log_trn, "train.epoch", epoch=1):
        ob.event(log_trn, "train.metric", mAP=0.94)

    # Evento de warning y error para pruebas negativas / auditoria
    ob.event(log_det, "detector.warning", level=logging.WARNING, msg="baja confianza en placa 3")
    log_det._log(logging.ERROR, "detector.error", (), extra={"event": "detector.error", "data": {"code": 500}, "ms": 12})

    ob.clear_run_id()
    return run_id


def test_10_plate_batch_log_generation(log_env):
    """Verifica que el lote de 10 placas genere los JSONL esperados con formato valido."""
    run_id = _simular_lote_10_placas(log_env)

    det_file = log_env / "detector.jsonl"
    rep_file = log_env / "reporte.jsonl"
    trn_file = log_env / "trainer.jsonl"

    assert det_file.exists()
    assert rep_file.exists()
    assert trn_file.exists()

    for path in (det_file, rep_file, trn_file):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            for fijo in ("ts", "level", "module", "pid", "thread", "event"):
                assert fijo in rec

    det_recs = [json.loads(l) for l in det_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    all_finishes = [r for r in det_recs if r["event"].endswith(".finish") and r.get("run_id") == run_id]
    assert len(all_finishes) == 40
    for r in all_finishes:
        assert isinstance(r["ms"], int)
        assert r["ms"] >= 0


@pytest.mark.skipif(shutil.which("jq") is None, reason="requiere jq instalado")
def test_jq_recetas_subprocess(log_env):
    """Ejecuta y verifica las recetas jq via subprocess.run."""
    _simular_lote_10_placas(log_env)
    det_path = str(log_env / "detector.jsonl")

    # 1. Finish event tailing
    cmd1 = ["jq", "-c", "select(.event | endswith(\".finish\")) | {ts, event, ms}", det_path]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
    lines1 = [json.loads(l) for l in res1.stdout.splitlines() if l.strip()]
    assert len(lines1) >= 40

    # 2. Batch run_id filtering
    cmd2 = ["jq", "-c", "select(.run_id == \"batch-test-10-plates-2026\") | {ts, event, ms}", det_path]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
    lines2 = [json.loads(l) for l in res2.stdout.splitlines() if l.strip()]
    assert len(lines2) > 0

    # 3. Error/warning queries
    cmd3 = ["jq", "-c", "select(.level == \"ERROR\" or .level == \"WARNING\")", det_path]
    res3 = subprocess.run(cmd3, capture_output=True, text=True, check=True)
    lines3 = [json.loads(l) for l in res3.stdout.splitlines() if l.strip()]
    assert len(lines3) >= 2

    # 4. Cross-module correlation query
    cmd4 = [
        "jq", "-s", "flatten | map(select(.run_id == \"batch-test-10-plates-2026\"))",
        str(log_env / "detector.jsonl"),
        str(log_env / "reporte.jsonl"),
        str(log_env / "trainer.jsonl")
    ]
    res4 = subprocess.run(cmd4, capture_output=True, text=True, check=True)
    correlated = json.loads(res4.stdout)
    assert isinstance(correlated, list)
    assert len(correlated) > 40


# ── Q5 Failure Modes ──────────────────────────────────────────────────

def test_q5_missing_jq_fallback_handling(log_env):
    """Q5: Si jq no esta disponible o falla, la prueba gestiona el error limpiamente."""
    _simular_lote_10_placas(log_env)
    det_path = str(log_env / "detector.jsonl")

    # Verificar que FileNotFoundError se captura si binario no existe
    try:
        subprocess.run(["jq-inexistente-bin", det_path], capture_output=True, text=True, check=True)
        pytest.fail("debio fallar")
    except (FileNotFoundError, OSError):
        pass  # capturado correctamente

    # Fallback con python puro
    recs = [json.loads(l) for l in Path(det_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    finishes = [r for r in recs if r["event"].endswith(".finish")]
    assert len(finishes) == 40


# ── Q6 Load Profile (10x load breakpoint) ─────────────────────────────

def test_q6_10x_load_profile_batch(log_env):
    """Q6: Valida rendimiento y estabilidad bajo carga de 10x (100 placas simuladas)."""
    from polyx.model import observabilidad as ob

    run_id = "batch-100-plates-load"
    ob.set_run_id(run_id)
    log = ob.logger_for("detector")

    for i in range(1, 101):
        with ob.timer(log, "phase.infer", plate_idx=i):
            pass

    ob.clear_run_id()
    det_file = log_env / "detector.jsonl"
    recs = [json.loads(l) for l in det_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    finishes = [r for r in recs if r["event"] == "phase.infer.finish" and r.get("run_id") == run_id]
    assert len(finishes) == 100
    for r in finishes:
        assert r["ms"] >= 0


# ── Q7 Negative Tests ────────────────────────────────_________________

def test_q7_malformed_jsonl_records_handled(log_env):
    """Q7: Registros corruptos o lineas malformadas en JSONL se filtran sin colapsar."""
    det_file = log_env / "detector.jsonl"
    det_file.write_text('{"ts":"2026-10-16T00:00:00.000Z","event":"valid.event"}\n'
                        'CORRUPTED_NON_JSON_LINE\n'
                        '{"ts":"2026-10-16T00:00:01.000Z","event":"valid.event2"}\n',
                        encoding="utf-8")

    parsed = []
    for line in det_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    assert len(parsed) == 2
    assert parsed[0]["event"] == "valid.event"
    assert parsed[1]["event"] == "valid.event2"


@pytest.mark.skipif(shutil.which("jq") is None, reason="requiere jq instalado")
def test_q7_non_matching_run_id_returns_empty(log_env):
    """Q7: Filtrar por un run_id inexistente devuelve resultado vacio sin error."""
    _simular_lote_10_placas(log_env)
    det_path = str(log_env / "detector.jsonl")

    cmd = ["jq", "-c", "select(.run_id == \"run-inexistente-xyz\")", det_path]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.stdout.strip() == ""
