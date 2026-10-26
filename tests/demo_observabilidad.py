#!/usr/bin/env python3
"""Demo auto-verificada de la capa de observabilidad (S03-T04).

Simula UN lote correlacionado de 10 placas a traves de las costuras REALES
de ``polyx.core.observability`` — ``logger_para``, ``set_run_id``,
``cronometro``, ``evento`` — sin torch, sin ventana: el mismo camino que
reescriben detector/reporte/trainer en una corrida real. Al terminar,
vuelve a leer los JSONL y chequea SOLO el tramo que esta corrida appendeo:
cada linea de ese tramo es JSON con los campos nucleo (ts, level, module,
event, run_id en ingles), las fases comparten ``run_id`` y los ``.finish``
llevan ``ms`` numerico. La carpeta real trae historico anterior al renombro
(registros en espanol y corridas sin run_id): el historico no es negocio de
la demo, por eso se mide el tamano previo de cada archivo y se lee desde ahi.

Uso::

    python tests/demo_observabilidad.py [--log-dir DIR] [--placas 10]

``--log-dir`` por defecto usa la carpeta real (``~/.polyx/logs`` o
``POLYX_LOG_DIR``); pasar uno temporario para que no toque nada personal.
Sale 0 si todo cumple y 1 si alguna verificacion falla. Los registros
quedan listos para seguir en vivo::

    jq -c 'select(.event | endswith(".finish")) | {ts, event, ms}' detector.jsonl

Docstring en ingles no se aplica aqui: es script de demo, comentarios en
espanol segun la convencion del repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyx.core import observability as ob

CAMPOS_NUCLEO = ("ts", "level", "module", "event")
FALSOS = []  # ruta absoluta de cada mensaje de falla, para el resumen

FASES = ("phase.decode", "phase.detect", "phase.morphology", "phase.measure")


def _falla(mensaje: str) -> None:
    FALSOS.append(mensaje)


def _simular_lote(ruta_dir: Path, placas: int, ts: str) -> dict:
    """Recorre las costuras reales de un lote de deteccion de placas."""
    os.environ["POLYX_LOG_DIR"] = str(ruta_dir)  # vale tambien si explicito
    ob._loggers.clear()  # el cache honra el nuevo dir
    run_id = f"demo_{ts}"
    ob.clear_run_id()
    ob.set_run_id(run_id)
    log = ob.logger_for("detector")
    ob.event(log, "run.config", placas=placas, params={"min_confianza": 0.6, "forma_alta": True})
    for i in range(1, placas + 1):
        for fase in FASES:
            with ob.timer(log, fase, placa=i):
                time.sleep(0.002)  # latencia minima del 2 AM
    # el reporte corre en el mismo run (misma corrida, otro modulo en la vida real)
    log_reporte = ob.logger_for("reporte")
    with ob.timer(log_reporte, "report.build", placas=placas):
        time.sleep(0.002)
    # el trainer queda SUSPENDIDO en su run separado por el mismo lote
    ob.set_run_id(f"demo_{ts}_train")
    log_trainer = ob.logger_for("trainer")
    ob.event(log_trainer, "train.config", epochs=20)
    with ob.timer(log_trainer, "train.run", epochs=20):
        time.sleep(0.002)
    ob.clear_run_id()
    return {
        "run_id": run_id,
        "logger_detector": log,
        "logger_reporte": log_reporte,
        "logger_trainer": log_trainer,
    }


def _offsets_previos(ruta_dir: Path) -> dict[str, int]:
    """Tamano previo de cada JSONL: lo que esta corrida appendea es lo suyo."""
    return {
        m: (ruta_dir / f"{m}.jsonl").stat().st_size if (ruta_dir / f"{m}.jsonl").exists() else 0
        for m in ("detector", "reporte", "trainer")
    }


def _tramo_propio(ruta: Path, desde: int) -> list[dict]:
    """Lee solo lo appendeado desde ``desde`` y valida que sea JSON valido.

    Si el archivo previo no terminaba en salto de linea, el primer registro
    propio quedaria pegado al anterior: en ese caso se descarta hasta el
    primer corte limpio.
    """
    bruto = ruta.read_bytes()
    ini = desde
    if ini > 0 and bruto[ini - 1 : ini] != b"\n":
        salto = bruto.find(b"\n", ini - 1)
        ini = len(bruto) if salto == -1 else salto + 1
    lineas = [l for l in bruto[ini:].decode("utf-8").splitlines() if l.strip()]  # noqa: E741
    recs: list[dict] = []
    for n, linea in enumerate(lineas):
        try:
            recs.append(json.loads(linea))
        except json.JSONDecodeError as e:
            _falla(f"{ruta.name} (tramo propio linea {n}): no es JSON: {e}")
    return recs


def _verificar(ruta_dir: Path, run_id: str, placas: int, desde: dict[str, int]) -> list[dict]:
    """Re-lee el tramo que esta corrida escribio y verifica el contrato nucleo."""
    todos: list[dict] = []
    for modulo in ("detector", "reporte", "trainer"):
        ruta = ruta_dir / f"{modulo}.jsonl"
        if not ruta.exists():
            _falla(f"falta el archivo {ruta.name} que la demo debio escribir")
            continue
        recs = _tramo_propio(ruta, desde[modulo])
        if not recs:
            _falla(f"{ruta.name}: la demo no appendeo nada")
        for n, rec in enumerate(recs):
            for campo in CAMPOS_NUCLEO:
                if campo not in rec:
                    _falla(f"{ruta.name}:{n}: sin campo nucleo '{campo}'")
            if "run_id" not in rec:
                _falla(f"{ruta.name}:{n}: sin run_id (el sobre debe llevarlo)")
        todos.extend(recs)

    # el nombre del module es el del logger completo (polyx.<modulo>)
    det = [r for r in todos if r.get("module", "").endswith("detector")]
    det_run = [r for r in det if r.get("run_id") == run_id]
    if len(det_run) != len(det):
        _falla("detector: eventos sin el run_id compartido del lote")
    runs = {r.get("run_id") for r in det}
    if runs != {run_id}:
        _falla(f"detector: run_ids inesperados {runs}")

    por_evento: dict[str, int] = {}
    for r in det_run:
        por_evento[r["event"]] = por_evento.get(r["event"], 0) + 1
    if por_evento.get("run.config", 0) != 1:
        _falla("detector: falta (o abunda) el run.config del lote")
    for fase in FASES:
        if por_evento.get(f"{fase}.start") != placas or por_evento.get(f"{fase}.finish") != placas:
            _falla(f"detector: faltan pares start/finish de {fase}")

    terminados = [r for r in det_run if r["event"].endswith(".finish")]
    for r in terminados:
        ms = r.get("ms")
        if not isinstance(ms, (int, float)) or ms < 0:
            _falla(f"finish sin ms numerico: {r['event']} ms={ms}")

    train = [r for r in todos if r.get("module", "").endswith("trainer")]
    if not train:
        _falla("trainer: la demo no escribio nada en trainer.jsonl")
    elif any(r.get("run_id") != f"{run_id}_train" for r in train):
        _falla("trainer: el run del entrenador no quedo separado del lote")
    return todos


def _tabla_fase(recs: list[dict]) -> str:
    lineas = []
    ms_fase: dict[str, list[float]] = {}
    for r in recs:
        if r["event"].endswith(".finish") and isinstance(r.get("ms"), (int, float)):
            ms_fase.setdefault(r["event"], []).append(r["ms"])
    for evento, valores in sorted(ms_fase.items()):
        lineas.append(
            f"  {evento:<28} n={len(valores):3}  "
            f"min={min(valores):6.1f}  max={max(valores):6.1f} ms"
        )
    return "\n".join(lineas)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo auto-verificada del esquema de logs de Poly-X"
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        metavar="DIR",
        help="carpeta destino (defecto: la carpeta real de logs)",
    )
    parser.add_argument(
        "--placas",
        type=int,
        default=10,
        metavar="N",
        help="placas simuladas en el lote (defecto: 10)",
    )
    args = parser.parse_args()

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    if args.log_dir:  # noqa: SIM108
        ruta_dir = Path(args.log_dir)
    else:
        ruta_dir = ob.folder_logs()
    ruta_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] carpeta de logs: {ruta_dir}")
    print(f"[demo] simulando un lote de {args.placas} placas, run_id demo_{ts}")

    desde = _offsets_previos(ruta_dir)  # el historico ajeno no se juzga
    _simular_lote(ruta_dir, args.placas, ts)
    recs = _verificar(ruta_dir, f"demo_{ts}", args.placas, desde)

    if FALSOS:
        print(f"[demo] FALLÓ: {len(FALSOS)} verificaciones rotas:")
        for m in FALSOS:
            print(f"  - {m}")
        return 1

    por_modulo: dict[str, int] = {}
    for r in recs:
        por_modulo[r["module"].split(".")[-1]] = por_modulo.get(r["module"].split(".")[-1], 0) + 1
    print(f"[demo] OK: {len(recs)} registros escritos y verificados:")
    for modulo, n in sorted(por_modulo.items()):
        print(f"  {modulo}.jsonl: {n} registros")
    print(f"[demo] timings por fase ({args.placas} placas):")
    print(_tabla_fase(recs))
    print()
    print("Recetas jq para seguir el log en vivo (tarjeta de la 2 AM):")
    print(
        "  jq -c 'select(.event | endswith(\".finish\")) | {ts, event, ms}' "
        f"'{ruta_dir / 'detector.jsonl'}'"
    )
    print(
        f"  jq -c 'select(.run_id==\"demo_{ts}\") | {{ts, event, ms}}' "
        f"'{ruta_dir / 'detector.jsonl'}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
