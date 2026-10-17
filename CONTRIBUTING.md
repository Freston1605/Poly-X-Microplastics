
## Diagnóstico a las 2 AM (observabilidad)

- Cada módulo deja un diario JSONL en `~/.polyx/logs/<modulo>.jsonl`
  (`POLYX_LOG_DIR` lo redirige). Rotación: 5 MB × 3 copias.
- Leer con jq:
  ```bash
  jq -r '"\(.ts) \(.level) \(.evento) \(.datos)"' ~/.polyx/logs/detector.jsonl | tail -40
  ```
- Los eventos son `verbo.objeto` en inglés (run.finish, imgsz.fallback,
  report.error, train.error, update.available). Los fallos van a nivel
  ERROR y llevan el traceback en `exc`.
- Más detalle: `POLYX_LOG=debug python -m polyx.gui.detector`
- Las señales que deben alertar: `imgsz.fallback` (el lote corrió a menos
  resolución de la pedida), `run.sin_forma_alta` (muchas partículas sin
  medir), y cualquier evento `*.error`.
