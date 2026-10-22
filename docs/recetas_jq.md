# Operator Runbook: Batch Monitoring & `jq` Recipes for Poly-X Observability

Poly-X emits structured JSONL logs under `~/.polyx/logs/` (or `POLYX_LOG_DIR`) across three core modules:
- `detector.jsonl`: Plate inference, segmentation, and feature extraction events.
- `reporte.jsonl`: PDF/HTML report generation and spatial distribution statistics.
- `trainer.jsonl`: YOLO model training and validation cycles.

Every log record follows a strict JSON envelope containing timestamp (`ts`), log level (`level`), module (`module`), process ID (`pid`), thread name (`thread`), event name (`event`), optional execution run identifier (`run_id`), duration in milliseconds (`ms`), and optional structured payload (`data`).

---

## 1. Real-Time Tailing & Filtering on Linux / macOS (`jq`)

Ensure `jq` is installed (`sudo apt install jq` or `brew install jq`).

### A. Live Latency Tracking (`*.finish` events)
To stream all completion events and their wall-clock execution time in milliseconds (`ms`) in real time:
```bash
tail -F ~/.polyx/logs/detector.jsonl | jq -c 'select(.event | endswith(".finish")) | {ts, event, ms}'
```

### B. Batch Filtering by `run_id`
To filter logs for a specific batch run:
```bash
jq -c 'select(.run_id == "batch-2026-10-16-001") | {ts, module, event, ms}' ~/.polyx/logs/detector.jsonl
```

### C. Error and Warning Auditing across All Modules
To catch any unexpected warnings or errors across all modules instantly:
```bash
jq -c 'select(.level == "ERROR" or .level == "WARNING") | {ts, module, level, event, data, exc}' ~/.polyx/logs/*.jsonl
```

### D. Cross-Module Correlation by `run_id`
To correlate detector, reporte, and trainer activities for a single run:
```bash
jq -s 'flatten | map(select(.run_id == "batch-2026-10-16-001"))' ~/.polyx/logs/*.jsonl
```

---

## 2. Windows PowerShell Equivalents (Without `jq`)

For Windows operators running PowerShell where `jq` is not installed:

### A. Live Tailing of Detector Logs
```powershell
Get-Content -Path "$env:USERPROFILE\.polyx\logs\detector.jsonl" -Wait -Tail 50
```

### B. Filtering Finish Events and Parsing JSON
```powershell
Get-Content -Path "$env:USERPROFILE\.polyx\logs\detector.jsonl" -Wait | ConvertFrom-Json | Where-Object { $_.event -like "*.finish" } | Select-Object ts, event, ms
```

---

## 3. Live UAT Batch Execution Runbook Checklist

1. **Pre-Flight**:
   - Verify `POLYX_LOG_DIR` is set correctly or confirm write access to `~/.polyx/logs/`.
   - Confirm disk space quota (`SIZE_TOTAL` defaults to 200 MB with automatic FIFO pruning of oldest logs).
2. **Batch Launch**:
   - Open a monitoring terminal and start tailing `detector.jsonl` with `jq`.
3. **Execution Monitoring**:
   - Verify `phase.start` and `phase.finish` telemetry pairs arrive for each plate.
   - Confirm non-negative numeric `ms` values on all finish events.
4. **Post-Batch Audit**:
   - Run the error query across `*.jsonl` to ensure 0 unhandled exceptions or ERROR records.
