# SIGTERM Diagnostics and Memory Telemetry in Poly-X

## 1. Executive Summary
During the full-suite verification run in M001, SIGTERM events were recorded that initially raised concerns about possible memory leaks or interventions from the `systemd-oomd` daemon. This document compiles the empirical telemetry obtained in the execution environment, analyzing memory usage (RSS), cgroup pressure statistics and the systemd journal records.

The formal conclusion (recorded as decision **D009**) is that **the SIGTERM events are not caused by memory leaks nor by systemd-oomd actions**, but by:
1. The user's desktop session finishing (closing the Konsole terminal / `exit.target` / `LogoutPrompt`).
2. Timeouts in automated harnesses (e.g., 30s limits against a full-suite duration of ~31.8s).

---

## 2. Memory and Load Profile (Load Profile - Q6)
Memory consumption of the full test suite (281 tests in `pytest`) was profiled using `resource.getrusage(resource.RUSAGE_SELF)` and `/proc/self/status`.

- **Peak RSS (Resident Set Size):** ~383 MB to ~390 MB in the main process and child threads.
- **Test system:** Linux environment with 16 GB of physical RAM.
- **Behavior under load (10x Load Profile):** memory consumption stays flat and stable across the 281 unit and integration tests. No memory leaks or runaway growth were detected under repeated batches.

---

## 3. Memory Pressure Telemetry (cgroups and systemd-oomd)
Kernel and systemd metrics were inspected to verify whether memory pressure or OOM intervention occurred:

- **`/proc/pressure/memory` and `/sys/fs/cgroup/memory.pressure`:**
  ```
  some avg10=0.00 avg60=0.00 avg300=0.00 total=856299
  full avg10=0.00 avg60=0.00 avg300=0.00 total=834853
  ```
  Sustained memory pressure is exactly **0.00%**, which indicates a total absence of contention or (*thrashing*).
- **systemd-oomd status / `oomctl`:** zero processes killed by oomd since system boot.

---

## 4. Analysis of the Systemd Journal Logs
Reviewing the systemd events of user sessions confirmed that the intermittent SIGTERMs coincided exactly with:
- Logout of the user's graphical session (`LogoutPrompt`, `Shutdown`, `exit.target`).
- Teardown of the Konsole scope (`app-org.kde.konsole-*.scope`), which propagates `SIGTERM` to every descendant process (including `pytest` instances or dev harnesses) before escalating to `SIGKILL` after 15 seconds.
- Expiration of strict timeout thresholds set in CI/automation scripts (30.0s thresholds on tasks that finished in 31.7s).

---

## 5. Failure Modes and Mitigation (Failure Modes - Q5)
- **Restricted access to `/proc` or `journalctl`:**
  - *Failure:* insufficient permissions or containerized environments without `/proc/pressure/memory` mounted.
  - *Handling:* the system falls back transparently to standard Python `resource.getrusage()`, guaranteeing that RSS monitoring never blocks execution.
- **Timeouts in CI / automation:**
  - *Mitigation:* configure generous timeout margins (e.g., >= 60s) for full concurrent test suites.

---

## 6. Negative Tests and Robustness (Negative Tests - Q7)
- Verified through tests with simulated environment variables and defensive reading of `/proc` on systems without cgroup v2 support.
- Exception coverage in telemetry capture with no impact on test results (`tests/test_recetas_jq.py`, etc.).

---

## 7. Operational Guide
1. **Timeout configuration:** make sure integration runners have a timeout above 45 seconds for the full 281-test suite.
2. **Session hygiene:** avoid closing interactive terminal sessions during long background test runs.
3. **Continuous monitoring:** check `cat /proc/pressure/memory` if performance degradation is suspected.
