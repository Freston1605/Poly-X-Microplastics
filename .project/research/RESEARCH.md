# Research Handoff

Phase: research
Status: complete
Intent: `.project/intent/INTENT.md`

## Dispatch

- `domain` — skipped → none — no [RESEARCH] question maps to domain; HPCC dataset location is a pitfalls/ops question.
- `stack` — dispatched → `.project/research/evidence-stack.md` — PyInstaller hooks for torch/ultralytics per OS, src/ layout impact on launchers, Singularity vs Apptainer.
- `pitfalls` — dispatched → `.project/research/evidence-pitfalls.md` — HPCC dataset location/sync, credential handling, packaging failure modes already in codebase.
- `similar` — skipped → none — no [RESEARCH] question about comparable products.

## Question assignments

- `` [RESEARCH] Which PyInstaller hooks are needed for ultralytics/torch on each OS? `` → `stack`
- `` [RESEARCH] Does `src/` layout break the existing `lanzar_polyx.py` / `Poly-X.vbs` launchers? `` → `stack`, `pitfalls`
- `` [RESEARCH] Singularity vs Apptainer vs conda-pack for the cluster image? `` → `stack`
- `` [RESEARCH] Where do datasets live on the HPCC (shared scratch vs. user home) and how are they synced? `` → `pitfalls`
