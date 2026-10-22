# GSD directives — Poly-X takeover and environment remediation

**Prepared:** 2026-10-20
**Repo:** `Freston1605/Poly-X-Microplastics` (fork of `CrissFerrada/Poly-X-Microplastics`)
**Handed to:** GSD, milestone pipeline (`gsd-path/v2`)
**Companion documents:** `docs/TAKEOVER_PLAN.md` (architecture + science), `docs/ENVIRONMENT_PLAN.md` (resource snapshot + runtime analysis)

---

## READ FIRST — three corrections to the prior analysis

The prior documents were written without reading `.gsd/DECISIONS.md` and `.gsd/KNOWLEDGE.md`. Those registers already contain decisions that **override** parts of the plan. Implementing the prior documents as written would violate recorded decisions. Read this section before planning anything.

### C1 — The AMD/ROCm question is already decided (D031)

D031 (revisable) pins the GPU strategy:

> ROCm 7.2.4 + torch 2.10.0+rocm7.2.4 + torchvision 0.25.0+rocm7.2.4 + ultralytics 8.4.182, in a **Docker GPU worker** over loopback HTTP + WebSocket, with a native PySide6 GUI. "Do not adopt ROCm 10.1 + torch 2.14 until NMS and AMP are re-probed."

D031 was based on a **live probe on this exact RX 9070 XT**: ultralytics inference at `device=0`, `torchvision::nms` working, a one-epoch training run completing, 14.44 ms at 640 px vs 37.16 ms on CPU (17.44 vs 104.33 at 1280). L043 records that `torch.version.hip` — not `torch.version.cuda` — is the real ROCm discriminator on gfx1201, and that `get_device_capability()` returns `(12, 0)` there, so capability must not be used as a GPU validation gate. L044 records that the **ROCm 10.1 + torch 2.14 line has a live `torchvision::nms` breakage** and that AMP auto-disables in ultralytics.

**This directly contradicts the ROCm 10.1 / torch 2.14 recommendation in `docs/ENVIRONMENT_PLAN.md` §4.** That recommendation was derived from AMD's published wheel index without knowing D031 or L044 had already probed the exact pairing and found NMS broken. **D031 and L044 win. Do not install ROCm 10.1 + torch 2.14.** The stable 7.2.4 / 2.10.0 pair in D031 is the supported one.

Consequence: the resource findings in `ENVIRONMENT_PLAN.md` §4 about `/dev/kfd` permissions and gfx1201 wheel availability are still true and still useful, but they describe a path D031 has already chosen differently (Docker worker, not a host venv).

### C2 — `requirements.txt` must NOT gain a torch or sam2 pin (D039)

D039: "no code under `polyx/` may import sam2/torch for the mask path and **requirements.txt must not gain a torch or sam2 pin**, so the runtime has to live outside the project interpreter yet stay reproducible from one pinned command." The SAM teacher lives in a **gitignored worktree-local `.sam2env/` venv** provisioned by `tools/provision_sam2_runtime.py`.

`ENVIRONMENT_PLAN.md` §4 and §0.2 recommend adding `torch==2.14.1+cpu` and an `--extra-index-url` to `requirements.txt`. **That violates D039 and must not be done.** D045 likewise says "no code under `polyx/` or `tools/` imports mlflow, and requirements.txt gains no tracker pin."

What survives from §0.2: the *observation* that `requirements.txt` pins no torch at all, so torch arrives transitively via `ultralytics==8.3.40` and would resolve to the **CUDA-bundled PyPI wheel** (554.6 MB + ~2.5 GB of `nvidia-*`/`triton`) on a machine with no NVIDIA GPU. That is a real install-cost and disk hazard. It is fixed **inside the isolated venv** by index selection, not by editing the project `requirements.txt`. See GIT-04.

### C3 — The suite *does* run on this Linux worktree, without a project venv

`ENVIRONMENT_PLAN.md` §2 claims "no venv for the Poly-X project at all" and that "the 210-passing-tests claim was produced somewhere else." **That is wrong.** The M007 worktree runs the suite with the **system `/usr/bin/python3`** (3.14.8), which already has PySide6 6.11.2, cv2 4.14.0, matplotlib, Pillow, psutil, PyYAML, pytest and numpy 2.5.3 — but **not** torch and **not** ultralytics. L029 and L003 document this exact arrangement: the suite runs without the heavy stack, and the M004 laziness gate is explicitly acknowledged to be trivially satisfied in the Linux worktree for that reason.

The important consequence: **numpy 2.5.3 is what the suite actually runs against**, while `requirements.txt` pins `numpy==1.26.4` (which has no cp314 wheel — verified: cp tags are cp39–cp312 only, classifiers stop at 3.12). The Python-3.12 standardization proposed in `ENVIRONMENT_PLAN.md` §3 is therefore still a real recommendation for a *reproducible* environment, but it is **not** a precondition for the suite to run, and it is **not** a blocker on M007/S03–S06.

**Do not re-plan the environment around C3.** The environment work below is additive and non-blocking.

---

## Scope of these directives

Three buckets, in the order they must be executed:

- **GIT** — version-control operations: branches, merges, commits, and what must be true in the tree before the next milestone runs.
- **SHIP** — what leaves the repo: releases, tags, published artifacts, license position, provenance, and the proposal interface.
- **DONE** — the definition-of-done gates that close a milestone, in the repo's own idiom (a pytest gate, not a prose promise).

**Everything here is subordinate to the active milestone.** `.gsd/STATE.md` reads: Active Milestone M007, Active Slice S03, Phase `executing`, Next Action "Execute T04: Run the live CPU training of both variants and commit the comparison with a standing gate in slice S03." D048 records that S03/T04's epoch assertion was repaired to accept `8 <= epochs_run <= 20` with patience 10 recorded, and that the committed artifact legitimately carries `verdict REVIEW` with 10 failing gate ids. S06 exists specifically to remediate the learned-path detection collapse (S04 depends on S06).

**None of these directives may reorder, gate, or delay S03/T04 → S06 → S04.** They are new work, scheduled after the active milestone closes.

---

# GIT

### GIT-01 — Merge M007 into `main` at milestone close, not now

`milestone/M007` is 15 commits ahead of the merge base with zero conflicts, and `main` is 7 commits ahead of the merge base. Merging now, mid-S03, would put a half-finished S03 artifact in `main` and invalidate the standing gates D044 and D048 are built on.

**Directive:** leave `milestone/M007` as the bound branch until M007 ships. At ship, integrate per `.gsd/STATE.md`'s `integration: direct` and `default_branch: main`. The merge of `main`'s parallel CI/security line into the milestone branch is the established precedent (M006's `549723c`).

**Do not** rebase `milestone/M007` onto `main`. The M006 integration merged `main` *into* the milestone branch; keep that shape.

### GIT-02 — Reconcile the two `main` histories before shipping M007

`main` and `milestone/M007` have **both** changed `polyx/gui/report_html.py` since the M006 integration. On `main`, commit `ab7e53a` ("escapar el HTML de textos ajenos en el informe") applied `_esc = html.escape(str(v), quote=True)`. On the milestone branch, D004 rejected `quote=True` and chose `quote=False`, because the report interpolates values between tags and never inside attributes, and `quote=True` breaks byte-identity for benign filenames containing apostrophes (`prueba's.png` → `prueba&#x27;s.png`).

**Directive:** at the M007 ship merge, `main`'s `report_html.py` change must **lose**. The milestone branch's `_esc(..., quote=False)` and its byte-identity baseline are the sanctioned values (D004, D017, D034). Verify the merged `polyx/gui/report_html.py` byte-identity gate still passes against the pinned baseline (`bbb1c7a5…905edde` / 139503 B benign; `fc175d4f…42ab0f` / 139758 B hostile) — if `main`'s version won the merge, the report bytes change and the pin test fails loudly, which is the intended guard.

Note that `main` also carries `e058e6a` ("auto-commit after workflow-template"), which vendored 83,408 lines of agent skills (`agent/skills/`) and a `skills-lock.json`. This is unrelated to the science and inflates the repo. **Directive:** decide deliberately whether to keep it in the history or to drop it in the next release. If dropping, do it as an explicit, recorded decision (new D-row) — not as a silent history rewrite. Flag it to the milestone owner; it is a judgement call, not an agent decision.

### GIT-03 — Commit the two plan documents and the literature review

`docs/TAKEOVER_PLAN.md`, `docs/ENVIRONMENT_PLAN.md` and `docs/literature/` are untracked. The literature review (60 studies, PRISMA-documented) is the evidence base for the science plan and currently lives outside version control.

**Directive:**
- Commit `docs/literature/` including the PRISMA artifacts (`prisma.json`, `prisma_stage1.json`, `extraction_table.csv`, `sources/`). These are small and are the review's substance.
- **Do not** commit `runs/litreview_v1/records_dedup.jsonl` (4.1 MB). Confirm `.gitignore` covers `runs/`.
- Commit the two plan documents, but **amend them first** per C1/C2/C3 above — `ENVIRONMENT_PLAN.md` §0.2, §2, §3 and §4 currently contain statements that D031/D039/L029 make wrong or superseded. Do not commit a document that contradicts the decisions register.
- `docs/literature/microplastics_review_v1.md` references `runs/litreview_v1/`. If the committed copy must be self-contained, note the path in the commit message; do not restructure the review.

### GIT-04 — Do not touch `requirements.txt`; fix the torch index inside the isolated venv only

Per C2, `requirements.txt` must not gain a torch or sam2 pin (D039, D045). The hazard is real but belongs to the isolated venv.

**Directive:** any environment work that installs torch must (a) live in a gitignored venv outside the project interpreter, and (b) select the CPU index explicitly so the 554.6 MB CUDA-bundled wheel and its `nvidia-*`/`triton` tree never land. `tools/provision_sam2_runtime.py` already encodes this pattern (`CPU_INDEX = "https://download.pytorch.org/whl/cpu"`, `TORCH_SPEC = "torch==2.14.1+cpu"`, `SAM2_BUILD_FLAGS = ("--no-build-isolation",)`). L075 and L077 record why the isolated-build path is mandatory on this box: sam2's `pyproject` declares `requires = ["setuptools>=61.0", "torch>=2.5.1"]`, so a default isolated build resolves torch against PyPI and pulls the CUDA wheel plus ~2.5 GB of `nvidia-*`/`cuda-toolkit`/`triton`, which trips `OSError: [Errno 122] Disk quota exceeded` on this machine's ~6.3 GB per-user `/tmp` tmpfs.

**Directive:** keep `TMPDIR` pointed off `/tmp` for any provisioning run (L077), and keep `.sam2env/` and any new venv gitignored.

### GIT-05 — Keep the branch topology the preferences declare

`.gsd/PREFERENCES.md` declares `git.isolation: worktree`, `git.main_branch: main`, `git.auto_push: false`, `git.merge_strategy: squash`, `mode: team`, `verification_commands: [pytest]`, `verification_timeout_ms: 600000` (raised because the full suite measures ~187 s and the 120 s default killed the gate with exit 124 at M007/S02/T06).

**Directive:** new milestone work goes on its own `gsd-path/M<NNN>` branch in a `.gsd-worktrees/M<NNN>` worktree, never directly on `main`. `auto_push: false` means **do not push** — the owner pushes. Do not raise `verification_timeout_ms`; 600000 is the engine's clamp ceiling.

### GIT-06 — Record the two new decisions this directive set implies

`.gsd/DECISIONS.md` is append-only; new rows, never edits.

**Directive:** add rows for:
- **(new)** The environment standardization: system `/usr/bin/python3` (3.14.8, numpy 2.5.3, no torch, no ultralytics) is the sanctioned interpreter for the Linux worktree test suite; `numpy==1.26.4` in `requirements.txt` has no cp314 wheel and is therefore a CI-and-Windows pin, not a Linux-worktree pin; the divergence is documented rather than resolved. Rationale: L029 already records that the Linux worktree runs without the heavy stack and that the laziness gate is trivially satisfied there for that reason; making that explicit prevents a future agent from "fixing" the environment and breaking the gate's meaning.
- **(new)** Whether `agent/skills/` (83,408 lines from `e058e6a`) stays in the history. Rationale: it is unrelated to the science, inflates the repo and the diff surface, and its disposition is the milestone owner's call.

### GIT-07 — Add the three missing enforcement gates to CI, at M008 time

These are the gates whose absence the sidecar evidence and `ENVIRONMENT_PLAN.md` both flagged. Each is a pytest file, matching the repo's established pattern of durable contract gates (P028/P032/P039: structural budgets and boundaries are enforced by tests, not review discipline).

- **`tests/test_entorno_reproducible.py`** — asserts the project's dependency pins resolve for the interpreter in use, that `numpy` is at a version consistent with the ultralytics pin in the environment where training runs, and that no `nvidia-*` package is present in the venv that holds torch. Anti-vacuity guard required (P039): the test must fail on a synthetic violation, not pass vacuously.
- **`tests/test_ci_ejecuta_ruta_ml.py`** — imports `polyx.model.yolo_wrap` and runs a single inference, so CI is not green while the ML path is broken. The sidecar finding stands: *"A green test run here does not validate the YOLO pipeline."* Note the M007 worktree has no ultralytics, so this gate must be conditional on the heavy stack being importable — and must **assert the skip is honest** (P006 pattern: a skip must be visible and explained, never silent), so it cannot be mistaken for coverage.
- **`tests/test_bench_phantom_fidelity.py`** — runs the M007 fidelity bench in `--check` mode against the committed phantom corpus, recomputing the baseline from committed pixels with no torch. The M007 tools already implement this shape; wire it as a CI step so the segmentation claim is machine-checkable by a reviewer rather than asserted in a Markdown table.

**Directive:** these land in M008, not M007. S05 ("Publish benchmark and settle licensing") already owns the benchmark artifact; do not duplicate it in CI under a different name — the M008 gate must call the same `tools/bench_mask_fidelity.py` entry point the S05 artifact was produced by.

### GIT-08 — `.gitignore` and stale-artifact cleanup

**Directive:** confirm `.gitignore` covers `__pycache__/`, `*.pyc`, `.sam2env/`, `mlruns/`, `mlflow.db` and `runs/`. D045 records that `mlflow.db` and `mlruns/` were found untracked and unreferenced; confirm they are ignored, not merely untracked. Delete the empty `polyx/core/`, `polyx/detector/`, `polyx/trainer/` directories on disk (each has 0 `.py` files — only stale `__pycache__` left over from the M005 split, D016). This is cosmetic but is the kind of residue that reads as confusion to a new contributor, and the repo has one contributor.

### GIT-09 — The M007 worktree is not a safe place to run ad-hoc verification

Observed live on 2026-10-20: a full-suite `pytest` run launched in `.gsd-worktrees/M007` was terminated by `FileNotFoundError: [Errno 2] No such file or directory: '…/.gsd-worktrees/M007'` on pytest's own `chdir` to the session start path. The worktree had been removed and recreated under it by an active GSD execution, and the branch advanced `5452666 -> 60d4562` in the same window (adding `tools/run_seg_training.py`, 481 lines, and `tests/test_seg_training_pipeline.py`, 453 lines).

**Directive:**
- **Never run verification in a milestone worktree while GSD is executing in it.** The GSD execution owns that directory; it can be torn down and rebuilt mid-run. Use the project root (`main`), or a scratch clone, for any out-of-band measurement.
- **Any measured figure taken while the branch was advancing must be re-measured on a quiet tree before it is recorded.** The 1191-collected figure in DONE-01 was taken in exactly this window and is explicitly marked provisional for that reason.
- This is an instance of L019 (dual projection roots, artifact-db-status-divergence) with a sharper edge: a torn-down worktree breaks an in-flight verification, not just the artifact row. If GSD's own verification gate ever hits this, the failure mode is a `FileNotFoundError` on the start path, **not** a test failure — read it as "the tree moved", not "the code is broken".


---

# SHIP

### SHIP-01 — License the repository (blocks any public release)

**There is no LICENSE file.** README and CLAUDE.md still read `**Author:** Cristofher Ferrada · PhD (c) in Chemistry · PUCV · 2026` and point at `github.com/CrissFerrada/Poly-X-Microplastics`. For a repository whose artifact is cited in a paper's Methods, this is the blocking item for external reuse.

**Directive — decide, do not decide unilaterally.** The choice is between AGPL-3.0 (copyleft, consistent with the ultralytics AGPL question M007/S05 already owns), MIT/Apache-2.0 (adoption), CC-BY-4.0 for docs, and an explicit "all rights reserved, internal use only" notice. Present the options with their consequences to the milestone owner (Waldo Quiroz) and Ferrada, record the outcome as a D-row, and only then add the file.

**Sequencing note:** M007/S05 already owns "the ultralytics AGPL distribution decision." The ultralytics AGPL question and the repo's own license are the same question seen from two sides — if the runtime depends on AGPL ultralytics and the app links it, the distribution position of the whole repo is affected. **Ship the repo license decision and the ultralytics AGPL decision together, in S05, as one recorded position.** Do not let S05 publish a benchmark while the license of the thing that produced it is unresolved.

### SHIP-02 — Authorship and provenance note (blocks any public release)

**Directive:** update README and CLAUDE.md to name both Franco and Ferrada, point the canonical repository at `Freston1605/Poly-X-Microplastics`, and add an explicit **"Original work / provenance"** section stating that the base repository is Ferrada's, that the fork is the maintained continuation, and which substantial changes belong to whom (M005 package split, M006 English migration, M007 learned segmentation, and Franco's CI/security/hardening line). Add a `CITATION.cff` and a "How to cite" section.

This is a courtesy obligation with a technical consequence: a Methods section that cites the tool must be able to point at a versioned release with an unambiguous author list. Write it plainly and kindly — this is a handover between people, and the note is worth more than the metadata it carries.

### SHIP-03 — Tag a first release with the exact producing environment

**Directive:** at M007 close, tag `v0.1.0` (or the owner's chosen version) and record alongside it: the Python version, the exact `requirements.txt` pins, the ultralytics version, the hardware, the model weights by SHA-256, the dataset version, and the SAM checkpoint hash (`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`, 156,008,466 bytes, per D039). The repo already has `polyx/model/procedencia.py` for run-level traceability; extend the same discipline to the release. `auto_push: false` means the tag is created locally and the owner pushes it.

### SHIP-04 — Publish the weights and the dataset manifest (not the raw data)

`models/` and `data_microplastico/` are gitignored, which is correct for size but means **the published numbers are not reproducible from the repo alone.**

**Directive:** publish, as release assets with a DOI: the trained segmentation weights (or a pointer to them), a dataset manifest (plate IDs, annotation provenance, per-image hash, the exact train/val split per D041's plate-level rule), and the code + environment pin. Do not commit the raw plates. D035's pattern is the precedent: commit the manifest and the small canonical artifacts, pin the heavy bytes by hash.

### SHIP-05 — Ship the segmentation benchmark with its limits stated on its face

**Directive:** M007/S05's benchmark report must carry, in the artifact itself and not only in surrounding prose:
- the **real-image ceiling** — Park et al. (2022): mean F1 ≈ 0.736, mean IoU ≈ 0.617 for fluorescence microscopy of real microplastics. A synthetic-phantom result must never be presented as field performance.
- the **detection-probability-vs-size curve** (Tanoiri et al. 2026: 50 % at 40 µm, 90 % at 100 µm) so that a count is not reported without its censoring.
- the **CPU-vs-GPU provenance** of every number, kept separate per the D031 split, and never mixed in one table without saying which produced which.
- the **honest REVIEW verdict** where that is the outcome. D044's gates and D048's repair exist precisely so a failing run is recorded rather than re-tuned. S06 exists because the learned path currently loses 15–21 of 21 gated particles. That is a finding, and it must reach the published artifact.

### SHIP-06 — Reconcile the FONDECYT proposal with the code and the literature review

> **Superseded in scope by the 2026 outcome (see below), and partly by [[FONDECYT New Proposal — Strategy]].** The 2026 attempt was **rejected as *fuera de bases*** at administrative admissibility for a missing Bibliographical References section, and never reached a panel. There is therefore no panel feedback to reconcile — only a reusable scientific review and a new proposal to build. This directive now covers **the repo-side obligations of the new proposal**; the proposal-side work is the PI's, outside the queue. See [[FONDECYT 2026 Attempt — Rejected at Admissibility]] and [[FONDECYT New Proposal — Strategy]].

The 2026 Formulación promised **selective identification of five polymers (PE, PET, PP, PVC, PA) by Nile Red colour**. The literature review and the code both say otherwise, and they say it independently:

- Nile Red is a **polarity** signal (Ho 2023, Stanton 2019). The honest output is a polarity class plus abstention.
- `polyx/model/asignacion.py` already implements abstention, and the measured per-class recall (PET 0.98, PP 0.70, LDPE 0.54) matches the solvatochromism explanation.
- Polymer identity belongs to µFTIR or Raman. van Mourik et al. (2021): polymer typing agrees in 53–100 % of labs — the tail is the honest headline.
- Park et al. (2022) sets the expected F1 at ~0.736, not ~0.98.
- The group's own J. Chil. Chem. Soc. 2024 paper reports PE as bright yellow and distinguishable from PS/PP/PVC/PET **by eye** — a far more modest claim than the proposal's automated selective identification.

**Directive — repo-side, and it is the same two reconciliations:**

1. **Polymer scope.** The proposal targeted PE, PET, PP, PVC, PA; the code has PET, PP, LDPE; the 2024 papers cover PET (sediment) and PE (fish). PE and LDPE are different polymers. Pick one scope and make proposal, code and dataset agree. **Scientific decision for the milestone owner, not an agent decision.**
2. **Claim type.** Either the proposal narrows to *counting + sizing + morphology + polarity class with abstention*, or it adds a documented µFTIR/Raman confirmation subset. **Do not ship a polymer label from fluorescence alone.** The recommended framing is in the strategy note.

**The repo-side obligation that makes either framing true:** the abstention becomes an explicit contract (DONE-05) and the published benchmark states its own limits (SHIP-05). The proposal is the external face of this repo; if the code is more honest than the proposal, the proposal is what must change — not the code.

**Sequencing:** proposal-side work is the PI's, outside the repo queue, and must not be interleaved with M007. It does not gate M007/S03–S06 and is not gated by them.

---

# DONE

The repo's own idiom, per AGENTS.md: *"Run the tests with `.venv\Scripts\python.exe -m pytest -q` and wait until the **full suite passes with zero failures and zero skips** before calling anything done."* And per D001: the bar is **suite-green, not a static count** — a stale number is worse than no number.

### DONE-01 — The test-count discrepancy is resolved by measurement, not by picking a number

Figures in circulation, none currently right for the current tree:

| Source | Figure | Status |
|---|---|---|
| `AGENTS.md` | 128 | stale (pre-dates M005/M006) |
| Sidecar evidence | 210 | not reproducible on this tree |
| `.gsd/PROJECT.md` | **767 passed / 0 failed / 0 skipped** | at M006 close, on the merged tree |
| D001 | 140 test functions, 2026-10-14 | static count, pre-parametrize |
| L027 | 507 at M005/S03 | superseded by 767 at M006 close |
| **Measured 2026-10-20** | **1191 collected** on `milestone/M007` @ `60d4562` (803 static test functions across 62 files) | current, pre-run |

**Directive:** the bar is **suite-green with 0 failures and 0 skips**, and the **count is measured at close and recorded** in the milestone's Summary — the executor records the collected figure, per D001. Then **fix `AGENTS.md` to the measured value.** A wrong test count in the document an evaluator reads is a small but real credibility hit, and it is the kind of thing that has already been allowed to rot twice in this repo.

**Note on the 1191 figure:** it is a `--collect-only` count, taken on the M007 worktree with the system `/usr/bin/python3`. It is the *collected* figure (parametrize expanded), which is the right unit. It is **not** a pass count — the full run did not complete in this session (see Appendix note), and the branch was advancing under an active GSD execution while the measurement was taken, so the figure must be re-measured at close on a quiet tree. Treat 1191 as the current expected magnitude, not as the recorded result.

### DONE-02 — M007 closes with its REVIEW verdict intact

**Directive:** M007's success criteria are the M007 ROADMAP's, and they include *"the final benchmark and the ultralytics AGPL distribution decision are documented and reproducible."* The learned path currently loses 15–21 of 21 gated particles; S06 exists to remediate it.

**Do not** close M007 by re-tuning the D044 gates, the D041 split, the matching rule, or the metric definitions after seeing the numbers. D044 is explicit: *"thresholds, epochs, the split, the matching rule and the metric definitions are never adjusted after seeing the numbers."* A REVIEW verdict with the failing gate ids and observed values recorded is a **valid** M007 outcome. A passing artifact obtained by moving the gate is not.

**Directive:** the M007 close Summary states which of the two paths was taken — (a) S06 remediated the collapse and the re-run passes, or (b) the re-run still carries REVIEW with one named root cause recorded — and both are acceptable closes. What is not acceptable is a green artifact whose gates moved.

### DONE-03 — Every new boundary is a pytest gate, not prose

Per P028/P032/P039 and the repo's established pattern, each structural constraint introduced by this work is enforced by a test with an anti-vacuity guard. Concretely, the new gates:

| Gate | Asserts |
|---|---|
| `tests/test_ci_ejecuta_ruta_ml.py` | the ML path is exercised, not merely importable; skip is honest and visible |
| `tests/test_entorno_reproducible.py` | dependency pins resolve; no `nvidia-*` in the torch venv |
| `tests/test_bench_phantom_fidelity.py` | the committed benchmark is recomputable from committed pixels |
| boundary test for the new `core/detect/` seam | `polyx.core.detect` imports no `ultralytics` and no `PySide6`; a synthetic violation must be flagged (P028) |
| label-format test | each `LabelFormat` adapter round-trips, and the adapters are independent of the GUI |

**Directive:** each new gate ships with (a) a synthetic-violation test, (b) a structure-existence test, and (c) an inventory floor — or it is a vacuous gate and does not count (P039).

### DONE-04 — The provenance of every published number is reconstructible

**Directive:** a number is not DONE unless the following are recorded with it: the code revision, the environment (Python version, dependency pins, hardware), the model weights by hash, the dataset manifest version, the split, and the resource envelope that produced it (peak RSS, worker count, wall-clock). The resource envelope is new and it matters: on this box the planner's own verdict for an 8 GiB worker is `insufficient_memory`, so a published number carries an obligation to state the machine that produced it. The repo's `procedencia.py` pattern is extended, not replaced.

### DONE-05 — The abstention contract is explicit, not implicit

**Directive:** `Detection` (or its successor) carries `polarity_class`, `assigned: bool` and `assigned_conf` as part of its contract, and the report states the **unassignable fraction as a result**, not as a silent filter. The trainer's class list is at most `PET | PP | LDPE | unassignable`. This is the code-side half of SHIP-06's second reconciliation, and it is what makes the repo's honesty survive the move from "the code abstains" to "the code is *specified* to abstain."

### DONE-06 — The measurement code is unchanged, and proven so

**Directive:** the `core/measure/` and `core/quality/` moves are **moves, not rewrites**. Every measurement test must pass unchanged, byte-for-byte in behaviour. P004 already pins `_propagar_geodesico` bit-exactly against a frozen reference body, and P006's fixture corpus exists for exactly this. **Do not** re-measure the nine-shape taxonomy or the detection-probability curves until the pipeline is frozen — on a CPU-only teacher that is a multi-hour-to-multi-day job, and the resource budget allows **one corpus re-measurement per pipeline change, not three.** This is the resource constraint from `ENVIRONMENT_PLAN.md` §3 becoming a scheduling rule.

### DONE-07 — M008's scope is the convergence, and it inherits these gates

M008 is "Converge the two measurement pipelines" and already owns the field-plate intake contract, the runtime-free check, and the drift statistic against the synthetic corpus (D047). **Directive:** M008's Roadmap must explicitly inherit GIT-07's three gates and DONE-03's seam/label gates, so they are not lost at the milestone boundary. M008 is the natural home for the `core/detect/` seam, the `LabelFormat` adapters, the unified `Detection` type, and the headless CLI (the "detect / evaluate / audit" entry points) — because a CLI is what lets a reviewer, a CI job, and a script exercise the pipeline without a display, and M008's drift statistic needs exactly that.

**Directive:** the CLI must emit a machine-readable JSON run record alongside the HTML report, and it must be streaming and bounded — batch size 1, `num_workers=0` for the dataloader, pinned `OMP_NUM_THREADS` — so that a 2,000–4,800 px plate never has to fit in RAM alongside the corpus. Record peak RSS in the run record.

---

# Execution order

```
NOW     M007 S03/T04 -> S06 -> S04 -> S05        (uninterrupted; nothing below may gate this)
        S05 ships the benchmark AND the license position together (SHIP-01 + SHIP-05)

AFTER   M007 close
        GIT-01  merge M007 into main, main's report_html change loses
        GIT-02  decide the agent/skills history (owner call)
        GIT-03  amend + commit the two plan docs and docs/literature/
        GIT-06  record the two new D-rows
        SHIP-02 authorship + provenance note + CITATION.cff
        SHIP-03 tag v0.1.0 with the producing environment
        SHIP-04 publish weights + dataset manifest
        DONE-01 measure the suite count, fix AGENTS.md

M008    GIT-07  the three new CI gates
        GIT-08  .gitignore + stale-artifact cleanup
        DONE-03 seam + label-format gates
        DONE-05 abstention contract
        DONE-07 CLI (detect/evaluate/audit), streaming + bounded, JSON run record
        DONE-02  M007 close Summary states which path was taken

PARALLEL (PI, not the queue)
        SHIP-06  reconcile the FONDECYT proposal: polymer scope + claim type
```

---

# What this directive set does **not** do, and why

- **It does not touch the M007 milestone queue.** M007 is mid-execution with a live gate and a remediation slice. Reordering it would invalidate D044/D048's standing gates.
- **It does not add a torch or sam2 pin to `requirements.txt`.** D039 forbids it; the fix belongs in the isolated venv (GIT-04).
- **It does not adopt ROCm 10.1 + torch 2.14.** D031 and L044 already probed that exact pairing on this GPU and found `torchvision::nms` broken. The supported pair is 7.2.4 / 2.10.0.
- **It does not mandate Python 3.12 for the worktree.** C3: the suite runs on the system 3.14.8 with numpy 2.5.3 today, and L029 records that arrangement. Python 3.12 is a *reproducibility* recommendation, not a blocker.
- **It does not create a new milestone for the model-agnostic seam.** That is M008's job ("Converge the two measurement pipelines") and M008's own scope already includes the field-plate intake contract and the drift statistic. A new milestone would duplicate it.
- **It does not decide the license or the polymer scope.** Both are owner decisions with external consequences, and both are recorded as such rather than made by an agent.

---

## Appendix — every factual claim above, with its source

**Fork and branch state**
```
$ git log --oneline upstream/main..main
e797fa8 ab7e53a 5a6578d 0be5629 c639a40 48598d9 e058e6a
$ git log --oneline main..upstream/main      # (empty)
$ git diff --stat upstream/main..main | tail -1
 391 files changed, 113657 insertions(+), 13008 deletions(-)
$ git worktree list
  .../Poly-X-Microplastics                e797fa8 [main]
  .../Poly-X-Microplastics/.gsd-worktrees/M007 5452666 [milestone/M007]
$ git branch -a
  main, gsd-path/M001, gsd/local-main-sync, milestone/M007
```

**Records that override the earlier plans**
```
$ sed -n '/^| D031 |/p' .gsd/DECISIONS.md
  ... "Do not adopt ROCm 10.1 + torch 2.14 until NMS and AMP are re-probed." ...
$ sed -n '/^| D039 |/p' .gsd/DECISIONS.md
  ... "requirements.txt must not gain a torch or sam2 pin" ...
$ grep -E 'L043|L044|L075|L077' .gsd/KNOWLEDGE.md
  L043 torch.version.hip is the ROCm discriminator on gfx1201; get_device_capability() -> (12,0)
  L044 ROCm 7.2.4 + torch 2.10.0 + ultralytics 8.4.182: nms OK, epoch completes, AMP auto-disabled;
       "La línea ROCm 10.1 + torch 2.14 tiene torchvision::nms roto"
  L075 sam2 must install with --no-build-isolation or it pulls the 554 MB CUDA torch + ~2.5 GB
       nvidia-*/triton -> "OSError: [Errno 122] Disk quota exceeded" on the 6.3 GB /tmp tmpfs
  L077 pin TMPDIR outside /tmp; provisioning was run with TMPDIR=/var/home/... on the 150 GB root fs
$ grep -E 'L029|L003' .gsd/KNOWLEDGE.md
  L029 "torch and ultralytics are not installed for the system python3 there" - the laziness gate is
       trivially satisfied in the Linux worktree
```

**The worktree interpreter (correcting ENVIRONMENT_PLAN.md §2)**
```
$ /usr/bin/python3 -c "import importlib.util as u; ..."
 3.14.8  PySide6 6.11.2, cv2 4.14.0, matplotlib, Pillow, psutil, PyYAML, pytest, numpy 2.5.3
  ultralytics: False    torch: False
$ ls -d .gsd-worktrees/M007/.sam2env    -> none (not provisioned in this checkout)
$ ls /home/Franco/.venv/bin/python ...   -> yaml, pytest only; no torch/ultralytics/cv2
```

**Collected test count on the M007 branch (2026-10-20)**
```
$ cd .gsd-worktrees/M007 && QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m pytest -q --collect-only
 1191 tests collected in 2.23s
$ ls tests/*.py | wc -l            -> 62
$ grep -rhoE "^def test_[a-zA-Z0-9_]+" tests/*.py | wc -l  -> 803
```
**Caveat, stated plainly:** this is a *collected* count, not a pass count. The full suite run started in this session did **not** complete — the M007 worktree directory was removed and recreated by an active GSD execution while pytest was running, and pytest terminated with `FileNotFoundError` on its `chdir` to the (now-recreated) worktree path. The branch also advanced `5452666 -> 60d4562` during the same window, adding `tools/run_seg_training.py` (481 lines) and `tests/test_seg_training_pipeline.py` (453 lines). **The figure must be re-measured on a quiet tree at milestone close.** The recorded directive is suite-green with 0 failures and 0 skips; the count is recorded at close, per D001.

**numpy 1.26.4 vs Python 3.14 (unchanged, still true)**
```
$ python3 -c "... numpy/1.26.4/json ..."
 cp tags: ['cp310','cp311','cp312','cp39','pp39']    has cp314: False
 classifiers: 3.9, 3.10, 3.11, 3.12
```

**Resource snapshot** (schema 1.1, `valid: true`, mode 0600, `/home/Franco/.hermes/cache/scratch/resource-snapshot.json`)
```
 8 physical / 16 logical cores; RAM 16.67 GB total, 8.68 GB available; swap 8.34 GB
 AMD Radeon RX 9070 XT, gfx1201, 17.10 GB device-reported pool, runtime_usable_devices: null
 Disk 929 GB total / 147 GB free (85% used), single filesystem
 planner: 8 GiB/worker -> insufficient_memory, 0 workers
         1 GiB/worker CPU -> 7 workers x 2 threads (binding: memory)
         512 MiB/worker I/O -> 14 workers x 1 thread (binding: memory)
```

**Literature grounding (from `docs/literature/microplastics_review_v1.md`, §4–6)**
- Ho 2023 / Stanton 2019 — Nile Red is polarity-based; honest output is a polarity class with abstention
- Park 2022 — mean F1 0.736, mean IoU 0.617 on real fluorescence microscopy
- Tanoiri 2026 — detection probability 50 % at 40 µm, 90 % at 100 µm
- Schnepf 2023 — report size as maximum Feret's diameter
- Liu 2023 — nine validated shape categories, scored by five experts over 11,042 particles
- Isobe 2019 — errors in both directions; running-mean smoothing above 0.5 mm
- Vitali 2023 — no fully validated quantification method; recovery, blanks, detection limit must be modelled
- Kukkola 2023 — calibration protocol, nine-polymer key, five natural false-positive substrates
- van Mourik 2021 — polymer typing agrees in 53–100 % of labs
