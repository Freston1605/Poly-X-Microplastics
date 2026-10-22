# Poly-X: fork-vs-restart decision and takeover plan

> **READ `docs/GSD_DIRECTIVES.md` FIRST.** It carries three corrections to this document, and it is the document to hand to GSD. The corrections concern: (a) the AMD/ROCm strategy, already decided by D031 and L043/L044; (b) `requirements.txt` not gaining a torch or sam2 pin, already decided by D039; and (c) the worktree interpreter question, already settled in practice by L029. The architecture (Block B), the science (Block D), the provenance and licensing (Block E) and the proposal reconciliation (Block F) below are unaffected and stand as written.

**Date:** 2026-10-20
**Context:** Taking over `CrissFerrada/Poly-X-Microplastics` from Cristofher Ferrada; fork at `Freston1605/Poly-X-Microplastics`.
**Verdict up front:** Do **not** restart from scratch. Restart is the more expensive and less defensible path, and most of the problems you listed are already fixed in your fork. The real problem is not the base repo's architecture — it is that **two large pieces of unfinished work are sitting in different branches and neither is integrated**, and the repo has no license, no citation, and no provenance.

---

## 1. What I found in your working copy

### 1.1 Your fork is already 7 commits ahead of upstream, and upstream is dormant

```
merge-base: 6e749d1  (== upstream/main HEAD)
upstream/main..main:  7 commits, 391 files, +113,657 / −13,008
main..upstream/main:  0 commits
```

Upstream's last commit is **2026-09-04** — six weeks stale, with the bulk of its 103 commits landing in a single August 2026 burst. Ferrada is leaving the line, so **upstream will not merge anything, and there is no upstream to reconcile against**. Your fork is the de facto trunk.

The 7 commits you have are not cosmetic:

| Commit | What it actually is |
|---|---|
| `e797fa8` | **The restructure you thought was still pending.** `polyx/core/` → `polyx/model/` (13 files, Qt-free) + `polyx/gui/` (57 files). Plus the Spanish→English translation of identifiers, docstrings, comments and headings, with 5 dedicated test files guarding it. 174 files, +30,165/−13,124. |
| `c639a40` | GitHub Actions CI with exact pins, headless Qt libs, `QT_QPA_PLATFORM=offscreen`. |
| `5a6578d` | Pillow 10.4.0 → 12.3.0 (33 known advisories). |
| `ab7e53a` | HTML escaping of untrusted text in the report generator. |
| `0be5629` | CONTRIBUTING guide. |
| `48598d9` | Test-suite language pin. |
| `e058e6a` | Vendored agent skills (unrelated to the science — consider dropping from history if you want a clean provenance). |

### 1.2 Your five concerns, checked against the code

| Your concern | Verdict | Evidence |
|---|---|---|
| ML pipeline not separated from GUI | **Already fixed** | `polyx/model/` (13 files) imports with **PySide6 blocked** — verified. `polyx/gui/` imports `polyx.model` 95 times; `polyx.model` imports `polyx.gui` **zero** times. Clean one-way dependency. |
| Vast sections in Spanish | **Already fixed** | `auditar_traduccion.py` → 701/705 translated, 4 untranslated (0.6 %), both in trainer pages. Identifiers/docstrings/comments/headings are English; Spanish survives only inside the `tr()` UI layer, which is the correct place for it. |
| In-house segmentation instead of tried-and-true two-step models | **Real, and you are already fixing it** | `polyx/model/morfologia.py` (1,045 lines) does Otsu + connected-component inside each detection box. **But `milestone/M007` (15 commits, 198 files, +49,327) already builds exactly the replacement you want**: SAM 2.1 `hiera-tiny` as a frozen mask teacher → polygon export → YOLO-seg student, with a fidelity bench against a pinned analytic phantom corpus. |
| Didn't use standard libraries for the GUI, like OpenCV | **Inverted** | OpenCV *is* the standard library here and it is used properly: `cv2.imdecode`/`imencode` over `np.fromfile` (the Windows-accented-path fix), `cv2.rectangle`/`putText` for annotation, `cv2` + numpy only in `morfologia.py` — with `scipy` deliberately *not* relied on (it is only a transitive dep of ultralytics, and `report_html.py` hand-implements Kruskal–Wallis and the t-distribution rather than assume it). The 103 `QPainter` calls are theme-consistent icon and canvas drawing, which is the right tool for that job. **Nothing to undo here.** |
| ML pipeline too opinionated, heavily YOLO-based | **Real, and mostly in the GUI, not the core** | See §3. The real seam problem is concentrated in the trainer, the labeller's label format, and the detector's runner — not in the measurement science. |

**So: two of your five concerns are already solved in your fork, one was never a problem, and two are real — and the harder of the two is already 49k lines into development on `milestone/M007`.**

### 1.3 What is genuinely still broken (and none of it is "the base repo")

1. **M007 is unmerged.** 15 commits, 198 files, +49,327 lines — the SAM-teacher → YOLO-seg pipeline, a resolution sweep, a fidelity bench, a pinned mask corpus, and 11 new test files — sitting on a branch that is not on `main`. `main` does not contain any of it.
2. **The trainer is the YOLO-opinionated hotspot.** `polyx/gui/trainer/` has ultralytics imports in `runner.py`, `dominio.py`, `pages/evaluar.py`, `pages/exportar.py`. `report_html.py` also touches ultralytics. The labeller writes YOLO `.txt` labels, so the label format is the de-facto data contract. `model/yolo_wrap.py` is 503 lines and is the *only* place `ultralytics` is imported for inference — that part is already a clean seam.
3. **Two parallel data types.** `main` has `model/yolo_wrap.Detection` (bbox, no mask). M007 has `tools/train_yolo_seg.Instance` (box + mask + score) behind a `predict(frame_bgr) -> list[Instance]` protocol with `make_predictor()` as the factory seam. They are not yet unified.
4. **No license, no citation, no provenance.** No `LICENSE` file. README and CLAUDE.md both still say **"Author: Cristofher Ferrada"** and point at `CrissFerrada/Poly-X-Microplastics`. For a paper's Methods section that is a citable artifact, this matters.
5. **The scientific gap, which is the real one.** Your own literature review (60 studies, PRISMA-documented) is more demanding than the code: Nile Red is a *polarity* signal (Ho 2023, Stanton 2019), so the honest output is "polarity class + abstention"; the real-image segmentation ceiling is **mean F1 ≈ 0.736** (Park 2022), not ≈0.98; **detection probability is 50% at 40 µm and 90% at 100 µm** (Tanoiri 2026), so a count without that curve is reporting unquantified censoring; size should be **maximum Feret** (Schnepf 2023) and shape should use **Liu 2023's nine validated categories**; and no fully validated quantification method exists (Vitali 2023), so recovery, blanks and detection limit must be modelled explicitly.

Your repo already handles the *abstention* insight better than the proposal does — `model/asignacion.py` refuses to attribute a polymer below a confidence threshold, and the measured per-class recall (PET 0.98, PP 0.70, LDPE 0.54) is consistent with the solvatochromism explanation. **Your review's §4.1 and §6.1–6.3 are, in effect, the scientific specification your code is already 80% built against.** That is the strongest argument against a restart: the remaining work is *alignment*, and alignment does not require throwing away validated measurement code.

---

## 2. The decision

**Keep the fork. Do not restart.** Reasons, in order of weight:

1. **A restart throws away the expensive, correct parts.** `morfologia.py`'s Feret/geodesic machinery has a measured median error of 0.6% and worst case 4.7% against known-size synthetic shapes, pinned in `tests/test_morfologia.py` and `tests/test_geodesia.py`. The report generator is 3,038 lines with 12+ dedicated test files. The theme system, the i18n layer, the 40 vector icons, the OS-portability isolation, the CI, the 47-file test suite. Restarting means re-deriving all of it.
2. **A restart has no scientific baseline.** You would begin with no measured recall, no pinned corpus, no phantom bench, no annotated particle set. Since the whole deliverable is a *measurement*, losing the measurement history makes every future number unanchored.
3. **A restart produces two repos.** You would need the old one to keep producing comparable numbers during the transition anyway — which is a de-facto fork with worse tooling.
4. **The proposal depends on this repo's continuity.** The FONDECYT Formulación (Waldo Quiroz as IR) and the PLoS ONE methods lineage both reference a working, citable tool. A restart mid-proposal creates a reproducibility gap in the exact artifact the proposal claims to build on.
5. **Your own review already decided the scientific direction.** It says what the pipeline must be able to do. A rewrite that changes nothing scientifically still costs months; a rewrite that changes something scientifically needs the measurement code to stay.

**What "not restarting" must not mean**, though: it must not mean "patch the base forever," and it must not mean "keep two codebases." The plan below converges to one tree.

---

## 3. The architecture that satisfies your concern #5 without a rewrite

The target is a **three-layer** tree with a hard one-way dependency. The key insight: your measurement science (`morfologia`, `calibracion`, `metrics`, `asignacion`) is already model-agnostic — it consumes *boxes* and *masks*, and knows nothing about YOLO. The opinionation lives in three places only: the **detector** (which produces boxes), the **labeller** (which produces labels), and the **trainer** (which produces weights). Confine the opinion to those, and the core becomes replaceable.

```
polyx/
  core/                 # model-agnostic science. NO Qt, NO ultralytics. (rename model/ -> core/? see note)
    detect/             # <-- NEW: the seam
      protocol.py       #   Detection, Instance, DetectionSource (Protocol)
      native.py         #   threshold/connected-component (today's morfologia.segment)
      yolo.py           #   UltralyticsDetector  (the only ultralytics import)
      sam.py            #   Sam2Segmenter        (M007's teacher, promoted to a runtime backend)
      factory.py        #   make_detector(name, ...) -> DetectionSource
    measure/            # morfologia.py, calibracion.py  (unchanged science; new home)
    quality/            # metrics.py, asignacion.py, dataset_audit.py
    provenance/         # procedencia.py, observabilidad.py
    data/               # the label/annotation format lives HERE, not in the GUI
      labels.py         #   read/write the format, one place
  gui/                  # everything Qt
  cli/                  # <-- NEW: headless entry points so the pipeline is scriptable/CI-able
```

Three rules make it stick:

- **R1 — one direction.** `core` never imports `gui` or `ultralytics`. Enforce it with a test (you already have `tests/test_model_package.py` doing exactly this kind of import guard; extend it).
- **R2 — one detector protocol.** Every backend implements `predict(frame_bgr) -> list[Instance]`. M007 already has this exact seam (`make_predictor`, `Instance(box, mask, score)`) — promote it to `polyx/core/detect/protocol.py` instead of inventing a new one. The trainer's `UltralyticsSegPredictor` and a future `Sam2Segmenter` are then interchangeable, and a `ThresholdDetector` is the control arm.
- **R3 — one label format, in `core/data/`.** Today the labeller writes YOLO `.txt` and the trainer consumes it, so the format is welded to the tool. Put the reader/writer in `core/data/labels.py`, and make YOLO-txt one *adapter* among several (COCO, Pascal VOC, and a plain JSON for your own corpus). Then the labeller can emit COCO and the trainer can read COCO without either knowing about the other.

**A fourth, non-negotiable rule, from your own review:**

- **R4 — the output is a polarity class with abstention, never a polymer name.** `asignacion.py` already implements this. Make it a *contract*, not a feature: `Detection.polarity_class` + `Detection.assigned: bool` + `Detection.assigned_conf`, and have the report state the unassignable fraction as a result (Vitali 2023 / Ho 2023). This also means the trainer's class list becomes `PET | PP | LDPE | unassignable` at most, and the polymer claim in the FONDECYT proposal gets its needed softening.

---

## 4. Actionable plan

### Block A — consolidate what you already have (week 1–2)

Do this *before* any new architecture work. It is the part that is already paid for and currently unbanked.

- **A1.** Merge `milestone/M007` into `main`. It is 15 commits ahead of the merge base with zero conflicts (it is the only branch with new files plus one touched file). First read the branch's own artifacts — `tests/fixtures/mask_corpus/teacher_v1.json`, `tools/mask_corpus.py`, `tools/bench_mask_fidelity.py` — and check the committed resolution sweep verdict ("tiled-1280") is recorded in `InferenceParams` as the branch claims.
- **A2.** Run the full suite on the merged tree and record the number. The sidecar evidence says 210 tests passing; your tree has since gained the translation tests and M007's 11 new files. **Fix the stale count in AGENTS.md** — it still says 128, and the sidecar flagged exactly this. A wrong test count in the doc an evaluator reads is a small but real credibility hit.
- **A3.** Add an **ultralytics-present CI job**. The sidecar finding is important and still true: *"A green test run here does not validate the YOLO pipeline."* Your CI installs pinned requirements including `ultralytics==8.3.40` — confirm it actually does, then add a one-line smoke test that imports `polyx.core.detect.yolo` and runs a single inference on a 1×1 image. Without it, CI cannot catch a break in the one module that matters most to the paper.
- **A4.** Add a **second CI job that runs the M007 fidelity bench** on the pinned phantom corpus, so the segmentation claim is machine-checkable by a reviewer rather than asserted in a Markdown table. The branch already builds for this (`--check` recomputes the baseline from committed pixels with no torch) — that is exactly a CI-shaped design; wire it up.
- **A5.** Re-home `docs/literature/` into the repo (it is currently untracked) and commit the PRISMA artifacts under `runs/litreview_v1/` **or** a tracked `docs/literature/data/`. Right now the review that justifies the architecture lives outside version control. The extraction table and PRISMA JSON are small and should be committed; the 4.1 MB `records_dedup.jsonl` can stay ignored.
- **A6.** Delete `polyx/core/`, `polyx/detector/`, `polyx/trainer/` if they are empty leftovers on disk (verified: 0 `.py` files in each), and remove the stale `__pycache__` dirs. Also add `__pycache__/` and `*.pyc` to `.gitignore` if not already there — 228 `.pyc` files are currently untracked noise in your tree.

### Block B — the seam, done properly (week 2–4)

- **B1.** Create `polyx/core/detect/protocol.py` holding `Instance` (box + mask + score) and `DetectionSource` (a `Protocol` with `predict(frame_bgr) -> list[Instance]`). Move M007's `Instance` there; keep `tools/train_yolo_seg.py` importing it, so there is one definition, not two.
- **B2.** Implement three backends against that one protocol:
  - `ThresholdDetector` — wraps today's `morfologia.segment` (Otsu + connected component). This is your **control arm** and it must stay in the tree permanently; it is the baseline every learned method is measured against, and the M007 bench already assumes it exists.
  - `UltralyticsDetector` — wraps today's `yolo_wrap`.
  - `Sam2Segmenter` — promotes M007's SAM 2.1 teacher to a runtime backend (it is already a `Sam2Backend` with lazy `torch`/`sam2` imports).
- **B3.** `make_detector(name, **kwargs)` as the single factory. This is the "model-agnostic" property, made concrete and testable: a test that parametrizes over backends and asserts identical output shape.
- **B4.** Introduce `polyx/core/measure/` as the new home of `morfologia.py` + `calibracion.py` (moved, not rewritten), and `polyx/core/quality/` for `metrics.py` + `asignacion.py` + `dataset_audit.py`. Move only; every measurement test must still pass unchanged, byte-for-byte in behaviour. **The point of this move is to make the model-agnostic core legible as a unit** — not to change a line of measurement logic.
- **B5.** Unify `Detection` and `Instance` into one type. `Detection` is bbox-only; `Instance` is bbox+mask+score. The measurement code needs the mask (it segments inside the box today) and the classifier needs the score. One type, `Detection`, with optional `mask`, consumed everywhere.
- **B6.** Move the label format into `polyx/core/data/labels.py` with a `LabelFormat` adapter per format (YOLO-txt, COCO, JSON). The labeller and trainer then depend on `core.data`, not on each other.

### Block C — make the pipeline headless and scriptable (week 4–6)

- **C1.** Add `polyx/cli/` with `polyx.cli.detect`, `polyx.cli.evaluate`, `polyx.cli.audit`. Today every capability is reachable only through a Qt window, which is why the CI cannot test the ML path (sidecar finding: *"none instantiate Qt windows, run inference, or train a model"*). A CLI entry point is what lets a reviewer, a CI job, and your own scripts run the pipeline without a display.
- **C2.** Make the CLI emit a **machine-readable run record** (JSON) as well as the HTML report: per-image, per-class, per-particle — count, box, mask area, Feret max, geodesic length, aspect, solidity, polarity class, assigned or unassigned, confidence, detection limit, calibrated µm/px, uncertainty. This is the artifact a paper's Methods and Results section can cite, and it is what makes your counts comparable to anyone else's.
- **C3.** Add **detection probability as a function of particle size** to the CLI output, using Tanoiri et al. (2026)'s curve as the external prior and your own measured curve as the estimate. Your review's implication 3 says a count without this is "reporting an unquantified censoring." Make the curve a first-class output, not a footnote.
- **C4.** Add the **blanks and recovery** inputs to the CLI so a count can be corrected: blank subtraction, spike recovery, and a detection limit per polymer. Vitali (2023) and Ciornii (2025) make this the norm; without it your numbers are not comparable to the interlaboratory literature.

### Block D — the science, aligned to your own review (week 6–10)

This is where your literature review becomes a specification rather than a reading list.

- **D1.** Replace the two-class `fibra`/`fragmento` vocabulary with **Liu et al. (2023)'s nine validated categories** — fiber, rod, ellipse, oval, sphere, quadrilateral, triangle, free-form, unidentifiable — scored the way Liu scored them. Note the cost: your current `morfotipo` is a two-way split with a 1.1% fibre rate (measured over 6,638 particles). Adopting the nine-category taxonomy will *change* your measured shape distribution, so it is a results-affecting decision — do it deliberately, with a before/after comparison in the report.
- **D2.** Adopt **maximum Feret** as the reported size, per Schnepf (2023). You already compute it (`feret_maximum`) and already prefer the larger of Feret/geodesic — this is mostly a *reporting* change, and it is cheap. Keep the geodesic as the additional descriptor; do not report it as the size.
- **D3.** State the **detection limit per polymer and per false-positive substrate**, per Kukkola et al. (2023), who publish both a microscopy calibration protocol and an identification key for nine polymers *and five natural substrates that produce false-positive Nile Red signals*. Your calibration code already supports line and circle calibration — extend it to publish the limits.
- **D4.** Decide the **polymer-typing scope** deliberately. Your review's implication 1 and the code's `asignacion.py` agree: Nile Red yields a polarity class plus abstention, and polymer identity belongs to µFTIR or Raman. So either (a) the pipeline reports polarity class + abstention only, and the polymer claim moves to a documented µFTIR/Raman confirmation subset — or (b) the pipeline claims polymer identity and you must add an orthogonal confirmation path. **Do not ship a polymer label from fluorescence alone.** This is also the exact defect the FONDECYT review flagged in the proposal's M7.
- **D5.** Benchmark against the **real-image ceiling**, not the synthetic one. Park et al. (2022) put mean F1 at 0.736 and mean IoU at 0.617 for fluorescence microscopy of real microplastics. Your report should carry that number as the reference point, and your synthetic-phantom results (which M007's fidelity bench produces) should never be presented as field performance. Your review's gap section flags exactly this: *"No retrieved study trains only on synthetic/rendered microplastic images and reports the performance drop on real filters."* — **you can be the study that does.**
- **D6.** Report errors **in both directions** — false recognition *and* overlooking (Isobe et al. 2019) — and implement the >0.5 mm running-mean smoothing they recommend for size distributions. `metrics.py` already computes misclassification as a separate cell from TP/FP/FN; make sure the report surfaces it.

### Block E — provenance, licensing, and the handover (week 10–12, and it gates the paper)

- **E1.** **Add a LICENSE.** There is none. Without one, nobody may legally reuse, modify, or redistribute — including you, in a fork. If the group intends it to stay internal, say so explicitly in the README ("All rights reserved, internal use only"); if it is meant to be citable and reusable, pick a license (AGPL-3.0 if you want copyleft like the scientific-instrument ecosystem, MIT/Apache-2.0 if you want adoption, CC-BY-4.0 for the docs). This is a decision only the group can make — raise it with Waldo Quiroz and Ferrada before you publish anything.
- **E2.** **Fix authorship and pointers.** README and CLAUDE.md still read "Author: Cristofher Ferrada" and point at `CrissFerrada/Poly-X-Microplastics`. Update to name both you and Ferrada, point at your fork as the working repo, and add an **"Original work" / provenance note** stating that the base repo is Ferrada's, that the fork is the maintained continuation, and which substantial changes are yours. This is the difference between a fork and a fork-and-claim. Given you are taking over someone's PhD line, write it explicitly and kindly; a future reader of the Methods section will want to know who did what.
- **E3.** **Add a CITATION.cff** and a "How to cite" section. If the tool is going into a paper's Methods, it needs a citable version. Pin it to a release tag.
- **E4.** **Tag a first release** (`v0.1.0-fork` or similar) with the exact environment that produced your reported numbers: Python version, `requirements.txt` pins, ultralytics version, hardware, the model weights (by checksum), and the dataset version. The repo already has a `procedencia.py` module for run-level traceability — extend the same discipline to releases.
- **E5.** **Provenance for the dataset and the weights.** `models/bestdetectormedium.pt` and `data_microplastico/` are gitignored. That is right for size, but it means the published numbers are not reproducible from the repo alone. Publish: the trained weights (or a DOI to a release), the dataset manifest (plate IDs, annotation provenance, per-image hash), and the exact split. This is what turns "we measured X" into "you can verify X."

### Block F — the proposal interface (parallel, week 1+)

- **F1.** Your literature review and the proposal must tell the **same story**. The Formulación currently promises selective identification of five polymers (PE, PET, PP, PVC, PA) by Nile Red colour — and your own review says the honest output is a polarity class with abstention, with 76–88 % accuracy and PE/PP confusion as the dominant failure. The proposal's M7 major comment says the same. Reconcile before submission: either the proposal narrows to counting + sizing + morphology + polarity class with abstention, or it adds the µFTIR/Raman confirmation subset. **Your repo already implements the abstention; the proposal should be brought in line with the code.**
- **F2.** Note the mismatch in polymer scope: the proposal targets PE, PET, PP, PVC, PA; the code has PET, PP, LDPE. PE and LDPE are different polymers; the measured recall table is PET 0.98 / PP 0.70 / LDPE 0.54. Decide the scope and make proposal, code, and dataset agree.
- **F3.** Report the F1 = 0.736 real-image ceiling in the proposal's expected-results section. A reviewer who knows Park et al. (2022) will discount an uncalibrated expectation; one who reads the ceiling quoted will trust the rest.

---

## 5. Sequencing, in one view

```
Week 1–2   A: merge M007 · run suite · fix test count · ultralytics CI job
                 bench CI job · track the literature review · clean pycache
Week 2–4   B: protocol.py + 3 backends + factory · move measure/quality · unify
                 Detection/Instance · labels.py adapters
Week 4–6   C: polyx/cli/{detect,evaluate,audit} · JSON run record · detection
                 probability vs size · blanks/recovery/LOD inputs
Week 6–10  D: Liu 9-shape taxonomy · Feret as reported size · LOD per polymer and
                 false-positive substrate · polymer-scope decision · Park-anchored
                 benchmark · errors in both directions + 0.5 mm smoothing
Week 10–12 E: LICENSE · authorship + provenance note · CITATION.cff · v0.1.0 tag ·
                 weights + dataset manifest + split published
Parallel    F: proposal reconciled with review and code · polymer scope · F1 0.736
```

**Critical path:** B1→B2→C1→C2. Everything in D and E depends on the seam existing, because D needs a backend-agnostic way to re-measure, and E needs a release artifact that contains a settled measurement path.

**Do not start D before B.** Nine-shape taxonomy and detection-probability curves are measurements over a settled pipeline. Measuring them against a pipeline you are still moving is how you get numbers you have to re-measure.

---

## 6. What I would tell a reviewer who asked me this

The base repo is not the problem. It is a competent single-purpose instrument with a genuinely good measurement core — the Feret/geodesic work with its measured error bounds and the abstention logic are better than most published microplastics pipelines, and better than the group's own current proposal.

What is missing is not architecture but **integration and provenance**: 49k lines of the right replacement sitting unmerged on a branch, two parallel detection types, a label format welded to the GUI, no license, no citation, authorship still pointing at someone else's repo, and a review-in-progress whose findings are not yet the specification the code is built against.

Restart would cost you the measurement history and the tested code, and buy a cleaner folder structure you would spend a year rebuilding anyway. The plan above converges to one tree in roughly three months, and every step produces something citable.

**The one thing that would change my advice:** if the group has decided to abandon the Nile Red fluorescence approach entirely — for example, moving to hyperspectral or Raman-based identification as the primary method — then the *detector* and *trainer* layers become near-worthless and a scoped rewrite of only the measurement-and-identification core is justified. That is a scientific decision, not a software one, and it belongs to Waldo and Ferrada. Your literature review already contains the evidence for it: Nile Red is a screening method whose signal is polarity-based, and the paper you reviewed says so. Read §4.1 and §6.1–6.3 again with that question in mind. But that decision is about the *method*, and even then the measurement code, the calibration, the report generator, the CI, and the test suite all survive.

---

## Appendix — evidence for each claim above

**Fork state**
```
$ git log --oneline upstream/main..main
e797fa8 feat: Translate code-facing text to English
ab7e53a fix(security): escapar el HTML de textos ajenos en el informe
5a6578d fix(security): Pillow 10.4.0 -> 12.3.0 (33 asesorías conocidas)
0be5629 docs: guia de contribucion (que hace el CI y como depurarlo)
c639a40 ci: anadir GitHub Actions (instala pins exactos y corre pytest)
48598d9 test: fijar el idioma de la suite a espanol (conftest)
e058e6a chore: auto-commit after workflow-template
$ git log --oneline main..upstream/main     # (empty)
$ git diff --stat upstream/main..main | tail -1
 391 files changed, 113657 insertions(+), 13008 deletions(-)
```

**Upstream dormancy**
```
$ git log --format="%ad" --date=format:"%Y-%m" upstream/main | sort | uniq -c
     11 2026-05
      8 2026-06
     80 2026-08
      4 2026-09     <- last commit 2026-09-04
```

**Layer separation (verified by execution, not by reading)**
```
$ python3 -c "block PySide6; import polyx.model"
polyx.model imported OK without PySide6
$ grep -rn "from polyx.gui|import polyx.gui" polyx/model/ | wc -l
0
$ grep -rn "from polyx.model|import polyx.model" polyx/gui/ | wc -l
95
```

**Restructure provenance**
```
$ git log --oneline --follow --name-status -- polyx/model/yolo_wrap.py
e797fa8 feat: Translate code-facing text to English
R054  polyx/core/yolo_wrap.py  polyx/model/yolo_wrap.py
```

**Translation status**
```
$ python3 auditar_traduccion.py
cadenas en tr(): 823 (705 distintas)
traducidas:      701
SIN TRADUCIR:    4   (0.6 %)
```

**ultralytics import surface**
```
model/ : yolo_wrap.py (the only inference import) · procedencia.py (version string) · dataset_audit.py, i18n.py, asignacion.py, morfologia.py, plataforma.py (comments only)
gui/   : trainer/runner.py · trainer/dominio.py · trainer/pages/{evaluar,exportar,informe}.py · report_html.py
```

**M007 scope**
```
$ git diff --stat main..milestone/M007 | tail -1
 198 files changed, 49327 insertions(+), 8 deletions(-)
new: tools/{sam_teacher,seg_dataset,train_yolo_seg,export_yolo_seg,mask_corpus,
           teacher_masks,bench_mask_fidelity,provision_sam2_runtime,run_teacher_pipeline}.py
     tests/{test_sam_teacher,test_seg_dataset,test_train_yolo_seg,test_export_yolo_seg,
            test_mask_corpus,test_teacher_masks,test_teacher_pipeline_e2e,test_teacher_artifact,
            test_seg_compare_artifact,test_bench_mask_fidelity,test_benchmark_artifact,
            test_resolution_policy,test_provision_sam2,test_provision_yolo_seg}.py
     tests/fixtures/mask_corpus/teacher_v1.json + 20 teacher_masks/*.png
```

**Licensing / authorship**
```
$ ls LICENSE* COPYING*      -> (no such file)
README.md:6  **Author:** Cristofher Ferrada · PhD (c) in Chemistry · PUCV · 2026
README.md:623 Repository: https://github.com/CrissFerrada/Poly-X-Microplastics
```

**Literature grounding for Block D** (your own `docs/literature/microplastics_review_v1.md`, §4–6)
- Ho et al. 2023, Stanton et al. 2019 → Nile Red is polarity-based; "honest output of a Nile Red pipeline is a polarity class with abstention"; a second stain (DAPI) is needed to exclude biological material.
- Kukkola et al. 2023 → calibration protocol, nine-polymer identification key, five natural false-positive substrates, detection limits.
- Park et al. 2022 → mean F1 = 0.736, mean IoU = 0.617 on real fluorescence microscopy of microplastics.
- Tanoiri et al. 2026 → detection probability 50 % at 40 µm, 90 % at 100 µm.
- Schnepf et al. 2023 → report size as maximum Feret's diameter.
- Liu et al. 2023 → nine validated shape categories, scored by five experts over 11,042 particles.
- Isobe et al. 2019 → errors in both directions; running-mean smoothing above 0.5 mm.
- Vitali et al. 2023 → no fully validated quantification method; recovery, blanks, detection limit must be modelled.
- van Mourik et al. 2021 → polymer typing agrees in 53–100 % of labs — the tail is the honest headline.
