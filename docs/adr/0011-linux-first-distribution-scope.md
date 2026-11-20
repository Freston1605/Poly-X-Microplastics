# ADR-0011: A Linux-first research instrument distributed as MICRA

- **Status:** Accepted
- **Date:** 2026-11-20

## Context

Poly-X was written by Cristofher Ferrada as a PhD course project and is now
continued in the Laboratorio de Química Ambiental (Instituto de Química, PUCV)
by Franco Valdés, to train, validate and test models for a paper. The next
deliverable is reproducible headless results over SSH on OCÉANO, not a
downloadable desktop application. The fork is a derived work under the MIT
licence; the original author keeps authorship and the upstream repository.

The repository currently promises a general-audience product it does not have.
`README.md` opens with `pip install polyx-microplastics` while the name returns
404 on PyPI; there is no release tag (the only `v*` tag is `v0.0.0-smoketest`)
and no GitHub release, so `polyx-download-models` requests
`releases/latest/download/bestdetectormedium.pt` and gets a 404. Around that
missing channel sit two installer generations (`install/bootstrap.py` and its
wrappers, `install/update.py` and four `actualizar*` scripts), a
PyInstaller bundle, a Windows uninstaller and migrator, a Linux `.desktop`
template, and a macOS `.command` path that `LEEME_macOS.md` itself says was
never run on a Mac.

That surface is unsustainable for one maintainer and it is not what the paper
needs. Its costs are concrete:

- **The update channel executes whatever is on `main`.** `install/update.py`
  downloads `archive/refs/heads/main.zip` and overwrites program files, and
  `polyx.gui.updates` points users at it. A push to `main` is therefore remote
  code execution on every installed copy, with no tag, digest or rollback.
- **Three platforms are claimed and one is tested.** Every CI job runs on
  `ubuntu-24.04`; the Windows installer is the largest unmaintained surface and
  the macOS path is unverified by its own documentation.
- **The advertised Python range is wider than the stack.** `requires-python`
  says `>=3.11` with no upper bound, while `PySide6==6.7.2` declares `<3.13` and
  `numpy==1.26.4` publishes no cp313 wheel: on 3.13 or 3.14 installation fails
  at resolution.
- **The distribution identity is borrowed.** The package name, repository URLs
  and report footer all point at the upstream project, which the lab does not
  own and does not maintain.

## Decision

1. **Scope is a research instrument with named users.** The supported audience
   is the maintainer and named collaborators who receive instructions directly.
   The project's status is **pre-alpha** until the paper's artifact is archived;
   nothing in the documentation offers general-public support.
2. **Supported means exercised by CI.** Tier 1 is Linux x86_64 (headless and
   offscreen Qt). Tier 2 is Windows, documented as best effort through
   `uv sync --extra gui`, with no installer and no CI. Tier 3 is macOS: the code
   stays portable, but no launcher, bundle or platform claim ships.
3. **The project is MICRA.** Display name `MICRA` (*Microplastic Identification,
   Counting and Recognition Application*), distribution name `micra`, repository
   `Freston1605/micra`. The import package stays `polyx` and the `POLYX_*`
   environment variables and user data directory stay unchanged until a
   post-paper rename; `micra` is a new console script over the existing
   `polyx.cli.main` dispatcher, and the `polyx*` scripts remain as legacy
   aliases for the same period. `pipeline` joins the dispatcher so both commands
   expose `train`, `detect`, `eval`, `report`, `pipeline`, `download-models` and
   `gui` as subcommands. The help screen derives the program name from
   `argv[0]` instead of hardcoding `polyx`.
4. **Python is `>=3.11,<3.13`.** 3.11 is the container base and 3.12 is the
   development interpreter; newer interpreters are added only when the locked
   stack supports them and CI exercises them.
5. **Channels are source checkout and container image.** A lab machine runs
   `uv sync --locked --extra gui`; a compute node runs the Apptainer image built
   from the offline wheelhouse (ADR-0008). PyPI publication is deferred until
   the cluster smoke passes and the metadata above is truthful. The first upload
   is a pre-release (`2.1.0a1`), which `pip` skips unless `--pre` is given; it
   proves Trusted Publishing against a throwaway version number and reserves the
   name. The paper's stable release is tagged from `main` and archived.
6. **Weights are a citable artifact with a mandatory digest.** The canonical
   record lives on Zenodo (versioned, DOI, CC BY 4.0, with a model card covering
   dataset provenance, protocol and metrics), mirrored as a GitHub release asset;
   code stays MIT. Every download verifies a `sha256` from the release manifest,
   never from the bytes just fetched.
7. **The consumer-distribution track is retired.** `install/`, the auto-updater
   and its notice, `requirements.txt`, the PyInstaller spec, the Spanish
   end-user docs and the generated manual's installer chapters leave the
   supported surface. `empaquetar_dataset_para_drive.bat` moves to
   `tools/data/` because it packages data, not software.
8. **Credit is split by role.** Cristofher Ferrada is the author of the original
   work and keeps the `LICENSE` notice, with one added line for modifications by
   the current maintainer. `pyproject.toml` lists him as author and Franco
   Valdés as maintainer; `SECURITY.md` and the GUI/report footers name the
   maintainer, the original design and the laboratory. The README carries an
   Origin paragraph. `CITATION.cff` is created at the first archived release,
   lists both, and references the upstream repository. The documentation makes
   no claim about whether the upstream project is maintained.

## Considered options

- **Ask the original author to transfer the repository.** Rejected: the lab
  continues the work under its own identity and does not need to take ownership
  of the upstream project. The fork's permanent "forked from" label is accepted
  as an accurate description of the relationship.
- **Keep the three-OS installers and the updater.** Rejected: two of the three
  platforms are untested, and the updater turns `main` into a code-execution
  channel for a tool with no release to update to.
- **Publish to PyPI now.** Rejected: the package is not installable on the
  Python versions its metadata advertises, the cluster path is unverified, and a
  version number on PyPI is permanent. The pre-release rehearsal above gives the
  same proof without spending a stable version.
- **Retarget the updater to tagged releases.** Rejected: nobody in the supported
  audience needs it, and deleting removes the risk instead of managing it.
- **Rename the import package, console scripts and environment variables now.**
  Rejected for this window: it touches every module, test, script and the user
  data path for no scientific benefit before the paper. Recorded as a post-paper
  follow-up.
- **Acronym alternatives (`SMACK`, `MAPS`, `PACE`, `FLAP`).** Rejected: `SMACK`
  names a spectral method the code does not implement and reads as slang;
  `MAPS`, `PACE` and `FLAP` are taken on PyPI. `micra` is free, pronounceable in
  Spanish and English, and is the plural of *micron*, the unit of the sizing
  output.

## Consequences

- `README.md` loses its PyPI quickstart, its second installation section, the
  updating and migration sections, and the installer promises (Desktop shortcut,
  Linux `.desktop` entry, macOS `.command`). It gains the pre-alpha banner, the
  support table and two install paths: container, then `uv sync`.
- Two defects disappear with the code instead of being repaired: the uninstaller
  and migrator look for `<dir>\polyx\launcher.py`, which the `src/` layout never
  creates, and the portable bundle copies `SETUP.bat` without `bootstrap.py`.
- `ci.yml` drops the installer smoke step; `tools/check.py` and
  `CONTRIBUTING.md` drop `install/` from the ruff paths; `tests/test_install.py`,
  `tests/test_updater_observability.py` and the updater assertions in
  `tests/test_platform.py` and `tests/test_gui_package.py` go with the code.
  The launcher's README button opens `README.md`, since `LEEME.txt` is gone.
- `install/update.py`, `src/polyx/gui/updates.py`, `platform.name_updater()`
  and `platform.launch_updater()` disappear, and with them the two remaining
  hardcoded references to the upstream repository. The
  remaining `CrissFerrada/...` URLs in `pyproject.toml`, `README.md`,
  `CLAUDE.md`, the manual generators and `download_models.py` move to the MICRA
  repository. `SECURITY.md` already names the fork; `docs/agents/issue-tracker.md`
  and the runbook follow the rename like every other reference.
- `micra download-models` keeps returning 404 until the first release or Zenodo
  record exists. That is a release-checklist item, not a code defect.
- No component updates itself. A collaborator re-runs `uv sync --locked` or
  pulls a new image digest; nothing executes a branch tip.
- The name `micra` is not reserved until the first upload. Anyone may take it on
  PyPI before `2.1.0a1` ships; the rehearsal is also the reservation.
- Adding a distribution channel (PyPI beyond the rehearsal, conda, Homebrew, a
  bundled installer or an auto-updater) now requires a new ADR. `CONTRIBUTING.md`
  says so in one line, so the next expansion is a decision rather than a drift.
