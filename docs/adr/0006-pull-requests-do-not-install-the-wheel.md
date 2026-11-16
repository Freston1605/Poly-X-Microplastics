# ADR-0006: A pull request does not install the wheel

- **Status:** Accepted
- **Date:** 2026-11-15
- **Landed in:** #104

## Context

Every pull request built the wheel, then installed it twice with the system
`pip` into a fresh virtual environment — once bare, once with the `[gui]`
extra — and ran each console script with `--help`. It was the only step that
looked at the distribution rather than the source tree, and it was honest work.
It was also in the wrong place, for three reasons.

**A version number on PyPI is permanent.** A release can be yanked, but the
version is spent: `2.0.0` cannot be uploaded again, by this project or anyone.
The failure the install catches — a wheel that does not import, an asset that
did not ship, an entry point that resolves to nothing — is therefore cheap in a
pull request and expensive at a tag, and the pipeline spent the cheap proof on
every pull request while the permanent one waited for the tag to be pushed.

**The pull request still needs the packaging proof the source tree cannot
see.** The gate imports `src/` through `PYTHONPATH`, so it cannot observe what
Hatchling actually packages. Once, a compiled `messages.mo` silently missed the
wheel because it is gitignored and Hatchling filters VCS-ignored paths; the
checker loads the Spanish catalog out of the built archive precisely because
no import of the source tree could have failed. That check stays on every pull
request — it is the one distribution invariant a source run cannot establish.

**The install was mostly a re-download.** Measured on three runs of the merged
pipeline on 2026-11-15, the packaging job took 165–184 s and 101 s of that was
`pip` fetching torch again, while the uv archive restored in the same job made
the equivalent `uv sync` take 3 s. The pull-request verdict was 380–418 s, and
43% of that wait was the packaging job waiting for `quality` and `test`
instead of working.

The entry-point failure the install was mainly guarding had also changed
owner: #100 added a test that reads the console scripts from the installed
distribution metadata, compares them with `[project.scripts]` and imports
every target. It runs inside the gate, in both contexts, and costs no wall
clock.

## Decision

**A pull request proves the source and the packaging invariant. A tag proves
the distribution.**

On a pull request the packaging job builds the wheel and sdist, runs
`tools/check_wheel.py` against the built wheel — the Spanish catalog is inside,
the pseudo-locale is not — and uploads the wheel so a reviewer can download
what the run produced. It installs no Qt system libraries (nothing there
imports PySide6), creates no virtual environment, and starts no console script.
It depends on the gate only, so it runs beside the 3.12 leg instead of after
it.

The wheel install and the console-script smoke are conditioned on
`github.event_name != 'pull_request'`, not on the ref being a tag. They run on
a tag, on a push to `main` and on a manual dispatch of `publish.yml`, from one
definition in `ci.yml`; a dispatch is therefore a real rehearsal of the tag
path rather than a different, weaker path. The install uses uv:

```bash
uv venv /tmp/smoke --python 3.11
uv pip install --python /tmp/smoke/bin/python "dist/*.whl[gui]"
```

One install with the extra proves the base distribution and the GUI surface
together, and uv resolves from the archive the job already restores instead of
downloading torch through `pip`. This deliberately leads the intended move of
distribution to uv; the project installer still uses pip.

`publish.yml`'s reusable-workflow job is named `ci`: "gate" names the local
check runner (`quality` in `ci.yml`) and nothing else. The artifact names stay
distinct — `wheel` for pull requests, `dist` for publishing — because a
reusable workflow shares the caller's run and names must not collide.

## Considered options

- **Keep installing the wheel on every pull request.** Rejected: it spends 101
  s proving the cheap failure and leaves the permanent one to the tag, which is
  the ordering this ADR is about. The entry-point class of failure it caught is
  now a test in the gate, and the catalog invariant stays a pull-request step.
- **Condition the install on the ref being a tag.** Rejected: the first tag
  would then be the first run of the path that matters, and a manual dispatch
  could not rehearse it without lying about the ref. The event is the honest
  condition.
- **Keep the system `pip`.** Rejected: it re-downloads torch that the same job
  already has in the uv cache, and it points the release path away from the
  resolver the lockfile is written for.
- **Install bare, then install the extra, as before.** Rejected: the second
  install re-resolves the same graph to add PySide6. One `[gui]` install leaves
  a superset environment, which is what the smoke needs to check both the CLI
  and the Qt import.
- **A separate packaging job for the smoke, running only at the tag.**
  Rejected: it would check out, build and upload a second wheel, so the
  artifact proved at the tag would not be the artifact uploaded. The condition
  belongs on the steps.
- **Drop the packaged-catalog check from pull requests as well.** Rejected: it
  is the only proof a source run cannot make, and a missing catalog is not a
  permanent-mistake argument — it is a pull-request argument.

## Consequences

- A pull request whose 3.12 leg fails still uploads a wheel. Its packaging check
  is green, the artifact is not a verdict, and retention is seven days. If the
  gate itself fails, the packaging job is skipped instead — `needs: quality`
  means no artifact is built for a commit the suite has already rejected.
- A broken console-script entry point fails the gate through
  `tests/test_console_scripts.py`, not through a workflow step. Adding a
  distribution check to pull requests means adding a test to the suite, which
  is the cheaper place to put it.
- A mistake that only appears in an installed distribution — a missing data
  file, a wrong extra, a script whose import needs the real install — is found
  at the tag, or at the dispatch that rehearses it. Run one dispatch of
  `publish.yml` after this lands, and after any change to the packaging steps.
- The tag path and the dispatch path are the same steps, so a rehearsal cannot
  pass while the release fails for a reason the rehearsal never ran.
- The packaging job no longer needs Qt shared libraries on a pull request; the
  gate and the 3.12 leg still install them, because the suite imports PySide6.
  The tag smoke installs them too, because importing `polyx.gui.launcher` loads
  QtWidgets and Qt wants libGL even without a display.
