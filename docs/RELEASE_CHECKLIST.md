# Release checklist

The steps that make a MICRA [release](../GLOSSARY.md#release) repeatable, in the
order they must happen. Most of this list is ordinary work that can be redone;
the steps marked **irreversible** cannot, and they come after everything that
can still change the artifacts.

The first [publication](../GLOSSARY.md#publication) is the pre-release
`2.1.0a1`: `pip` skips it unless `--pre` is given, so it proves Trusted
Publishing without spending a stable version number. No stable version is
published before the cluster smoke passes and the paper's artifact is archived
(ADR-0011, decisions 5 and 6).

## Before the tag

Repeatable, and cheap to redo. A failure here costs nothing.

1. **Bump the version.** Edit `__version__` in `src/polyx/__init__.py` — the
   single source Hatchling reads — to `2.1.0a1`. Run `uv sync --locked --extra
   gui`; the lockfile does not carry the project version, so no lock change is
   expected. `publish.yml` compares the wheel's version with the tag and fails
   the release when they differ.
2. **Write the changelog entry.** In `CHANGELOG.md`, move the items under
   `[Unreleased]` into a dated `## [2.1.0a1] - YYYY-MM-DD` section. The entry
   describes what ships, not the steps that shipped it.
3. **Run the gate on the commit you will tag.** Locally,
   `uv run --locked --extra gui python tools/check.py` must finish with zero
   failures and zero skips. Push to `main` and wait for CI to pass on that exact
   commit; the publish workflow re-runs CI from the tag, and a red commit would
   spend a version number on a failed release.
4. **Rehearse the publish path.** Run the **Publish** workflow manually
   (`workflow_dispatch`) against `main`. It runs the full CI workflow — gate,
   suite, wheel and sdist build, installed-wheel smoke — and uploads nothing to
   PyPI. This is the free rehearsal of everything below.

## Irreversible steps

From here on, a mistake is fixed by the next version, never by editing this one
([`GLOSSARY.md`](../GLOSSARY.md#release)).

5. **Tag on `main`.** **Irreversible.** The tag is what names the release:
   `git tag -a v2.1.0a1 -m "MICRA 2.1.0a1"`, then `git push origin v2.1.0a1`.
   `publish.yml` rejects a tag whose commit `main` does not carry. Never move or
   re-push a release tag.
6. **Publish the pre-release to PyPI.** **Irreversible:** a version number on
   PyPI is permanent and can only be yanked, not reused. The tag push makes
   `publish.yml` build from the proven `dist/` artifact and upload
   `2.1.0a1`. Check the rehearsal produced an installable pre-release:
   `pip install --pre micra==2.1.0a1`. A defect gets `2.1.0a2`, not a re-cut.
7. **Upload the weights with `SHA256SUMS`.** **Irreversible once the digest is
   published.** Hash the artifact you are about to upload, publish that hash,
   and only then move the file:
   `sha256sum bestdetectormedium.pt > SHA256SUMS`, verify with
   `sha256sum -c SHA256SUMS`, attach both to the GitHub release
   (`gh release upload v2.1.0a1 bestdetectormedium.pt SHA256SUMS`). The digest in
   `SHA256SUMS` is the one every `micra download-models --sha256` call must use;
   a digest computed from the uploaded copy proves nothing. Update
   `weights.sha256` in the HPC configuration with the same digest.
8. **Archive on Zenodo.** **Irreversible:** a DOI is permanent. Publish the
   versioned record with the model card (dataset provenance, protocol,
   metrics), the weights and the same `SHA256SUMS`. Code stays MIT; weights are
   CC BY 4.0. The record's version DOI is the citation for this release.
9. **Create `CITATION.cff` with the DOI.** Add `CITATION.cff` naming both
   authors, the laboratory, the repository and the DOI from step 8, then commit
   it to `main`. It is metadata for future citations; the release is not re-cut
   to carry it.
10. **Verify the attestation.** The release must ship a Sigstore attestation for
   the wheel and the sdist. Verify the file PyPI actually served, not your local
   build:
    `gh attestation verify dist/micra-2.1.0a1-py3-none-any.whl -R Freston1605/micra`.
    A missing or unverifiable attestation is a failed release, not a warning.

## Why this order

- The version, the changelog and a green CI are all reversible; the tag is the
  first thing that is not, so it comes after them.
- The PyPI publication precedes the weights because the release page that holds
  the weight assets is created by that workflow.
- Zenodo follows the weight upload so the record and the GitHub release carry
  the same `SHA256SUMS`; `CITATION.cff` follows Zenodo because the DOI does not
  exist before it.
- Verification is last: it checks the bytes a user downloads, not the ones the
  maintainer built.

## After the release

- Install from the published channel, and copy the digest a collaborator would
  use into the release notes.
- No component updates itself. Collaborators re-run `uv sync --locked --extra
  gui` or pull a new image digest ([ADR-0011](adr/0011-linux-first-distribution-scope.md)).
