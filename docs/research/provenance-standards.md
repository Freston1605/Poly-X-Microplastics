# Provenance mechanisms for a small OSS Python project on PyPI + GitHub Releases

**Ticket:** research #154 (Wayfinder map issue #153).
**Saved at:** `docs/research/provenance-standards.md` in this worktree (branch `research/provenance-standards`).
**Question:** Against primary sources, what provenance mechanisms are available, what does each actually prove, and which SLSA build level is realistically reachable with the existing tag-triggered workflow? What does the pinned publish action do *by default*?

**Method.** Primary sources only: the official SLSA specification, the PEPs, Sigstore documentation, and the docs *and source at the exact pinned commit* of `pypa/gh-action-pypi-publish`. No blogs or secondary write-ups. Sources accessed 2026-11-17. Every factual claim carries a citation; `[S#]` labels resolve to URLs in [Sources](#sources).

> **Note on scope.** This worktree has no `.ok/` directory, so the OpenKnowledge MCP server is not registered for it (confirmed by a failed `config`/`exec` call); the file was therefore written with native tools as a plain git-tracked Markdown artifact. OK's "ingest external URLs before citing" rule cannot apply here, and the ticket explicitly requires URL citations.

---

## 1. What the existing workflow actually does

The file under study is `.github/workflows/publish.yml` (tag trigger `v*`, plus `workflow_dispatch`). [S1]

- Top-level permission: `contents: read`. [S1]
- `publish` job: `permissions: id-token: write`, with the comment *"Trusted Publishing, plus the Sigstore attestations"*; it downloads the `dist` artifact built by the `ci`/`build` jobs and runs `pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086 # v1.14.3` with exactly one input: `print-hash: true`. There is **no** `attestations:` input. [S1]
- `release` job: `permissions: contents: write`; creates the GitHub Release with `gh release create … dist/*`. [S1]
- Trusted Publishing is configured through the `pypi` environment (`environment: name: pypi`, URL `https://pypi.org/p/polyx-microplastics`). [S1] The action's docs say Trusted Publishing authenticates without an API token/username-password and is entered by granting `id-token: write` *without* an explicit username or password — which is this job's configuration. [S3]

### 1.1 Exact action inputs at the pinned commit

From `action.yml` at `7245906…` (v1.14.3): [S2]

| Input | Default in `action.yml` | Set by this workflow |
|---|---|---|
| `attestations` | **`'true'`** | *not set* → default applies |
| `print-hash` | no explicit default on the canonical key (the deprecated `print_hash` alias defaults to `'true'`) | `true` |
| `user` | `__token__` | not set |
| `password` | none | not set (Trusted Publishing) |
| `repository-url` / `repository_url` | `https://upload.pypi.org/legacy/` | not set |
| `packages-dir` / `packages_dir` | `dist` | not set |
| `verify-metadata` / `verify_metadata` | `'true'` | not set |
| `skip-existing` / `skip_existing` | `'false'` | not set |
| `verbose` | `'true'` | not set |

### 1.2 What the pinned action does by default — the comment is correct

**Finding: the comment is accurate.** At the pinned commit, `attestations` defaults to `'true'` [S2], and the action's README states that generating and uploading signed digital attestations for *all* distribution files "is now on by default for all projects using Trusted Publishing" and must be disabled with `attestations: false`. [S3] PyPI's own docs independently confirm that with the official PyPA action "attestations are generated and uploaded automatically by default, with no additional configuration necessary". [S12] Because this job uses Trusted Publishing and `id-token: write`, the action generates attestations without any explicit input. [S2][S3]

Mechanically, the action's `attestations.py` collects sdists/wheels from `dist/`, obtains the ambient GitHub OIDC token (`detect_credential()`, audience `sigstore`), signs each distribution with `pypi_attestations.Attestation.sign(signer, dist)`, and writes files named `.publish.attestation`. [S4] In the `pypi-attestations` library, `Attestation.sign` hardcodes `.predicate_type(AttestationType.PYPI_PUBLISH_V1)`, i.e. `https://docs.pypi.org/attestations/publish/v1`. [S5]

**Crucial consequence:** the attestation emitted by this workflow is a **PyPI Publish Attestation**, *not* SLSA Provenance. PyPI accepts both predicates, but this action only produces the former. [S9][S5] `print-hash: true` merely prints SHA-256/MD5/BLAKE2-256 values of the files in the CI log; it is informational, not a signed, distributed attestation. [S2][S3]

---

## 2. Mechanism catalogue — what each one actually proves

### 2.1 Trusted Publishing (OIDC) — authentication, not provenance

Trusted Publishing exchanges a short-lived OIDC token from the CI provider for an upload credential; no long-lived PyPI token exists to leak or rotate. [S3] PyPI's publish attestation is described as proving a package was "published via a short-lived credential corresponding to a specific machine identity (such as a GitHub Actions workflow)". [S10]

- **Proves:** the upload was performed by the configured Trusted Publisher identity.
- **Does not prove:** how the artifact was built, from what source, or by what build process.

### 2.2 PEP 740 / PyPI Publish Attestation — the one this repo gets today

PEP 740 defines the attestation *format*: an in-toto v1 Statement bound to one distribution by SHA-256 digest, signed with DSSE using an X.509 certificate and backed by transparency-log entries; PyPI serves these as "provenance objects" through its simple/JSON APIs. [S8] The PyPI-specific `publish/v1` predicate is deliberately minimal: its predicate body **must be empty (`{}`) or null**, and its purpose is "a minimal, 'implicit' digital attestation for PyPI packages published via Trusted Publishing". [S10]

- **Proves (per PyPI's own docs):** (1) the file was uploaded via a Trusted Publisher rather than a locally-held API token; (2) a *specific* Trusted Publisher identity was used — e.g. a particular GitHub Actions workflow. [S10] The identity lives in the Sigstore certificate: the verifier checks OIDC issuer `https://token.actions.githubusercontent.com`, source repository URI, workflow filename, and source ref/digest. [S5]
- **Does not prove:** *how* the file was built. The predicate is empty, so there is no build platform, build process, or input list. [S10]
- **Explicit index-trust limit:** PEP 740 states it "does not increase (or decrease) trust in the index itself" — a dishonest index could also modify or omit attestations. [S8]

### 2.3 Sigstore keyless signing (Fulcio + Rekor) — the cryptographic substrate

Sigstore signs with ephemeral keys: a client requests a short-lived certificate from Fulcio using an OIDC identity token (email, service account, or *workflow information* in CI), discards the private key after one signature, and persists the digest, signature and certificate in the append-only Rekor transparency log. Verification checks the signature, the certificate identity against an expected identity, the certificate chain against Sigstore's root of trust, and proof of inclusion in Rekor. [S14]

- **Proves:** the artifact was signed by a holder of the stated identity at a recorded time and has not changed since signing; signing events are publicly auditable. [S14]
- **Does not prove:** that the build process was correct or untampered. Note also GitHub's own warning that in an attestation, only `signature.certificate` and `verifiedTimestamps` are non-manipulable by the workflow; `statement.predicate` can be falsified by anyone with access to the workflow execution context. [S18]

### 2.4 GitHub artifact attestations / `actions/attest-build-provenance` — real SLSA build provenance

`actions/attest-build-provenance` (v4 is a wrapper over `actions/attest`) produces a **SLSA build provenance** predicate bound to a named artifact and digest in in-toto format, signed with a short-lived Sigstore certificate, and uploaded to the GitHub attestations API. [S15][S16] For a public repo the public-good Sigstore instance is used. [S15] Required permissions: `id-token: write`, `contents: read`, `attestations: write` (`packages: write` for container images). [S17]

- **Proves:** a hosted platform (GitHub Actions) generated and signed a statement tying the artifact digest to source repository/ref and the producing workflow. [S15][S17][S18]
- **Does not prove:** L3-style tamper resistance — GitHub's docs recommend a "trusted builder" (build + signing in an uninfluenceable reusable workflow) when stronger guarantees are needed. [S18]
- **Not present in this repo's workflow.** [S1]

### 2.5 PEP 458 / PEP 480 (TUF end-to-end signing) — proposed index-compromise defence, not available

PEP 480 extends PEP 458 with TUF "end-to-end signing" so developers sign their own distributions and a compromised PyPI cannot serve forged claimed-project files; it uses offline developer keys and `claimed`/`recently-claimed` roles. [S13] Its status is **Draft**, and its own PEP Status section says discussion was deferred and implementation of PEP 458 was still in progress. [S13] PEP 740 names PEP 458/480 as the mechanisms that would address index trust, which it explicitly does not. [S8]

- **Proves (if implemented):** protection against a compromised index, using developer-held offline keys.
- **Availability:** not an option for this project today — it is a draft proposal requiring PyPI infrastructure changes. [S13]

### 2.6 Mechanisms explicitly unavailable

PGP signature uploads were disabled on PyPI in May 2023 after most signatures proved unverifiable; wheel signatures are specified but unsupported in practice (official tooling deprecated them in 2018). [S8]

---

## 3. SLSA build levels and what is realistically reachable

The current SLSA specification is v1.2; the Build track levels are unchanged from v1.0 (the v1.0 levels page is marked *Retired*). [S6][S7]

| Level | Requirement (summary) | Source |
|---|---|---|
| L0 | No requirements. | [S6] |
| L1 | Provenance exists: what entity built, what process, what top-level inputs; may be incomplete/unsigned. | [S6] |
| L2 | L1 + hosted platform generates and signs provenance; consumer validates authenticity; prevents tampering *after* the build. | [S6] |
| L3 | L2 + hardened platform preventing runs from influencing one another and preventing signing secret from being accessible to user-defined build steps; prevents tampering *during* the build. | [S6] |

**Assessment for this repo's tag-triggered workflow:**

1. **As-is: no SLSA Build level — the emitted attestation is not SLSA provenance.** L1 requires provenance describing the build platform, process and top-level inputs; the action emits an *empty-predicate* PyPI Publish Attestation and nothing that describes a build. [S6][S10][S5] `print-hash` is only a CI-log printout. [S2] So on the SLSA Build track the project is at L0 with respect to *producing SLSA provenance*, even though it already has signed, consumer-verifiable identity attestations.

2. **Realistically reachable without changing the tag-triggered model: Build L2.** Adding one step (`actions/attest-build-provenance` v4 / `actions/attest` v4 with `subject-path: dist/*`) plus `attestations: write` would make GitHub's hosted platform generate and sign a SLSA v1 provenance statement over the exact same artifacts, verifiable by consumers with `gh attestation verify`. [S15][S17][S18] That satisfies L2 (signed provenance from a hosted platform + consumer authenticity validation). [S6] PyPI also accepts SLSA Provenance attestations, so this can be layered alongside — not instead of — the existing publish attestation. [S9]

3. **Not realistically reachable: Build L3.** L3 requires strong platform isolation: runs cannot influence one another and signing material must not be accessible to user-defined build steps. [S6] This workflow explicitly grants the OIDC signing credential to `id-token: write` in the publish job, and GitHub's own docs say predicate contents can be falsified by workflow execution context and recommend a "trusted builder" for stronger guarantees. [S1][S18] Reaching L3 would require a different, hardened builder — a substantial platform change, not a workflow tweak.

**I could not verify from a primary source** that stock GitHub-hosted runners satisfy SLSA L3 (no such certification was found); the L3 conclusion above is a reasoned inference from the L3 requirements [S6] and GitHub's trusted-builder guidance [S18], not a quoted certification.

---

## 4. Verification commands (primary-source documented)

- **PyPI / PEP 740 publish attestation** — PyPI documents: [S11]
  ```bash
  export WHEEL_DIRECT_URL=https://files.pythonhosted.org/packages/.../sampleproject-4.0.0-py3-none-any.whl
  pypi-attestations verify pypi --repository https://github.com/pypa/sampleproject "$WHEEL_DIRECT_URL"
  ```
  This downloads the file and its provenance JSON, checks the Trusted Publisher in the provenance matches `--repository`, and cryptographically verifies the file against the attestations. [S11]

- **GitHub build-provenance attestation** — GitHub documents: [S17]
  ```bash
  gh attestation verify PATH/TO/ARTIFACT -R OWNER/REPO
  ```
  Default predicate is `https://slsa.dev/provenance/v1`; tighten identity with `--signer-workflow` / `--source-ref` / `--cert-identity` for a meaningful policy. [S18]

- **Printed hashes** — no verification command exists; `print-hash` only emits digests into the Actions log. [S2][S3]

---

## Sources

- **[S1]** Local file under study: `.github/workflows/publish.yml` (branch `research/provenance-standards`); upstream: https://github.com/Freston1605/Poly-X-Microplastics/blob/main/.github/workflows/publish.yml
- **[S2]** `pypa/gh-action-pypi-publish` `action.yml` at pinned commit: https://raw.githubusercontent.com/pypa/gh-action-pypi-publish/7245906fb61fa33a13554f9d5fb856fe61758086/action.yml
- **[S3]** `pypa/gh-action-pypi-publish` `README.md` at pinned commit: https://raw.githubusercontent.com/pypa/gh-action-pypi-publish/7245906fb61fa33a13554f9d5fb856fe61758086/README.md
- **[S4]** `pypa/gh-action-pypi-publish` `attestations.py` at pinned commit: https://raw.githubusercontent.com/pypa/gh-action-pypi-publish/7245906fb61fa33a13554f9d5fb856fe61758086/attestations.py
- **[S5]** `pypi/pypi-attestations` `src/pypi_attestations/_impl.py` (`Attestation.sign` → `PYPI_PUBLISH_V1`): https://raw.githubusercontent.com/pypi/pypi-attestations/main/src/pypi_attestations/_impl.py
- **[S6]** SLSA v1.2, Build track basics: https://slsa.dev/spec/v1.2/build-track-basics
- **[S7]** SLSA v1.0, Security levels (Retired): https://slsa.dev/spec/v1.0/levels
- **[S8]** PEP 740 — Index support for digital attestations: https://peps.python.org/pep-0740/
- **[S9]** PyPI docs — Introduction to Attestations: https://docs.pypi.org/attestations/
- **[S10]** PyPI docs — PyPI Publish Attestation (v1): https://docs.pypi.org/attestations/publish/v1/
- **[S11]** PyPI docs — Consuming attestations (verification command): https://docs.pypi.org/attestations/consuming-attestations/
- **[S12]** PyPI docs — Producing attestations: https://docs.pypi.org/attestations/producing-attestations/
- **[S13]** PEP 480 — Surviving a Compromise of PyPI (Draft): https://peps.python.org/pep-0480/
- **[S14]** Sigstore docs — Overview: https://docs.sigstore.dev/about/overview/
- **[S15]** `actions/attest-build-provenance` README: https://github.com/actions/attest-build-provenance/blob/main/README.md
- **[S16]** `actions/attest-build-provenance` `action.yml`: https://raw.githubusercontent.com/actions/attest-build-provenance/main/action.yml
- **[S17]** GitHub docs — Using artifact attestations to establish provenance for builds: https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds
- **[S18]** GitHub CLI manual — `gh attestation verify`: https://cli.github.com/manual/gh_attestation_verify
