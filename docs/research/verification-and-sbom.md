---
title: What an ordinary user can verify (PEP 740, GitHub Releases, signed manifests, SHA-256) and CycloneDX vs SPDX
description: Primary-source research for Wayfinder ticket #155 — copy-pasteable verification commands and an SBOM-format comparison for polyx-microplastics.
tags: [research, verification, sbom, supply-chain, pep-740, cyclonedx, spdx]
status: provisional
---

# Verification and SBOM research (ticket #155)

**File location:** `docs/research/verification-and-sbom.md`, in worktree `/tmp/opencode/research-verification`, branch `research/verification-sbom`.

**Method.** Every claim below is cited to a primary source (official spec, official project documentation, or tool source) with a URL. No secondary write-ups are used as evidence. Where I could not confirm something from a primary source, I say so explicitly. Commands are copy-pasteable and come from the cited tool's own documentation.

**Why this matters here.** This project is a Python distribution, `polyx-microplastics`, published to PyPI from GitHub Actions with Trusted Publishing (`id-token: write`) and `pypa/gh-action-pypi-publish`, and the same built files are attached to a GitHub Release (`.github/workflows/publish.yml` in this checkout). It also consumes YOLO `.pt` weights at runtime (`SECURITY.md`). That means the PyPI (a), GitHub-Release (b), and published-hash (d) flows below map directly onto this project's real release path.

---

## (a) PyPI artifacts with PEP 740 attestations

**What the thing is.** PEP 740 defines index-hosted digital attestations: a signed [in-toto v1 Statement](https://github.com/in-toto/attestation/blob/v1.0/spec/v1.0/statement.md) whose single subject is the distribution filename bound to a **SHA-256 digest** of the file's contents, signed over DSSE with an X.509 (Fulcio) certificate and logged in a transparency log (Rekor); PyPI verifies attestations at upload and serves them through its "Integrity API" ([PEP 740, Specification and Attestation objects](https://peps.python.org/pep-0740/), [PyPI attestation intro](https://docs.pypi.org/attestations/), [PyPI blog, 14 Nov 2024](https://blog.pypi.org/posts/2024-11-14-pypi-now-supports-digital-attestations)). PyPI accepts two predicates — SLSA Provenance and PyPI Publish — from Trusted Publisher identities GitHub Actions, GitLab CI/CD and Google Cloud (ActiveState is supported as a publisher but does not currently produce attestations) ([PyPI: Supported attestations](https://docs.pypi.org/attestations/)). Attestations are produced automatically, with no extra configuration, when a project publishes from GitHub Actions via Trusted Publishing using `pypa/gh-action-pypi-publish` ([PyPI: Producing attestations](https://docs.pypi.org/attestations/producing-attestations/)) — which is exactly this project's publish workflow.

**What it proves and what it does not.** PyPI states plainly that an attestation tells you where a package came from but **not whether you should trust it**: a valid signature proves access to a signing identity, not that the code is safe or free of malicious/vulnerable code ([PyPI: Security model and considerations](https://docs.pypi.org/attestations/security-model)). PEP 740 explicitly does not require installing clients such as `pip` to verify attestations ([PEP 740, Abstract](https://peps.python.org/pep-0740/)), so verification is an explicit action by the user or their build pipeline, not something that happens on `pip install`.

**On the command in the ticket.** I could **not** find any primary source documenting a command literally named `pypi attestation verify`. The official, PyPI-documented verifier is the `pypi-attestations` CLI (PyPI warns its CLI is "intended primarily for experimentation" and not a stable interface) ([pypi/pypi-attestations README](https://github.com/pypi/pypi-attestations), [CHANGELOG](https://github.com/pypi/pypi-attestations/blob/main/CHANGELOG.md)).

**Copy-pasteable verification (PyPI's own documented flow):**

```bash
python -m pip install pypi-attestations

# Verify a wheel on PyPI against the publisher identity you expect.
# The tool downloads the file and its provenance via the Integrity API,
# checks the Trusted Publisher against --repository, and verifies the signature.
pypi-attestations verify pypi \
  --repository github.com/pypa/sampleproject \
  pypi:sampleproject-4.0.0-py3-none-any.whl

# A direct URL or an already-downloaded local path also work;
# for a local artifact the provenance JSON is supplied explicitly.
pypi-attestations verify pypi \
  --repository github.com/pypa/sampleproject \
  ./sampleproject-4.0.0-py3-none-any.whl \
  --provenance-file ./sampleproject-4.0.0.whl.provenance
```

Sources: [PyPI: Consuming attestations](https://docs.pypi.org/attestations/consuming-attestations/), [pypi-attestations README — Verifying a PyPI package](https://github.com/pypi/pypi-attestations), [CHANGELOG v0.0.22 / v0.0.23 (`pypi:` prefix, local paths, `--provenance-file`, `--offline`)](https://github.com/pypi/pypi-attestations/blob/main/CHANGELOG.md). For a downloaded `.publish.attestation` / `.slsa.attestation` file, the same CLI offers:

```bash
pypi-attestations verify attestation \
  --identity github.com/trailofbits/pypi-attestations/.github/workflows/release.yml@refs/tags/v0.0.19 \
  test/assets/pypi_attestations-0.0.19.tar.gz
```

Source: [pypi-attestations README — Verifying a PEP 740 Attestation](https://github.com/pypi/pypi-attestations). PyPI itself strongly discourages hand-implementing verification and recommends `pypi-attestations` or `sigstore-python` ([PyPI: Internals and Technical Details](https://dev.pypi.org/security/attestation-internals)).

**Raw/API path (for a script).** PyPI's Integrity API route is `GET /integrity/<project>/<version>/<filename>/provenance`, returning a PEP 740 provenance object; the same data is exposed in the HTML/JSON simple APIs and on a per-file web UI page ([PyPI: Integrity API](https://docs.pypi.org/api/integrity), [PyPI blog — How to view a file's attestations](https://blog.pypi.org/posts/2024-11-14-pypi-now-supports-digital-attestations)).

**Related GitHub flow.** For artifacts bearing GitHub Artifact Attestations, the official CLI verifier is:

```bash
gh attestation verify ./artifact.bin --repo owner/repo
```

`gh attestation verify` checks the signer identity (default predicate `https://slsa.dev/provenance/v1`) and can emit `--format json` for policy engines ([GitHub CLI manual: gh attestation verify](https://cli.github.com/manual/gh_attestation_verify)).

---

## (b) Files attached to a GitHub Release

Two verification paths exist, depending on what the release provides.

**1. Attestation-based (strongest).** GitHub's CLI has dedicated release commands:

```bash
# Verify the release itself has a valid, GitHub-signed attestation;
# prints metadata (including digests) for every asset it references.
gh release verify v1.2.3

# Verify a specific downloaded asset against the release attestation:
# checks the asset's digest matches the attestation's subject and that the
# attestation is associated with the release.
gh release verify-asset v1.2.3 ./dist/my-asset.zip
```

Sources: [GitHub CLI manual: gh release verify](https://cli.github.com/manual/gh_release_verify) ("Verify that a GitHub Release is accompanied by a valid cryptographically signed attestation… an attestation is a claim made by GitHub regarding a release and its assets") and [GitHub CLI manual: gh release verify-asset](https://cli.github.com/manual/gh_release_verify-asset) ("ensures the asset's integrity by validating that the asset's digest matches the subject in the attestation and that the attestation is associated with the release"). Both commands accept `-R OWNER/REPO`; omitting the tag uses the latest release.

**2. Digest-based (works without an attestation).** GitHub's REST API exposes a `digest` field on release-asset objects ([GitHub REST: Get a release asset / List release assets](https://docs.github.com/en/rest/releases/assets)). A consumer can fetch that digest and compare a locally computed SHA-256:

```bash
# Read the digest GitHub records for each asset of a release.
gh api repos/OWNER/REPO/releases/latest/assets --jq '.[] | [.name, .digest]'

# Independently compute and compare (GNU coreutils sha256sum).
sha256sum ./dist/my-asset.zip
```

Source for `sha256sum`: [GNU coreutils manual, §6.7 "sha2 utilities"](https://www.gnu.org/software/coreutils/manual/html_node/sha2-utilities.html) ("`sha256sum`… compute checksums… can also determine whether a file and checksum are consistent"). Caveat: a digest served by the same host as the file protects against transfer corruption, not against a compromised host that can replace both; for origin authenticity prefer the attestation-based path above (the same distinction GnuPG draws between integrity and trustworthiness, see §(c)).

---

## (c) A signed manifest

The common pattern is a checksum list (e.g. `SHA256SUMS`) published together with a detached cryptographic signature over that list.

**OpenPGP / GPG detached signature.** Verify the signature on the manifest first, then check the files against it:

```bash
# Verify the detached signature over the manifest (specify both files explicitly).
gpg --verify SHA256SUMS.asc SHA256SUMS

# Then verify the downloaded files against the now-authenticated manifest.
sha256sum --check --ignore-missing SHA256SUMS
```

Sources: [GnuPG — Operational GPG Commands (`--verify`, detached form)](https://www.gnupg.org/documentation/manuals/gnupg/Operational-GPG-Commands.html), [The GNU Privacy Handbook — Detached signatures (`gpg --verify doc.sig doc`)](https://gnupg.org/gph/en/manual/x135.html), [GnuPG FAQ 8.19 — How can I use GnuPG to verify a downloaded file?](https://www.gnupg.org/faq/gnupg-faq.html), [GNU coreutils — sha2 utilities](https://www.gnu.org/software/coreutils/manual/html_node/sha2-utilities.html).

Two constraints stated by GnuPG's own docs: you must obtain the signer's public key through a trusted source (anyone can impersonate anyone online), and a good signature does **not** mean the software is trustworthy — only that it was not tampered with ([GnuPG FAQ 8.19](https://www.gnupg.org/faq/gnupg-faq.html)). For scripts, GnuPG recommends the machine-parseable `--status-fd` interface or `gpgv` rather than the return code alone ([GnuPG gpg(1) manual](https://www.gnupg.org/documentation/manuals/gnupg26/gpg.1.html)).

**Sigstore / Cosign alternative (keyless, identity-based).** If the manifest was signed with Sigstore and a bundle was published:

```bash
cosign verify-blob SHA256SUMS \
  --bundle SHA256SUMS.sigstore.json \
  --certificate-identity=name@example.com \
  --certificate-oidc-issuer=accounts.example.com
```

The bundle holds the signature, certificate, signed timestamp and proof of Rekor inclusion; Cosign checks the timestamp against the short-lived Fulcio certificate and verifies transparency-log inclusion ([Sigstore Quickstart with Cosign](https://docs.sigstore.dev/quickstart/quickstart-cosign), [Sigstore: Verifying Signatures](https://docs.sigstore.dev/cosign/verifying/verify)). GitHub Actions identities use `--certificate-oidc-issuer=token.actions.githubusercontent.com` ([Sigstore CI Quickstart](https://docs.sigstore.dev/quickstart/quickstart-ci)).

---

## (d) Model weights with a published SHA-256

**Generic, publisher-independent check:**

```bash
sha256sum ./weights/model.pt
# Compare the output to the SHA-256 published by the model author.

# If the author publishes a SHA256SUMS file, check all files at once:
sha256sum --check --ignore-missing SHA256SUMS
```

Sources: [GNU coreutils manual, §6.7 "sha2 utilities"](https://www.gnu.org/software/coreutils/manual/html_node/sha2-utilities.html). Note the same trust caveat as §(c): a hash on the same page as the download authenticates bytes, not the publisher; bind the published hash to an authenticated channel or a signature ([GnuPG FAQ 8.19](https://www.gnupg.org/faq/gnupg-faq.html)).

**Hugging Face–hosted weights.** `huggingface_hub` can verify cached files against the checksums on the Hub:

```bash
hf cache verify meta-llama/Llama-3.2-1B-Instruct
hf cache verify meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659
```

Source: [Hugging Face — Understand caching: "Verify your cache"](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) ("✅ Verified… All checksums match"). Hugging Face's cache stores Git-LFS files under their **SHA-256** (Git-tracked files under SHA-1) ([Hugging Face Hub — Local cache](https://huggingface.co/docs/hub/en/local-cache)); ETag-based integrity checking during download is described in [Hugging Face — file_download](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download).

---

## CycloneDX vs SPDX for this project

Both are **data formats describing a software inventory**; neither is itself a signature or an attestation. If a consumer wants to verify *who produced* an SBOM, sign or attest the SBOM file using one of the mechanisms in §(a)–(c). (CycloneDX additionally defines a "CycloneDX Attestations / CDXA" capability in its own standard — [CycloneDX specification README](https://github.com/CycloneDX/specification) — but the SBOM document alone is not proof of origin.)

| | **CycloneDX** | **SPDX** |
|---|---|---|
| Standard / governance | OWASP + Ecma International TC54; published as **ECMA-424**, royalty-free patent policy; JSON Schema is the reference implementation ([spec README](https://github.com/CycloneDX/specification), [cyclonedx.org](https://cyclonedx.org/)). | Linux Foundation project; spec version 3.0.1, Community Specification License ([SPDX 3.0.1 spec](https://spdx.github.io/spdx-spec/v3.0.1/)). |
| Scope | "Full-stack BOM" beyond software: SBOM, SaaSBOM, HBOM, **ML-BOM**, CBOM, MBOM, OBOM, VDR, VEX, CycloneDX Attestations ([spec README](https://github.com/CycloneDX/specification)). | Software-centric model with Core, Software, Security, Licensing, Dataset, **AI**, Build and Lite profiles; classes such as `Sbom`, `Package`, `File`, `Relationship`, ` hash`, `Vulnerability`, `AIPackage` ([SPDX 3.0.1 spec — model](https://spdx.github.io/spdx-spec/v3.0.1/)). |
| Serialisations / file names | JSON, XML, Protobuf; IANA media types `application/vnd.cyclonedx+json` / `+xml`; conventional `bom.json` / `bom.xml`, `*.cdx.json` ([spec README](https://github.com/CycloneDX/specification)). | "Model and serializations" section of the spec ([SPDX 3.0.1](https://spdx.github.io/spdx-spec/v3.0.1/)); tag-value, JSON, YAML, RDF and XML are the SPDX serialisations. |
| What it contains | Metadata (timestamp, tools), components with names/versions/PURLs/hashes/licenses, dependency graph, services, vulnerabilities and VEX; designed for risk/vulnerability workflows ([spec README](https://github.com/CycloneDX/specification), [Capabilities](https://cyclonedx.org/capabilities)). | Creation info, packages and files with checksums, relationships, annotations, and first-class SPDX license expressions/license list — with dedicated licensing profiles ([SPDX 3.0.1 spec](https://spdx.github.io/spdx-spec/v3.0.1/)). |
| Publishers (tooling) | [CycloneDX Tool Center](https://cyclonedx.org/tool-center/) (project says 260+ compatible tools); e.g. `syft <dir> -o cyclonedx-json` ([Anchore syft README](https://github.com/anchore/syft)); GitHub "SBOM Dependency Submission Action" uploads CycloneDX via the dependency-submission API ([GitHub: Exporting an SBOM](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/exporting-a-software-bill-of-materials-for-your-repository)). | GitHub exports a repository's dependency graph as an SPDX SBOM from the UI or REST API ([GitHub: Exporting an SBOM](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/exporting-a-software-bill-of-materials-for-your-repository)); `syft <dir> -o spdx-json=./spdx.json`; Microsoft SBOM Tool → SPDX 2.2 ([syft README](https://github.com/anchore/syft), [GitHub SBOM doc](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/exporting-a-software-bill-of-materials-for-your-repository)). |
| Consumers / what a consumer can do | Feed vulnerability-management platforms and scanners (tool center), convert between formats with syft, express VEX/exploitability and ML-BOM ([CycloneDX use cases](https://cyclonedx.org/use-cases), [syft README](https://github.com/anchore/syft)). | License-compliance and inventory tooling, GitHub dependency-graph import, security/vulnerability and AI/dataset modelling in the 3.x profiles ([GitHub SBOM doc](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/exporting-a-software-bill-of-materials-for-your-repository), [SPDX 3.0.1 spec](https://spdx.github.io/spdx-spec/v3.0.1/)). |

**Recommendation for `polyx-microplastics`.** It is a Python package with strict dependency pins on a scientific/YOLO stack and externally sourced `.pt` weights. Both formats can be produced from the same source with one tool: syft explicitly outputs **both** `cyclonedx-json` and `spdx-json` and can convert between SBOM formats ([syft README](https://github.com/anchore/syft)), so this is not an either/or at generation time. Choose **CycloneDX** if the first consumer is vulnerability triage/scanning or you want VEX/ML-BOM and its larger tool ecosystem ([cyclonedx.org](https://cyclonedx.org/), [tool center](https://cyclonedx.org/tool-center/)). Choose **SPDX** if the priority is license compliance/SPDX license expressions or publishing through GitHub's native dependency-graph SBOM export ([GitHub SBOM doc](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/exporting-a-software-bill-of-materials-for-your-repository), [SPDX spec](https://spdx.github.io/spdx-spec/v3.0.1/)). In either case, if the decision question is "what can an ordinary user *verify*", the SBOM answers "what is in the release"; authenticity comes from the PEP 740 / release-attestation / signed-manifest flows in §(a)–(c).

---

## Primary sources (all URLs used above)

- [PEP 740 — Index support for digital attestations](https://peps.python.org/pep-0740/)
- [PyPI — Introduction to Attestations](https://docs.pypi.org/attestations/)
- [PyPI — Producing attestations](https://docs.pypi.org/attestations/producing-attestations/)
- [PyPI — Consuming attestations](https://docs.pypi.org/attestations/consuming-attestations/)
- [PyPI — Security Model and Considerations](https://docs.pypi.org/attestations/security-model)
- [PyPI — Integrity API](https://docs.pypi.org/api/integrity)
- [PyPI — Internals and Technical Details](https://dev.pypi.org/security/attestation-internals)
- [PyPI blog — PyPI now supports digital attestations (2024-11-14)](https://blog.pypi.org/posts/2024-11-14-pypi-now-supports-digital-attestations)
- [pypi/pypi-attestations (README + CLI examples)](https://github.com/pypi/pypi-attestations)
- [pypi-attestations CHANGELOG](https://github.com/pypi/pypi-attestations/blob/main/CHANGELOG.md)
- [GitHub CLI — gh attestation verify](https://cli.github.com/manual/gh_attestation_verify)
- [GitHub CLI — gh release verify](https://cli.github.com/manual/gh_release_verify)
- [GitHub CLI — gh release verify-asset](https://cli.github.com/manual/gh_release_verify-asset)
- [GitHub REST — release assets (`digest` field)](https://docs.github.com/en/rest/releases/assets)
- [GnuPG — Operational GPG Commands](https://www.gnupg.org/documentation/manuals/gnupg/Operational-GPG-Commands.html)
- [GnuPG — gpg(1) manual](https://www.gnupg.org/documentation/manuals/gnupg26/gpg.1.html)
- [GNU Privacy Handbook — Making and verifying signatures](https://gnupg.org/gph/en/manual/x135.html)
- [GnuPG FAQ — verifying a downloaded file](https://www.gnupg.org/faq/gnupg-faq.html)
- [Sigstore — Quickstart with Cosign](https://docs.sigstore.dev/quickstart/quickstart-cosign)
- [Sigstore — Verifying Signatures](https://docs.sigstore.dev/cosign/verifying/verify)
- [Sigstore — CI Quickstart](https://docs.sigstore.dev/quickstart/quickstart-ci)
- [GNU coreutils — sha2 utilities (`sha256sum`)](https://www.gnu.org/software/coreutils/manual/html_node/sha2-utilities.html)
- [Hugging Face — manage cache ("Verify your cache")](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [Hugging Face Hub — Local cache (SHA-256 for LFS)](https://huggingface.co/docs/hub/en/local-cache)
- [Hugging Face — file_download (ETag/integrity)](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)
- [CycloneDX specification (ECMA-424) README](https://github.com/CycloneDX/specification)
- [cyclonedx.org (capabilities, tool center, use cases)](https://cyclonedx.org/)
- [SPDX Specification 3.0.1](https://spdx.github.io/spdx-spec/v3.0.1/)
- [Anchore syft README (CycloneDX + SPDX output/conversion)](https://github.com/anchore/syft)
- [GitHub — Exporting a software bill of materials](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/exporting-a-software-bill-of-materials-for-your-repository)
