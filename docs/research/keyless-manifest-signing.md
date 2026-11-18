---
title: Keyless signing of a GitHub Actions release manifest
description: Primary-source comparison of sigstore/cosign keyless, GitHub artifact attestations, in-toto, and ephemeral keys for signing a release manifest with no code-signing budget.
status: provisional
tags:
  - research
  - security
  - supply-chain
  - signing
ticket: 156
---
# Keyless signing of a GitHub Actions release manifest

Research ticket #156 (Wayfinder map issue #153).

Status: provisional research note. Nothing here is merged into the project; this
branch is throwaway.

Saved at: `docs/research/keyless-manifest-signing.md` in the
`research/keyless-manifest` worktree (`/tmp/opencode/research-keyless`).

`## Scope and decision context

Ticket #156 (Wayfinder map issue #153). Constraint: no budget for a code-signing certificate, and the trust anchor is the build's own cryptographic identity (the GitHub Actions workflow), not a maintainer-held key. The question is how to sign a release manifest produced inside GitHub Actions, and what that forces the manifest to contain.

Every finding below is taken from first-party documentation, specifications, or CLI reference and carries its source URL. Where I could not confirm something from a primary source, I say so explicitly at the end.

## Primary sources cited

- **S1** Sigstore, "Signing Overview" (keyless model, Fulcio/Rekor, TUF root of trust) — https://docs.sigstore.dev/cosign/signing/overview/
- **S2** Sigstore, "Verifying Signatures" (keyless verify commands) — https://docs.sigstore.dev/cosign/verifying/verify/
- **S3** Sigstore, "OIDC Verification Cheat Sheet" (exact identity/issuer strings) — https://docs.sigstore.dev/quickstart/verification-cheat-sheet/
- **S4** GitHub, "Using artifact attestations to establish provenance for builds" — https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds
- **S5** `actions/attest` README (signature, in-toto, which Sigstore instance) — https://github.com/actions/attest
- **S6** GitHub CLI manual, `gh attestation verify` (identity model, flags, timestamp/certificate caveat) — https://cli.github.com/manual/gh_attestation_verify
- **S7** GitHub, "Verifying attestations offline" — https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/verify-attestations-offline
- **S8** Sigstore, "Bundle Format" (self-contained bundle, SET/RFC3161, DSSE/in-toto) — https://docs.sigstore.dev/about/bundle/
- **S9** cosign source, `cosign_verify-blob` CLI reference — https://github.com/sigstore/cosign/blob/main/doc/cosign_verify-blob.md
- **S10** Sigstore, "CI Quickstart" (GitHub Actions OIDC signing/verifying commands) — https://docs.sigstore.dev/quickstart/quickstart-ci/
- **S11** Sigstore, "Signing Blobs" and "Signing with Self-Managed Keys" — https://docs.sigstore.dev/cosign/signing/signing_with_blobs/ , https://docs.sigstore.dev/cosign/key_management/signing_with_self-managed_keys/
- **S12** GitHub, "OpenID Connect" (OIDC token/issuer) — https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
- **S13** in-toto: "What is in-toto?" and attestation spec v1 — https://in-toto.io/docs/what-is-in-toto/ , https://github.com/in-toto/attestation/tree/main/spec/v1 , https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md
- **S14** GitHub, "Immutable releases" and `gh release verify` / `gh release verify-asset` manual — https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases , https://cli.github.com/manual/gh_release_verify , https://cli.github.com/manual/gh_release_verify-asset , https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity

## Comparison at a glance

`| Option | Identity bound into the signature | What the verifier must have | Offline |
|---|---|---|---|
| cosign keyless | Fulcio short-lived cert binds an ephemeral key to the GitHub Actions OIDC identity; Rekor log entry | cosign binary, the bundle, the long identity URL + issuer, trust material via TUF | Needs pre-staged trusted root + bundle; no documented `--offline` flag |
| GitHub artifact attestations | X.509 cert fields SourceRepository / SourceRepositoryOwner / SAN populated from GitHub OIDC token | only `gh` + owner/repo (optionally signer workflow) | Supported via downloaded bundle + trusted root; revocation/rename caveats |
| in-toto | none by itself (it is a format; authentication lives in the Envelope layer) | whatever the chosen signer requires | format is plain JSON; authenticity depends on signer |
| Ephemeral/self-managed key | none — verification is against a supplied public key | cosign + the public key out-of-band | easiest technically, but trust bootstrap is unsolved |
| Immutable release attestation | GitHub-issued claim about the release (tag, commit SHA, assets) | only `gh` + tag/repo | no offline path found in primary docs |

## A. Sigstore / cosign keyless (public-good instance)

**How it works / identity bound.** Keyless signing "associates identities, rather than keys, with an artifact signature. Fulcio issues short-lived certificates binding an ephemeral key to an OpenID Connect identity. Signing events are logged in Rekor, a signature transparency log." At signing time an in-memory keypair is created, the identity token is retrieved, Fulcio issues a certificate binding the identity to the public key, and the private key is destroyed shortly after; Rekor witnesses a timestamped entry (S1). The root of trust — Fulcio's root CA certificate and Rekor's public key — is distributed by TUF (S1).

For GitHub Actions specifically, the two strings a verifier must know are (S3, S10):
- OIDC issuer: `https://token.actions.githubusercontent.com`
- certificate identity (SAN): `https://github.com/OWNER/REPO/.github/workflows/WORKFLOW@refs/heads/BRANCH`

A real signed bundle in the Sigstore docs shows the tag form of the same identity: `https://github.com/sigstore/sigstore-java/.github/workflows/release-sigstore-java-from-tag.yaml@refs/tags/v1.0.0`, with the issuer extension `https://token.actions.githubusercontent.com` (S8). Note the distinction: interactive GitHub login uses issuer `https://github.com/login/oauth`, whereas GitHub **Actions** OIDC tokens use `https://token.actions.githubusercontent.com` (S3, S12). The cheat sheet also says that if you are unsure of the values, search the project's README, documentation, or website (S3) — so the project must publish these strings.

**Signing inside GitHub Actions** (exact workflow fragments from S10):

`yaml
permissions:
  id-token: write   # required to retrieve the OIDC token

steps:
  - uses: sigstore/cosign-installer@v4.0.0
  - name: Sign Blob
    run: cosign sign-blob to_be_signed.txt --bundle artifact.sigstore.json --yes
`

Sigstore also offers `sigstore/gh-action-sigstore-python@v3.2.0` as the "easiest way to generate Sigstore signatures within your CI system"; it uses `id-token: write` and returns the ephemeral certificate and the transparency-log index (S10). `cosign sign-blob` with `--bundle` is the recommended form for blobs; the bundle holds the signature, certificate and proof of transparency-log inclusion (S11).

**Verifying — exactly what a user installs/runs.** Install cosign ("To verify a signed artifact or blob, first install Cosign", S2; installation page: https://docs.sigstore.dev/cosign/system_config/installation/). Then (S2, S9, S10):

`bash
cosign verify-blob <blob> --bundle artifact.sigstore.json   --certificate-identity https://github.com/OWNER/REPO/.github/workflows/WORKFLOW@refs/heads/BRANCH   --certificate-oidc-issuer https://token.actions.githubusercontent.com
`

The `cosign verify-blob` reference confirms `--bundle` is required, that keyless flows require `--certificate-identity` (or `--certificate-identity-regexp`) **and** `--certificate-oidc-issuer` (or regexp), and that "A trusted root verifies service-provided material in the bundle (certificates, transparency log entries, and RFC 3161 timestamps). Cosign fetches it from TUF, unless you provide one via `--trusted-root`" (S9). It also offers GitHub-specific shortcuts such as `--certificate-github-workflow-repository`, `--certificate-github-workflow-ref` and `--certificate-github-workflow-sha` (S9).

**Network access.** A cold verifier must reach TUF to obtain the Fulcio root/Rekor key (S1, S9), and transparency-log inclusion is checked against Rekor unless the bundle's signed entry timestamp (SET) is used (S1, S8, S9). Because the Fulcio certificate is short-lived, a bundle "must include at least one transparency log's signed entry timestamp or an RFC3161 timestamp to provide proof that signing occurred during the certificate's validity window" (S8).

**What fails offline.** TUF retrieval fails with an empty cache, so verification cannot establish trust material; the current `verify-blob` reference lists no `--offline` flag — the offline-capable path is to supply a previously downloaded trusted root with `--trusted-root` plus a self-contained bundle (S9). Revocation/freshness cannot be checked offline. `--insecure-ignore-tlog` disables log verification, but the CLI warns "Artifacts cannot be publicly verified when not included in a log" (S9). Signatures are also publicly auditable by design (S1), so do not put secrets in a cosign-signed manifest.

## B. GitHub artifact attestations (`actions/attest`)

**Identity bound.** An attestation binds a subject (named artifact + digest) to a predicate using the in-toto format, and "a verifiable signature is generated for the attestation using a short-lived Sigstore-issued signing certificate" (S5). Public repositories use the public-good Sigstore instance; private/internal repositories use GitHub's private Sigstore instance (S5). The verifier manual states that the certificate's `SourceRepository`, `SourceRepositoryOwner` and `SubjectAlternativeName` fields are populated directly from the GitHub OIDC token, and that the default expected issuer is `https://token.actions.githubusercontent.com` (S6, S12). Identity is enforced with `--owner` or `--repo` (required), optionally tightened with `--signer-workflow`, `--signer-repo`, `--cert-identity`, `--source-ref` or `--source-digest` (S6).

**Signing inside GitHub Actions** (S4, S5):

`yaml
permissions:
  id-token: write
  contents: read
  attestations: write

steps:
  - uses: actions/attest@v4
    with:
      subject-path: 'PATH/TO/MANIFEST.json'
`

To put the manifest itself under the signature, pass it as `subject-path` (or `subject-digest` + `subject-name`). `actions/attest@v4` also supports a custom predicate via `predicate-type` + `predicate`/`predicate-path`, and saves attestations as JSON-serialized Sigstore bundles (S5). By default it generates SLSA build provenance (S4, S5).

**Verifying — what a user installs/runs.** Only the GitHub CLI (S4):

`bash
gh attestation verify PATH/TO/MANIFEST.json -R OWNER/REPO
# tighten to the exact workflow that signed it:
gh attestation verify PATH/TO/MANIFEST.json -R OWNER/REPO   --signer-workflow OWNER/REPO/.github/workflows/release.yml
`

By default this fetches attestations from the GitHub API using the owner/repo values, enforces predicate type `https://slsa.dev/provenance/v1` (use `--predicate-type` for others), and `gh` must already be authenticated for OCI registries when using `oci://` (S6). A custom predicate needs an explicit `--predicate-type` (S4).

**Offline.** GitHub documents a three-step offline path (S7):

`bash
# online machine:
gh attestation download PATH/TO/ARTIFACT -R OWNER/REPO
gh attestation trusted-root > trusted_root.jsonl
# offline machine (import gh CLI, artifact, bundle and trusted root):
gh attestation verify PATH/TO/ARTIFACT -R OWNER/REPO   --bundle sha256:ae57936def59bc4c75edd3a837d89bcefc6d3a5e31d55a6fa7a71624f92c3c3b.jsonl   --custom-trusted-root trusted_root.jsonl
`

Caveats from S7: the trusted root "does not have a built-in expiration date"; anything signed after you generate it verifies until that Sigstore instance rotates keys (a few times a year), and "You will not know if key material has been revoked since you last generated the trusted root file." Offline, `-R` is compared only with names in the signing certificate, so it cannot detect a repo/owner rename; compare `sourceRepositoryIdentifier` against `gh api repos/OWNER/REPO --jq .id` (S7).

**Critical constraint on manifest contents.** `gh attestation verify --format json` exposes `signature.certificate` and `verifiedTimestamps`, and the manual states: "IMPORTANT: ... only the `signature.certificate` and the `verifiedTimestamps` properties contain values that cannot be manipulated by the workflow that originated the attestation" and "special care must be taken when examining the contents of the `statement.predicate` property: should an attacker gain access to your workflow's execution context, they could then falsify the contents of the `statement.predicate`." The recommended mitigation is a "trusted builder": make the build and attestation signing happen in a reusable workflow whose execution cannot be influenced by caller input, and pin it with `--signer-workflow` (S6).

## C. in-toto — a format, not a trust mechanism

in-toto is "a framework to secure the integrity of software supply chains" that makes transparent "what steps were performed, by whom and in what order" (S13). The attestation spec (v1.2) defines four layers: **Envelope** (authentication and serialization), **Statement** (binds the attestation to a subject and identifies the predicate type), **Predicate** and **Bundle** (S13). A Statement is simply JSON with `_type` = `https://in-toto.io/Statement/v1`, a `subject` array of name+digest, `predicateType`, and optional `predicate` (S13).

**Identity bound:** none by in-toto itself — authentication is delegated to the Envelope/signature layer (S13). In a Sigstore bundle the DSSE envelope has `payloadType: application/vnd.in-toto+json` and an in-toto statement as its payload (S8); GitHub artifact attestations use in-toto with an SLSA provenance predicate (S4, S5).

**Verification / offline:** there is no in-toto-specific tool or identity string to install or know; you verify with the signer above. The statement is plain JSON and readable offline, but its authenticity still follows the signer's offline path. Conclusion: use in-toto as the *payload shape*, not as the trust anchor.

## D. Ephemeral / self-managed key pair (the naive "ephemeral key")

`cosign generate-key-pair` writes a password-protected `cosign.key` / `cosign.pub` (RSA, ECDSA or ED25519), and `cosign sign-blob --key cosign.key --bundle bundle.sigstore.json README.md` signs a file; `--yes` is for CI (S11). Verification is `cosign verify-blob --bundle bundle.sigstore.json --key cosign.pub <blob>` (S9).

**Identity bound:** none. Sigstore's own contrast is that keyless "associates identities, rather than keys, with an artifact signature" (S1), while key verification succeeds against "the specified public key" (S2); the docs explicitly say "While keyless signing is recommended, you may specify your own keys for signing" (S11). So a throwaway key pair generated in the job does not satisfy "the build's own cryptographic identity": the verifier must obtain and trust the public key out of band, and there is no OIDC identity string to check. This trust-bootstrap gap is my inference from the cited contrast, not a verbatim claim in the docs. Technically verification is the *most* offline-friendly (no TUF fetch needed once the public key is in hand), but it defeats the stated requirement.

## E. Better suited for a release manifest: immutable releases / release attestations

GitHub states that "creating an immutable release automatically generates a **release attestation**, which is a cryptographically verifiable record of a release containing the release tag, commit SHA, and release assets", and that immutable releases lock the Git tag and prevent modification/deletion of assets, including protection against repository-resurrection tag reuse (S14). The `gh` manual describes an attestation as "a claim made by GitHub regarding a release and its assets" (S14) — i.e. the trust anchor is GitHub plus the release/repository, not a maintainer key.

User verification (S14):

`bash
gh release verify v1.2.3 -R OWNER/REPO
gh release verify-asset v1.2.3 ./dist/manifest.json -R OWNER/REPO
`

Only `gh` is needed. The `gh release verify` / `verify-asset` manual lists no `--bundle` or `--custom-trusted-root` flag, so **I found no documented offline path** for release attestations; the commands fetch the attestation from GitHub. `gh release verify-asset` also cannot verify a release's auto-generated source zip/tarball (S14). This is the lowest-friction option *if* the manifest is published as an asset on an immutable GitHub Release.

## Recommendation and consequences for the manifest

1. Best fit for "the build's own cryptographic identity" with the lightest user burden is a **GitHub artifact attestation** (`actions/attest@v4`) with the manifest as the `subject`; verify with `gh attestation verify ... -R OWNER/REPO` and pin `--signer-workflow` (S4, S5, S6). If the manifest ships as a release asset, an **immutable release attestation** is even simpler for consumers (`gh release verify-asset`) but has no documented offline mode (S14).
2. Choose **cosign keyless** if consumers are not GitHub-centric or need a portable bundle; the price is that each verifier must install cosign and know the exact identity URL + issuer, and must stage a trusted root for offline use (S1, S2, S3, S9).
3. Use **in-toto** as the statement/predicate format underneath either choice; it is not a signing or trust mechanism (S13).
4. Reject the **ephemeral/self-managed key**: it binds no identity and just moves the unsolvable trust bootstrap to the public key (S1, S2, S11).

Constraints this places on *what the release manifest contains*:
- Anything whose truth must be guaranteed by the builder belongs in a field the signature actually protects: make the manifest the attestation **subject** (its digest is signed), or generate it in a trusted-builder reusable workflow — do not rely on `statement.predicate`, which the originating workflow can falsify (S6, S13).
- Publish the verifier-facing identity string and issuer alongside the release (cheat sheet explicitly tells verifiers to look in a project's README/docs) (S3).
- Treat the manifest as public: Sigstore signing events are written to a public transparency log (S1) and GitHub attestations to the attestations API (S5). No secrets.
- If offline verification is a requirement, the release must also publish the bundle and a trusted root, and accept that revocation and repo renames cannot be detected (S7, S9).

## Verification status / not confirmed from primary sources

- I found **no `--offline` flag** in the current `cosign verify-blob` reference; the documented offline-capable inputs are a self-contained bundle plus `--trusted-root` (S9), and older release text says the protobuf bundle "supports offline verification" (S2). I did not find a single canonical "cosign offline verify" command.
- I found **no documented offline path** for `gh release verify` / `gh release verify-asset` (S14).
- The `@refs/heads/BRANCH` identity form is from S3/S10; the `@refs/tags/TAG` form is confirmed by the real bundle in S8.
- GitHub notes that as of 15 July 2026 new repositories use an immutable default OIDC `sub` format containing owner/repository IDs (S12); this is the JWT `sub` claim, distinct from the Fulcio SAN workflow-ref URL used for cosign verification (S3, S8).
- Saved at: `docs/research/keyless-manifest-signing.md` in the `research/keyless-manifest` worktree (`/tmp/opencode/research-keyless`). This branch is throwaway and will never be merged.

