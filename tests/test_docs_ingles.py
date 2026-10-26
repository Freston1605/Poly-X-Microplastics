"""Contract test: developer-facing documentation reads English (M006 S01).

Encodes the M006 language convention as a pytest gate instead of review
discipline: every developer-facing markdown document must contain no Spanish
prose outside code fences and inline code spans, and every relative link in
those documents must resolve on disk.

Spanish is allowed to survive only inside:
- fenced code blocks (```` ``` ... ``` ````) and inline code spans
  (`` `...` ``) — e.g. quoted runtime UI strings kept Spanish by the tr()
  layer, and real filenames/identifiers such as ``iniciar_polyx.bat``;
- a small explicit allowlist of proper nouns (author and institution names
  in the licence/contact and publications sections).

Anti-vacuity guards (P028): a synthetic Spanish sentence must be flagged by
the same scanner used on the real docs, identifier-like tokens with
underscores must NOT be flagged (word boundaries do not fire inside them),
and every scoped file must exist and be non-empty so the scan can never
silently pass over an empty set.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Developer-facing documents in scope (paths relative to the repo root).
DOC_PATHS = [
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "LEEME_macOS.md",
    "docs/diagnostico_sigterm.md",
    "docs/recetas_jq.md",
    "tests/fixtures/baseline_informe/README.md",
]

# Curated high-signal Spanish words, word-boundary anchored. Deliberately
# narrow: only words that cannot appear in legitimate technical English
# prose. Anything ambiguous is left to the accented-letter heuristic below.
_SPANISH_WORDS = re.compile(
    r"\b(?:"
    r"detecci\w*|instalaci\w*|configuraci\w*|caracterizaci\w*"
    r"|m[óo]dulos?|archivos?|part[íi]culas?|pol[íi]meros?|informes?"
    r"|im[áa]genes|im[áa]gen|caracter[íi]sticas|etiquetad\w*|capturas?"
    r"|tama[ñn]os?|ventanas?|directorios?|pruebas|c[óo]digos?"
    r"|comentarios|resultados|ejecutar|entrenamientos?|biling[üu]es?"
    r"|espa[ñn]ol|p[áa]ginas?"
    r")\b",
    re.IGNORECASE,
)

# Accented-letter heuristic: any alphabetic word containing one of the
# Spanish accented characters is treated as a Spanish remnant unless it is
# in the allowlist (proper nouns are matched case-insensitively).
_ACCENTED_WORD = re.compile(r"[A-Za-zÀ-ÿ]*[áéíóúñüÁÉÍÓÚÑÜ][A-Za-zÀ-ÿ]*")

# Author and institution names that legitimately carry Spanish accents in
# the licence/contact and publications sections of English documents.
_ACCENT_ALLOWLIST = {
    "pérez",
    "sepúlveda",
    "católica",
    "valparaíso",
}


def _strip_code(text: str) -> str:
    """Remove fenced code blocks first, then inline code spans."""
    # Normalize to NFC so allowlist comparisons are robust to decomposed
    # accents (e.g. 'P' + U+0301 vs the composed 'P\u00e9').
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", " ", text)
    return text


def _spanish_findings(prose: str) -> list[str]:
    """Return the Spanish marker hits found in prose (empty means clean)."""
    findings: list[str] = []
    for match in _SPANISH_WORDS.finditer(prose):
        findings.append(f"spanish-word:{match.group(0).lower()}")  # noqa: PERF401
    for match in _ACCENTED_WORD.finditer(prose):
        word = match.group(0)
        if word.lower() not in _ACCENT_ALLOWLIST:
            findings.append(f"accented:{word}")
    return findings


# Regex over markdown links: [text](target) — target captured without the
# optional trailing title. Also matches image links (![alt](target)).
_MD_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _relative_link_targets(text: str) -> list[str]:
    """Markdown link targets that must resolve to a file on disk.

    Skips http(s) URLs, mailto: links, pure anchors and empty targets;
    strips the #fragment from the rest.
    """
    targets: list[str] = []
    for raw in _MD_LINK.findall(text):
        if not raw or raw.startswith(("#", "http://", "https://", "mailto:")):
            continue
        targets.append(raw.split("#", 1)[0])
    return targets


def test_scoped_docs_exist_and_are_nonempty() -> None:
    """Structure guard: the scanner must never pass over an empty file set."""
    assert len(DOC_PATHS) >= 1
    for rel in DOC_PATHS:
        path = REPO_ROOT / rel
        assert path.is_file(), f"scoped doc missing: {rel}"
        assert path.stat().st_size > 0, f"scoped doc is empty: {rel}"


@pytest.mark.parametrize("rel", DOC_PATHS)
def test_doc_prose_has_no_spanish(rel: str) -> None:
    """Spanish-absence scan over the prose of each scoped document."""
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    findings = _spanish_findings(_strip_code(text))
    assert not findings, f"Spanish remnants in {rel}: {findings}"


@pytest.mark.parametrize("rel", DOC_PATHS)
def test_doc_relative_links_resolve(rel: str) -> None:
    """Every non-external markdown link target must exist on disk."""
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    for target in _relative_link_targets(text):
        resolved = (REPO_ROOT / rel).parent / target
        assert resolved.exists(), f"broken link in {rel}: {target}"


def test_readme_en_twin_is_gone() -> None:
    """D019: README.md is the single README; no twin, no references."""
    assert not (REPO_ROOT / "README.en.md").exists()
    for rel in DOC_PATHS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "README.en" not in text, f"{rel} still references README.en"


def test_scanner_flags_synthetic_spanish_sentence() -> None:
    """Anti-vacuity: a deliberately Spanish sentence must be detected."""
    synthetic = (
        "La detección de partículas genera imágenes del informe en el directorio de resultados."
    )
    findings = _spanish_findings(synthetic)
    assert findings, "scanner failed to flag a synthetic Spanish sentence"
    assert any(f.startswith("accented:detecci") for f in findings)
    assert any(f.startswith("spanish-word:") for f in findings)


def test_scanner_ignores_spanish_inside_identifier_tokens() -> None:
    """Negative test: underscore-bearing filenames must not be flagged.

    ``\\b`` word boundaries do not fire inside a single word-character run,
    so ``empaquetar_para_llevar.bat`` contains no ``para`` hit and
    ``Lanzar_macOS.command`` contains no ``macOS`` hit.
    """
    identifiers = (
        "Run LEEME.txt with iniciar_polyx.bat, empaquetar_para_llevar.bat, "
        "Lanzar_macOS.command, actualizar.bat and crear_icono_macOS.command."
    )
    assert _spanish_findings(identifiers) == []
