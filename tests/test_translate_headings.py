"""Tests for the report-heading machine-translation bridge (M006/S05/T01).

Covers the prose path added to ``tools/translate_identifiers.py`` and the
``translate-headings`` subcommand of ``tests/translate_code_text.py``. The
fake backend proves glossary/cache precedence and batching without touching
the network; the validation cases prove a Spanish or structurally corrupted
result can never be written into a heading manifest.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """Load a module from file path, registering it before execution."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The harness loads the bridge itself, so use its instance for CLI tests.
tct = _load("translate_code_text", REPO_ROOT / "tests" / "translate_code_text.py")
ti = tct.ti


class _FakeBackend(ti.TranslationBackend):
    """Deterministic prose backend that records its batches."""

    name = "fake"
    model = "fake-1"

    def __init__(
        self,
        translations: Optional[Dict[str, str]] = None,
        *,
        fail: bool = False,
    ) -> None:
        self.translations = dict(translations or {})
        self.fail = fail
        self.calls: List[List[str]] = []

    def translate_many(self, texts):
        if self.fail:
            raise AssertionError("the backend must not be called")
        self.calls.append(list(texts))
        return [self.translations[text] for text in texts]


def _cache(tmp_path: Path) -> ti.TranslationCache:
    return ti.TranslationCache(tmp_path / "translation_cache.json")


def _plan(
    tmp_path: Path,
    queries,
    *,
    cache: Optional[ti.TranslationCache] = None,
    backend=None,
    offline: bool = False,
    glossary=None,
):
    return ti.translate_heading_queries(
        queries,
        cache or _cache(tmp_path),
        backend=backend,
        offline=offline,
        glossary=glossary,
    )


# ---------------------------------------------------------------------------
# Glossary and cache precedence.
# ---------------------------------------------------------------------------


def test_en_dictionary_hit_wins_with_zero_backend_calls(tmp_path):
    backend = _FakeBackend(fail=True)
    plan = _plan(
        tmp_path,
        ["Imagen", "Clase"],
        backend=backend,
        glossary={"Imagen": "Image", "Clase": "Class"},
    )
    assert plan.mapping == {"Imagen": "Image", "Clase": "Class"}
    assert plan.sources == {"Imagen": "en-dictionary", "Clase": "en-dictionary"}
    assert backend.calls == []
    assert plan.stats.glossary_hits == 2
    assert plan.stats.backend_calls == 0


def test_default_glossary_is_the_project_en_dictionary():
    glossary = ti._default_heading_glossary()
    assert glossary["Imagen"] == "Image"


def test_cache_hit_avoids_the_backend(tmp_path):
    cache = _cache(tmp_path)
    cache.store(ti._heading_key("Clase"), "Class", "fake", "fake-1")
    backend = _FakeBackend(fail=True)
    plan = _plan(tmp_path, ["Clase"], cache=cache, backend=backend)
    assert plan.mapping == {"Clase": "Class"}
    assert plan.sources == {"Clase": "cache"}
    assert backend.calls == []
    assert plan.stats.cache_hits == 1
    assert plan.stats.backend_calls == 0


def test_unresolved_queries_are_batched_once_and_cached(tmp_path):
    cache = _cache(tmp_path)
    backend = _FakeBackend({"Clase": "Class", "Fibras": "Fibers"})
    plan = _plan(tmp_path, ["Clase", "Fibras"], cache=cache, backend=backend)
    assert plan.mapping == {"Clase": "Class", "Fibras": "Fibers"}
    assert plan.sources == {"Clase": "mt", "Fibras": "mt"}
    assert backend.calls == [["Clase", "Fibras"]]
    assert plan.stats.backend_calls == 1
    assert plan.stats.cache_misses == 2
    for query, english in plan.mapping.items():
        entry = cache.lookup(ti._heading_key(query))
        assert entry is not None
        assert entry.translation == english
        assert entry.backend == "fake"
        assert entry.model == "fake-1"
        assert entry.timestamp
        assert entry.usable is True
    # A second run is served from the cache with no backend call.
    second = _FakeBackend(fail=True)
    again = _plan(tmp_path, ["Clase", "Fibras"], cache=cache, backend=second)
    assert again.mapping == plan.mapping
    assert again.sources == {"Clase": "cache", "Fibras": "cache"}
    assert second.calls == []


def test_duplicate_queries_are_queried_once(tmp_path):
    backend = _FakeBackend({"Clase": "Class"})
    plan = _plan(tmp_path, ["Clase", "Clase"], backend=backend)
    assert list(plan.mapping) == ["Clase"]
    assert backend.calls == [["Clase"]]


def test_offline_cache_miss_fails_loudly(tmp_path):
    backend = _FakeBackend(fail=True)
    with pytest.raises(ti.OfflineCacheMiss, match="Clase"):
        _plan(tmp_path, ["Clase"], backend=backend, offline=True)


def test_stale_unusable_cache_entry_is_requeried(tmp_path):
    cache = _cache(tmp_path)
    cache.store(ti._heading_key("Clase"), "Clase", "fake", "fake-1", usable=False)
    backend = _FakeBackend({"Clase": "Class"})
    plan = _plan(tmp_path, ["Clase"], cache=cache, backend=backend)
    assert plan.mapping == {"Clase": "Class"}
    assert plan.sources == {"Clase": "mt"}
    assert backend.calls == [["Clase"]]
    assert plan.stats.rejected_candidates == 1


# ---------------------------------------------------------------------------
# Heading validation (negative surface).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "result"),
    [
        ("Clase", "Clase"),  # unchanged
        ("Clase", "clase"),  # unchanged after casefold
        ("Clase", ""),  # empty
        ("Clase", "   "),  # whitespace only
        ("Clase", "Clase más"),  # accented Spanish letter and stopword
        ("Muestra (imagen)", "Muestra de la imagen"),  # Spanish grammar
        ("Total {n}", "Total"),  # lost placeholder
        ("Total {n}", "Total {count}"),  # changed placeholder
        ("<b>Clase</b>", "Class"),  # lost HTML tag
        ("<b>Clase</b>", "<i>Class</i>"),  # changed tag
        ("menor (µm)", "smaller (m)"),  # lost µm
        ("Clase 2", "Class"),  # lost digit run
        (" Clase ", "Class"),  # lost surrounding whitespace
    ],
)
def test_unusable_heading_is_rejected(source, result):
    assert ti._is_usable_heading(result, source) is False


def test_unusable_backend_result_raises_and_is_marked_unusable(tmp_path):
    cache = _cache(tmp_path)
    backend = _FakeBackend({"Clase": "Clase más"})
    with pytest.raises(ti.TranslationError, match="Clase"):
        _plan(tmp_path, ["Clase"], cache=cache, backend=backend)
    entry = cache.lookup(ti._heading_key("Clase"))
    assert entry is not None
    assert entry.usable is False
    # The rejected text is not silently pinned into a later offline run.
    with pytest.raises(ti.OfflineCacheMiss):
        _plan(tmp_path, ["Clase"], cache=cache, offline=True)


@pytest.mark.parametrize(
    ("source", "result"),
    [
        ("Clase", "Class"),
        ("Muestra (imagen)", "Sample (image)"),
        ("Total {n}", "Total {n} items"),
        ("<b>Clase</b>", "<b>Class</b>"),
        ("menor (µm)", "smallest (µm)"),
        ("Clase 2", "Class 2"),
        (" Clase ", " Class "),
    ],
)
def test_usable_heading_is_accepted(source, result):
    assert ti._is_usable_heading(result, source) is True


# ---------------------------------------------------------------------------
# Manifest writer.
# ---------------------------------------------------------------------------


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_writer_round_trips_and_records_provenance(tmp_path):
    plan = ti.HeadingTranslationPlan(
        mapping={"Clase": "Class", "Imagen": "Image"},
        sources={"Clase": "mt", "Imagen": "en-dictionary"},
        stats=ti.TranslationStats(glossary_hits=1, backend_calls=1),
    )
    path = ti.write_heading_manifest(
        tmp_path / "headings.py", plan, command="translate-headings --input x.json"
    )
    module = _load_module(path, "heading_manifest_round_trip")
    assert module.HEADING_TRANSLATIONS == plan.mapping
    assert module.HEADING_TRANSLATION_SOURCES == plan.sources
    text = path.read_text(encoding="utf-8")
    assert "Provenance:" in text
    assert "translate-headings --input x.json" in text
    assert "do not edit by hand" in text
    assert not list(path.parent.glob("*.tmp"))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_writes_a_loadable_manifest_from_json(tmp_path, monkeypatch, capsys):
    payload = tmp_path / "headings.json"
    payload.write_text(json.dumps({"queries": ["Imagen", "Clase"]}), encoding="utf-8")
    manifest = tmp_path / "generated.py"
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(
        tct.ti, "make_backend", lambda spec=None: _FakeBackend({"Clase": "Class"})
    )
    code = tct.main(
        [
            "translate-headings",
            "--input",
            str(payload),
            "--write-manifest",
            str(manifest),
            "--cache",
            str(cache),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Imagen -> Image" in output
    assert "Clase -> Class" in output
    assert "manifest written to" in output
    module = _load_module(manifest, "heading_manifest_cli")
    assert module.HEADING_TRANSLATIONS == {"Imagen": "Image", "Clase": "Class"}
    assert module.HEADING_TRANSLATION_SOURCES == {
        "Imagen": "en-dictionary",
        "Clase": "mt",
    }
    stored = json.loads(cache.read_text(encoding="utf-8"))
    assert "@heading:Clase" in stored["entries"]


def test_cli_accepts_a_bare_json_array(tmp_path, monkeypatch, capsys):
    payload = tmp_path / "headings.json"
    payload.write_text(json.dumps(["Clase"]), encoding="utf-8")
    monkeypatch.setattr(
        tct.ti, "make_backend", lambda spec=None: _FakeBackend({"Clase": "Class"})
    )
    assert (
        tct.main(
            [
                "translate-headings",
                "--input",
                str(payload),
                "--cache",
                str(tmp_path / "cache.json"),
            ]
        )
        == 0
    )
    assert "Clase -> Class" in capsys.readouterr().out


def test_cli_exits_1_on_malformed_input(tmp_path, capsys):
    payload = tmp_path / "headings.json"
    payload.write_text("{not json", encoding="utf-8")
    code = tct.main(
        ["translate-headings", "--input", str(payload), "--cache", str(tmp_path / "c.json")]
    )
    assert code == 1
    assert "ERROR" in capsys.readouterr().err


def test_cli_exits_1_on_unusable_translation(tmp_path, monkeypatch, capsys):
    payload = tmp_path / "headings.json"
    payload.write_text(json.dumps(["Clase"]), encoding="utf-8")
    monkeypatch.setattr(
        tct.ti, "make_backend", lambda spec=None: _FakeBackend({"Clase": "Clase más"})
    )
    code = tct.main(
        [
            "translate-headings",
            "--input",
            str(payload),
            "--cache",
            str(tmp_path / "cache.json"),
        ]
    )
    assert code == 1
    assert "unusable heading translation" in capsys.readouterr().err
