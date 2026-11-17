"""The model alias is a folder key; the display name is interface prose.

One name used to do both jobs, so the same weights landed in ``run/Modelo 1/``
under Spanish and ``run/Model 1/`` under English. These tests pin the two
values apart: the resolver owns alias safety and uniqueness, and the display
name is resolved from ``(alias, alias_generated)`` without ever translating
the alias itself.
"""

from __future__ import annotations

from pathlib import Path

from polyx.core.detection_run import (
    ModelIdentity,
    display_name,
    generated_alias,
    resolve_identity,
    sanitize_alias,
)


def _resolved(typed: str = "", weights: Path | None = None, index: int = 0, taken=None):
    return resolve_identity(
        typed=typed,
        weights=weights,
        index=index,
        taken=set() if taken is None else taken,
    )


# ── The two values ──────────────────────────────────────────────────────


def test_generated_alias_is_a_stable_folder_key(selected_language):
    """The fallback is invariant; only its display label follows the language."""
    identity = _resolved()

    assert identity == ModelIdentity(alias="model_1", alias_generated=True)
    assert generated_alias(0) == "model_1"
    assert generated_alias(2) == "model_3"

    selected_language("es")
    assert identity.alias == "model_1"
    assert display_name(identity.alias, identity.alias_generated, 0) == "Modelo 1"

    selected_language("en")
    assert identity.alias == "model_1"
    assert display_name(identity.alias, identity.alias_generated, 0) == "Model 1"


def test_typed_alias_is_kept_verbatim_and_never_translated(selected_language):
    identity = _resolved(typed="my-model")

    assert identity == ModelIdentity(alias="my-model", alias_generated=False)
    selected_language("es")
    assert display_name(identity.alias, identity.alias_generated, 0) == "my-model"
    selected_language("en")
    assert display_name(identity.alias, identity.alias_generated, 0) == "my-model"


def test_an_old_translated_alias_is_not_re_translated(selected_language):
    """A run written before #124 keeps the folder name it recorded."""
    selected_language("en")
    assert display_name("Modelo 1", False, 0) == "Modelo 1"


def test_weights_stem_names_the_folder_without_becoming_a_generated_label():
    identity = _resolved(weights=Path("weights/polyx_pp.pt"), index=1)

    assert identity == ModelIdentity(alias="polyx_pp", alias_generated=False)
    assert display_name(identity.alias, identity.alias_generated, 0) == "polyx_pp"


# ── Uniqueness ──────────────────────────────────────────────────────────


def test_duplicate_stems_disambiguate_instead_of_colliding():
    taken: set[str] = set()
    first = resolve_identity(typed="", weights=Path("a/polyx_pp.pt"), index=0, taken=taken)
    taken.add(first.alias)
    second = resolve_identity(typed="", weights=Path("b/polyx_pp.pt"), index=1, taken=taken)
    taken.add(second.alias)
    third = resolve_identity(typed="", weights=Path("c/polyx_pp.pt"), index=2, taken=taken)

    assert (first.alias, second.alias, third.alias) == ("polyx_pp", "polyx_pp_2", "polyx_pp_3")


def test_uniqueness_is_case_insensitive_for_windows_folders():
    """``Run`` and ``run`` are one folder on Windows, so they must not collide."""
    identity = _resolved(typed="run", taken={"Run"})

    assert identity.alias == "run_2"


# ── Safety, one owner ───────────────────────────────────────────────────


def test_path_separators_and_reserved_characters_are_replaced():
    assert sanitize_alias("../x") == ".._x"
    assert sanitize_alias("a/b") == "a_b"
    assert sanitize_alias("a\\b") == "a_b"
    assert sanitize_alias("x<y>") == "x_y_"
    assert sanitize_alias('a:b"c|d?e*f') == "a_b_c_d_e_f"


def test_windows_device_names_are_reserved():
    assert sanitize_alias("CON") == "CON_"
    assert sanitize_alias("nul") == "nul_"
    assert sanitize_alias("COM1") == "COM1_"
    assert sanitize_alias("lpt9") == "lpt9_"


def test_dot_names_and_empty_names_fall_back_to_the_generated_alias():
    assert sanitize_alias(".") == ""
    assert sanitize_alias("..") == ""
    assert sanitize_alias("   ") == ""
    assert _resolved(typed=".", index=1).alias == "model_2"
    assert _resolved(typed="..", index=1).alias == "model_2"


def test_nul_is_replaced_like_any_other_invalid_character():
    assert sanitize_alias("\x00") == "_"
    assert "\x00" not in sanitize_alias("a\x00b")


def test_surrounding_whitespace_trailing_dots_and_unicode_are_normalised():
    assert sanitize_alias("  best model  ") == "best model"
    assert sanitize_alias("name. ") == "name"
    assert sanitize_alias("e\u0301") == "\u00e9"


# ── ModelSpec delegates; the run record persists the two values ─────────


def test_model_spec_folder_name_uses_the_same_resolver():
    from polyx.core.detection_run import ModelSpec

    assert ModelSpec(detector=object(), alias="my-model").folder_name(0) == "my-model"
    assert ModelSpec(detector=object(), alias="../x").folder_name(0) == ".._x"
    assert ModelSpec(detector=object(), path=Path("w/polyx_pp.pt")).folder_name(1) == "polyx_pp"


def test_run_json_round_trips_alias_and_alias_generated(tmp_path):
    import json

    from polyx.core.detection_run import ModelInfo, RunResult
    from polyx.core.detection_run.schema import build_run_payload, load_run

    run = RunResult(
        run_dir=tmp_path,
        models=[
            ModelInfo(alias="model_1", alias_generated=True),
            ModelInfo(alias="my-model", alias_generated=False),
        ],
    )
    payload = build_run_payload(run)
    assert [entry["alias_generated"] for entry in payload["models"]] == [True, False]
    (tmp_path / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_run(tmp_path)

    assert [(model.alias, model.alias_generated) for model in loaded.models] == [
        ("model_1", True),
        ("my-model", False),
    ]


def test_old_run_json_defaults_alias_generated_to_false(tmp_path):
    """A folder written before #124 keeps its alias and is shown verbatim."""
    import json

    from polyx.core.detection_run import ModelInfo, RunResult
    from polyx.core.detection_run.schema import build_run_payload, load_run

    payload = build_run_payload(RunResult(run_dir=tmp_path, models=[ModelInfo(alias="Modelo 1")]))
    del payload["models"][0]["alias_generated"]
    (tmp_path / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_run(tmp_path)

    assert loaded.models[0].alias == "Modelo 1"
    assert loaded.models[0].alias_generated is False
    assert display_name(loaded.models[0].alias, loaded.models[0].alias_generated, 0) == "Modelo 1"
