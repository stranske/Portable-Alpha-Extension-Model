"""Focused exact-output tests for run diff formatting."""

from __future__ import annotations

from pa_core.llm.compare_runs import format_config_diff


def test_format_config_diff_no_differences_exact_string() -> None:
    manifest = {
        "seed": 7,
        "cli_args": {"capital": 750000, "distribution": "student_t"},
        "wizard_config": {"risk": {"cvar_limit": 0.09}},
    }

    text = format_config_diff(manifest, manifest)

    assert text == "No config differences detected."


def test_format_config_diff_simple_one_line_difference_exact_string() -> None:
    current_manifest = {"seed": 2}
    prior_manifest = {"seed": 1}

    text = format_config_diff(current_manifest, prior_manifest)

    assert text == "- seed: 1 -> 2"


def test_format_config_diff_multiline_text_difference_exact_string() -> None:
    current_manifest = {"wizard": {"note": "line one\nline three"}}
    prior_manifest = {"wizard": {"note": "line one\nline two"}}

    text = format_config_diff(current_manifest, prior_manifest)

    assert text == "- wizard.note: line one\nline two -> line one\nline three"


def test_format_config_diff_unicode_difference_exact_string() -> None:
    current_manifest = {"wizard_inputs": {"comment": "Cafe 😀 東京"}}
    prior_manifest = {"wizard_inputs": {"comment": "Café 😐 東京"}}

    text = format_config_diff(current_manifest, prior_manifest)

    assert text == "- wizard_inputs.comment: Café 😐 東京 -> Cafe 😀 東京"
