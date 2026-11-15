import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin

import pytest

from pa_core.config import ModelConfig
from pa_core.fees import FeeSchedule
from pa_core.schema import (
    _dump_payload,
    _escape_markdown,
    _stringify_type,
    export_schema_definitions,
    generate_parameter_dictionary,
    generate_schema_templates,
    main,
)


def test_export_schema_includes_aliases() -> None:
    payload = export_schema_definitions(schema="config")
    model = payload["models"]["ModelConfig"]
    fields = model["fields"]
    assert fields["N_SIMULATIONS"]["alias"] == "Number of simulations"
    assert "Active share" in fields["active_share"]["aliases"]


def test_templates_match_schema(tmp_path: Path) -> None:
    generate_schema_templates(tmp_path)
    template_dir = Path("templates")
    template_files = [
        "parameters_template.csv",
        "params_template.yaml",
        "scenario_template.yaml",
        "scenario_example.yaml",
    ]
    for filename in template_files:
        generated = (tmp_path / filename).read_text()
        committed = (template_dir / filename).read_text()
        assert generated == committed, (
            f"{filename} template drift detected; re-run "
            "`python -m pa_core.schema --generate-templates`."
        )


def test_user_yaml_template_omits_backend() -> None:
    template = Path("templates/params_template.yaml").read_text()
    assert not any(line.lstrip().startswith("backend:") for line in template.splitlines())


def test_parameter_dictionary_matches_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "PARAMETER_DICTIONARY.md"
    generate_parameter_dictionary(output_path, schema="config")
    committed_path = Path("docs/guides/PARAMETER_DICTIONARY.md")
    committed = committed_path.read_text()
    generated = output_path.read_text()
    assert generated == committed, (
        "Parameter dictionary drift detected; re-run "
        "`python -m pa_core.schema --generate-parameter-dictionary`."
    )
    assert output_path.read_bytes() == committed_path.read_bytes()


def _discover_interpreter(minor: str) -> str | None:
    candidates: list[str] = []
    if f"{sys.version_info.major}.{sys.version_info.minor}" == minor:
        candidates.append(sys.executable)
    which_hit = shutil.which(f"python{minor}")
    if which_hit:
        candidates.append(which_hit)
    toolcache = Path("/opt/hostedtoolcache/Python")
    if toolcache.is_dir():
        for exe in sorted(toolcache.glob(f"{minor}.*/x64/bin/python"), reverse=True):
            candidates.append(str(exe))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        probe = subprocess.run(
            [candidate, "-c", "import pa_core.schema"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
    return None


def test_parameter_dictionary_byte_identical_across_313_and_314(tmp_path: Path) -> None:
    committed_bytes = Path("docs/guides/PARAMETER_DICTIONARY.md").read_bytes()

    # Simulate Python 3.13 typing.Union/Optional + pathlib._local.Path vs
    # Python 3.14 types.UnionType (PEP 604) + pathlib.Path field annotations.
    orig_annotations = {name: field.annotation for name, field in ModelConfig.model_fields.items()}
    try:
        for name, annotation in orig_annotations.items():
            if get_origin(annotation) in (Union, getattr(types, "UnionType", Union)):
                args = get_args(annotation)
                ModelConfig.model_fields[name].annotation = Union[tuple(args)]  # type: ignore[misc]  # noqa: UP007
        out_313_style = tmp_path / "dict_313_style.md"
        main(
            [
                "--generate-parameter-dictionary",
                "--dictionary-output",
                str(out_313_style),
            ]
        )

        for name, annotation in orig_annotations.items():
            if get_origin(annotation) in (Union, getattr(types, "UnionType", Union)):
                args = get_args(annotation)
                pep604 = args[0]
                for arg in args[1:]:
                    pep604 = pep604 | arg
                ModelConfig.model_fields[name].annotation = pep604
        out_314_style = tmp_path / "dict_314_style.md"
        main(
            [
                "--generate-parameter-dictionary",
                "--dictionary-output",
                str(out_314_style),
            ]
        )
    finally:
        for name, annotation in orig_annotations.items():
            ModelConfig.model_fields[name].annotation = annotation

    assert out_313_style.read_bytes() == out_314_style.read_bytes()
    assert out_313_style.read_bytes() == committed_bytes

    interpreters = {
        minor: exe
        for minor in ("3.13", "3.14")
        if (exe := _discover_interpreter(minor)) is not None
    }
    assert interpreters, "Expected at least the active Python interpreter to be discoverable"
    outputs_by_minor: dict[str, bytes] = {}
    for minor, exe in interpreters.items():
        out_file = tmp_path / f"PARAMETER_DICTIONARY_py{minor.replace('.', '')}.md"
        subprocess.run(
            [
                exe,
                "-m",
                "pa_core.schema",
                "--generate-parameter-dictionary",
                "--dictionary-output",
                str(out_file),
            ],
            check=True,
        )
        outputs_by_minor[minor] = out_file.read_bytes()
        assert outputs_by_minor[minor] == committed_bytes
    if "3.13" in outputs_by_minor and "3.14" in outputs_by_minor:
        assert outputs_by_minor["3.13"] == outputs_by_minor["3.14"]


def test_parameter_dictionary_has_generation_header(tmp_path: Path) -> None:
    output_path = tmp_path / "PARAMETER_DICTIONARY.md"
    generate_parameter_dictionary(output_path, schema="config")
    content = output_path.read_text()
    assert "Generated by `python -m pa_core.schema --generate-parameter-dictionary`" in content


def test_schema_export_supports_all_formats_and_cli_writes_requested_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = export_schema_definitions(schema="all")

    assert {"ModelConfig", "Scenario"}.issubset(payload["models"])
    assert '"models"' in _dump_payload(payload, "json")
    assert "models:" in _dump_payload(payload, "yaml")
    assert _escape_markdown("a|b\nc") == "a\\|b c"
    with pytest.raises(ValueError, match="Unsupported export format"):
        _dump_payload(payload, "toml")
    with pytest.raises(ValueError, match="Unknown schema selection"):
        export_schema_definitions(schema="unknown")

    output = tmp_path / "schema.yaml"
    main(["--export", "--schema", "scenario", "--format", "yaml", "--output", str(output)])
    assert output.read_text() == _dump_payload(export_schema_definitions(schema="scenario"), "yaml")

    main(["--export", "--schema", "config", "--format", "json"])
    printed = capsys.readouterr().out
    assert printed == _dump_payload(export_schema_definitions(schema="config"), "json") + "\n"

    template_dir = tmp_path / "templates"
    dictionary_path = tmp_path / "dictionary.md"
    main(
        [
            "--generate-templates",
            "--template-dir",
            str(template_dir),
            "--generate-parameter-dictionary",
            "--dictionary-output",
            str(dictionary_path),
            "--dictionary-schema",
            "scenario",
        ]
    )
    assert (template_dir / "params_template.yaml").exists()
    assert "## Scenario" in dictionary_path.read_text()
    assert "## ModelConfig" not in dictionary_path.read_text()
    assert "Generated by `python -m pa_core.schema --generate-parameter-dictionary`" in (
        dictionary_path.read_text()
    )

    with pytest.raises(SystemExit) as help_exit:
        main([])
    assert help_exit.value.code == 2
    assert "Scenario schema utilities" in capsys.readouterr().out


def test_format_default_for_markdown_handles_collections() -> None:
    from pa_core.schema import _format_default_for_markdown

    assert _format_default_for_markdown({"required": True}) == ""
    assert _format_default_for_markdown({"required": False, "default": None}) == "None"
    assert _format_default_for_markdown({"required": False, "default": [1, 2]}) == "[1, 2]"
    assert _format_default_for_markdown({"required": False, "default": {"a": 1}}) == '{"a": 1}'
    assert (
        _format_default_for_markdown(
            {"required": False, "default_factory": "list", "default": None}
        )
        == "default_factory: list"
    )


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (
            Optional[Dict[str, FeeSchedule]],  # noqa: UP045
            "Dict[str, pa_core.fees.FeeSchedule] | None",
        ),
        (Dict[str, FeeSchedule] | None, "Dict[str, pa_core.fees.FeeSchedule] | None"),
        (
            Dict[str, Optional[FeeSchedule]],  # noqa: UP045
            "Dict[str, pa_core.fees.FeeSchedule | None]",
        ),
        (Dict[str, FeeSchedule | None], "Dict[str, pa_core.fees.FeeSchedule | None]"),
        (Optional[List[List[float]]], "List[List[float]] | None"),  # noqa: UP007, UP045
        (List[List[float]] | None, "List[List[float]] | None"),
        (Optional[str], "str | None"),  # noqa: UP007, UP045
        (str | None, "str | None"),
        (Union[int, str, None], "int | str | None"),  # noqa: UP007, UP045
        (int | str | None, "int | str | None"),
        (Optional[Path], "pathlib.Path | None"),  # noqa: UP007, UP045
        (Path | None, "pathlib.Path | None"),
        (List[Path], "List[pathlib.Path]"),
    ],
)
def test_stringify_type_renders_unions_independent_of_spelling(
    annotation: object, expected: str
) -> None:
    assert _stringify_type(annotation) == expected


def test_stringify_type_preserves_non_union_spelling() -> None:
    assert _stringify_type(None) == "Any"
    assert _stringify_type(Dict[str, Any]) == "Dict[str, Any]"
    assert _stringify_type(Tuple[int, ...]) == "Tuple[int, ...]"
    assert _stringify_type(Dict[str, FeeSchedule]) == "Dict[str, pa_core.fees.FeeSchedule]"
    assert _stringify_type(float) == "<class 'float'>"
