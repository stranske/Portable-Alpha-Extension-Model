from __future__ import annotations

import ast
from pathlib import Path


def _is_captured_output_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "captured"
        and node.attr in {"out", "err"}
    )


def test_test_main_avoids_hardcoded_cli_output_strings() -> None:
    source_path = Path(__file__).resolve().parent / "test_main.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sides = [node.left, *node.comparators]
        if not any(_is_captured_output_attr(side) for side in sides):
            continue
        for side in sides:
            if _is_captured_output_attr(side):
                continue
            if isinstance(side, ast.Constant) and isinstance(side.value, str):
                raise AssertionError(
                    "tests/test_main.py should compare CLI outputs against constants from "
                    "tests/expected_cli_outputs.py, not hardcoded strings."
                )
