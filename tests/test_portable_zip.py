import inspect
import sys
import zipfile
from pathlib import Path

from scripts.make_portable_zip import (
    DEFAULT_EMBEDDED_PYTHON,
    build_windows_portable_zip,
    create_filtered_zip,
    get_default_excludes,
    should_exclude_path,
)


def test_should_exclude_basic_patterns(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    # Create files and directories that should be excluded
    (root / ".git").mkdir()
    (root / ".venv").mkdir()
    (root / "__pycache__").mkdir()
    (root / "build").mkdir()
    (root / "notes.ipynb").write_text("{}")
    (root / "keep.py").write_text("print('x')")

    excludes = get_default_excludes()

    # Excluded dirs
    assert should_exclude_path(root / ".git", root, excludes)
    assert should_exclude_path(root / ".venv", root, excludes)
    assert should_exclude_path(root / "__pycache__", root, excludes)
    assert should_exclude_path(root / "build", root, excludes)
    # Excluded file patterns
    assert should_exclude_path(root / "notes.ipynb", root, excludes)
    # Non-excluded source file
    assert not should_exclude_path(root / "keep.py", root, excludes)


def test_create_filtered_zip_includes_and_excludes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pa_core").mkdir()
    (root / "pa_core" / "__init__.py").write_text("")
    (root / "pa_core" / "mod.py").write_text("x=1")
    (root / "build").mkdir()
    (root / "build" / "lib").mkdir()
    (root / "build" / "lib" / "stale.py").write_text("x=2")
    (root / ".git").mkdir()
    (root / "scratch.log").write_text("debug")

    out = tmp_path / "portable.zip"
    create_filtered_zip(root, out, get_default_excludes().union({"*.log"}))

    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
    # Included package file
    assert "pa_core/mod.py" in names
    # Excluded items
    assert ".git/" not in names
    assert "build/lib/stale.py" not in names
    assert "scratch.log" not in names


def test_embedded_python_default_meets_the_supported_floor() -> None:
    major, minor, *_ = (int(part) for part in DEFAULT_EMBEDDED_PYTHON.split("."))
    assert (major, minor) >= (3, 13)


def test_build_function_defaults_to_the_embedded_python_constant() -> None:
    parameter = inspect.signature(build_windows_portable_zip).parameters["python_version"]
    assert parameter.default == DEFAULT_EMBEDDED_PYTHON


def test_cli_defaults_to_the_embedded_python_constant(monkeypatch, tmp_path: Path) -> None:
    import scripts.make_portable_zip as module

    seen = {}

    def fake_build(root, output_path, *, python_version, verbose):
        seen["python_version"] = python_version

    monkeypatch.setattr(module, "build_windows_portable_zip", fake_build)
    monkeypatch.setattr(
        sys, "argv", ["pa-make-zip", "--with-python", "--output", str(tmp_path / "out.zip")]
    )
    assert module.main() == 0
    assert seen == {"python_version": DEFAULT_EMBEDDED_PYTHON}
