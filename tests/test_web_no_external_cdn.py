from __future__ import annotations

import json
import re
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"
WEB_HTML = WEB_DIR / "index.html"
VENDOR_DIR = WEB_DIR / "vendor"
PYODIDE_VENDOR_DIR = VENDOR_DIR / "pyodide-0.26.2"
STLITE_VENDOR_DIR = VENDOR_DIR / "stlite@0.76.0"
PYPI_VENDOR_DIR = VENDOR_DIR / "pypi"

APP_REQUIREMENTS = (
    "numpy",
    "pandas",
    "scipy",
    "plotly",
    "python-pptx",
    "lxml",
    "Pillow",
    "openpyxl",
    "XlsxWriter",
    "pypdf",
    "pydantic",
    "rich",
    "PyYAML",
    "streamlit",
)
EXTERNAL_URL_RE = re.compile(r"https?://")


def _lock_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _assert_non_empty(path: Path) -> None:
    assert path.is_file(), f"missing vendored asset: {path.relative_to(REPO_ROOT)}"
    assert path.stat().st_size > 0, f"vendored asset is empty: {path.relative_to(REPO_ROOT)}"


def _pyodide_lock_packages() -> dict[str, dict]:
    return json.loads((PYODIDE_VENDOR_DIR / "pyodide-lock.json").read_text())["packages"]


def _wheel_metadata_requires(wheel: Path) -> list[Requirement]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    requirements: list[Requirement] = []
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker is None or requirement.marker.evaluate(
            {"python_version": "3.12", "extra": ""}
        ):
            requirements.append(requirement)
    return requirements


def _dependency_closure(seed_names: set[str], packages: dict[str, dict]) -> set[str]:
    closure: set[str] = set()
    stack = [_lock_key(name) for name in seed_names]
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        assert name in packages, f"requirement not found in pyodide lock: {name}"
        closure.add(name)
        stack.extend(_lock_key(dep) for dep in packages[name].get("depends", []))
    return closure


def _pyodide_wheel_metadata_requirements(
    seed_names: set[str], packages: dict[str, dict]
) -> list[Requirement]:
    requirements: list[Requirement] = []
    for name in sorted(_dependency_closure(seed_names, packages)):
        file_name = str(packages[name]["file_name"])
        if file_name.endswith(".whl"):
            requirements.extend(_wheel_metadata_requires(PYODIDE_VENDOR_DIR / file_name))
    return requirements


def _expanded_pyodide_seed_names(
    seed_names: set[str], packages: dict[str, dict]
) -> tuple[set[str], list[Requirement]]:
    seed_names = set(seed_names)
    metadata_requirements: list[Requirement] = []
    while True:
        metadata_requirements = _pyodide_wheel_metadata_requirements(seed_names, packages)
        metadata_seed_names = {
            requirement.name
            for requirement in metadata_requirements
            if _lock_key(requirement.name) in packages
        }
        if metadata_seed_names <= seed_names:
            return seed_names, metadata_requirements
        seed_names.update(metadata_seed_names)


def _pypi_wheel_names() -> set[str]:
    return {
        _lock_key(path.name.split("-")[0])
        for path in PYPI_VENDOR_DIR.glob("*.whl")
        if path.is_file()
    }


def _pypi_wheel_for_name(name: str) -> Path:
    candidates = [
        path
        for path in PYPI_VENDOR_DIR.glob("*.whl")
        if _lock_key(path.name.split("-")[0]) == _lock_key(name)
    ]
    assert candidates, f"missing pure-PyPI wheel for {name}"
    return candidates[0]


def test_web_html_has_no_external_network_references() -> None:
    html = WEB_HTML.read_text()
    assert "cdn.jsdelivr.net" not in html
    assert "cdn.plot.ly" not in html
    assert EXTERNAL_URL_RE.search(html) is None


def test_app_controlled_web_assets_have_no_external_cdn_references() -> None:
    worker_paths = list((STLITE_VENDOR_DIR / "assets").glob("worker-*.js"))
    assert len(worker_paths) == 1
    paths = [WEB_HTML, *worker_paths]
    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "cdn.jsdelivr.net" in text or "cdn.plot.ly" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_web_html_uses_vendored_runtime_assets() -> None:
    html = WEB_HTML.read_text()
    assert "./vendor/stlite@0.76.0/style.css" in html
    assert './vendor/stlite@0.76.0/stlite.js"' in html
    assert "./vendor/plotly-2.35.2.min.js" in html
    # Vendored runtime + wheel paths are referenced...
    assert "/vendor/pyodide-0.26.2/pyodide.mjs" in html
    assert "/vendor/stlite@0.76.0/wheels/stlite_lib-0.1.0-py3-none-any.whl" in html
    assert "/vendor/stlite@0.76.0/wheels/streamlit-1.41.0-cp312-none-any.whl" in html
    # ...and resolved relative to document.baseURI rather than the server root,
    # so the bundle boots when served from a subdirectory (web/). A regression to
    # a root-absolute 'pyodideUrl: "/vendor/..."' would 404 the worker's Pyodide
    # and wheel fetches. Guards the offline-boot fix.
    assert "document.baseURI" in html
    assert "pyodideUrl: __abs(" in html
    assert "stliteLib: __abs(" in html
    assert "streamlit: __abs(" in html


def test_required_runtime_files_are_vendored() -> None:
    for path in (
        VENDOR_DIR / "plotly-2.35.2.min.js",
        STLITE_VENDOR_DIR / "stlite.js",
        STLITE_VENDOR_DIR / "style.css",
        STLITE_VENDOR_DIR / "wheels" / "stlite_lib-0.1.0-py3-none-any.whl",
        STLITE_VENDOR_DIR / "wheels" / "streamlit-1.41.0-cp312-none-any.whl",
        PYODIDE_VENDOR_DIR / "pyodide.mjs",
        PYODIDE_VENDOR_DIR / "pyodide.asm.js",
        PYODIDE_VENDOR_DIR / "pyodide.asm.wasm",
        PYODIDE_VENDOR_DIR / "python_stdlib.zip",
        PYODIDE_VENDOR_DIR / "pyodide-lock.json",
    ):
        _assert_non_empty(path)


def test_pyodide_wheel_closure_is_vendored() -> None:
    packages = _pyodide_lock_packages()
    streamlit_wheel = STLITE_VENDOR_DIR / "wheels" / "streamlit-1.41.0-cp312-none-any.whl"
    seed_names = {
        Requirement(requirement).name
        for requirement in APP_REQUIREMENTS
        if _lock_key(Requirement(requirement).name) in packages
    }
    seed_names.update(
        requirement.name
        for requirement in _wheel_metadata_requires(streamlit_wheel)
        if _lock_key(requirement.name) in packages
    )
    seed_names.update({"micropip", "packaging"})
    seed_names, _ = _expanded_pyodide_seed_names(seed_names, packages)

    missing: list[str] = []
    for name in sorted(_dependency_closure(seed_names, packages)):
        file_name = packages[name]["file_name"]
        if not (PYODIDE_VENDOR_DIR / file_name).is_file():
            missing.append(f"{name}: {file_name}")
    assert missing == []


def test_pure_pypi_wheel_closure_is_vendored() -> None:
    packages = _pyodide_lock_packages()
    streamlit_wheel = STLITE_VENDOR_DIR / "wheels" / "streamlit-1.41.0-cp312-none-any.whl"
    queue = list(
        _lock_key(Requirement(requirement).name)
        for requirement in APP_REQUIREMENTS
        if _lock_key(Requirement(requirement).name) not in packages
        and _lock_key(Requirement(requirement).name) != "streamlit"
    )
    queue.extend(
        _lock_key(requirement.name)
        for requirement in _wheel_metadata_requires(streamlit_wheel)
        if _lock_key(requirement.name) not in packages
    )
    pyodide_seed_names = {
        Requirement(requirement).name
        for requirement in APP_REQUIREMENTS
        if _lock_key(Requirement(requirement).name) in packages
    }
    pyodide_seed_names.update(
        requirement.name
        for requirement in _wheel_metadata_requires(streamlit_wheel)
        if _lock_key(requirement.name) in packages
    )
    pyodide_seed_names.update({"micropip", "packaging"})
    _, pyodide_metadata_requirements = _expanded_pyodide_seed_names(
        pyodide_seed_names, packages
    )
    queue.extend(
        _lock_key(requirement.name)
        for requirement in pyodide_metadata_requirements
        if _lock_key(requirement.name) not in packages
    )
    wheel_names = _pypi_wheel_names()
    required: set[str] = set()
    missing: list[str] = []
    while queue:
        name = queue.pop(0)
        if name in required:
            continue
        required.add(name)
        if name not in wheel_names:
            missing.append(name)
            continue
        for dependency in _wheel_metadata_requires(_pypi_wheel_for_name(name)):
            dep_name = _lock_key(dependency.name)
            if dep_name not in packages and dep_name not in required:
                queue.append(dep_name)
    assert missing == []

    html = WEB_HTML.read_text()
    missing_from_html = [
        path.name
        for path in sorted(PYPI_VENDOR_DIR.glob("*.whl"))
        if f"/vendor/pypi/{path.name}" not in html
    ]
    assert missing_from_html == []
