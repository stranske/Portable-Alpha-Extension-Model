#!/usr/bin/env python3
"""Regenerate committed offline assets for the browser-only stlite app."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_VENDOR_DIR = REPO_ROOT / "web" / "vendor"
STLITE_VERSION = "0.76.0"
PYODIDE_VERSION = "0.26.2"
PLOTLY_JS_VERSION = "2.35.2"
STLITE_VENDOR_DIR = WEB_VENDOR_DIR / f"stlite@{STLITE_VERSION}"
PYODIDE_VENDOR_DIR = WEB_VENDOR_DIR / f"pyodide-{PYODIDE_VERSION}"
PYPI_VENDOR_DIR = WEB_VENDOR_DIR / "pypi"
PYODIDE_CDN_BASE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full"
NPM_PACKAGE_URL = f"https://registry.npmjs.org/@stlite/browser/-/browser-{STLITE_VERSION}.tgz"
PLOTLY_JS_URL = f"https://cdn.plot.ly/plotly-{PLOTLY_JS_VERSION}.min.js"
DOWNLOAD_TIMEOUT_SECONDS = 60

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
    "PyYAML",
    "streamlit",
)
PYODIDE_RUNTIME_FILES = (
    "pyodide.mjs",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, expected_sha: str | None = None) -> None:
    try:
        display_target = str(target.relative_to(REPO_ROOT))
    except ValueError:
        display_target = str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if expected_sha is None or _sha256(target) == expected_sha:
            print(f"skip existing {display_target}")
            return
        target.unlink()

    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                shutil.copyfileobj(response, tmp)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    if expected_sha is not None and _sha256(tmp_path) != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {target.name}")
    tmp_path.replace(target)
    print(f"downloaded {display_target}")


def _lock_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _package_name(requirement: str) -> str:
    return Requirement(requirement).name


def _load_pyodide_lock() -> dict[str, dict[str, object]]:
    return json.loads((PYODIDE_VENDOR_DIR / "pyodide-lock.json").read_text())["packages"]


def _dependency_closure(seed_names: set[str], packages: dict[str, dict]) -> set[str]:
    closure: set[str] = set()
    stack = [_lock_key(name) for name in seed_names]
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        if name not in packages:
            continue
        closure.add(name)
        stack.extend(_lock_key(dep) for dep in packages[name].get("depends", []))
    return closure


def _safe_file_name(file_name: str) -> str:
    path = Path(file_name)
    if path.is_absolute() or path.name != file_name or ".." in path.parts:
        raise ValueError(f"unsafe package file_name: {file_name!r}")
    return file_name


def _download_pyodide_runtime() -> None:
    for name in PYODIDE_RUNTIME_FILES:
        _download(f"{PYODIDE_CDN_BASE}/{name}", PYODIDE_VENDOR_DIR / name)


def _download_pyodide_wheels(seed_names: set[str]) -> None:
    packages = _load_pyodide_lock()
    for name in sorted(_dependency_closure(seed_names, packages)):
        package = packages[name]
        file_name = _safe_file_name(str(package["file_name"]))
        _download(
            f"{PYODIDE_CDN_BASE}/{file_name}",
            PYODIDE_VENDOR_DIR / file_name,
            expected_sha=str(package.get("sha256") or "") or None,
        )


def _vendor_stlite() -> None:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
        archive = Path(tmp) / "stlite-browser.tgz"
        _download(NPM_PACKAGE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(Path(tmp), filter="data")
        package_build = Path(tmp) / "package" / "build"
        if STLITE_VENDOR_DIR.exists():
            shutil.rmtree(STLITE_VENDOR_DIR)
        shutil.copytree(package_build, STLITE_VENDOR_DIR)

    worker_paths = list((STLITE_VENDOR_DIR / "assets").glob("worker-*.js"))
    if len(worker_paths) != 1:
        raise RuntimeError(f"expected one stlite worker, found {worker_paths}")
    worker = worker_paths[0]
    worker_text = worker.read_text(encoding="utf-8")
    worker_text = worker_text.replace(
        f'"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/pyodide.mjs"',
        f'"/vendor/pyodide-{PYODIDE_VERSION}/pyodide.mjs"',
    )
    worker.write_text(worker_text, encoding="utf-8")
    for source_map in STLITE_VENDOR_DIR.rglob("*.map"):
        source_map.unlink()


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


def _pip_download_no_deps(requirement: str) -> Path:
    before = {path.name for path in PYPI_VENDOR_DIR.glob("*.whl")}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(PYPI_VENDOR_DIR),
            requirement,
        ],
        check=True,
    )
    after = {path.name for path in PYPI_VENDOR_DIR.glob("*.whl")}
    new_files = sorted(after - before)
    if new_files:
        return PYPI_VENDOR_DIR / new_files[-1]

    name = _lock_key(_package_name(requirement))
    candidates = [
        path
        for path in PYPI_VENDOR_DIR.glob("*.whl")
        if _lock_key(path.name).startswith(f"{name}-")
    ]
    if not candidates:
        raise RuntimeError(f"pip did not produce a wheel for {requirement!r}")
    return sorted(candidates)[-1]


def _download_pypi_wheel_closure(seed_requirements: set[str]) -> set[str]:
    packages = _load_pyodide_lock()
    queue = list(seed_requirements)
    seen: set[str] = set()
    vendored_names: set[str] = set()
    PYPI_VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    while queue:
        requirement_text = queue.pop(0)
        requirement = Requirement(requirement_text)
        name = _lock_key(requirement.name)
        if name == "streamlit":
            continue
        if name in seen or name in packages:
            continue
        seen.add(name)
        wheel = _pip_download_no_deps(requirement_text)
        vendored_names.add(name)
        for dependency in _wheel_metadata_requires(wheel):
            dep_name = _lock_key(dependency.name)
            if dep_name not in packages and dep_name not in seen:
                queue.append(str(dependency))
    return vendored_names


def _stlite_streamlit_requirements() -> set[str]:
    requirements = {"blinker", "tenacity"}
    for wheel in (STLITE_VENDOR_DIR / "wheels").glob("streamlit-*.whl"):
        requirements.update(str(req) for req in _wheel_metadata_requires(wheel))
    return requirements


def main() -> int:
    WEB_VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    _vendor_stlite()
    _download(PLOTLY_JS_URL, WEB_VENDOR_DIR / f"plotly-{PLOTLY_JS_VERSION}.min.js")
    _download_pyodide_runtime()

    pyodide_seed_names = {_package_name(requirement) for requirement in APP_REQUIREMENTS}
    pyodide_seed_names.update(
        _package_name(requirement) for requirement in _stlite_streamlit_requirements()
    )
    pyodide_seed_names.update({"micropip", "packaging"})
    _download_pyodide_wheels(pyodide_seed_names)

    if PYPI_VENDOR_DIR.exists():
        shutil.rmtree(PYPI_VENDOR_DIR)
    pypi_requirements = {
        requirement
        for requirement in APP_REQUIREMENTS
        if _lock_key(_package_name(requirement)) != "streamlit"
    }
    pypi_requirements.update(_stlite_streamlit_requirements())
    vendored = _download_pypi_wheel_closure(pypi_requirements)
    print(f"vendored pure-PyPI wheels: {', '.join(sorted(vendored))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
