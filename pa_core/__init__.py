"""Core portable alpha utilities (lightweight package import).

This package intentionally avoids importing heavy submodules at import time to
keep `python -m pa_core.cli` fast and robust in subprocess tests. Import submodules
directly as needed, e.g. `from pa_core import sensitivity` or `import pa_core.cli`.
"""

try:  # pragma: no cover - optional dependency
    from . import viz  # noqa: F401
except (
    ImportError,
    ModuleNotFoundError,
):  # pragma: no cover - viz may require heavy deps
    viz = None  # type: ignore[assignment]

__all__: list[str] = [
    "viz",
]
