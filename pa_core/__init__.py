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

# Re-export commonly used helpers for CLI scripts
from .run_flags import RunFlags
from .sim import draw_financing_series, draw_joint_returns
from .reporting.excel import export_to_excel
from .config import load_config
from .data.loaders import load_index_returns

__all__: list[str] = [
    "viz",
    "RunFlags",
    "draw_financing_series",
    "draw_joint_returns",
    "export_to_excel",
    "load_config",
    "load_index_returns",
]
