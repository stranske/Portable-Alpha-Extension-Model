"""Core portable alpha utilities (lightweight package import).

This package intentionally avoids importing heavy submodules at import time to
keep `python -m pa_core.cli` fast and robust in subprocess tests. Import submodules
directly as needed, e.g. `from pa_core import sensitivity` or `import pa_core.cli`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .config import load_config as load_config
    from .data.loaders import load_index_returns as load_index_returns
    from .reporting.excel import export_to_excel as export_to_excel
    from .run_flags import RunFlags as RunFlags
    from .sim import draw_financing_series as draw_financing_series
    from .sim import draw_joint_returns as draw_joint_returns

__all__: list[str] = [
    "viz",
    "RunFlags",
    "draw_financing_series",
    "draw_joint_returns",
    "export_to_excel",
    "load_config",
    "load_index_returns",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "load_config": ("pa_core.config", "load_config"),
    "load_index_returns": ("pa_core.data.loaders", "load_index_returns"),
    "export_to_excel": ("pa_core.reporting.excel", "export_to_excel"),
    "RunFlags": ("pa_core.run_flags", "RunFlags"),
    "draw_financing_series": ("pa_core.sim", "draw_financing_series"),
    "draw_joint_returns": ("pa_core.sim", "draw_joint_returns"),
}


def __getattr__(name: str) -> Any:
    if name == "viz":
        try:  # pragma: no cover - optional dependency
            return importlib.import_module(".viz", __name__)
        except (ImportError, ModuleNotFoundError):
            return None
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    module = importlib.import_module(module_name)
    return getattr(module, attr)
