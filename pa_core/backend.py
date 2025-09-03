from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import ModelConfig

__all__ = ["xp", "set_backend", "get_backend", "resolve_and_set_backend"]

xp = importlib.import_module("numpy")


def set_backend(name: str) -> None:
    """Set numeric backend to 'numpy' or 'cupy'."""
    global xp
    if name == "numpy":
        xp = importlib.import_module("numpy")
    elif name == "cupy":
        try:
            xp = importlib.import_module("cupy")
        except (
            ImportError,
            ModuleNotFoundError,
        ) as e:  # pragma: no cover - depends on optional dep
            raise ImportError("CuPy backend requested but not installed") from e
    else:
        raise ValueError(f"Unknown backend: {name}")


def get_backend() -> str:
    """Return the current backend name."""
    return "cupy" if xp.__name__.startswith("cupy") else "numpy"


def resolve_and_set_backend(
    cli_backend: Optional[str], config: Optional["ModelConfig"] = None
) -> str:
    """Resolve backend choice from CLI args and config, then set it.
    
    Args:
        cli_backend: Backend specified via CLI (can be None)
        config: ModelConfig object (can be None)
        
    Returns:
        The resolved backend name that was set
        
    The resolution priority is:
    1. CLI argument (if provided)
    2. Config file setting (if config provided)
    3. Default "numpy" (fallback)
    """
    # Priority: CLI arg > config setting > default
    if cli_backend is not None:
        backend_choice = cli_backend
    elif config is not None:
        backend_choice = config.backend
    else:
        backend_choice = "numpy"
    
    set_backend(backend_choice)
    return backend_choice
