from __future__ import annotations

import importlib

__all__ = ["xp", "set_backend", "get_backend"]

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
