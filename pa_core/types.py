from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class BitGeneratorLike(Protocol):
    """Protocol for NumPy-style bit generator state handling."""

    state: Any


class GeneratorLike(Protocol):
    """Protocol for RNGs compatible with the simulation sampling interface."""

    bit_generator: BitGeneratorLike

    def normal(self, *args: Any, **kwargs: Any) -> Any: ...

    def random(self, *args: Any, **kwargs: Any) -> Any: ...

    def chisquare(self, *args: Any, **kwargs: Any) -> Any: ...

    def multivariate_normal(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for array-likes (numpy/cupy compatible)."""

    shape: tuple[int, ...]
    ndim: int
    size: int

    def __array__(self, dtype: Any | None = ...) -> Any: ...
