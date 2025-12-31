from __future__ import annotations

from typing import Any, Protocol


class GeneratorLike(Protocol):
    """Protocol for RNGs compatible with the simulation sampling interface."""

    def normal(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def random(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def chisquare(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def multivariate_normal(self, *args: Any, **kwargs: Any) -> Any:
        ...
