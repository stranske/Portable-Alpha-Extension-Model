from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

__all__ = ["ModelConfig", "load_config"]


class ModelConfig(BaseModel):
    """Validated simulation parameters."""

    N_SIMULATIONS: int = Field(gt=0, alias="N_SIMULATIONS")
    N_MONTHS: int = Field(gt=0, alias="N_MONTHS")
    mu_H: float = 0.04
    sigma_H: float = 0.01

    class Config:
        allow_population_by_field_name = True
        frozen = True


def load_config(path: str | Path | dict[str, Any]) -> ModelConfig:
    """Return ``ModelConfig`` parsed from YAML/CSV dictionary."""
    if isinstance(path, dict):
        data = path
    else:
        data = yaml.safe_load(Path(path).read_text())
    try:
        return ModelConfig(**data)
    except ValidationError as e:  # pragma: no cover - explicit failure
        raise ValueError(str(e)) from e
