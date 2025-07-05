from __future__ import annotations

from pathlib import Path
import yaml
import plotly.graph_objects as go

# Load thresholds for traffic-light styling
_THRESH_PATH = Path(__file__).resolve().parents[1] / "config_thresholds.yaml"
_THEME_PATH = Path(__file__).resolve().parents[1] / "config_theme.yaml"

if _THRESH_PATH.exists():
    with open(_THRESH_PATH, "r", encoding="utf-8") as fh:
        THRESHOLDS: dict[str, float] = yaml.safe_load(fh) or {}
else:
    THRESHOLDS = {}

def _load_theme(path: Path) -> tuple[list[str], str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        colors = cfg.get("colorway") or [
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
        ]
        font = cfg.get("font", "Roboto")
    else:
        colors = [
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
        ]
        font = "Roboto"
    return colors, font


_COLORWAY, _FONT = _load_theme(_THEME_PATH)
TEMPLATE = go.layout.Template(
    layout=dict(colorway=_COLORWAY, font=dict(family=_FONT))
)

# Map agent class -> category name used for consistent colours
CATEGORY_BY_AGENT = {
    "InternalPAAgent": "Internal Portable Alpha",
    "InternalBetaAgent": "Internal Portable Alpha",
    "ExternalPAAgent": "External Portable Alpha",
    "ActiveExtensionAgent": "External Portable Alpha",
    "BaseAgent": "Benchmark / Passive",
}


def reload_thresholds(path: str | Path = _THRESH_PATH) -> None:
    """Reload traffic-light thresholds from a YAML file.

    Parameters
    ----------
    path:
        Location of the YAML config. Defaults to ``config_thresholds.yaml`` next
        to ``pa_core``.
    """
    global THRESHOLDS
    with open(path, "r", encoding="utf-8") as fh:
        THRESHOLDS = yaml.safe_load(fh) or {}


def reload_theme(path: str | Path = _THEME_PATH) -> None:
    """Reload colour palette and font from a YAML file."""
    global _COLORWAY, _FONT, TEMPLATE
    _COLORWAY, _FONT = _load_theme(Path(path))
    TEMPLATE = go.layout.Template(
        layout=dict(colorway=_COLORWAY, font=dict(family=_FONT))
    )
