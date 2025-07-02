from __future__ import annotations

from pathlib import Path
import yaml
import plotly.graph_objects as go

# Load thresholds for traffic-light styling
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_thresholds.yaml"
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        THRESHOLDS: dict[str, float] = yaml.safe_load(fh) or {}
else:
    THRESHOLDS = {}

# Colour-blind friendly palette
_COLORWAY = [
    "#377eb8",  # blue
    "#ff7f00",  # orange
    "#4daf4a",  # green
    "#f781bf",  # pink
    "#a65628",  # brown
    "#984ea3",  # purple
]

TEMPLATE = go.layout.Template(
    layout=dict(colorway=_COLORWAY, font=dict(family="Roboto"))
)

# Map agent class -> category name used for consistent colours
CATEGORY_BY_AGENT = {
    "InternalPAAgent": "Internal Portable Alpha",
    "InternalBetaAgent": "Internal Portable Alpha",
    "ExternalPAAgent": "External Portable Alpha",
    "ActiveExtensionAgent": "External Portable Alpha",
    "BaseAgent": "Benchmark / Passive",
}


def reload_thresholds(path: str | Path = _CONFIG_PATH) -> None:
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
