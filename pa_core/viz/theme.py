from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import yaml

# Default threshold constant for buffer ratio
LOW_BUFFER_THRESHOLD = 0.1  # 10% - used as fallback for shortfall_amber threshold

# Load thresholds for traffic-light styling
_THRESH_PATH = Path(__file__).resolve().parents[1] / "config_thresholds.yaml"
_THEME_PATH = Path(__file__).resolve().parents[1] / "config_theme.yaml"

if _THRESH_PATH.exists():
    with open(_THRESH_PATH, encoding="utf-8") as fh:
        THRESHOLDS: dict[str, float] = yaml.safe_load(fh) or {}
else:
    THRESHOLDS = {}


def _load_theme(path: Path) -> tuple[list[str], str, str, str]:
    if path.exists():
        with open(path, encoding="utf-8") as fh:
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
        paper_bg = cfg.get("paper_bgcolor", "#ffffff")
        plot_bg = cfg.get("plot_bgcolor", "#ffffff")
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
        paper_bg = "#ffffff"
        plot_bg = "#ffffff"
    return colors, font, paper_bg, plot_bg


_COLORWAY, _FONT, _PAPER_BG, _PLOT_BG = _load_theme(_THEME_PATH)
TEMPLATE = go.layout.Template(
    layout=dict(
        colorway=_COLORWAY,
        font=dict(family=_FONT),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
    )
)

# Default value for missing ShortfallProb in visualizations and exports
DEFAULT_SHORTFALL_PROB = 0.0

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
    with open(path, encoding="utf-8") as fh:
        THRESHOLDS = yaml.safe_load(fh) or {}


def reload_theme(path: str | Path = _THEME_PATH) -> None:
    """Reload colour palette and font from a YAML file."""
    global _COLORWAY, _FONT, _PAPER_BG, _PLOT_BG, TEMPLATE
    _COLORWAY, _FONT, _PAPER_BG, _PLOT_BG = _load_theme(Path(path))
    TEMPLATE = go.layout.Template(
        layout=dict(
            colorway=_COLORWAY,
            font=dict(family=_FONT),
            paper_bgcolor=_PAPER_BG,
            plot_bgcolor=_PLOT_BG,
        )
    )
