from __future__ import annotations

"""Generate heatmaps for a small alpha-share parameter grid.

This helper script demonstrates the parameter sweep engine by running a
5×5 grid over the external PA alpha fraction (`external_pa_alpha_min_pct`, `external_pa_alpha_max_pct`, `external_pa_alpha_step_pct`) and
active share (`active_share_min_pct`, `active_share_max_pct`, `active_share_step_pct`). Summary metrics for the combined portfolio are
rendered as heatmaps and written to `plots/`.

The grid is deterministic thanks to a fixed random seed so results are
repeatable across runs.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from pa_core.config import ModelConfig
from pa_core.sweep import run_parameter_sweep_cached, sweep_results_to_dataframe
from pa_core.viz import grid_heatmap, grid_panel

METRICS = ["AnnReturn", "AnnVol", "TE", "CVaR", "BreachProb"]


def run_grid(seed: int, out_dir: Path) -> None:
    """Run the 5×5 grid sweep and export heatmaps."""
    cfg = ModelConfig(
        N_SIMULATIONS=500,
        N_MONTHS=12,
        external_pa_capital=50.0,
        active_ext_capital=0.0,
        internal_pa_capital=950.0,
        total_fund_capital=1000.0,
        analysis_mode="alpha_shares",
        external_pa_alpha_min_pct=30.0,
        external_pa_alpha_max_pct=70.0,
        external_pa_alpha_step_pct=10.0,
        active_share_min_pct=20.0,
        active_share_max_pct=100.0,
        active_share_step_pct=20.0,
    )

    rng = np.random.default_rng(seed)
    index_series = pd.Series(rng.normal(0, 0.01, cfg.N_MONTHS))

    results = run_parameter_sweep_cached(cfg, index_series, seed)
    df = sweep_results_to_dataframe(results)
    df_base = df[df["Agent"] == "Base"]

    figures = []
    for metric in METRICS:
        fig = grid_heatmap.make(df_base, x="theta_extpa", y="active_share", z=metric)
        fig.write_image(str(out_dir / f"grid_{metric}.png"))
        figures.append(fig)

    panel = grid_panel.make(figures, cols=3)
    panel.write_image(str(out_dir / "scenario_grid_panel.png"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run alpha-share grid demo")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--output", type=Path, default=Path("plots"), help="Output directory"
    )
    args = parser.parse_args(argv)
    args.output.mkdir(exist_ok=True)
    run_grid(args.seed, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
