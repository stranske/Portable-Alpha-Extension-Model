from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from pa_core.cli import main


def test_cli_overrides_return_distribution(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        yaml.safe_dump({"N_SIMULATIONS": 2, "N_MONTHS": 1, "financing_mode": "broadcast"})
    )
    idx_path = tmp_path / "index.csv"
    idx_path.write_text("Return\n0.01\n0.02\n")
    out_path = tmp_path / "out.xlsx"

    with (
        patch("pa_core.sim.draw_joint_returns") as mock_draws,
        patch("pa_core.sim.draw_financing_series") as mock_financing,
        patch("pa_core.simulations.simulate_agents") as mock_simulate,
        patch("pa_core.sim.metrics.summary_table") as mock_summary,
        patch("pa_core.reporting.export_to_excel"),
        patch("pa_core.agents.registry.build_from_config") as mock_build_agents,
        patch("pa_core.sim.covariance.build_cov_matrix") as mock_build_cov,
    ):
        mock_draws.return_value = ([], [], [], [])
        mock_financing.return_value = ([], [], [])
        mock_build_agents.return_value = []
        mock_simulate.return_value = {"Base": [[0.01]]}
        mock_summary.return_value = pd.DataFrame(
            {"Agent": ["Base"], "terminal_AnnReturn": [0.1], "monthly_AnnVol": [0.1]}
        )
        mock_build_cov.return_value = np.eye(4)

        main(
            [
                "--config",
                str(cfg_path),
                "--index",
                str(idx_path),
                "--output",
                str(out_path),
                "--sensitivity",
                "--return-distribution",
                "student_t",
                "--return-t-df",
                "7",
                "--return-copula",
                "t",
            ]
        )

    params = mock_draws.call_args.kwargs["params"]
    assert params["return_distribution"] == "student_t"
    assert params["return_t_df"] == 7.0
    assert params["return_copula"] == "t"
