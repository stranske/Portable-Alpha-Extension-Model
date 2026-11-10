"""The single-run export must survive a failure of detailed attribution.

`pa_core.cli` imports NumPy for type checking only (`np`); at runtime `_main` binds it as `_np`.
The attribution fallback once called `np.asarray`, so any failure of
`compute_sleeve_return_attribution` turned into an uncaught `NameError` and aborted the run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pa_core.cli import main

INDEX = Path(__file__).resolve().parents[1] / "data/sp500tr_fred_divyield.csv"


def test_attribution_fallback_exports_annualised_totals(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "pa_core.sim.sensitivity.one_factor_deltas", lambda *_args, **_kwargs: pd.DataFrame()
    )

    def unavailable(*_args):
        raise TypeError("detailed attribution unavailable")

    monkeypatch.setattr(
        "pa_core.reporting.attribution.compute_sleeve_return_attribution", unavailable
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 3,
                "financing_mode": "broadcast",
                "analysis_mode": "single_with_sensitivity",
            }
        )
    )
    output = tmp_path / "result.xlsx"
    # `--sensitivity` keeps the CLI on the single-run path, where the fallback lives.
    main(
        [
            "--config",
            str(config),
            "--index",
            str(INDEX),
            "--output",
            str(output),
            "--seed",
            "123",
            "--sensitivity",
        ]
    )

    attribution = pd.read_excel(output, sheet_name="Attribution")
    assert set(attribution["Sub"]) == {"Total"}
    assert set(attribution["Agent"]) == set(pd.read_excel(output, sheet_name="Summary")["Agent"])
    assert attribution["Agent"].is_unique
    for row in attribution.itertuples():
        raw = pd.read_excel(output, sheet_name=row.Agent, index_col=0)
        # Three distinct months, so an annualised mean differs from an annualised sum.
        assert raw.shape[1] == 3
        assert raw.to_numpy().sum() != pytest.approx(raw.to_numpy().mean())
        assert row.Return == pytest.approx(12 * raw.to_numpy().mean())
