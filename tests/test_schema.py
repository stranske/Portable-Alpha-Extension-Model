from __future__ import annotations

import pytest
from pydantic import ValidationError

from pa_core.schema import Asset, Correlation, Index, Scenario


def test_scenario_rejects_index_in_assets() -> None:
    with pytest.raises(ValidationError, match="assets must not include index id"):
        Scenario(
            index=Index(id="IDX", mu=0.05, sigma=0.1),
            assets=[Asset(id="IDX", mu=0.04, sigma=0.08)],
            correlations=[Correlation(pair=("IDX", "IDX"), rho=0.0)],
        )
