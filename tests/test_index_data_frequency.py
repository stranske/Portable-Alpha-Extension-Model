from pathlib import Path

import pandas as pd


def test_sp500tr_csv_is_monthly_returns() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    df = pd.read_csv(path)
    assert {"Date", "Monthly_TR"}.issubset(df.columns)

    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    diffs = dates.sort_values().diff().dropna()
    day_deltas = diffs.dt.days
    assert day_deltas.between(28, 31).all()
