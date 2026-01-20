import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pa_core.config import ModelConfig
from pa_core.facade import RunArtifacts, export
from pa_core.reporting import export_to_excel

openpyxl: Any = pytest.importorskip("openpyxl")


def test_export_to_excel_sheets(tmp_path: Path):
    inputs = {"a": 1, "b": 2}
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1])}
    file_path = tmp_path / "out.xlsx"
    export_to_excel(inputs, summary, raw, filename=str(file_path))
    wb = openpyxl.load_workbook(file_path)
    assert set(wb.sheetnames) == {"Inputs", "Summary", "Base"}


def test_export_to_excel_pivot(tmp_path: Path):
    inputs = {"x": 1}
    summary = pd.DataFrame({"Total": [0.2]})
    raw = {
        "Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1]),
        "Ext": pd.DataFrame([[0.3, 0.4]], columns=[0, 1]),
    }
    file_path = tmp_path / "pivot.xlsx"
    export_to_excel(inputs, summary, raw, filename=str(file_path), pivot=True)
    wb = openpyxl.load_workbook(file_path)
    assert set(wb.sheetnames) == {"Inputs", "Summary", "AllReturns"}
    ws = wb["AllReturns"]
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    assert header == ["Sim", "Month", "Agent", "Return"]


def test_export_to_excel_adds_attribution_and_risk_sheets(tmp_path: Path) -> None:
    inputs = {
        "_attribution_df": pd.DataFrame({"Agent": ["Base"], "Sub": ["Core"], "Return": [0.01]}),
        "_risk_attr_df": pd.DataFrame(
            {
                "Agent": ["Base"],
                "BetaVol": [0.1],
                "AlphaVol": [0.05],
                "CorrWithIndex": [0.8],
                "AnnVolApprox": [0.12],
                "TEApprox": [0.03],
            }
        ),
    }
    summary = pd.DataFrame({"Total": [0.2]})
    raw = {"Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1])}
    file_path = tmp_path / "with_attr.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    wb = openpyxl.load_workbook(file_path)
    assert {"Attribution", "RiskAttribution"} <= set(wb.sheetnames)


def test_export_to_excel_sets_correlation_repair_metadata(tmp_path: Path) -> None:
    inputs = {"correlation_repair_applied": True}
    summary = pd.DataFrame({"Total": [0.2]})
    raw = {"Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1])}
    file_path = tmp_path / "repair.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    wb = openpyxl.load_workbook(file_path)
    assert "correlation_repair_applied=true" in (wb.properties.keywords or "")


def test_export_to_excel_writes_rng_metadata_sheet(tmp_path: Path) -> None:
    inputs = {"a": 1}
    summary = pd.DataFrame({"Total": [0.2]})
    raw = {"Base": pd.DataFrame([[0.1, 0.2]], columns=[0, 1])}
    metadata = {"rng_seed": 123, "substream_ids": {"Base": "abc123"}}
    file_path = tmp_path / "meta.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path), metadata=metadata)

    wb = openpyxl.load_workbook(file_path)
    assert "Metadata" in wb.sheetnames
    ws = wb["Metadata"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    meta_map = {key: value for key, value in rows}
    assert meta_map["rng_seed"] == 123
    assert meta_map["substream_ids"] == json.dumps(metadata["substream_ids"], sort_keys=True)


def test_export_to_excel_adds_agent_semantics_sheet(tmp_path: Path) -> None:
    inputs = {
        "total_fund_capital": 1000.0,
        "agents": [
            {
                "name": "Base",
                "capital": 600.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "ExternalPA",
                "capital": 200.0,
                "beta_share": 0.2,
                "alpha_share": 0.0,
                "extra": {"theta_extpa": 0.25},
            },
            {
                "name": "ActiveExt",
                "capital": 100.0,
                "beta_share": 0.1,
                "alpha_share": 0.0,
                "extra": {"active_share": 0.5},
            },
            {
                "name": "InternalPA",
                "capital": 50.0,
                "beta_share": 0.0,
                "alpha_share": 0.05,
                "extra": {},
            },
            {
                "name": "InternalBeta",
                "capital": 50.0,
                "beta_share": 0.05,
                "alpha_share": 0.0,
                "extra": {},
            },
        ],
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "agent_semantics.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    required_cols = [
        "Agent",
        "capital_mm",
        "implied_capital_share",
        "beta_coeff_used",
        "alpha_coeff_used",
        "financing_coeff_used",
        "notes",
        "mismatch_flag",
    ]
    assert set(required_cols) <= set(df.columns)
    expected_agents = {"Base", "ExternalPA", "ActiveExt", "InternalPA", "InternalBeta"}
    assert expected_agents <= set(df["Agent"].tolist())


def test_export_attaches_agent_semantics_from_config(tmp_path: Path) -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        reference_sigma=0.0,
        agents=[
            {
                "name": "Base",
                "capital": 1000.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            }
        ],
    )
    artifacts = RunArtifacts(
        config=cfg,
        index_series=pd.Series([0.0]),
        returns={"Base": [0.0]},
        summary=pd.DataFrame({"Base": [0.0]}),
        inputs={},
        raw_returns={"Base": pd.DataFrame([[0.0]], columns=[0])},
    )
    file_path = tmp_path / "export_agent_semantics.xlsx"

    export(artifacts, file_path)

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert "Base" in set(df["Agent"].tolist())
