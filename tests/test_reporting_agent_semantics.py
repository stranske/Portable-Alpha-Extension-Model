from types import SimpleNamespace

import pandas as pd
import pytest

from pa_core.config import ModelConfig
from pa_core.facade import _serialize_agent_semantics_input
from pa_core.reporting import export_to_excel
from pa_core.reporting.agent_semantics import build_agent_semantics


def test_build_agent_semantics_coeffs_and_mismatch() -> None:
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
                "capital": 150.0,
                "beta_share": 0.15,
                "alpha_share": 0.0,
                "extra": {"active_share": 0.5},
            },
            {
                "name": "InternalPA",
                "capital": 100.0,
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
    )

    df = build_agent_semantics(cfg)

    assert list(df.columns) == [
        "Agent",
        "capital_mm",
        "implied_capital_share",
        "beta_coeff_used",
        "alpha_coeff_used",
        "financing_coeff_used",
        "notes",
        "mismatch_flag",
    ]

    lookup = df.set_index("Agent")

    base = lookup.loc["Base"]
    assert base["beta_coeff_used"] == pytest.approx(0.6)
    assert base["alpha_coeff_used"] == pytest.approx(0.4)
    assert base["financing_coeff_used"] == pytest.approx(-0.6)
    assert bool(base["mismatch_flag"]) is False

    external = lookup.loc["ExternalPA"]
    assert external["beta_coeff_used"] == pytest.approx(0.2)
    assert external["alpha_coeff_used"] == pytest.approx(0.05)
    assert external["financing_coeff_used"] == pytest.approx(-0.2)
    assert bool(external["mismatch_flag"]) is False

    active = lookup.loc["ActiveExt"]
    assert active["beta_coeff_used"] == pytest.approx(0.15)
    assert active["alpha_coeff_used"] == pytest.approx(0.075)
    assert active["financing_coeff_used"] == pytest.approx(-0.15)
    assert bool(active["mismatch_flag"]) is False

    internal = lookup.loc["InternalPA"]
    assert internal["beta_coeff_used"] == pytest.approx(0.0)
    assert internal["alpha_coeff_used"] == pytest.approx(0.05)
    assert internal["financing_coeff_used"] == pytest.approx(0.0)
    assert bool(internal["mismatch_flag"]) is True

    internal_beta = lookup.loc["InternalBeta"]
    assert internal_beta["beta_coeff_used"] == pytest.approx(0.05)
    assert internal_beta["alpha_coeff_used"] == pytest.approx(0.0)
    assert internal_beta["financing_coeff_used"] == pytest.approx(-0.05)
    assert bool(internal_beta["mismatch_flag"]) is False


def test_build_agent_semantics_mismatch_flags() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        reference_sigma=0.0,
        agents=[
            {
                "name": "Base",
                "capital": 500.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "ExternalPA",
                "capital": 300.0,
                "beta_share": 0.2,
                "alpha_share": 0.0,
                "extra": {"theta_extpa": 0.25},
            },
            {
                "name": "ActiveExt",
                "capital": 50.0,
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
                "capital": 200.0,
                "beta_share": 0.05,
                "alpha_share": 0.0,
                "extra": {},
            },
        ],
    )

    df = build_agent_semantics(cfg)
    lookup = df.set_index("Agent")

    assert bool(lookup.loc["Base"]["mismatch_flag"]) is False
    assert bool(lookup.loc["ExternalPA"]["mismatch_flag"]) is True
    assert bool(lookup.loc["ActiveExt"]["mismatch_flag"]) is True
    assert bool(lookup.loc["InternalPA"]["mismatch_flag"]) is False
    assert bool(lookup.loc["InternalBeta"]["mismatch_flag"]) is True


def test_build_agent_semantics_mismatch_tolerance() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        reference_sigma=0.0,
        agents=[
            {
                "name": "Base",
                "capital": 599.999,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "ExternalPA",
                "capital": 200.0005,
                "beta_share": 0.2,
                "alpha_share": 0.0,
                "extra": {"theta_extpa": 0.25},
            },
            {
                "name": "InternalPA",
                "capital": 200.0005,
                "beta_share": 0.0,
                "alpha_share": 0.2,
                "extra": {},
            },
        ],
    )

    df = build_agent_semantics(cfg)
    lookup = df.set_index("Agent")

    assert bool(lookup.loc["ExternalPA"]["mismatch_flag"]) is False
    assert bool(lookup.loc["InternalPA"]["mismatch_flag"]) is False


def test_build_agent_semantics_adds_internalbeta_for_margin_requirement(monkeypatch) -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        reference_sigma=0.01,
        agents=[
            {
                "name": "Base",
                "capital": 1000.0,
                "beta_share": 1.0,
                "alpha_share": 0.0,
                "extra": {},
            }
        ],
    )

    def _margin_requirement(*_args, **_kwargs) -> float:
        return 25.0

    monkeypatch.setattr(
        "pa_core.reporting.agent_semantics.calculate_margin_requirement", _margin_requirement
    )

    df = build_agent_semantics(cfg)
    lookup = df.set_index("Agent")

    internal_beta = lookup.loc["InternalBeta"]
    assert internal_beta["capital_mm"] == pytest.approx(25.0)
    assert internal_beta["beta_coeff_used"] == pytest.approx(0.025)
    assert bool(internal_beta["mismatch_flag"]) is False


def test_build_agent_semantics_custom_agent_defaults() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=500.0,
        reference_sigma=0.0,
        agents=[
            {
                "name": "Base",
                "capital": 250.0,
                "beta_share": 0.5,
                "alpha_share": 0.5,
                "extra": {},
            },
            {
                "name": "CustomSleeve",
                "capital": 250.0,
                "beta_share": 0.55,
                "alpha_share": 0.45,
                "extra": {},
            },
        ],
    )

    df = build_agent_semantics(cfg)
    row = df.set_index("Agent").loc["CustomSleeve"]

    assert row["beta_coeff_used"] == pytest.approx(0.55)
    assert row["alpha_coeff_used"] == pytest.approx(0.45)
    assert row["financing_coeff_used"] == pytest.approx(-0.55)
    assert row["notes"] == "Semantics depend on the specific agent implementation"
    assert bool(row["mismatch_flag"]) is False


def test_build_agent_semantics_base_and_custom_ignore_mismatch() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        reference_sigma=0.0,
        agents=[
            {
                "name": "Base",
                "capital": 600.0,
                "beta_share": 0.9,
                "alpha_share": 0.1,
                "extra": {},
            },
            {
                "name": "CustomSleeve",
                "capital": 400.0,
                "beta_share": 0.95,
                "alpha_share": 0.05,
                "extra": {},
            },
        ],
    )

    df = build_agent_semantics(cfg)
    lookup = df.set_index("Agent")

    assert bool(lookup.loc["Base"]["mismatch_flag"]) is False
    assert bool(lookup.loc["CustomSleeve"]["mismatch_flag"]) is False


def test_build_agent_semantics_percent_inputs() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        reference_sigma=0.0,
        agents=[
            {
                "name": "Base",
                "capital": 300.0,
                "beta_share": 0.3,
                "alpha_share": 0.7,
                "extra": {},
            },
            {
                "name": "ExternalPA",
                "capital": 400.0,
                "beta_share": 0.4,
                "alpha_share": 0.0,
                "extra": {"theta_extpa": 25},
            },
            {
                "name": "ActiveExt",
                "capital": 300.0,
                "beta_share": 0.3,
                "alpha_share": 0.0,
                "extra": {"active_share": 50},
            },
        ],
    )

    df = build_agent_semantics(cfg)
    lookup = df.set_index("Agent")

    assert lookup.loc["ExternalPA"]["alpha_coeff_used"] == pytest.approx(0.1)
    assert lookup.loc["ActiveExt"]["alpha_coeff_used"] == pytest.approx(0.15)


def test_build_agent_semantics_percent_inputs_base_internalpa() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        total_fund_capital=1000.0,
        reference_sigma=0.0,
        agents=[
            {
                "name": "Base",
                "capital": 600.0,
                "beta_share": 60,
                "alpha_share": 40,
                "extra": {},
            },
            {
                "name": "InternalPA",
                "capital": 50.0,
                "beta_share": 0.0,
                "alpha_share": 5,
                "extra": {},
            },
        ],
    )

    df = build_agent_semantics(cfg)
    lookup = df.set_index("Agent")

    base = lookup.loc["Base"]
    assert base["beta_coeff_used"] == pytest.approx(0.6)
    assert base["alpha_coeff_used"] == pytest.approx(0.4)
    assert base["financing_coeff_used"] == pytest.approx(-0.6)

    internal = lookup.loc["InternalPA"]
    assert internal["beta_coeff_used"] == pytest.approx(0.0)
    assert internal["alpha_coeff_used"] == pytest.approx(0.05)
    assert internal["financing_coeff_used"] == pytest.approx(0.0)


def test_build_agent_semantics_normalizes_agent_share_inputs() -> None:
    cfg = SimpleNamespace(
        total_fund_capital=1000.0,
        agents=[
            {
                "name": "Base",
                "capital": 600.0,
                "beta_share": 60,
                "alpha_share": 40,
                "extra": {},
            }
        ],
    )

    df = build_agent_semantics(cfg)
    base = df.set_index("Agent").loc["Base"]

    assert base["beta_coeff_used"] == pytest.approx(0.6)
    assert base["alpha_coeff_used"] == pytest.approx(0.4)


def test_export_uses_serialized_agent_semantics(tmp_path, monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": [
            {
                "Agent": "Base",
                "capital_mm": 1000.0,
                "implied_capital_share": 1.0,
                "beta_coeff_used": 0.6,
                "alpha_coeff_used": 0.4,
                "financing_coeff_used": -0.6,
                "notes": "",
                "mismatch_flag": False,
            }
        ],
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "serialized_agent_semantics.xlsx"

    def _fail(*args, **kwargs):
        raise AssertionError("build_agent_semantics should not be called")

    monkeypatch.setattr("pa_core.reporting.agent_semantics.build_agent_semantics", _fail)

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert not df.empty


def test_export_serializes_agent_semantics_dataframe(tmp_path, monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": pd.DataFrame(
            [
                {
                    "Agent": "Base",
                    "capital_mm": 1000.0,
                    "implied_capital_share": 1.0,
                    "beta_coeff_used": 0.6,
                    "alpha_coeff_used": 0.4,
                    "financing_coeff_used": -0.6,
                    "notes": "",
                    "mismatch_flag": False,
                }
            ]
        )
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "dataframe_agent_semantics.xlsx"

    def _fail(*args, **kwargs):
        raise AssertionError("build_agent_semantics should not be called")

    monkeypatch.setattr("pa_core.reporting.agent_semantics.build_agent_semantics", _fail)

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"


def test_export_serializes_agent_semantics_series(tmp_path, monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": pd.Series(
            {
                "Agent": "Base",
                "capital_mm": 1000.0,
                "implied_capital_share": 1.0,
                "beta_coeff_used": 0.6,
                "alpha_coeff_used": 0.4,
                "financing_coeff_used": -0.6,
                "notes": "",
                "mismatch_flag": False,
            }
        )
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "series_agent_semantics.xlsx"

    def _fail(*args, **kwargs):
        raise AssertionError("build_agent_semantics should not be called")

    monkeypatch.setattr("pa_core.reporting.agent_semantics.build_agent_semantics", _fail)

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"


def test_export_uses_single_row_agent_semantics_dict(tmp_path, monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": {
            "Agent": "Base",
            "capital_mm": 1000.0,
            "implied_capital_share": 1.0,
            "beta_coeff_used": 0.6,
            "alpha_coeff_used": 0.4,
            "financing_coeff_used": -0.6,
            "notes": "",
            "mismatch_flag": False,
        },
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "single_row_agent_semantics.xlsx"

    def _fail(*args, **kwargs):
        raise AssertionError("build_agent_semantics should not be called")

    monkeypatch.setattr("pa_core.reporting.agent_semantics.build_agent_semantics", _fail)

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert not df.empty


def test_export_uses_agent_semantics_dict_of_columns(tmp_path, monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": {
            "Agent": ["Base", "ExternalPA"],
            "capital_mm": [1000.0, 200.0],
            "implied_capital_share": [1.0, 0.2],
            "beta_coeff_used": [0.6, 0.2],
            "alpha_coeff_used": [0.4, 0.05],
            "financing_coeff_used": [-0.6, -0.2],
            "notes": ["", ""],
            "mismatch_flag": [False, False],
        },
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "column_dict_agent_semantics.xlsx"

    def _fail(*args, **kwargs):
        raise AssertionError("build_agent_semantics should not be called")

    monkeypatch.setattr("pa_core.reporting.agent_semantics.build_agent_semantics", _fail)

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert not df.empty
    assert list(df["Agent"]) == ["Base", "ExternalPA"]


def test_export_uses_agent_semantics_dict_of_series(tmp_path, monkeypatch) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": {
            "Agent": pd.Series(["Base", "ExternalPA"]),
            "mismatch_flag": pd.Series([False, False]),
        },
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "series_dict_agent_semantics.xlsx"

    def _fail(*args, **kwargs):
        raise AssertionError("build_agent_semantics should not be called")

    monkeypatch.setattr("pa_core.reporting.agent_semantics.build_agent_semantics", _fail)

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert list(df["Agent"]) == ["Base", "ExternalPA"]


def test_export_builds_agent_semantics_when_missing(tmp_path) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "total_fund_capital": 1000.0,
        "agents": [
            {
                "name": "Base",
                "capital": 1000.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            }
        ],
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "fallback_agent_semantics.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert not df.empty
    assert "Agent" in df.columns


def test_export_builds_agent_semantics_when_empty_list(tmp_path) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": [],
        "total_fund_capital": 1000.0,
        "agents": [
            {
                "name": "Base",
                "capital": 1000.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            }
        ],
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "empty_list_agent_semantics.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert not df.empty
    assert "Agent" in df.columns


def test_export_builds_agent_semantics_when_empty_dict(tmp_path) -> None:
    pytest.importorskip("openpyxl")
    inputs = {
        "_agent_semantics_df": {},
        "total_fund_capital": 1000.0,
        "agents": [
            {
                "name": "Base",
                "capital": 1000.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            }
        ],
    }
    summary = pd.DataFrame({"Base": [0.1]})
    raw = {"Base": pd.DataFrame([[0.1]], columns=[0])}
    file_path = tmp_path / "empty_dict_agent_semantics.xlsx"

    export_to_excel(inputs, summary, raw, filename=str(file_path))

    df = pd.read_excel(file_path, sheet_name="AgentSemantics")
    assert not df.empty
    assert "Agent" in df.columns


def test_serialize_agent_semantics_input_dataframe() -> None:
    inputs = {
        "_agent_semantics_df": pd.DataFrame(
            [
                {
                    "Agent": "Base",
                    "capital_mm": 1000.0,
                    "implied_capital_share": 1.0,
                    "beta_coeff_used": 0.6,
                    "alpha_coeff_used": 0.4,
                    "financing_coeff_used": -0.6,
                    "notes": "",
                    "mismatch_flag": False,
                }
            ]
        )
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"


def test_serialize_agent_semantics_input_series() -> None:
    inputs = {
        "_agent_semantics_df": pd.Series(
            {
                "Agent": "Base",
                "capital_mm": 1000.0,
                "implied_capital_share": 1.0,
                "beta_coeff_used": 0.6,
                "alpha_coeff_used": 0.4,
                "financing_coeff_used": -0.6,
                "notes": "",
                "mismatch_flag": False,
            }
        )
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"


def test_serialize_agent_semantics_input_dict() -> None:
    inputs = {
        "_agent_semantics_df": {
            "Agent": "Base",
            "capital_mm": 1000.0,
            "implied_capital_share": 1.0,
            "beta_coeff_used": 0.6,
            "alpha_coeff_used": 0.4,
            "financing_coeff_used": -0.6,
            "notes": "",
            "mismatch_flag": False,
        }
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"


def test_serialize_agent_semantics_input_dict_of_columns() -> None:
    inputs = {
        "_agent_semantics_df": {
            "Agent": ["Base", "ExternalPA"],
            "capital_mm": [1000.0, 200.0],
            "implied_capital_share": [1.0, 0.2],
            "beta_coeff_used": [0.6, 0.2],
            "alpha_coeff_used": [0.4, 0.05],
            "financing_coeff_used": [-0.6, -0.2],
            "notes": ["", ""],
            "mismatch_flag": [False, False],
        }
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"
    assert inputs["_agent_semantics_df"][1]["Agent"] == "ExternalPA"


def test_serialize_agent_semantics_input_dict_of_series() -> None:
    inputs = {
        "_agent_semantics_df": {
            "Agent": pd.Series(["Base", "ExternalPA"]),
            "capital_mm": pd.Series([1000.0, 200.0]),
        }
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"
    assert inputs["_agent_semantics_df"][1]["Agent"] == "ExternalPA"


def test_serialize_agent_semantics_input_list_passthrough() -> None:
    inputs = {
        "_agent_semantics_df": [
            {
                "Agent": "Base",
                "capital_mm": 1000.0,
                "implied_capital_share": 1.0,
                "beta_coeff_used": 0.6,
                "alpha_coeff_used": 0.4,
                "financing_coeff_used": -0.6,
                "notes": "",
                "mismatch_flag": False,
            }
        ]
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"


def test_serialize_agent_semantics_input_list_of_series() -> None:
    inputs = {
        "_agent_semantics_df": [
            pd.Series(
                {
                    "Agent": "Base",
                    "capital_mm": 1000.0,
                    "implied_capital_share": 1.0,
                    "beta_coeff_used": 0.6,
                    "alpha_coeff_used": 0.4,
                    "financing_coeff_used": -0.6,
                    "notes": "",
                    "mismatch_flag": False,
                }
            ),
            pd.Series(
                {
                    "Agent": "ExternalPA",
                    "capital_mm": 200.0,
                    "implied_capital_share": 0.2,
                    "beta_coeff_used": 0.2,
                    "alpha_coeff_used": 0.05,
                    "financing_coeff_used": -0.2,
                    "notes": "",
                    "mismatch_flag": False,
                }
            ),
        ]
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"
    assert inputs["_agent_semantics_df"][1]["Agent"] == "ExternalPA"


def test_serialize_agent_semantics_input_tuple() -> None:
    inputs = {
        "_agent_semantics_df": (
            {
                "Agent": "Base",
                "capital_mm": 1000.0,
                "implied_capital_share": 1.0,
                "beta_coeff_used": 0.6,
                "alpha_coeff_used": 0.4,
                "financing_coeff_used": -0.6,
                "notes": "",
                "mismatch_flag": False,
            },
        )
    }

    _serialize_agent_semantics_input(inputs)

    assert isinstance(inputs["_agent_semantics_df"], list)
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"
