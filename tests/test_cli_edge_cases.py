import pandas as pd
import pytest

from pa_core.cli import main
from pa_core.config import ModelConfig


def _minimal_config() -> ModelConfig:
    return ModelConfig(N_SIMULATIONS=1, N_MONTHS=1)


def _patch_core(monkeypatch, cfg: ModelConfig) -> None:
    monkeypatch.setattr("pa_core.config.load_config", lambda _: cfg)
    monkeypatch.setattr("pa_core.backend.resolve_and_set_backend", lambda *_: "numpy")


def test_cli_rejects_non_series_index(monkeypatch):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    bad_index = pd.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.04]})
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _: bad_index)

    with pytest.raises(
        ValueError, match="Index data must be convertible to pandas Series"
    ):
        main(["--config", "cfg.yaml", "--index", "idx.csv"])


def test_cli_rejects_non_series_type(monkeypatch):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _: [0.01, 0.02])

    with pytest.raises(ValueError, match="Index data must be a pandas Series"):
        main(["--config", "cfg.yaml", "--index", "idx.csv"])


def test_cli_suggest_sleeves_empty_exits(monkeypatch, capsys):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _: pd.Series([0.01]))
    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.suggest_sleeve_sizes",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    main(["--config", "cfg.yaml", "--index", "idx.csv", "--suggest-sleeves"])
    captured = capsys.readouterr()
    assert "No feasible sleeve allocations found." in captured.out


def test_cli_suggest_sleeves_blank_selection_aborts(monkeypatch, capsys):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _: pd.Series([0.01]))
    suggestions = pd.DataFrame(
        {
            "external_pa_capital": [10.0],
            "active_ext_capital": [5.0],
            "internal_pa_capital": [2.5],
        }
    )
    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.suggest_sleeve_sizes",
        lambda *_args, **_kwargs: suggestions,
    )
    monkeypatch.setattr("builtins.input", lambda *_args: "")

    main(["--config", "cfg.yaml", "--index", "idx.csv", "--suggest-sleeves"])
    captured = capsys.readouterr()
    assert "Aborting run." in captured.out


def test_cli_suggest_sleeves_invalid_selection_aborts(monkeypatch, capsys):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _: pd.Series([0.01]))
    suggestions = pd.DataFrame(
        {
            "external_pa_capital": [10.0],
            "active_ext_capital": [5.0],
            "internal_pa_capital": [2.5],
        }
    )
    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.suggest_sleeve_sizes",
        lambda *_args, **_kwargs: suggestions,
    )
    monkeypatch.setattr("builtins.input", lambda *_args: "not-a-number")

    main(["--config", "cfg.yaml", "--index", "idx.csv", "--suggest-sleeves"])
    captured = capsys.readouterr()
    assert "Invalid selection. Aborting run." in captured.out


def test_cli_suggest_sleeves_out_of_range_aborts(monkeypatch, capsys):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _: pd.Series([0.01]))
    suggestions = pd.DataFrame(
        {
            "external_pa_capital": [10.0],
            "active_ext_capital": [5.0],
            "internal_pa_capital": [2.5],
        }
    )
    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.suggest_sleeve_sizes",
        lambda *_args, **_kwargs: suggestions,
    )
    monkeypatch.setattr("builtins.input", lambda *_args: "5")

    main(["--config", "cfg.yaml", "--index", "idx.csv", "--suggest-sleeves"])
    captured = capsys.readouterr()
    assert "Invalid selection. Aborting run." in captured.out


def test_cli_prev_manifest_invalid_json_is_ignored(monkeypatch, capsys, tmp_path):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    monkeypatch.setattr(
        "pa_core.data.load_index_returns",
        lambda *_: pd.DataFrame({"only": [0.01, 0.02]}),
    )
    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.suggest_sleeve_sizes",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text("{not json")

    main(
        [
            "--config",
            "cfg.yaml",
            "--index",
            "idx.csv",
            "--suggest-sleeves",
            "--prev-manifest",
            str(bad_manifest),
        ]
    )
    captured = capsys.readouterr()
    assert "No feasible sleeve allocations found." in captured.out


def test_cli_log_json_setup_warning(monkeypatch, caplog):
    cfg = _minimal_config()
    _patch_core(monkeypatch, cfg)
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _: pd.Series([0.01]))
    monkeypatch.setattr(
        "pa_core.logging_utils.setup_json_logging",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")),
    )
    monkeypatch.setattr(
        "pa_core.sleeve_suggestor.suggest_sleeve_sizes",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    caplog.set_level("WARNING")
    main(
        [
            "--config",
            "cfg.yaml",
            "--index",
            "idx.csv",
            "--suggest-sleeves",
            "--log-json",
        ]
    )
    assert "Failed to set up JSON logging" in caplog.text
