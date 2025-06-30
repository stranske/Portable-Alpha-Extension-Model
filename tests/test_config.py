from pa_core.config import ModelConfig, load_config
import yaml


def test_load_yaml(tmp_path):
    data = {
        "N_SIMULATIONS": 10,
        "N_MONTHS": 6,
        "mu_H": 0.04,
        "sigma_H": 0.01,
        "external_pa_capital": 100.0,
        "active_ext_capital": 200.0,
        "internal_pa_capital": 300.0,
        "total_fund_capital": 1000.0,
    }
    path = tmp_path / "conf.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    assert isinstance(cfg, ModelConfig)
    assert cfg.N_SIMULATIONS == 10
    assert cfg.N_MONTHS == 6
    assert cfg.external_pa_capital == 100.0
    assert cfg.active_ext_capital == 200.0
    assert cfg.internal_pa_capital == 300.0
    assert cfg.total_fund_capital == 1000.0


def test_load_dict():
    data = {"N_SIMULATIONS": 5, "N_MONTHS": 3, "mu_H": 0.05}
    cfg = load_config(data)
    assert cfg.N_SIMULATIONS == 5
    assert cfg.N_MONTHS == 3
    assert cfg.mu_H == 0.05


def test_invalid_capital(tmp_path):
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "external_pa_capital": 800.0,
        "active_ext_capital": 800.0,
        "internal_pa_capital": 800.0,
        "total_fund_capital": 1000.0,
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    try:
        load_config(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected validation failure")
