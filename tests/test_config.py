from pa_core.config import ModelConfig, load_config
import yaml


def test_load_yaml(tmp_path):
    data = {
        "N_SIMULATIONS": 10,
        "N_MONTHS": 6,
        "mu_H": 0.04,
        "sigma_H": 0.01,
    }
    path = tmp_path / "conf.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    assert isinstance(cfg, ModelConfig)
    assert cfg.N_SIMULATIONS == 10
    assert cfg.N_MONTHS == 6
