import numpy as np

from pa_core.config import ModelConfig, RegimeConfig
from pa_core.random import spawn_rngs
from pa_core.sim.paths import draw_joint_returns
from pa_core.sim.regimes import (
    build_regime_draw_params,
    resolve_regime_start,
    simulate_regime_paths,
)


def test_simulate_regime_paths_alternates() -> None:
    rng = spawn_rngs(123, 1)[0]
    transition = [[0.0, 1.0], [1.0, 0.0]]
    paths = simulate_regime_paths(
        n_sim=1,
        n_months=4,
        transition=transition,
        start_state=0,
        rng=rng,
    )
    assert paths.tolist() == [[0, 1, 0, 1]]


def test_simulate_regime_paths_respects_transition_probabilities() -> None:
    rng = spawn_rngs(42, 1)[0]
    transition = [[0.25, 0.75], [0.1, 0.9]]
    paths = simulate_regime_paths(
        n_sim=5000,
        n_months=2,
        transition=transition,
        start_state=0,
        rng=rng,
    )
    next_states = paths[:, 1]
    prob_to_state_1 = float((next_states == 1).mean())
    assert abs(prob_to_state_1 - 0.75) < 0.03


def test_regime_switching_increases_corr_and_vol() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=2000,
        N_MONTHS=24,
        financing_mode="broadcast",
        return_unit="monthly",
        sigma_H=0.02,
        sigma_E=0.02,
        sigma_M=0.02,
        rho_idx_H=0.1,
        rho_idx_E=0.1,
        rho_idx_M=0.1,
        rho_H_E=0.2,
        rho_H_M=0.2,
        rho_E_M=0.2,
        regimes=[
            RegimeConfig(name="calm"),
            RegimeConfig(
                name="stress",
                idx_sigma_multiplier=2.0,
                sigma_H=0.05,
                sigma_E=0.05,
                sigma_M=0.05,
                rho_idx_H=0.8,
                rho_idx_E=0.8,
                rho_idx_M=0.8,
                rho_H_E=0.85,
                rho_H_M=0.85,
                rho_E_M=0.85,
            ),
        ],
        regime_transition=[[0.9, 0.1], [0.2, 0.8]],
        regime_start="calm",
    )
    params, _labels = build_regime_draw_params(
        cfg,
        mu_idx=0.0,
        idx_sigma=0.015,
        n_samples=120,
    )
    rng_regime = spawn_rngs(7, 1)[0]
    paths = simulate_regime_paths(
        n_sim=cfg.N_SIMULATIONS,
        n_months=cfg.N_MONTHS,
        transition=cfg.regime_transition or [],
        start_state=resolve_regime_start(cfg),
        rng=rng_regime,
    )
    rng_returns = spawn_rngs(11, 1)[0]
    _r_beta, r_H, r_E, _r_M = draw_joint_returns(
        n_months=cfg.N_MONTHS,
        n_sim=cfg.N_SIMULATIONS,
        params=params[0],
        rng=rng_returns,
        regime_paths=paths,
        regime_params=params,
    )

    calm_mask = paths == 0
    stress_mask = paths == 1
    h_calm = r_H[calm_mask]
    e_calm = r_E[calm_mask]
    h_stress = r_H[stress_mask]
    e_stress = r_E[stress_mask]

    calm_corr = float(np.corrcoef(h_calm, e_calm)[0, 1])
    stress_corr = float(np.corrcoef(h_stress, e_stress)[0, 1])
    calm_vol = float(np.std(h_calm))
    stress_vol = float(np.std(h_stress))

    assert stress_corr > calm_corr
    assert stress_vol > calm_vol


def test_regime_switching_seed_is_deterministic() -> None:
    cfg = ModelConfig(
        N_SIMULATIONS=128,
        N_MONTHS=12,
        financing_mode="broadcast",
        return_unit="monthly",
        sigma_H=0.02,
        sigma_E=0.02,
        sigma_M=0.02,
        rho_idx_H=0.1,
        rho_idx_E=0.1,
        rho_idx_M=0.1,
        rho_H_E=0.2,
        rho_H_M=0.2,
        rho_E_M=0.2,
        regimes=[
            RegimeConfig(name="calm"),
            RegimeConfig(
                name="stress",
                idx_sigma_multiplier=1.5,
                sigma_H=0.04,
                sigma_E=0.04,
                sigma_M=0.04,
                rho_idx_H=0.7,
                rho_idx_E=0.7,
                rho_idx_M=0.7,
                rho_H_E=0.75,
                rho_H_M=0.75,
                rho_E_M=0.75,
            ),
        ],
        regime_transition=[[0.8, 0.2], [0.2, 0.8]],
        regime_start="calm",
    )
    params, _labels = build_regime_draw_params(
        cfg,
        mu_idx=0.0,
        idx_sigma=0.015,
        n_samples=120,
    )

    def _run(seed: int) -> tuple[np.ndarray, np.ndarray]:
        paths = simulate_regime_paths(
            n_sim=cfg.N_SIMULATIONS,
            n_months=cfg.N_MONTHS,
            transition=cfg.regime_transition or [],
            start_state=resolve_regime_start(cfg),
            seed=seed,
        )
        _r_beta, r_H, r_E, _r_M = draw_joint_returns(
            n_months=cfg.N_MONTHS,
            n_sim=cfg.N_SIMULATIONS,
            params=params[0],
            seed=seed + 1,
            regime_paths=paths,
            regime_params=params,
        )
        return paths, np.stack([r_H, r_E])

    paths_a, returns_a = _run(101)
    paths_b, returns_b = _run(202)
    paths_c, returns_c = _run(101)

    assert np.array_equal(paths_a, paths_c)
    assert np.allclose(returns_a, returns_c)
    assert not np.array_equal(paths_a, paths_b)
    assert not np.allclose(returns_a, returns_b)
