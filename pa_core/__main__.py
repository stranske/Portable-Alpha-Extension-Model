from __future__ import annotations

import argparse
import warnings
from typing import Literal, Optional, Sequence, cast

from .backend import get_backend
from .config import load_config
from .data import load_index_returns
from .facade import RunOptions, export, run_single
from .units import get_index_series_unit, normalize_index_series
from .validators import select_vol_regime_sigma


def main(argv: Optional[Sequence[str]] = None) -> None:
    # Legacy entry point for `python -m pa_core`; warn and keep args aligned with `pa run`.
    warnings.warn(
        "Direct invocation via `python -m pa_core` is deprecated; use `pa run` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Keep legacy argument parsing stable for backward compatibility with documented behavior and
    # to mirror `pa run` flags/output expectations captured in tests/expected_cli_outputs.py (for
    # example MAIN_BACKEND_STDOUT) and any golden output fixtures. This parser is intentionally
    # smaller than pa_core.cli.main, but shared flags must remain consistent so the legacy entry
    # point emits the same anchored lines asserted by CLI tests.
    parser = argparse.ArgumentParser(description="Portable Alpha simulation")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "YAML config file (set financing_mode to broadcast for shared paths or "
            "per_path for independent draws)"
        ),
    )
    parser.add_argument("--index", required=True, help="Index returns CSV")
    parser.add_argument("--output", default="Outputs.xlsx", help="Output workbook")
    parser.add_argument(
        "--backend",
        choices=["numpy"],
        help="Computation backend (numpy only; cupy/GPU acceleration is not available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulations",
    )
    parser.add_argument(
        "--legacy-agent-rng",
        action="store_true",
        help="Use legacy order-dependent agent RNG streams (defaults to stable name-based streams)",
    )
    parser.add_argument(
        "--return-distribution",
        choices=["normal", "student_t"],
        help="Override return distribution (normal or student_t). student_t adds heavier tails and more compute",
    )
    parser.add_argument(
        "--return-t-df",
        type=float,
        help="Override Student-t degrees of freedom (requires student_t; lower df => heavier tails)",
    )
    parser.add_argument(
        "--return-copula",
        choices=["gaussian", "t"],
        help="Override return copula (gaussian or t). t adds tail dependence and extra compute",
    )
    # argparse returns a Namespace; translate parsed flags into RunOptions so this legacy entry
    # point delegates through the same facade pipeline as `pa run`, keeping output and behavior
    # aligned with constants in tests/expected_cli_outputs.py (e.g., MAIN_BACKEND_STDOUT).
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    idx_series = load_index_returns(args.index)
    idx_series = normalize_index_series(idx_series, get_index_series_unit())

    # Validate vol_regime before passing to run_single
    vol_regime_value = getattr(cfg, "vol_regime", "single")
    if vol_regime_value not in ("single", "two_state"):
        raise ValueError(f"vol_regime must be 'single' or 'two_state', got {vol_regime_value!r}")
    vol_regime = cast(Literal["single", "two_state"], vol_regime_value)
    vol_regime_window = getattr(cfg, "vol_regime_window", 12)
    select_vol_regime_sigma(
        idx_series,
        regime=vol_regime,
        window=vol_regime_window,
    )

    # Delegate through the same facade path as `pa run` to keep outputs aligned with
    # external expected-output artifacts (tests/expected_cli_outputs.py, golden files).
    options = RunOptions(
        seed=args.seed,
        backend=args.backend,
        legacy_agent_rng=args.legacy_agent_rng,
        return_distribution=args.return_distribution,
        return_t_df=args.return_t_df,
        return_copula=args.return_copula,
    )
    # Delegate the simulation to the facade, then pass artifacts to export.
    artifacts = run_single(cfg, idx_series, options)
    # Output line is asserted by CLI tests via tests/expected_cli_outputs.py::MAIN_BACKEND_STDOUT.
    print(f"[BACKEND] Using backend: {get_backend()}")
    export(artifacts, args.output)
