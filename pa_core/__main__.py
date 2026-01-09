from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .backend import get_backend
from .config import load_config
from .data import load_index_returns
from .facade import RunOptions, export, run_single
from .units import get_index_series_unit, normalize_index_series


def main(argv: Optional[Sequence[str]] = None) -> None:
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
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    idx_series = load_index_returns(args.index)
    idx_series = normalize_index_series(idx_series, get_index_series_unit())

    options = RunOptions(
        seed=args.seed,
        backend=args.backend,
        legacy_agent_rng=args.legacy_agent_rng,
        return_distribution=args.return_distribution,
        return_t_df=args.return_t_df,
        return_copula=args.return_copula,
    )
    artifacts = run_single(cfg, idx_series, options)
    print(f"[BACKEND] Using backend: {get_backend()}")
    export(artifacts, args.output)
