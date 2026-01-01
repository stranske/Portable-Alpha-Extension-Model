from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast


def _require_yaml() -> Any:
    """Import PyYAML lazily to keep `pa --help` lightweight."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyYAML is required for this command. Install it with `pip install pyyaml`."
        ) from exc
    return yaml


def _convert_csv_to_yaml(csv_path: str, yaml_path: str) -> None:
    """Convert legacy parameters CSV to YAML configuration."""

    # Import here to avoid circular imports
    from .config import get_field_mappings

    yaml = _require_yaml()

    # Get the field mappings from the ModelConfig Pydantic model
    label_map = get_field_mappings()

    data: dict[str, Any] = {}

    # Read CSV and convert
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            param_name = row.get("Parameter", "")
            value = row.get("Value", "")

            if param_name in label_map:
                key = label_map[param_name]

                # Handle different value types
                if key == "risk_metrics" and value:
                    data[key] = [v.strip() for v in value.split(";") if v.strip()]
                    continue

                # Try to convert to number
                num_val: float | int | str | None = None
                try:
                    if "." in str(value):
                        num_val = float(value)
                    else:
                        num_val = int(value)
                except (ValueError, TypeError):
                    # Try int first, then float, else leave as string
                    try:
                        num_val = int(value)
                    except (ValueError, TypeError):
                        try:
                            num_val = float(value)
                        except (ValueError, TypeError):
                            num_val = value

                # Convert percentages
                if "(%)" in param_name and isinstance(num_val, (int, float)):
                    num_val = float(num_val) / 100.0

                data[key] = num_val

    # Add default risk_metrics if not present
    if "risk_metrics" not in data:
        data["risk_metrics"] = ["Return", "Risk", "ShortfallProb"]

    # Write YAML
    Path(yaml_path).write_text(yaml.safe_dump(data, default_flow_style=False))
    print(f"✓ Converted {csv_path} to {yaml_path}")


def _load_calibration_overrides(path: str | Path) -> Mapping[str, Any]:
    """Return calibration overrides from a mapping template, if present."""

    yaml = _require_yaml()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        return {}
    calib = data.get("calibration")
    if isinstance(calib, dict):
        return calib
    return data


def _coerce_calibration_setting(value: Any, *, allowed: set[str], label: str) -> str | None:
    if value is None:
        return None
    value_str = str(value)
    if value_str not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)}")
    return value_str


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="pa")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run simulation")
    sub.add_parser("validate", help="Validate scenario YAML")
    calibrate_parser = sub.add_parser("calibrate", help="Calibrate parameters from returns data")
    io_group = calibrate_parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument(
        "--input", help="Input CSV/XLSX file with returns or prices (asset library output)"
    )
    io_group.add_argument(
        "--returns",
        help="Input CSV/XLSX file with manager returns (params output)",
    )
    calibrate_parser.add_argument("--index-id", help="Asset id to use as market index")
    calibrate_parser.add_argument(
        "--output",
        default="asset_library.yaml",
        help="Output YAML (default: asset_library.yaml)",
    )
    calibrate_parser.add_argument(
        "--mapping",
        help="Optional YAML mapping template for DataImportAgent",
    )
    calibrate_parser.add_argument(
        "--min-obs",
        type=int,
        default=36,
        help="Minimum observations per id when no mapping template is provided",
    )
    calibrate_parser.add_argument(
        "--date-col",
        default="Date",
        help="Date column for manager returns input (default: Date)",
    )
    calibrate_parser.add_argument(
        "--alpha-ids",
        nargs=3,
        metavar=("H", "E", "M"),
        help="Series ids for H/E/M alpha streams when using --returns",
    )
    calibrate_parser.add_argument(
        "--mean-shrinkage",
        type=float,
        default=0.2,
        help="Shrinkage intensity for means (0-1)",
    )
    calibrate_parser.add_argument(
        "--corr-shrinkage",
        type=float,
        default=0.2,
        help="Shrinkage intensity for correlations (0-1)",
    )
    calibrate_parser.add_argument(
        "--corr-target",
        choices=["identity", "constant"],
        default="identity",
        help="Correlation shrinkage target",
    )
    calibrate_parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for parameter intervals",
    )
    calibrate_parser.add_argument(
        "--scenario-output",
        help="Optional scenario YAML output from calibrated parameters",
    )
    calibrate_parser.add_argument(
        "--cov-shrinkage",
        choices=["none", "ledoit_wolf"],
        default=None,
        help="Covariance shrinkage mode",
    )
    calibrate_parser.add_argument(
        "--vol-regime",
        choices=["single", "two_state"],
        default=None,
        help="Volatility regime selection",
    )
    calibrate_parser.add_argument(
        "--vol-regime-window",
        type=int,
        default=None,
        help="Recent window length (months) for two-state regime",
    )

    # Add convert subparser with arguments
    convert_parser = sub.add_parser("convert", help="Convert legacy parameters CSV to YAML")
    convert_parser.add_argument("csv", help="Input parameters CSV file")
    convert_parser.add_argument("yaml", nargs="?", help="Output YAML file (default: <input>.yml)")

    args, remaining = parser.parse_known_args(argv)
    if args.command == "run":
        from .cli import main as run_main

        run_main(list(remaining))
    elif args.command == "validate":
        from .validate import main as validate_main

        validate_main(list(remaining))
    elif args.command == "calibrate":
        if args.returns:
            import pandas as pd

            from .calibration import (
                build_calibration_report,
                calibrate_returns,
                infer_series_ids,
            )
            from .schema import save_scenario

            returns_path = Path(args.returns)
            if returns_path.suffix.lower() == ".csv":
                raw = pd.read_csv(returns_path)
            elif returns_path.suffix.lower() in {".xlsx", ".xls"}:
                raw = pd.read_excel(returns_path)
            else:
                raise ValueError("unsupported returns file type")

            if args.date_col in raw.columns:
                raw[args.date_col] = pd.to_datetime(raw[args.date_col])
                returns_df = raw.set_index(args.date_col)
            else:
                returns_df = raw.copy()

            index_id, alpha_ids = infer_series_ids(
                returns_df.columns, index_id=args.index_id, alpha_ids=args.alpha_ids
            )

            result = calibrate_returns(
                returns_df,
                index_id=index_id,
                mean_shrinkage=args.mean_shrinkage,
                corr_shrinkage=args.corr_shrinkage,
                corr_target=cast(Literal["identity", "constant"], args.corr_target),
                confidence_level=args.ci_level,
            )
            alpha_map = {"H": alpha_ids[0], "E": alpha_ids[1], "M": alpha_ids[2]}
            payload = {
                "N_SIMULATIONS": 1000,
                "N_MONTHS": 12,
            }
            payload.update(result.to_model_config(alpha_map=alpha_map))
            report = build_calibration_report(result, alpha_map=alpha_map)
            payload["calibration"] = report

            yaml = _require_yaml()
            Path(args.output).write_text(
                yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
            )
            if args.scenario_output:
                save_scenario(result.to_scenario(), args.scenario_output)
            print(f"✓ Wrote calibrated parameters to {args.output}")
            if args.scenario_output:
                print(f"✓ Wrote scenario YAML to {args.scenario_output}")
            return

        from .data import CalibrationAgent, DataImportAgent

        calibration_overrides: Mapping[str, Any] = {}
        if args.mapping:
            importer = DataImportAgent.from_template(args.mapping)
            calibration_overrides = _load_calibration_overrides(args.mapping)
        else:
            importer = DataImportAgent(min_obs=int(args.min_obs))
        if args.index_id is None:
            raise ValueError("index-id is required when using --input")
        cov_shrinkage = cast(
            Literal["none", "ledoit_wolf"] | None,
            _coerce_calibration_setting(
                args.cov_shrinkage or calibration_overrides.get("covariance_shrinkage"),
                allowed={"none", "ledoit_wolf"},
                label="covariance_shrinkage",
            ),
        )
        vol_regime = cast(
            Literal["single", "two_state"] | None,
            _coerce_calibration_setting(
                args.vol_regime or calibration_overrides.get("vol_regime"),
                allowed={"single", "two_state"},
                label="vol_regime",
            ),
        )
        vol_regime_window = (
            args.vol_regime_window
            if args.vol_regime_window is not None
            else calibration_overrides.get("vol_regime_window")
        )
        if vol_regime_window is None:
            vol_regime_window = 12
        df = importer.load(args.input)
        cov_shrinkage_value: Literal["none", "ledoit_wolf"] = (
            cov_shrinkage if cov_shrinkage is not None else "none"
        )
        vol_regime_value: Literal["single", "two_state"] = (
            vol_regime if vol_regime is not None else "single"
        )
        calib = CalibrationAgent(
            min_obs=importer.min_obs,
            covariance_shrinkage=cov_shrinkage_value,
            vol_regime=vol_regime_value,
            vol_regime_window=int(vol_regime_window),
        )
        result = calib.calibrate(df, index_id=args.index_id)
        calib.to_yaml(result, args.output)
        print(f"✓ Wrote asset library to {args.output}")
    else:  # convert
        # Handle convert command with its specific arguments
        import warnings

        warnings.warn(
            "CSV parameter inputs are deprecated and will be removed in the next release. "
            "Please use YAML configurations going forward.",
            DeprecationWarning,
            stacklevel=2,
        )

        yaml_path = args.yaml or str(Path(args.csv).with_suffix(".yml"))
        _convert_csv_to_yaml(args.csv, yaml_path)


if __name__ == "__main__":  # pragma: no cover
    main()
