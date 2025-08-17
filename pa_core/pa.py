from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

def _convert_csv_to_yaml(csv_path: str, yaml_path: str) -> None:
    """Convert legacy parameters CSV to YAML configuration."""
    # Simple conversion logic to avoid complex imports
    import csv
    import yaml
    
    # Label mapping for CSV conversion
    label_map = {
        "Analysis mode": "analysis_mode",
        "Number of simulations": "N_SIMULATIONS",
        "Number of months": "N_MONTHS",
        "External PA capital (mm)": "external_pa_capital",
        "Active Extension capital (mm)": "active_ext_capital",
        "Internal PA capital (mm)": "internal_pa_capital",
        "Total fund capital (mm)": "total_fund_capital",
        "In-House beta share": "w_beta_H",
        "In-House alpha share": "w_alpha_H",
        "External PA alpha fraction": "theta_extpa",
        "Active share (%)": "active_share",
        "In-House annual return (%)": "mu_H",
        "In-House annual vol (%)": "sigma_H",
        "Alpha-Extension annual return (%)": "mu_E",
        "Alpha-Extension annual vol (%)": "sigma_E",
        "External annual return (%)": "mu_M",
        "External annual vol (%)": "sigma_M",
        "Corr index–In-House": "rho_idx_H",
        "Corr index–Alpha-Extension": "rho_idx_E",
        "Corr index–External": "rho_idx_M",
        "Corr In-House–Alpha-Extension": "rho_H_E",
        "Corr In-House–External": "rho_H_M",
        "Corr Alpha-Extension–External": "rho_E_M",
        "Internal financing mean (monthly %)": "internal_financing_mean_month",
        "Internal financing vol (monthly %)": "internal_financing_sigma_month",
        "Internal monthly spike prob": "internal_spike_prob",
        "Internal spike multiplier": "internal_spike_factor",
        "External PA financing mean (monthly %)": "ext_pa_financing_mean_month",
        "External PA financing vol (monthly %)": "ext_pa_financing_sigma_month",
        "External PA monthly spike prob": "ext_pa_spike_prob",
        "External PA spike multiplier": "ext_pa_spike_factor",
        "Active Ext financing mean (monthly %)": "act_ext_financing_mean_month",
        "Active Ext financing vol (monthly %)": "act_ext_financing_sigma_month",
        "Active Ext monthly spike prob": "act_ext_spike_prob",
        "Active Ext spike multiplier": "act_ext_spike_factor",
        "risk_metrics": "risk_metrics",
    }
    
    data = {}
    
    # Read CSV and convert
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            param_name = row.get('Parameter', '')
            value = row.get('Value', '')
            
            if param_name in label_map:
                key = label_map[param_name]
                
                # Handle different value types
                if key == "risk_metrics" and value:
                    data[key] = [v.strip() for v in value.split(";") if v.strip()]
                    continue
                
                # Try to convert to number
                try:
                    if '.' in str(value):
                        num_val = float(value)
                    else:
                        num_val = int(value)
                    
                    # Convert percentages
                    if "(%)" in param_name and isinstance(num_val, (int, float)):
                        num_val = float(num_val) / 100.0
                    
                    data[key] = num_val
                except (ValueError, TypeError):
                    data[key] = value
    
    # Add default risk_metrics if not present
    if "risk_metrics" not in data:
        data["risk_metrics"] = ["Return", "Risk", "ShortfallProb"]
    
    # Write YAML
    Path(yaml_path).write_text(yaml.safe_dump(data, default_flow_style=False))
    print(f"✓ Converted {csv_path} to {yaml_path}")

def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="pa")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run simulation")
    sub.add_parser("validate", help="Validate scenario YAML")
    
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
    else:  # convert
        # Handle convert command with its specific arguments
        import warnings
        warnings.warn(
            "CSV parameter inputs are deprecated and will be removed in the next release. "
            "Please use YAML configurations going forward.",
            DeprecationWarning,
            stacklevel=2
        )
        
        yaml_path = args.yaml or str(Path(args.csv).with_suffix('.yml'))
        _convert_csv_to_yaml(args.csv, yaml_path)


if __name__ == "__main__":  # pragma: no cover
    main()
