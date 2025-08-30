from __future__ import annotations

import argparse
import csv
import sys
import yaml
from pathlib import Path
from typing import Sequence

def _convert_csv_to_yaml(csv_path: str, yaml_path: str) -> None:
    """Convert legacy parameters CSV to YAML configuration."""
    
    # Import here to avoid circular imports
    from .config import get_field_mappings
    
    # Get the field mappings from the ModelConfig Pydantic model
    label_map = get_field_mappings()
    
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
                num_val = None  # Initialize to avoid undefined variable
                try:
                    if '.' in str(value):
                        num_val = float(value)
                    else:
                        num_val = int(value)
                    
                    # Convert percentages
                    if "(%)" in param_name:
                        num_val = float(num_val) / 100.0
                    
                    data[key] = num_val
                except (ValueError, TypeError):
                    # Try float, else leave as string
                    try:
                        num_val = float(value)
                        # Convert percentages
                        if "(%)" in param_name:
                            num_val = float(num_val) / 100.0
                        data[key] = num_val
                    except (ValueError, TypeError):
                        # Keep as string
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
