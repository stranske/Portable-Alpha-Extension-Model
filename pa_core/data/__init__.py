import csv
from pathlib import Path
from tkinter import filedialog, Tk
from ..backend import xp as np
import pandas as pd

__all__ = [
    "select_csv_file",
    "load_parameters",
    "get_num",
    "build_range",
    "build_range_int",
    "load_index_returns",
]

def select_csv_file():
    """Pop up a file picker to choose a CSV file and return its path."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not file_path:
        raise FileNotFoundError("No file selected.")
    return Path(file_path)


def load_parameters(csv_filepath, label_map):
    """Parse a parameters CSV using a label mapping."""
    params = {}
    lines = Path(csv_filepath).read_text(encoding="utf-8").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Parameter,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No header row starting with 'Parameter,' found in {csv_filepath}")
    header_and_data = lines[header_idx:]
    reader = csv.DictReader(header_and_data)
    for row in reader:
        friendly_key = row.get("Parameter", "").strip()
        if not friendly_key or friendly_key not in label_map:
            continue
        internal_key = label_map[friendly_key]
        raw_val = row.get("Value", "").strip()
        if ";" in raw_val:
            parts = [p.strip() for p in raw_val.split(";") if p.strip() != ""]
            parsed_list = []
            for p in parts:
                try:
                    if "." in p:
                        parsed_list.append(float(p))
                    else:
                        parsed_list.append(int(p))
                except ValueError:
                    parsed_list.append(p)
            params[internal_key] = parsed_list
        else:
            try:
                params[internal_key] = int(raw_val)
            except ValueError:
                try:
                    params[internal_key] = float(raw_val)
                except ValueError:
                    params[internal_key] = raw_val
    return params


def get_num(raw_params, key, default):
    """Return raw_params[key] if numeric else default."""
    v = raw_params.get(key, None)
    if isinstance(v, (int, float)):
        return v
    return default


def build_range(raw_params, key_base, default_midpoint):
    """Return a list of decimals for a percent range or fallback."""
    k_min = get_num(raw_params, f"{key_base}_min", None)
    k_max = get_num(raw_params, f"{key_base}_max", None)
    k_step = get_num(raw_params, f"{key_base}_step", None)
    if (k_min is not None) and (k_max is not None):
        step = k_step if (k_step is not None) else (k_max - k_min)
        if step <= 0:
            raise RuntimeError(f"Step for '{key_base}' must be positive.")
        start = k_min / 100.0
        stop = k_max / 100.0
        stepd = step / 100.0
        arr = np.arange(start, stop + 1e-9, stepd)
        return list(arr)
    flat_list = raw_params.get(f"{key_base}_list", None)
    if isinstance(flat_list, list):
        return flat_list
    return [default_midpoint]


def build_range_int(raw_params, key_base, default_midpoint):
    """Integer version of ``build_range``."""
    k_min = get_num(raw_params, f"{key_base}_min", None)
    k_max = get_num(raw_params, f"{key_base}_max", None)
    k_step = get_num(raw_params, f"{key_base}_step", None)
    if (k_min is not None) and (k_max is not None):
        step = int(k_step if (k_step is not None) else (k_max - k_min))
        if step <= 0:
            raise RuntimeError(f"Step for '{key_base}' must be positive.")
        return list(range(int(k_min), int(k_max) + 1, step))
    flat_list = raw_params.get(f"{key_base}_list", None)
    if isinstance(flat_list, list):
        return flat_list
    return [default_midpoint]


def load_index_returns(csv_path):
    """Load a CSV of monthly index returns into a Series."""
    csv_path = Path(csv_path)
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"Index CSV not found at {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if "Date" not in df.columns:
        raise ValueError(f"'Date' column is missing from {csv_path}")
    if "Monthly_TR" in df.columns:
        col = "Monthly_TR"
    elif "Return" in df.columns:
        col = "Return"
    else:
        raise ValueError(f"CSV must contain 'Monthly_TR' or 'Return'; found: {df.columns.tolist()}")
    df = df.sort_values("Date").reset_index(drop=True)
    df.set_index("Date", inplace=True)
    series = df[col].dropna().copy()
    series.index = pd.to_datetime(series.index)
    return series
