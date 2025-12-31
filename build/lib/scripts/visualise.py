from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd  # type: ignore

from pa_core.viz import (
    animation,
    corr_heatmap,
    fan,
    html_export,
    path_dist,
    pptx_export,
    risk_return,
    rolling_panel,
    sharpe_ladder,
    surface,
)

logging.basicConfig(level=logging.INFO)

PLOTS = {
    "risk_return": risk_return.make,
    "fan": fan.make,
    "path_dist": path_dist.make,
    "corr_heatmap": corr_heatmap.make,
    "sharpe_ladder": sharpe_ladder.make,
    "rolling_panel": rolling_panel.make,
    "surface": surface.make,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualise simulation outputs")
    parser.add_argument("--plot", choices=PLOTS.keys(), required=True)
    parser.add_argument("--xlsx", required=True, help="Outputs.xlsx file")
    parser.add_argument("--agent", action="append", help="Agent name for paths")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--pptx", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument(
        "--alt-text",
        dest="alt_text",
        help="Alt text for HTML/PPTX exports",
    )
    args = parser.parse_args(argv)

    df_summary = pd.read_excel(args.xlsx, sheet_name="Summary")
    parquet_path = Path(args.xlsx).with_suffix(".parquet")
    df_paths = None
    if parquet_path.exists():
        try:
            df_paths = pd.read_parquet(parquet_path)
        except ImportError:
            csv_path = parquet_path.with_suffix(".csv")
            if csv_path.exists():
                logging.info("pyarrow missing; using CSV path data")
                df_paths = pd.read_csv(csv_path, index_col=0)
            else:
                logging.info("Install pyarrow for Parquet support or provide a matching CSV file")

    if args.plot in {"fan", "path_dist"} and df_paths is None:
        raise FileNotFoundError(parquet_path)

    if args.plot == "corr_heatmap":
        if df_paths is None:
            raise FileNotFoundError(parquet_path)
        paths_map = {"All": df_paths}
        fig = PLOTS[args.plot](paths_map)
    elif args.plot in {"fan", "path_dist"}:
        fig = PLOTS[args.plot](df_paths)
    else:
        fig = PLOTS[args.plot](df_summary)

    base = Path("plots")
    base.mkdir(exist_ok=True)
    stem = base / args.plot
    if args.png:
        try:
            fig.write_image(f"{stem}.png", engine="kaleido")
        except Exception as e:
            logging.warning("PNG export failed: %s", e)
    if args.pdf:
        try:
            fig.write_image(f"{stem}.pdf", engine="kaleido")
        except Exception as e:
            logging.warning("PDF export failed: %s", e)
    if args.pptx:
        pptx_export.save(
            [fig], f"{stem}.pptx", alt_texts=[args.alt_text] if args.alt_text else None
        )
    if args.gif:
        if df_paths is None:
            raise FileNotFoundError(parquet_path)
        anim = animation.make(df_paths)
        try:
            anim.write_gif(f"{stem}.gif")
        except Exception as e:
            logging.warning("GIF export failed: %s", e)
    if args.html:
        html_export.save(fig, f"{stem}.html", alt_text=args.alt_text)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
