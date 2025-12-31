#!/usr/bin/env python3
"""
Headless capture of Streamlit Wizard screenshots.

Requirements:
  pip install playwright
  playwright install chromium

This script will:
  1) start the Streamlit dashboard on a given port (default 8501)
  2) wait for the server to be reachable
  3) open Chromium headless and navigate to Scenario Wizard
  4) capture screenshots into the output directory

Usage:
  python scripts/capture_wizard.py --out-dir docs/images --port 8501

If Playwright is not installed, the script will exit with instructions.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _wait_for_port(url: str, timeout: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as resp:  # noqa: S310 - local URL
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def _start_streamlit(port: int) -> subprocess.Popen[str]:
    # Ensure headless and consistent port
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.headless=true",
        f"--server.port={port}",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _capture_screens(port: int, out_dir: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        print(
            "Playwright is required. Install with: pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        sys.exit(2)

    url = f"http://localhost:{port}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()
        page.goto(url, wait_until="load")

        # Click the Scenario Wizard page link in the sidebar if present
        try:
            page.get_by_role("link", name="Scenario Wizard").click()
        except Exception:
            # Fallback: try text selector
            try:
                page.get_by_text("Scenario Wizard", exact=True).click()
            except Exception:
                pass

        page.wait_for_timeout(1000)
        out_dir.mkdir(parents=True, exist_ok=True)
        landing = out_dir / "wizard_landing.png"
        page.screenshot(path=str(landing), full_page=True)

        # Try to move to Review & Run if a Next or Continue button exists
        progressed = False
        for label in ("Next", "Continue", "Review & Run"):
            try:
                page.get_by_role("button", name=label).click()
                progressed = True
            except Exception:
                pass
        if progressed:
            page.wait_for_timeout(800)
            review = out_dir / "wizard_review.png"
            page.screenshot(path=str(review), full_page=True)

        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless capture for Scenario Wizard screenshots")
    ap.add_argument("--port", type=int, default=8501, help="Dashboard port")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/images"),
        help="Output directory for images",
    )
    args = ap.parse_args()

    proc = _start_streamlit(args.port)
    try:
        _wait_for_port(f"http://localhost:{args.port}", timeout=30.0)
        _capture_screens(args.port, args.out_dir)
        print(f"Saved screenshots to {args.out_dir}")
        return 0
    finally:
        _stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
