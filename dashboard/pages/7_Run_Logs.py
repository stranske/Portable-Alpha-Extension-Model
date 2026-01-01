from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from pa_core.contracts import (
    MANIFEST_FILENAME,
    RUN_END_FILENAME,
    RUN_LOG_FILENAME,
    RUNS_DIR_NAME,
    manifest_path_from_run_end,
)

st.set_page_config(page_title="Run Logs", page_icon="🧾")
st.title("Run Logs 🧾")

runs_dir = Path(RUNS_DIR_NAME)
if not runs_dir.exists():
    st.info("No runs directory found yet. Launch a run with --log-json to create logs.")
    st.stop()

run_ids = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
if not run_ids:
    st.info("No run directories available.")
    st.stop()

selected = st.selectbox("Select a run:", run_ids)
run_path = runs_dir / selected
log_file = run_path / RUN_LOG_FILENAME
run_end_file = run_path / RUN_END_FILENAME

cols = st.columns(2)
with cols[0]:
    st.subheader("Log preview")
    if log_file.exists():
        try:
            lines = (
                log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-500:]
            )
            for line in lines:
                # Try to pretty print JSON; fallback to raw line
                try:
                    obj = json.loads(line)
                    st.code(json.dumps(obj, indent=2), language="json")
                except Exception:
                    st.text(line)
        except Exception as e:
            st.error(f"Failed to read log: {e}")
    else:
        st.warning("run.log not found for this run.")

with cols[1]:
    st.subheader("Manifest link")
    # Prefer manifest path recorded in run_end.json for this run.
    found_manifest = manifest_path_from_run_end(run_end_file)
    if found_manifest is None or not found_manifest.exists():
        # Fallback: search nearby manifests in project root.
        for cand in Path.cwd().glob(MANIFEST_FILENAME):
            found_manifest = cand
            break
    if found_manifest is not None and found_manifest.exists():
        try:
            data = json.loads(found_manifest.read_text())
            st.write(found_manifest.name)
            st.code(json.dumps(data, indent=2), language="json")
        except Exception as e:
            st.error(f"Failed to read manifest: {e}")
    else:
        st.info("Manifest not found. Check the run's output directory.")
