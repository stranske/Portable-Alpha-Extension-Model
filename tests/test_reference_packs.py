"""Contract tests for curated agent reference packs."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TREND_PACK_PATHS = [
    "streamlit_app/components/llm_settings.py",
    "streamlit_app/components/explain_results.py",
    "streamlit_app/components/comparison_llm.py",
    "streamlit_app/pages/2_Model.py",
    "src/trend_analysis/llm/chain.py",
    "src/trend_analysis/llm/tracing.py",
    "streamlit_app/components/analysis_runner.py",
    "streamlit_app/components/nl_operation_viewer.py",
    "src/trend_analysis/llm/nl_logging.py",
    "src/trend_analysis/llm/replay.py",
    "src/trend_analysis/llm/validation.py",
    "docs/natural-language-config.md",
]


def _load_reference_packs_module():
    module_path = ROOT / "scripts" / "reference_packs.py"
    spec = importlib.util.spec_from_file_location("reference_packs", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reference_packs module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trend_streamlit_llm_pack_is_curated_and_pinned() -> None:
    reference_packs = _load_reference_packs_module()

    snapshot = reference_packs.load_reference_packs(ROOT)

    assert snapshot.exists
    assert len(snapshot.packs) == 1
    pack = snapshot.packs[0]
    assert pack.name == "trend_streamlit_llm"
    assert pack.repo == "stranske/Trend_Model_Project"
    assert re.fullmatch(r"[0-9a-f]{40}", pack.ref)
    assert pack.paths == EXPECTED_TREND_PACK_PATHS
    assert not any(path.startswith(".reference/") for path in pack.paths)


def test_reference_pack_checkout_plan_uses_runtime_reference_directory() -> None:
    reference_packs = _load_reference_packs_module()

    snapshot = reference_packs.load_reference_packs(ROOT)
    checkout_plan = reference_packs.build_checkout_plan(snapshot.packs)

    assert len(checkout_plan) == 1
    assert checkout_plan[0].checkout_path == ".reference/trend_streamlit_llm"
    assert checkout_plan[0].paths == EXPECTED_TREND_PACK_PATHS


def test_reference_directory_is_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", ".reference/trend_streamlit_llm/example.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ".reference/trend_streamlit_llm/example.py"
