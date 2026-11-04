#!/usr/bin/env python3
"""Run break/revert mutations for tests/test_coverage_round_2375.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv" / "bin" / "python"
OUT = ROOT / "docs/evidence/issue-2375-closer-repair/mutation-transcripts.md"


def run_pytest(nodeid: str) -> tuple[int, str]:
    proc = subprocess.run(
        [str(PY), "-m", "pytest", "-q", nodeid],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, text


def mutate(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"mutation anchor missing in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1))


def restore(path: Path, backup: Path) -> None:
    shutil.copy2(backup, path)
    backup.unlink()


def case(title: str, nodeid: str, mutations: list[tuple[Path, str, str]]) -> None:
    lines: list[str] = [f"## {title}", "", "```text", f"$ {PY.relative_to(ROOT)} -m pytest -q {nodeid}"]
    backups: list[tuple[Path, Path]] = []
    try:
        for path, old, new in mutations:
            backup = path.with_suffix(path.suffix + ".mutbak")
            shutil.copy2(path, backup)
            backups.append((path, backup))
            mutate(path, old, new)
        code, text = run_pytest(nodeid)
        lines.append(text.rstrip())
        lines.append(f"exit={code}")
        lines.append("```")
        lines.append("")
        lines.append("```text")
        lines.append(f"$ {PY.relative_to(ROOT)} -m pytest -q {nodeid}  # restored")
        for path, backup in backups:
            restore(path, backup)
        code2, text2 = run_pytest(nodeid)
        lines.append(text2.rstrip())
        lines.append(f"exit={code2}")
        lines.append("```")
        lines.append("")
        if code == 0:
            raise SystemExit(f"mutation did not fail test: {nodeid}")
        if code2 != 0:
            raise SystemExit(f"restored test still failing: {nodeid}")
    finally:
        for path, backup in backups:
            if backup.exists():
                restore(path, backup)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = [
        "# Mutation transcripts",
        "",
        "Check date: 2026-11-02 America/Chicago.",
        "",
    ]

    def record(title: str, nodeid: str, mutations: list[tuple[Path, str, str]]) -> None:
        case(title, nodeid, mutations)
        # re-run case to capture - case writes to stdout; refactor

    # Inline implementation writing to chunks
    def run(title: str, nodeid: str, mutations: list[tuple[Path, str, str]]) -> None:
        block: list[str] = [f"## {title}", "", "```text", f"$ .venv/bin/python -m pytest -q {nodeid}"]
        backups: list[tuple[Path, Path]] = []
        try:
            for path, old, new in mutations:
                backup = path.with_suffix(path.suffix + ".mutbak")
                shutil.copy2(path, backup)
                backups.append((path, backup))
                mutate(path, old, new)
            code, text = run_pytest(nodeid)
            block.append(text.rstrip())
            block.append(f"exit={code}")
            block.append("```")
            block.append("")
            block.append("```text")
            block.append(f"$ .venv/bin/python -m pytest -q {nodeid}  # restored")
            for path, backup in backups:
                restore(path, backup)
            code2, text2 = run_pytest(nodeid)
            block.append(text2.rstrip())
            block.append(f"exit={code2}")
            block.append("```")
            block.append("")
            chunks.extend(block)
            if code == 0:
                raise SystemExit(f"mutation did not fail: {nodeid}")
            if code2 != 0:
                raise SystemExit(f"restore failed: {nodeid}")
        finally:
            for path, backup in backups:
                if backup.exists():
                    restore(path, backup)

    compare = ROOT / "pa_core/llm/compare_runs.py"
    cli = ROOT / "pa_core/cli.py"
    bundle = ROOT / "pa_core/run_artifact_bundle.py"
    viz = ROOT / "pa_core/viz/utils.py"

    run(
        "test_compare_run_helpers_reject_unusable_artifacts",
        "tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts",
        [(compare, "    if not isinstance(loaded, dict):\n        return None, prev_manifest_path", "    if not isinstance(loaded, dict):\n        return {}, prev_manifest_path")],
    )
    run(
        "test_compare_runs_explains_missing_prior_summary",
        "tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary",
        [
            (
                compare,
                'raise ValueError(f"No readable prior summary found at expected path: {expected}")',
                "pass  # mutation",
            )
        ],
    )
    run(
        "test_compare_runs_rejects_missing_prior_manifest",
        "tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest",
        [
            (
                compare,
                'raise ValueError("No readable prior manifest found from manifest_data[\'previous_run\'].")',
                "pass  # mutation",
            )
        ],
    )
    run(
        "test_compare_runs_survives_fleet_event_failure",
        "tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure",
        [
            (
                compare,
                "    except Exception:\n        logger.warning(\"Failed to record run-comparison fleet event\", exc_info=True)",
                "    except Exception:\n        raise",
            )
        ],
    )
    run(
        "test_metric_catalog_rejects_non_dataframe_and_empty_delta",
        "tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta",
        [(compare, '        raise TypeError("summary_df must be a pandas DataFrame")', "        return {}")],
    )

    packet_msgs = [
        (
            "RuntimeError",
            "tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]",
            '                print(f"   📋 PowerPoint: {pptx_path}")\n            except RuntimeError as e:\n                print(f"❌ Export packet failed: {e}")\n                _emit_run_end()',
            '                print(f"   📋 PowerPoint: {pptx_path}")\n            except RuntimeError as e:\n                print(f"❌ Export packet failed (mutated): {e}")\n                _emit_run_end()',
        ),
        (
            "ImportError",
            "tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]",
            '                return\n            except (ImportError, ModuleNotFoundError) as e:\n                logger.error(f"Export packet failed due to missing dependency: {e}")\n                print(f"❌ Export packet failed due to missing dependency: {e}")\n                print(\n                    "💡 Install required packages: pip install plotly kaleido openpyxl python-pptx"\n                )',
            '                return\n            except (ImportError, ModuleNotFoundError) as e:\n                logger.error(f"Export packet failed due to missing dependency: {e}")\n                print(f"❌ Export packet failed due to missing dependency (mutated): {e}")\n                print(\n                    "💡 Install required packages: pip install plotly kaleido openpyxl python-pptx"\n                )',
        ),
        (
            "ValueError",
            "tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]",
            '            except (ValueError, TypeError, KeyError) as e:\n                logger.error(f"Export packet failed due to data/config issue: {e}")\n                print(f"❌ Export packet failed due to data or configuration issue: {e}")',
            '            except (ValueError, TypeError, KeyError) as e:\n                logger.error(f"Export packet failed due to data/config issue: {e}")\n                print(f"❌ Export packet failed due to data or configuration issue (mutated): {e}")',
        ),
        (
            "OSError",
            "tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]",
            '            except (OSError, PermissionError) as e:\n                logger.error(f"Export packet failed due to file system issue: {e}")\n                print(f"❌ Export packet failed due to file system issue: {e}")',
            '            except (OSError, PermissionError) as e:\n                logger.error(f"Export packet failed due to file system issue: {e}")\n                print(f"❌ Export packet failed due to file system issue (mutated): {e}")',
        ),
    ]
    for title, nodeid, old, new in packet_msgs:
        run(f"test_single_run_packet_failure ({title})", nodeid, [(cli, old, new)])

    run(
        "test_single_run_gif_conversion_error_is_reported",
        "tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported",
        [
            (
                cli,
                '            except (ValueError, TypeError) as e:\n                print(f"❌ GIF export failed: Data conversion error - {e}")',
                '            except (ValueError, TypeError) as e:\n                print(f"❌ GIF export failed: Data conversion error (mutated) - {e}")',
            )
        ],
    )
    run(
        "test_single_run_dashboard_missing_file_is_reported",
        "tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported",
        [
            (
                cli,
                '            except FileNotFoundError as e:\n                print(f"❌ Dashboard launch failed: {e}")',
                '            except FileNotFoundError as e:\n                print(f"❌ Dashboard launch failed (mutated): {e}")',
            )
        ],
    )
    run(
        "test_single_run_bundle_contains_the_completed_artifacts",
        "tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts",
        [(bundle, "        return True", "        return False",)],
    )

    OUT.write_text("\n".join(chunks))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
