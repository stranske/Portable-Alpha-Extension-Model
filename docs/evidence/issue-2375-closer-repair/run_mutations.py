#!/usr/bin/env python3
"""Run break/revert mutations for tests/test_coverage_round_2375.py."""

from __future__ import annotations

import shutil
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.is_file():
    PY = Path("/opt/anaconda3/bin/python3")
OUT = ROOT / "docs/evidence/issue-2375-closer-repair/current-mutation-transcripts.md"


def run_pytest(nodeid: str) -> tuple[int, str]:
    proc = subprocess.run(
        [str(PY), "-m", "pytest", "-q", "--tb=short", "--disable-warnings", nodeid],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=110,
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


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = [
        "# Mutation transcripts",
        "",
        "Check date: 2026-11-03 America/Chicago.",
        f"Interpreter: {PY}; Python {sys.version.split()[0]}. Production baseline is main 7613ffd9",
        "plus the one-token attribution fix (np -> _np). Historical evidence is retained separately.",
        "",
    ]

    def run(title: str, nodeid: str, mutations: list[tuple[Path, str, str]]) -> None:
        print(f"Mutate / restore: {title}", flush=True)
        command = shlex.join(
            [str(PY), "-m", "pytest", "-q", "--tb=short", "--disable-warnings", nodeid]
        )
        block: list[str] = [f"## {title}", ""]
        for path, old, new in mutations:
            block.extend(
                [
                    f"Production mutation: `{path.relative_to(ROOT)}`",
                    "",
                    "```diff",
                    "-" + old.replace("\n", "\n-"),
                    "+" + new.replace("\n", "\n+"),
                    "```",
                    "",
                ]
            )
        block.extend(
            [
                "Expected: the selected test reports FAILED (exit 1); restored: exit 0.",
                "",
                "```text",
                f"$ {command}",
            ]
        )
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
            block.append(f"$ {command}  # restored")
            for path, backup in backups:
                restore(path, backup)
            code2, text2 = run_pytest(nodeid)
            block.append(text2.rstrip())
            block.append(f"exit={code2}")
            block.append("```")
            block.append("")
            chunks.extend(block)
            OUT.write_text("\n".join(chunks))
            if code != 1 or f"FAILED {nodeid}" not in text:
                raise SystemExit(f"mutation did not fail: {nodeid}")
            if code2 != 0:
                raise SystemExit(f"restore failed: {nodeid}")
        finally:
            for path, backup in backups:
                if backup.exists():
                    restore(path, backup)

    compare = ROOT / "pa_core/llm/compare_runs.py"
    cli = ROOT / "pa_core/cli.py"

    run(
        "test_compare_run_helpers_reject_unusable_artifacts",
        "tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts",
        [
            (
                compare,
                "    if not isinstance(loaded, dict):\n        return None, prev_manifest_path",
                "    if not isinstance(loaded, dict):\n        return {}, prev_manifest_path",
            )
        ],
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
                "raise ValueError(\"No readable prior manifest found from manifest_data['previous_run'].\")",
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
                '    except Exception:\n        logger.warning("Failed to record run-comparison fleet event", exc_info=True)',
                "    except Exception:\n        raise",
            )
        ],
    )
    run(
        "test_metric_catalog_rejects_non_dataframe_and_empty_delta",
        "tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta",
        [
            (
                compare,
                '        raise TypeError("summary_df must be a pandas DataFrame")',
                "        return {}",
            )
        ],
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
        [
            (
                cli,
                '    _record_artifact(flags.save_xlsx or "Outputs.xlsx")',
                "    pass  # omitted completed workbook registration",
            )
        ],
    )

    def cli_case(name: str, parameter: str, old: str, new: str) -> None:
        node = f"tests/test_coverage_round_2375.py::{name}{parameter}"
        run(name + parameter, node, [(cli, old, new)])

    cli_case(
        "test_single_run_rejects_daily_index_before_simulation",
        "",
        '        raise SystemExit(f"Error: {e}") from None',
        '        raise ValueError(f"Error: {e}") from None',
    )
    dashboard_messages = [
        ("exit-code", 'print(f"❌ Dashboard launch failed with exit code {e.returncode}")'),
        ("dependency", 'print(f"❌ Dashboard launch failed: Streamlit not available - {e}")'),
        ("permission", 'print(f"❌ Dashboard launch failed: System/permission error - {e}")'),
    ]
    for parameter, message in dashboard_messages:
        cli_case(
            "test_single_run_dashboard_launch_failure",
            f"[{parameter}]",
            message,
            'print("mutated dashboard diagnosis")',
        )
    cli_case(
        "test_single_run_gif_export_failure",
        "[renderer]",
        '                    print("❌ GIF export failed: Chrome/Chromium required")',
        '                    print("mutated renderer diagnosis")',
    )
    cli_case(
        "test_single_run_gif_export_failure",
        "[encoder]",
        '                    print(f"❌ GIF export failed: {e}")',
        '                    print("mutated encoder diagnosis")',
    )
    cli_case(
        "test_single_run_attribution_fallback_exports_totals",
        "",
        "                mean_month = float(_np.asarray(arr).mean())",
        "                mean_month = float(np.asarray(arr).mean())",
    )
    cli_case(
        "test_single_run_tradeoff_table",
        "[export]",
        "trade_df.head(max(1, args.tradeoff_top)).reset_index(",
        "trade_df.reset_index(",
    )
    cli_case(
        "test_single_run_tradeoff_table",
        "[failure]",
        "Trade-off table computation failed.",
        "Trade-off table warning suppressed.",
    )
    cli_case(
        "test_single_run_registry_is_optional",
        "[registered]",
        'print(f"[REGISTRY] Scenario ID: {scenario_id}")',
        'print("registry ID lost")',
    )
    cli_case(
        "test_single_run_registry_is_optional",
        "[unavailable]",
        'logger.warning(f"Failed to register scenario: {exc}")',
        'logger.debug(f"Failed to register scenario: {exc}")',
    )
    for parameter, message in [
        ("grid-fallback", 'print(f"Optimizer unavailable; using grid fallback ({status}).")'),
        ("failed-fallback", 'print(f"Optimizer failed without grid fallback ({status}).")'),
    ]:
        cli_case(
            "test_single_run_suggestion_eof_aborts",
            f"[{parameter}]",
            message,
            'print("optimizer status lost")',
        )

    OUT.write_text("\n".join(chunks))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
