"""Reconcile a completed pytest coverage capture without hand-copied totals."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def validate_manifest(manifest_path: Path) -> None:
    """Require every declared evidence artifact to exist with its recorded bytes."""
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        path = manifest_path.parent / name
        if not path.is_file():
            raise ValueError(f"Missing evidence artifact: {name}")
        content = path.read_bytes()
        if len(content) != expected["bytes"]:
            raise ValueError(f"Evidence artifact size mismatch: {name}")
        if hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Evidence artifact hash mismatch: {name}")


def summarize(coverage_path: Path, junit_path: Path, returncode: int) -> dict:
    """Reject partial runs and inconsistent totals before reporting evidence."""
    if returncode not in (0, 1):
        raise ValueError(f"Incomplete pytest run (exit {returncode})")
    coverage = json.loads(coverage_path.read_text())
    modules = []
    for name, data in sorted(coverage["files"].items()):
        summary = data["summary"]
        modules.append(
            {
                "path": name,
                "covered_lines": summary["covered_lines"],
                "num_statements": summary["num_statements"],
                "missing_lines": summary["missing_lines"],
                "percent_covered": summary["percent_covered"],
                "missing_line_numbers": data["missing_lines"],
            }
        )
    totals = coverage["totals"]
    for key in ("covered_lines", "num_statements", "missing_lines"):
        if sum(module[key] for module in modules) != totals[key]:
            raise ValueError(f"Coverage file summaries disagree with totals.{key}")
    if totals["covered_lines"] + totals["missing_lines"] != totals["num_statements"]:
        raise ValueError("Covered plus missing statements disagree with total statements")

    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    nodes = root.findall(".//testcase")
    if not suites or not nodes or sum(int(s.get("tests", "0")) for s in suites) != len(nodes):
        raise ValueError("Incomplete JUnit testcase count")
    counts = dict(passed=0, failed=0, errors=0, skipped=0)
    failures = []
    for node in nodes:
        outcome = "passed"
        for tag, label in (("failure", "failed"), ("error", "errors"), ("skipped", "skipped")):
            detail = node.find(tag)
            if detail is not None:
                outcome = label
                if label != "skipped":
                    failures.append(
                        {
                            "node": node.get("classname", "").replace(".", "/")
                            + ".py::"
                            + node.get("name", ""),
                            "outcome": label,
                            "message": detail.get("message", ""),
                        }
                    )
                break
        counts[outcome] += 1
    for attribute, label in (("failures", "failed"), ("errors", "errors"), ("skipped", "skipped")):
        if sum(int(s.get(attribute, "0")) for s in suites) != counts[label]:
            raise ValueError(f"JUnit summary disagrees with testcase {attribute}")
    if (returncode == 0) != (counts["failed"] + counts["errors"] == 0):
        raise ValueError("Pytest exit disagrees with JUnit outcomes")
    return {
        "exit": returncode,
        "counts": counts,
        "totals": totals,
        "failures": failures,
        "modules": modules,
        "largest_gaps": sorted(modules, key=lambda module: -module["missing_lines"])[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, help="Verify retained artifacts before summarizing"
    )
    args = parser.parse_args()
    if args.manifest is not None:
        validate_manifest(args.manifest)
    report = summarize(args.coverage, args.junit, args.exit_code)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    totals = report["totals"]
    print(
        f"{totals['covered_lines']} / {totals['num_statements']} statements "
        f"({totals['percent_covered']}%); {report['counts']}"
    )


if __name__ == "__main__":
    main()
