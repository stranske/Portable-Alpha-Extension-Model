from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pa_core.run_artifact_bundle import RunArtifact, RunArtifactBundle


@pytest.mark.parametrize("mode", ["single_with_sensitivity", "vol_mult"])
@pytest.mark.parametrize("log_json", [False, True])
def test_cli_bundle_finalized_manifest_parity(tmp_path, mode, log_json) -> None:
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "N_SIMULATIONS": 100,
                "N_MONTHS": 2,
                "financing_mode": "broadcast",
                "analysis_mode": mode,
                "sd_multiple_min": 2.0,
                "sd_multiple_max": 2.0,
            }
        )
    )
    command = [
        sys.executable,
        "-m",
        "pa_core.cli",
        "--config",
        str(config),
        "--index",
        str(root / "data" / "sp500tr_fred_divyield.csv"),
        "--index-frequency",
        "daily",
        "--output",
        str(tmp_path / "out.xlsx"),
        "--bundle",
        str(tmp_path / "bundle"),
        "--seed",
        "123",
    ]
    if log_json:
        command.append("--log-json")
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert any("frequency mismatch" in w["message"].lower() for w in manifest["warnings"])
    assert manifest["cost"]["latency_seconds"] > 0
    assert manifest["run_timing"]["ended_at"]
    assert manifest["cost"]["latency_seconds"] == manifest["run_timing"]["duration_seconds"]
    bundle = RunArtifactBundle.load(tmp_path / "bundle")
    assert bundle.artifact.manifest == manifest
    assert bundle.verify()

    record = json.loads((tmp_path / "run.json").read_text())
    assert record["warnings"] == manifest["warnings"]
    assert record["cost"] == manifest["cost"]
    if log_json:
        run_end = json.loads((tmp_path / record["run_end_path"]).read_text())
        assert run_end["warnings"] == manifest["warnings"]
        assert run_end["cost"] == manifest["cost"]
        assert run_end["duration_seconds"] == manifest["run_timing"]["duration_seconds"]


def test_run_artifact_fields() -> None:
    artifact = RunArtifact(
        config="config: value\n",
        index_hash="abc123",
        seed=42,
        manifest={"seed": 42},
        outputs={"results.xlsx": "results.xlsx"},
    )

    assert artifact.config == "config: value\n"
    assert artifact.index_hash == "abc123"
    assert artifact.seed == 42
    assert artifact.manifest == {"seed": 42}
    assert artifact.outputs == {"results.xlsx": "results.xlsx"}


def test_run_artifact_bundle_save_load_verify(tmp_path) -> None:
    config_text = "config: value\n"
    output_file = tmp_path / "results.txt"
    output_file.write_text("ok")

    artifact = RunArtifact(
        config=config_text,
        index_hash="idx123",
        seed=7,
        manifest={"seed": 7},
        outputs={"results.txt": str(output_file)},
    )
    bundle = RunArtifactBundle(artifact)
    bundle_path = tmp_path / "bundle"
    bundle.save(bundle_path)

    assert (bundle_path / "config.yaml").read_text() == config_text

    loaded = RunArtifactBundle.load(bundle_path)
    assert loaded.artifact.config == config_text
    assert loaded.artifact.index_hash == "idx123"
    assert loaded.artifact.seed == 7
    assert loaded.artifact.manifest == {"seed": 7}

    bundled_output = Path(loaded.artifact.outputs["results.txt"])
    assert bundled_output.read_text() == "ok"
    assert loaded.verify()


def test_run_artifact_bundle_verify_fails_on_change(tmp_path) -> None:
    config_text = "config: value\n"
    output_file = tmp_path / "results.txt"
    output_file.write_text("ok")

    artifact = RunArtifact(
        config=config_text,
        index_hash="idx123",
        seed=7,
        manifest=None,
        outputs={"results.txt": str(output_file)},
    )
    bundle = RunArtifactBundle(artifact)
    bundle_path = tmp_path / "bundle"
    bundle.save(bundle_path)

    mutated_output = bundle_path / "outputs" / "results.txt"
    mutated_output.write_text("changed")

    loaded = RunArtifactBundle.load(bundle_path)
    assert not loaded.verify()


def test_run_artifact_bundle_verify_fails_on_config_change(tmp_path) -> None:
    config_text = "config: value\n"
    output_file = tmp_path / "results.txt"
    output_file.write_text("ok")

    artifact = RunArtifact(
        config=config_text,
        index_hash="idx123",
        seed=7,
        manifest=None,
        outputs={"results.txt": str(output_file)},
    )
    bundle = RunArtifactBundle(artifact)
    bundle_path = tmp_path / "bundle"
    bundle.save(bundle_path)

    mutated_config = bundle_path / "config.yaml"
    mutated_config.write_text("config: changed\n")

    loaded = RunArtifactBundle.load(bundle_path)
    assert not loaded.verify()


def test_run_artifact_bundle_verify_fails_on_manifest_change(tmp_path) -> None:
    config_text = "config: value\n"
    output_file = tmp_path / "results.txt"
    output_file.write_text("ok")

    artifact = RunArtifact(
        config=config_text,
        index_hash="idx123",
        seed=7,
        manifest={"seed": 7},
        outputs={"results.txt": str(output_file)},
    )
    bundle = RunArtifactBundle(artifact)
    bundle_path = tmp_path / "bundle"
    bundle.save(bundle_path)

    mutated_manifest = bundle_path / "manifest.json"
    mutated_manifest.write_text(json.dumps({"seed": 9}, indent=2))

    loaded = RunArtifactBundle.load(bundle_path)
    assert not loaded.verify()


def test_run_artifact_bundle_verify_fails_on_missing_hash(tmp_path) -> None:
    config_text = "config: value\n"
    output_file = tmp_path / "results.txt"
    output_file.write_text("ok")

    artifact = RunArtifact(
        config=config_text,
        index_hash="idx123",
        seed=7,
        manifest={"seed": 7},
        outputs={"results.txt": str(output_file)},
    )
    bundle = RunArtifactBundle(artifact)
    bundle_path = tmp_path / "bundle"
    bundle.save(bundle_path)

    meta_path = bundle_path / "bundle.json"
    meta = json.loads(meta_path.read_text())
    meta["hashes"]["outputs"].pop("results.txt")
    meta_path.write_text(json.dumps(meta, indent=2))

    loaded = RunArtifactBundle.load(bundle_path)
    assert not loaded.verify()
