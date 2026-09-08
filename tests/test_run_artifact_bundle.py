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
@pytest.mark.parametrize("bundle_failure", [False, True])
def test_cli_bundle_finalized_manifest_parity(tmp_path, mode, log_json, bundle_failure) -> None:
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
    if bundle_failure:
        (tmp_path / "bundle").write_text("not a directory")
    pythonpath = os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": pythonpath},
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
    if bundle_failure:
        assert any("Failed to write artifact bundle" in w["message"] for w in manifest["warnings"])
    else:
        (tmp_path / "bundle").rename(tmp_path / "moved_bundle")
        bundle = RunArtifactBundle.load(tmp_path / "moved_bundle")
        assert bundle.artifact.manifest == manifest
        assert all(Path(name).name != "run.json" for name in bundle.artifact.outputs)
        assert bundle.verify()

    record = json.loads((tmp_path / "run.json").read_text())
    assert record["bundle_path"] == (
        None if bundle_failure else str(tmp_path / "bundle" / "bundle.json")
    )
    assert record["warnings"] == manifest["warnings"]
    assert record["cost"] == manifest["cost"]
    if log_json:
        run_end = json.loads((tmp_path / record["run_end_path"]).read_text())
        assert run_end["warnings"] == manifest["warnings"]
        assert run_end["cost"] == manifest["cost"]
        assert run_end["duration_seconds"] == manifest["run_timing"]["duration_seconds"]


@pytest.mark.parametrize("reload_failure", ["unreadable", "invalid-json", "not-a-mapping"])
def test_cli_bundle_retains_manifest_when_finalized_reload_fails(
    tmp_path, monkeypatch, reload_failure
):
    from pa_core.cli import main

    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "N_SIMULATIONS": 100,
                "N_MONTHS": 2,
                "financing_mode": "broadcast",
                "analysis_mode": "vol_mult",
                "sd_multiple_min": 2.0,
                "sd_multiple_max": 2.0,
            }
        )
    )
    read_text = Path.read_text
    write_text = Path.write_text
    finalized = False
    injected = False

    def observe_write(path, data, *args, **kwargs):
        nonlocal finalized
        result = write_text(path, data, *args, **kwargs)
        if path == tmp_path / "manifest.json" and json.loads(data).get("warnings") is not None:
            finalized = True
        return result

    def transient_read(path, *args, **kwargs):
        nonlocal injected
        if path == tmp_path / "manifest.json" and finalized and not injected:
            injected = True
            if reload_failure == "unreadable":
                raise PermissionError("transient manifest read failure")
            return "{" if reload_failure == "invalid-json" else "[]"
        return read_text(path, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "write_text", observe_write)
    monkeypatch.setattr(Path, "read_text", transient_read)
    main(
        [
            "--config",
            str(config),
            "--index",
            str(root / "data/sp500tr_fred_divyield.csv"),
            "--output",
            str(tmp_path / "out.xlsx"),
            "--bundle",
            str(tmp_path / "bundle"),
            "--seed",
            "123",
        ]
    )
    assert injected
    manifest = json.loads(read_text(tmp_path / "manifest.json"))
    bundle = RunArtifactBundle.load(tmp_path / "bundle")
    assert bundle.artifact.manifest == manifest
    assert bundle.verify()
    record = json.loads(read_text(tmp_path / "run.json"))
    assert record["bundle_path"] == str(tmp_path / "bundle/bundle.json")


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
