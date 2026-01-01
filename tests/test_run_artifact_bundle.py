from __future__ import annotations

import json
from pathlib import Path

from pa_core.run_artifact_bundle import RunArtifact, RunArtifactBundle


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
