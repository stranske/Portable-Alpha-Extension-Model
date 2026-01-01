from __future__ import annotations

from pa_core.run_artifact_bundle import RunArtifact


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
