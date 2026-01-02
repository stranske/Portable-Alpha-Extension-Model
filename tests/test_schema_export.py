from pa_core.schema import export_schema_definitions


def test_export_schema_includes_aliases() -> None:
    payload = export_schema_definitions(schema="config")
    model = payload["models"]["ModelConfig"]
    fields = model["fields"]
    assert fields["N_SIMULATIONS"]["alias"] == "Number of simulations"
    assert "Active share" in fields["active_share"]["aliases"]
