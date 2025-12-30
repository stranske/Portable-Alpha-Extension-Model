from __future__ import annotations

from pathlib import Path

import dashboard.cli as dashboard_cli


def test_dashboard_cli_runs_streamlit(monkeypatch):
    called = {}

    def _fake_run(cmd, check):
        called["cmd"] = cmd
        called["check"] = check

    monkeypatch.setattr(dashboard_cli.subprocess, "run", _fake_run)

    dashboard_cli.main()

    expected_app = Path(dashboard_cli.__file__).with_name("app.py")
    assert called["cmd"] == ["streamlit", "run", str(expected_app)]
    assert called["check"] is True
