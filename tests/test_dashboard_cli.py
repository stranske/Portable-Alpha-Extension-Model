from pathlib import Path
from unittest.mock import Mock

from dashboard import cli


def test_cli_main_runs_streamlit(monkeypatch):
    run_mock = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run_mock)

    cli.main()

    expected_path = Path(cli.__file__).with_name("app.py")
    run_mock.assert_called_once_with(
        ["streamlit", "run", str(expected_path)], check=True
    )
