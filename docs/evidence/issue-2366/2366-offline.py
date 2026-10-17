import socket
import pytest
from pathlib import Path

out = Path(__file__).parent
attempts = []


class Offline:
    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        def blocked(sock, address):
            attempts.append(address)
            raise AssertionError("Network forbidden in credential qualification")

        monkeypatch.setattr(socket.socket, "connect", blocked)


rc = pytest.main(
    [
        "tests/test_dashboard_llm_settings.py",
        "tests/test_dashboard_explain_results.py",
        "tests/test_dashboard_comparison_llm.py",
        "tests/test_llm_provider_missing_keys.py",
        "tests/test_wizard_config_chat_flow.py",
        "--junitxml=" + str(out / "2366-offline.xml"),
    ],
    plugins=[Offline()],
)
print("OFFLINE_SOCKET_ATTEMPTS", len(attempts))
raise SystemExit(rc if rc else bool(attempts))
