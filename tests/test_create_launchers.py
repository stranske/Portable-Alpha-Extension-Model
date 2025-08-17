import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.create_launchers import (
    make_mac_launcher,
    make_windows_launcher,
)  # noqa: E402


def test_make_launchers(tmp_path):
    """Launchers for both platforms should be created in the target directory."""
    make_windows_launcher("pa-dashboard", tmp_path)
    cmd = make_mac_launcher("pa-dashboard", tmp_path)
    assert (tmp_path / "pa-dashboard.bat").exists()
    assert cmd.exists()
    assert cmd.stat().st_mode & 0o111
