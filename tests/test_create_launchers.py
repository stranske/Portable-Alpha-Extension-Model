from pathlib import Path

from scripts.create_launchers import _make_mac_launcher, _make_windows_launcher


def test_make_launchers(tmp_path):
    _make_windows_launcher("pa-dashboard", tmp_path)
    _make_mac_launcher("pa-dashboard", tmp_path)
    assert (tmp_path / "pa-dashboard.bat").exists()
    cmd = tmp_path / "pa-dashboard.command"
    assert cmd.exists()
    assert cmd.stat().st_mode & 0o111
