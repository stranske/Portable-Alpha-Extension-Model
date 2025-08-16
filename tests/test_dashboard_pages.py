import runpy
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(root / "pa_core")]
sys.modules.setdefault("pa_core", PKG)

PAGES = [
    Path("dashboard/pages/1_Asset_Library.py"),
    Path("dashboard/pages/2_Portfolio_Builder.py"),
    Path("dashboard/pages/3_Scenario_Wizard.py"),
    Path("dashboard/pages/4_Results.py"),
]


def test_pages_import() -> None:
    for page in PAGES:
        runpy.run_path(page)
