import runpy
from pathlib import Path

PAGES = [
    Path("dashboard/pages/1_Asset_Library.py"),
    Path("dashboard/pages/2_Portfolio_Builder.py"),
    Path("dashboard/pages/3_Scenario_Wizard.py"),
    Path("dashboard/pages/4_Results.py"),
]


def test_pages_import() -> None:
    for page in PAGES:
        runpy.run_path(page)
