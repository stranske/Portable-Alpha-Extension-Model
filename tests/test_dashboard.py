import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from dashboard.app import apply_theme
from pa_core.viz import theme


def test_apply_theme(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("colorway: ['#123456']\nfont: Foo\n")
    apply_theme(cfg)
    assert theme.TEMPLATE.layout.font.family == "Foo"
    assert list(theme.TEMPLATE.layout.colorway)[0] == "#123456"
