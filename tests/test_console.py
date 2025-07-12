import pandas as pd

from pa_core.reporting import print_summary


def test_print_summary(capsys):
    df = pd.DataFrame({"Metric": [1.0], "Value": [0.5]})
    print_summary(df)
    captured = capsys.readouterr()
    assert "Metric" in captured.out
    assert "0.5" in captured.out
