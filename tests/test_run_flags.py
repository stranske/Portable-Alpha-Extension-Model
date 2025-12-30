from __future__ import annotations

from pa_core.run_flags import RunFlags


def test_run_flags_defaults() -> None:
    flags = RunFlags()

    assert flags.save_xlsx == "Outputs.xlsx"
    assert flags.png is False
    assert flags.pdf is False
    assert flags.pptx is False
    assert flags.html is False
    assert flags.gif is False
    assert flags.dashboard is False
    assert flags.alt_text is None
    assert flags.packet is False


def test_run_flags_custom_values() -> None:
    flags = RunFlags(
        save_xlsx="custom.xlsx",
        png=True,
        pdf=True,
        pptx=True,
        html=True,
        gif=True,
        dashboard=True,
        alt_text="Alt text",
        packet=True,
    )

    assert flags.save_xlsx == "custom.xlsx"
    assert flags.png is True
    assert flags.pdf is True
    assert flags.pptx is True
    assert flags.html is True
    assert flags.gif is True
    assert flags.dashboard is True
    assert flags.alt_text == "Alt text"
    assert flags.packet is True
