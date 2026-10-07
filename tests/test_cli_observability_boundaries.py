"""Protect CLI diagnostics and warning collection at failure boundaries."""

import json
import logging
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pa_core import cli


@pytest.mark.parametrize("error", [KeyError, TypeError, ValueError], ids=["key", "type", "value"])
def test_run_diff_expected_error_is_logged_without_rendering(monkeypatch, caplog, error):
    from pa_core.reporting import console, run_diff

    def fail(*args):
        raise error("previous run unavailable")

    def cannot_render(*args):
        pytest.fail("A failed run comparison must not render a partial diff")

    monkeypatch.setattr(run_diff, "build_run_diff", fail)
    monkeypatch.setattr(console, "print_run_diff", cannot_render)
    with caplog.at_level(logging.WARNING, logger="pa_core.cli"):
        cli._maybe_print_run_diff(
            current_manifest={"run": "current"},
            prev_manifest={"run": "previous"},
            current_summary=pd.DataFrame({"Agent": ["Base"]}),
            prev_summary=None,
        )
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "pa_core.cli"
    assert caplog.records[0].levelno == logging.WARNING
    assert "Run diff unavailable:" in caplog.records[0].getMessage()
    assert "previous run unavailable" in caplog.records[0].getMessage()


def test_run_diff_unexpected_error_propagates(monkeypatch):
    from pa_core.reporting import run_diff

    def fail(*args):
        raise RuntimeError("unexpected comparison defect")

    monkeypatch.setattr(run_diff, "build_run_diff", fail)
    with pytest.raises(RuntimeError, match="unexpected comparison defect"):
        cli._maybe_print_run_diff(
            current_manifest={},
            prev_manifest={},
            current_summary=pd.DataFrame(),
            prev_summary=None,
        )


def test_warning_collector_double_install_captures_once_and_restores(monkeypatch):
    forwarded = []

    def previous_hook(*args):
        forwarded.append(args)

    monkeypatch.setattr(warnings, "showwarning", previous_hook)
    root = logging.getLogger()
    original_handlers = tuple(root.handlers)
    collector = cli._WarningCollector()
    try:
        collector.install()
        collector.install()
        logging.getLogger("pa.boundary").warning("run %s failed", "alpha")
        warning = UserWarning("source warning")
        warnings.showwarning(warning, UserWarning, "scenario.yml", 17)
        records = collector.snapshot()
        assert [(r["code"], r["severity"], r["message"]) for r in records] == [
            ("pa.boundary", "warning", "run alpha failed"),
            ("UserWarning", "warning", "source warning"),
        ]
        assert records[0]["context"]["source"] == "logging"
        assert records[1]["context"] == {
            "source": "warnings",
            "category": "UserWarning",
            "filename": "scenario.yml",
            "lineno": 17,
        }
        assert forwarded == [(warning, UserWarning, "scenario.yml", 17, None, None)]
        collector.uninstall()
        collector.uninstall()
        assert tuple(root.handlers) == original_handlers
        assert warnings.showwarning is previous_hook

        # Restoring the hook must also stop both capture paths.
        logging.getLogger("pa.boundary").warning("outside collection")
        outside_warning = UserWarning("outside source warning")
        warnings.showwarning(outside_warning, UserWarning, "scenario.yml", 18)
        assert collector.snapshot() == records
        assert forwarded[-1] == (outside_warning, UserWarning, "scenario.yml", 18)

        # A new run can reuse the collector without retaining stale hooks or handlers.
        collector.install()
        collector.install()
        logging.getLogger("pa.boundary").error("run %s failed again", "beta")
        next_warning = UserWarning("next source warning")
        warnings.showwarning(next_warning, UserWarning, "scenario.yml", 19)
        next_records = collector.snapshot()
        assert next_records[:2] == records
        assert [(r["code"], r["severity"], r["message"]) for r in next_records[2:]] == [
            ("pa.boundary", "error", "run beta failed again"),
            ("UserWarning", "warning", "next source warning"),
        ]
        assert forwarded == [
            (warning, UserWarning, "scenario.yml", 17, None, None),
            (outside_warning, UserWarning, "scenario.yml", 18),
            (next_warning, UserWarning, "scenario.yml", 19, None, None),
        ]
        collector.uninstall()
        collector.uninstall()
        assert tuple(root.handlers) == original_handlers
        assert warnings.showwarning is previous_hook
    finally:
        collector.uninstall()
        collector.uninstall()
        # Clean leaked handlers even if the deliberately broken implementation fails.
        root.handlers[:] = list(original_handlers)
        warnings.showwarning = previous_hook


def test_warning_snapshot_does_not_expose_mutable_message_records():
    collector = cli._WarningCollector()
    try:
        collector.install()
        logging.getLogger("pa.snapshot").warning("original warning")
        snapshot = collector.snapshot()
        snapshot[0]["message"] = "tampered"
        snapshot.clear()
        assert collector.snapshot()[0]["message"] == "original warning"
    finally:
        collector.uninstall()


def test_json_formatter_preserves_utc_and_interpolates_arguments(monkeypatch):
    class NonUtcDatetime(datetime):
        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            local_zone = timezone(timedelta(hours=-5))
            return super().fromtimestamp(timestamp, tz=tz if tz is not None else local_zone)

    # Model a non-UTC host without changing process-global TZ or requiring tzset.
    assert NonUtcDatetime.fromtimestamp(0).utcoffset() == timedelta(hours=-5)
    monkeypatch.setattr(cli, "datetime", NonUtcDatetime)
    record = logging.LogRecord("pa.output", logging.ERROR, __file__, 1, "scenario %s", ("α",), None)
    # Include fractional seconds so truncating real log timestamps cannot pass.
    for created, expected_time in [
        (0, datetime(1970, 1, 1, tzinfo=timezone.utc)),
        (1704164645.125, datetime(2024, 1, 2, 3, 4, 5, 125000, tzinfo=timezone.utc)),
    ]:
        record.created = created
        result = json.loads(cli.JsonFormatter().format(record))
        assert result == {
            "level": "ERROR",
            "timestamp": expected_time.isoformat(),
            "module": "pa.output",
            "message": "scenario α",
        }


def test_invalid_utf8_config_snapshot_is_unavailable(tmp_path, caplog):
    path = tmp_path / "private.yml"
    path.write_bytes(b"secret: \xff")
    with caplog.at_level(logging.DEBUG, logger="pa_core.cli"):
        assert cli._read_config_snapshot(path) == (None, None)
    assert "Unable to read config snapshot" in caplog.text
    assert "secret:" not in caplog.text


@pytest.mark.parametrize("source", ["logging", "warnings"])
def test_warning_snapshot_context_is_detached_from_collector(source, monkeypatch):
    monkeypatch.setattr(warnings, "showwarning", lambda *args: None)
    collector = cli._WarningCollector()
    try:
        collector.install()
        if source == "logging":
            logging.getLogger("pa.snapshot").warning("immutable diagnostic")
        else:
            warnings.showwarning(UserWarning("immutable diagnostic"), UserWarning, "cfg.yml", 23)
        before = json.loads(json.dumps(collector.snapshot()))
        snapshot = collector.snapshot()
        # A retained snapshot must also survive later edits to captured context.
        collector.records[0]["context"]["lineno"] += 1
        assert snapshot == before
        updated = json.loads(json.dumps(collector.snapshot()))
        assert updated[0]["context"]["lineno"] == before[0]["context"]["lineno"] + 1

        snapshot[0]["context"]["source"] = "tampered"
        snapshot[0]["context"]["injected"] = "must not leak into run.json"
        assert collector.snapshot() == updated
        assert collector.snapshot()[0]["context"]["source"] == source
    finally:
        collector.uninstall()


@pytest.mark.parametrize("benchmark", [None, "Index"], ids=["no-benchmark", "index-benchmark"])
def test_enhanced_summary_preserves_monthly_returns_and_benchmark(benchmark):
    import numpy as np

    index = np.array([[0.01, -0.02, 0.03, 0.04]])
    strategy = np.array([[0.02, -0.01, 0.01, 0.05]])
    summary = cli.create_enhanced_summary(
        {"Index": index, "Strategy": strategy}, benchmark=benchmark
    )
    rows = summary.set_index("Agent")
    assert {"Index", "Strategy"} <= set(rows.index)
    # Four monthly returns must compound to an annualized return, not be averaged.
    expected_return = float(np.prod(1.0 + strategy) ** 3 - 1.0)
    assert rows.loc["Strategy", "terminal_AnnReturn"] == pytest.approx(expected_return)
    if benchmark is None:
        assert rows["monthly_TE"].isna().all()
    else:
        assert pd.isna(rows.loc["Index", "monthly_TE"])
        expected_te = float(np.std(strategy - index, ddof=1) * np.sqrt(12))
        assert rows.loc["Strategy", "monthly_TE"] == pytest.approx(expected_te)


def test_config_snapshot_preserves_utf8_and_original_bytes(tmp_path):
    raw = "# café\nname: α\r\n".encode("utf-8")
    path = tmp_path / "scenario.yml"
    path.write_bytes(raw)
    text, captured = cli._read_config_snapshot(path)
    assert text == "# café\nname: α\r\n"
    assert captured == raw
    assert captured.decode("utf-8") == text


def test_run_timer_snapshot_uses_monotonic_elapsed_and_utc_wall_time(monkeypatch):
    instants = iter(
        [
            datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 3, 4, 7, tzinfo=timezone.utc),
        ]
    )

    class FixedDatetime:
        @staticmethod
        def now(tz):
            assert tz is timezone.utc
            return next(instants)

    ticks = iter([100.0, 102.5])
    monkeypatch.setattr(cli, "datetime", FixedDatetime)
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(ticks))
    assert cli.RunTimer().snapshot() == {
        "duration_seconds": 2.5,
        "started_at": "2026-01-02T03:04:05+00:00",
        "ended_at": "2026-01-02T03:04:07+00:00",
    }
