from __future__ import annotations

import asyncio
import json
from typing import Any, List

import plotly.graph_objects as go
import pytest

from pa_core.viz import live


class _FakeWebSocket:
    def __init__(self, messages: List[str]) -> None:
        self._messages = iter(messages)

    async def __aenter__(self) -> "_FakeWebSocket":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def __aiter__(self) -> "_FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeWebsockets:
    def __init__(self, messages: List[str]) -> None:
        self._messages = messages
        self.connected_url: str | None = None

    def connect(self, url: str) -> _FakeWebSocket:
        self.connected_url = url
        return _FakeWebSocket(self._messages)


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def test_connect_requires_websockets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "websockets", None)
    fig = go.Figure()

    with pytest.raises(RuntimeError, match="websockets package not installed"):
        _run_async(live.connect("ws://example", fig))


def test_connect_updates_figure_and_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [],
        "layout": {"title": {"text": "Live Update"}},
    }
    message = json.dumps(payload)
    fake_ws = _FakeWebsockets([message])
    monkeypatch.setattr(live, "websockets", fake_ws)

    fig = go.Figure()
    seen: list[go.Figure] = []

    def _on_update(updated: go.Figure) -> None:
        seen.append(updated)

    _run_async(live.connect("ws://example", fig, on_update=_on_update))

    assert fake_ws.connected_url == "ws://example"
    assert fig.layout.title.text == "Live Update"
    assert seen and seen[0] is fig
