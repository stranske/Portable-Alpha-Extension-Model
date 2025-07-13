from __future__ import annotations

import asyncio
import json
from typing import Callable

import plotly.graph_objects as go

try:
    import websockets
except Exception:  # pragma: no cover - optional dep
    websockets = None


async def connect(
    url: str, fig: go.Figure, on_update: Callable[[go.Figure], None] | None = None
) -> None:
    """Connect to ``url`` and stream JSON updates into ``fig``.

    Parameters
    ----------
    url : str
        WebSocket endpoint.
    fig : go.Figure
        Figure to update in place.
    on_update : callable, optional
        Callback invoked after each update.
    """
    if websockets is None:
        raise RuntimeError("websockets package not installed")

    async with websockets.connect(url) as ws:
        async for msg in ws:
            payload = json.loads(msg)
            data = go.Figure(data=payload.get("data"), layout=payload.get("layout"))
            fig.data = data.data
            fig.layout.update(data.layout)
            if on_update:
                on_update(fig)
            await asyncio.sleep(0)  # yield control
