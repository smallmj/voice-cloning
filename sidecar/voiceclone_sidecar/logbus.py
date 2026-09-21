"""Broadcasts generation log lines to every connected WebSocket client."""

from __future__ import annotations

import asyncio


class LogBus:
    """Broadcasts generation log lines to every connected WebSocket client."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, generation_id: str, message: str, level: str = "info") -> None:
        """Broadcast one log line. ``level`` is one of info/warn/error —
        issue #36: the sidebar's log pane filters on it client-side, so the
        classification happens where the failure is known."""
        event = {
            "type": "log",
            "generation_id": generation_id,
            "message": message,
            "ts": asyncio.get_running_loop().time(),
            "level": level if level in ("info", "warn", "error") else "info",
        }
        for q in list(self._subscribers):
            q.put_nowait(event)
