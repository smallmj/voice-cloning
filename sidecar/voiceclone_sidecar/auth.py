"""Auth dependencies for the sidecar HTTP surface (issue #19 policy).

Three gates, kept exactly as they were:

- ``verify_token`` — bearer-only auth for every non-media route.
- ``verify_media_query_token`` — media routes also accept a short-TTL,
  media-scoped query token (never the raw sidecar token).
- ``verify_token_ws`` — WebSocket handshake, header primary + media token
  fallback (browsers cannot set WS headers).
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, WebSocket, WebSocketDisconnect

from .media_token import verify_media_token


def _bearer_ok(auth: str, expected: str) -> bool:
    """Shared constant-time comparison of an Authorization header value."""
    return secrets.compare_digest(auth, f"Bearer {expected}")


def verify_token(expected: str):
    """Bearer-only auth for every non-media route (issue #19).

    The query-parameter form was historically accepted everywhere, but the
    process token is the single gate for the whole library and for paid
    cloud calls — it must never appear in a URL. Media elements get a
    separate, short-TTL, media-scoped token instead (see ``media_token``).
    """
    async def _verify(request: Request) -> None:
        if not _bearer_ok(request.headers.get("Authorization", ""), expected):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    return _verify


def verify_media_query_token(expected: str):
    """Auth for media routes: bearer auth OR a short-TTL media-scoped query
    token (issue #19). The raw sidecar token is NOT accepted in a query —
    only a value derived via ``media_token.make_media_token``."""

    async def _verify(request: Request) -> None:
        if _bearer_ok(request.headers.get("Authorization", ""), expected):
            return
        if verify_media_token(expected, request.query_params.get("token", "")):
            return
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    return _verify


def verify_token_ws(expected: str):
    async def _verify(websocket: WebSocket) -> None:
        # Browsers cannot set headers on WebSocket, so accept a valid
        # short-TTL media token as a query parameter (issue #19). The header
        # form stays the primary contract; the raw sidecar token is never
        # accepted here.
        auth = websocket.headers.get("Authorization", "")
        query_token = websocket.query_params.get("token", "")
        if not (
            _bearer_ok(auth, expected) or verify_media_token(expected, query_token)
        ):
            await websocket.accept()
            await websocket.close(code=4401, reason="invalid or missing bearer token")
            raise WebSocketDisconnect(code=4401)

    return _verify
