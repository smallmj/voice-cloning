"""Media-scoped, short-TTL tokens (issue #19).

The sidecar token is the single gate for read/write of the whole library and
for paid cloud calls. Media elements (``<img>`` / ``<audio>``) and the log
WebSocket cannot set Authorization headers, so historically the full token
was appended to their URLs — putting the process-wide credential into
browsing history, logs and DOM attributes.

Instead, media consumers fetch a narrow token over bearer auth:

- It only grants the media routes (audio files, reference/avatar images,
  ``/ws/logs``) — it is useless for anything else.
- It expires quickly, so a leaked value stops working on its own.
- It is derived from the real token via HMAC, never stored, and a token
  derived from a different root is rejected.
"""

from __future__ import annotations

import hashlib
import hmac
import time

DEFAULT_MEDIA_TTL_SECONDS = 300

_PURPOSE = "media"


def make_media_token(root_token: str, ttl_seconds: int = DEFAULT_MEDIA_TTL_SECONDS) -> str:
    """Derive a signed, expiring media token from the root sidecar token."""
    expires = int(time.time()) + int(ttl_seconds)
    payload = f"{_PURPOSE}|{expires}"
    sig = hmac.new(root_token.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{sig}"


def verify_media_token(root_token: str, value: str, now: float | None = None) -> bool:
    """Check a media token's signature and expiry against the root token."""
    if not value or "." not in value:
        return False
    try:
        expires_text, sig = value.rsplit(".", 1)
        expires = int(expires_text)
    except ValueError:
        return False
    current = time.time() if now is None else now
    if expires < current:
        return False
    payload = f"{_PURPOSE}|{expires}"
    expected = hmac.new(root_token.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def expires_at(value: str) -> int | None:
    """Extract the expiry epoch of a media token (or None if malformed)."""
    if "." not in value:
        return None
    try:
        return int(value.rsplit(".", 1)[0])
    except ValueError:
        return None


__all__ = [
    "DEFAULT_MEDIA_TTL_SECONDS",
    "expires_at",
    "make_media_token",
    "verify_media_token",
]
