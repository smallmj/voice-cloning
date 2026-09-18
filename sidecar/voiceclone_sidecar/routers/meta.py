"""Health check + media token issuance."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import _version
from ..context import AppContext
from ..media_token import DEFAULT_MEDIA_TTL_SECONDS, expires_at, make_media_token


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()

    @router.get("/health", dependencies=[Depends(ctx.require_auth)])
    async def health() -> dict:
        return {"status": "ok", "version": _version.SIDECAR_VERSION}

    @router.get("/media-token", dependencies=[Depends(ctx.require_auth)])
    async def media_token() -> dict:
        """Issue a short-TTL, media-scoped token (issue #19).

        Media elements (<img>/<audio>) and the log WebSocket cannot set
        Authorization headers, so the renderer fetches this narrow token
        over bearer auth and uses it in their URLs instead of the raw
        sidecar token. It only grants the media routes and expires on its
        own; fetch a fresh one when it lapses.
        """
        issued = make_media_token(ctx.token, DEFAULT_MEDIA_TTL_SECONDS)
        return {
            "media_token": issued,
            "expires_in_seconds": DEFAULT_MEDIA_TTL_SECONDS,
            "expires_at": expires_at(issued),
        }

    return router
