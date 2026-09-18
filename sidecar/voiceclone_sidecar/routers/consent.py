"""First-use voice consent (issue #15)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext
from ..generations import now_iso


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()

    @router.get("/consent", dependencies=[Depends(ctx.require_auth)])
    async def get_consent() -> dict:
        state = ctx.read_consent()
        return {
            "acknowledged": state.get("version") == ctx.consent_version,
            "confirmed_at": state.get("confirmed_at"),
            "version": ctx.consent_version,
        }

    @router.put("/consent", dependencies=[Depends(ctx.require_auth)])
    async def put_consent(body: dict) -> dict:
        if body.get("acknowledged") is not True:
            raise HTTPException(status_code=422, detail="acknowledged 必须为 true")
        state = {"version": ctx.consent_version, "confirmed_at": now_iso()}
        ctx.write_consent(state)
        return {
            "acknowledged": True,
            "confirmed_at": state["confirmed_at"],
            "version": ctx.consent_version,
        }

    return router
