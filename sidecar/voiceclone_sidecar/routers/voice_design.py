"""Voice design (issue #11): create a voice from a text description.

A designed voice is a first-class voice: the design engine's preview sample
becomes the reference (the source of truth every other engine binds from,
ADR-0001), so multi-engine bindings, generation and deletion all go through
the same paths as a cloned one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext
from ..engines.cloud_base import CloudEngineError


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    voice_store = ctx.voice_store
    _flag_placeholder = ctx.flag_placeholder

    @router.post("/voices/design", dependencies=[Depends(ctx.require_auth)])
    async def design_voice(body: dict) -> dict:
        engine_id = body.get("engine_id")
        name = body.get("name")
        description = body.get("description", "")
        voice_prompt = body.get("voice_prompt")
        preview_text = body.get("preview_text")

        engine = ctx.require_engine(engine_id if isinstance(engine_id, str) else "")
        if not engine.capabilities().voice_design:
            raise HTTPException(
                status_code=422,
                detail=f"引擎 {engine.engine_id} 未声明音色设计能力：不支持用文字描述创建音色",
            )
        if engine.requires_key and ctx.keys.get(engine.engine_id) is None:
            raise HTTPException(
                status_code=409,
                detail=f"引擎 {engine.engine_id} 需要 API Key；请在「设置」页配置后再设计音色",
            )
        if not isinstance(voice_prompt, str) or not voice_prompt.strip():
            raise HTTPException(status_code=422, detail="声音描述不能为空")
        if not isinstance(preview_text, str) or not preview_text.strip():
            raise HTTPException(status_code=422, detail="试听文本不能为空")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=422, detail="音色名称不能为空")

        voice = voice_store.create_designed(
            name, description if isinstance(description, str) else "",
            engine_id=engine.engine_id,
            voice_prompt=voice_prompt,
            preview_text=preview_text,
        )

        loop = asyncio.get_running_loop()
        _design_log = ctx.log_from_thread(f"design:{engine.engine_id}")

        try:
            extra = await loop.run_in_executor(
                None,
                lambda: engine.design_voice(voice_prompt, preview_text, _design_log),
            )
            voice = voice_store.attach_reference(
                voice["id"], Path(extra["sample_audio_path"]), extra.get("transcript")
            )
            voice_store.bind(voice["id"], engine.engine_id, status="ready",
                             extra={"voice_id": extra["voice_id"]})
            return _flag_placeholder(voice_store.get(voice["id"]))
        except CloudEngineError as exc:
            voice_store.delete(voice["id"])
            raise HTTPException(status_code=502, detail=f"设计音色失败：{exc}") from exc
        except (Exception, OSError) as exc:
            # ANY failure (vendor or local IO) leaves no orphaned,
            # reference-less voice behind — ADR-0010's hard guarantee.
            voice_store.delete(voice["id"])
            raise HTTPException(
                status_code=500, detail=f"设计音色失败：{exc}"
            ) from exc

    return router
