"""Voice CRUD + reference/avatar media routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..context import AppContext
from ..voices import AUDIO_MEDIA_TYPES, AVATAR_MEDIA_TYPES, VoiceValidationError


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    voice_store = ctx.voice_store
    _require_voice = ctx.require_voice
    _flag_placeholder = ctx.flag_placeholder
    _read_upload = ctx.read_upload

    @router.post("/voices", dependencies=[Depends(ctx.require_auth)])
    async def create_voice(
        file: UploadFile,
        name: str = Form(...),
        description: str = Form(""),
        avatar: UploadFile | None = None,
    ) -> dict:
        audio_tmp = avatar_tmp = None
        try:
            audio_tmp = await _read_upload(file)
            avatar_tmp = await _read_upload(avatar)
            if audio_tmp is None:
                raise VoiceValidationError("参考音频不能为空")
            return _flag_placeholder(voice_store.create(name, description, audio_tmp, avatar_tmp))
        except VoiceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            for tmp in (audio_tmp, avatar_tmp):
                if tmp is not None:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    @router.get("/voices", dependencies=[Depends(ctx.require_auth)])
    async def list_voices() -> dict:
        return {"voices": [_flag_placeholder(v) for v in voice_store.list()]}

    @router.get("/voices/{voice_id}", dependencies=[Depends(ctx.require_auth)])
    async def get_voice(voice_id: str) -> dict:
        return _flag_placeholder(_require_voice(voice_id))

    @router.patch("/voices/{voice_id}", dependencies=[Depends(ctx.require_auth)])
    async def update_voice(voice_id: str, body: dict) -> dict:
        _require_voice(voice_id)
        try:
            return _flag_placeholder(
                voice_store.update(
                    voice_id,
                    name=body.get("name") if "name" in body else None,
                    description=body.get("description")
                    if "description" in body
                    else None,
                )
            )
        except VoiceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/voices/{voice_id}/avatar", dependencies=[Depends(ctx.require_auth)])
    async def set_voice_avatar(voice_id: str, avatar: UploadFile) -> dict:
        _require_voice(voice_id)
        avatar_tmp = None
        try:
            avatar_tmp = await _read_upload(avatar)
            if avatar_tmp is None:
                raise VoiceValidationError("头像不能为空")
            return voice_store.set_avatar(voice_id, avatar_tmp)
        except VoiceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if avatar_tmp is not None:
                try:
                    avatar_tmp.unlink()
                except OSError:
                    pass

    @router.delete("/voices/{voice_id}", dependencies=[Depends(ctx.require_auth)])
    async def delete_voice(voice_id: str) -> dict:
        _require_voice(voice_id)
        voice_store.delete(voice_id)
        return {"deleted": voice_id}

    @router.get(
        "/voices/{voice_id}/reference", dependencies=[Depends(ctx.require_media_auth)]
    )
    async def voice_reference(voice_id: str) -> FileResponse:
        _require_voice(voice_id)
        path = voice_store.reference_path(voice_id)
        media = AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media, filename=path.name)

    @router.get(
        "/voices/{voice_id}/avatar", dependencies=[Depends(ctx.require_media_auth)]
    )
    async def voice_avatar(voice_id: str) -> FileResponse:
        _require_voice(voice_id)
        path = voice_store.avatar_path(voice_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="voice has no avatar")
        media = AVATAR_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media)

    return router
