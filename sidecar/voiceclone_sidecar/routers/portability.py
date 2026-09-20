"""Portability: voice packages + whole-library backup/restore (issue #14)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..context import AppContext
from ..portability import (
    PortabilityError,
    create_library_backup,
    export_voice_package,
    import_voice_package,
    restore_library_backup,
)
from ..voices import VoiceValidationError


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    voice_store = ctx.voice_store
    _require_voice = ctx.require_voice
    _read_upload = ctx.read_upload

    def _portability_error(exc: PortabilityError | VoiceValidationError) -> HTTPException:
        return HTTPException(status_code=422, detail=str(exc))

    @router.get("/voices/{voice_id}/export", dependencies=[Depends(ctx.require_auth)])
    async def export_voice(voice_id: str) -> FileResponse:
        """Download one voice as a shareable .zip package."""
        record = _require_voice(voice_id)
        out = ctx.new_temp_zip()
        try:
            export_voice_package(record, voice_store.root / voice_id, out)
        except PortabilityError as exc:
            raise _portability_error(exc) from exc
        import re

        filename = re.sub(r'[\\/:*?"<>|\s]+', "_", record.get("name") or voice_id)
        return FileResponse(
            out,
            media_type="application/zip",
            filename=f"{filename}.voice.zip",
            background=BackgroundTask(lambda: out.unlink(missing_ok=True)),
        )

    @router.post("/voices/import", dependencies=[Depends(ctx.require_auth)])
    async def import_voice(file: UploadFile) -> dict:
        """Import a voice package. Bindings are dropped; the user rebinds on
        demand against their own engines."""
        tmp = None
        try:
            tmp = await _read_upload(file)
            if tmp is None:
                raise PortabilityError("音色包不能为空")
            record = import_voice_package(tmp, voice_store)
            return {"voice": record}
        except (PortabilityError, VoiceValidationError) as exc:
            raise _portability_error(exc) from exc
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @router.post("/backup", dependencies=[Depends(ctx.require_auth)])
    async def backup_library() -> FileResponse:
        """Create and download a whole-library backup zip."""
        out = ctx.new_temp_zip()
        try:
            create_library_backup(ctx.data_root, out)
        except PortabilityError as exc:
            raise _portability_error(exc) from exc
        stamp = time.strftime("%Y%m%d", time.gmtime())
        return FileResponse(
            out,
            media_type="application/zip",
            filename=f"voice-library-backup-{stamp}.zip",
            background=BackgroundTask(lambda: out.unlink(missing_ok=True)),
        )

    @router.post("/restore", dependencies=[Depends(ctx.require_auth)])
    async def restore_library(file: UploadFile) -> dict:
        """Restore the whole library from a backup zip.

        Replaces the data directory with the archive's contents and reloads
        every durable store in place. Refused while a queued/running job is
        in flight: the swap would delete the directory its artifacts are
        written into. A synchronous generation mid-request is not detected
        and remains the user's responsibility.
        """
        running = [
            j["id"]
            for j in ctx.job_store.list(limit=100)
            if j["status"] in ("queued", "running")
        ]
        if running:
            raise HTTPException(
                status_code=409,
                detail="有正在进行的分段任务，请等待完成或取消后再恢复备份",
            )
        tmp = None
        try:
            tmp = await _read_upload(file)
            if tmp is None:
                raise PortabilityError("备份包不能为空")
            restored = restore_library_backup(tmp, ctx.data_root)
            # reload() reconnects each store's database to the swapped-in
            # file; a legacy JSON-format restore is migrated to SQLite by
            # that reopen path (ADR-0017).
            voice_store.reload()
            ctx.generation_store.reload()
            ctx.compare_store.reload()
            ctx.regression_store.reload()
            # TranscriptionSettings and the app-state blocks read the
            # settings/app_state tables on every access, so restored values
            # take effect immediately.
            restored["restored"] = True
            return restored
        except PortabilityError as exc:
            raise _portability_error(exc) from exc
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    return router
