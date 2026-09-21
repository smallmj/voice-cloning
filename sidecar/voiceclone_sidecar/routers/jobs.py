"""Long-text jobs: queue + cancellation (issue #13).

An over-length script is split into engine-sized segments (the engine's
declared max_chars_per_request), and each segment runs through the SAME
shared pipeline as a plain generation — one real generation record per
segment, full lineage. One worker drains the FIFO queue, so vendor
throttling is absorbed by pacing (plus the rate-limit retry in the
pipeline) instead of surfacing as user-facing errors.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext
from ..generations import now_iso
from ..jobs import is_cancel_requested, new_job, public_view
from ..pipeline import run_generation
from ..registry import Engine
from ..segmentation import concat_wav_files, split_text


class JobWorker:
    """Owns the FIFO drain task; restartable after a crash."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._task: asyncio.Task | None = None

    async def _run_job(self, job: dict) -> None:
        ctx = self.ctx
        job["status"] = "running"
        job["started_at"] = now_iso()
        completed: list[str] = []
        failure: str | None = None
        for seg in job["segments"]:
            if is_cancel_requested(job):
                break
            seg["status"] = "running"
            try:
                record = await run_generation(
                    ctx,
                    job["engine_id"],
                    seg["text"],
                    job["voice_id"] or None,
                    job["params"] or None,
                )
            except Exception as exc:  # noqa: BLE001 - recorded on the job
                seg["status"] = "failed"
                seg["error"] = str(getattr(exc, "detail", exc))
                failure = seg["error"]
                ctx.log_bus.publish(job["id"], f"分段 {seg['index'] + 1} 失败：{failure}", level="error")
                break
            seg["status"] = "succeeded"
            seg["generation_id"] = record["id"]
            completed.append(record["audio_file"])
            ctx.log_bus.publish(
                job["id"], f"分段 {seg['index'] + 1}/{len(job['segments'])} 完成"
            )

        cancelled = failure is None and (
            is_cancel_requested(job)
            # Queued segments left after a cancel request mid-run.
            or any(s["status"] == "pending" for s in job["segments"])
        )
        for seg in job["segments"]:
            if seg["status"] == "pending":
                seg["status"] = "skipped"

        # Cancelled jobs KEEP the audio of the segments that finished — a
        # partial result beats throwing completed synthesis away.
        if completed:
            try:
                out_name = f"job-{job['id']}.wav"
                loop = asyncio.get_running_loop()
                rate = await loop.run_in_executor(
                    None,
                    lambda: concat_wav_files(
                        [ctx.audio_dir / name for name in completed], ctx.audio_dir / out_name
                    ),
                )
                job["audio_file"] = out_name
                job["audio_url"] = f"/audio/{out_name}"
                job["sample_rate"] = rate
                # Issue #15: the concatenated artifact is as much an
                # AI-generated file as its segments — mark it too.
                ctx.mark_artifact(job, ctx.audio_dir / out_name, {"job": job["id"], "date": now_iso()})
            except Exception as exc:  # noqa: BLE001 - concat is the last step
                if failure is None and not cancelled:
                    failure = f"分段音频拼接失败：{exc}"

        job["finished_at"] = now_iso()
        if cancelled:
            job["status"] = "cancelled"
        elif failure is not None:
            job["status"] = "failed"
            job["error"] = failure
        else:
            job["status"] = "succeeded"
        ctx.log_bus.publish(
            job["id"],
            {
                "cancelled": "任务已取消",
                "failed": "任务失败",
                "succeeded": "任务完成",
            }.get(job["status"], f"任务 {job['status']}"),
            level={"failed": "error", "cancelled": "warn"}.get(job["status"], "info"),
        )

    async def _drain(self) -> None:
        # FIFO drain by polling: cancel-while-queued is then a plain flag
        # check, and a worker crash is healed by ensure()'s next run.
        ctx = self.ctx
        while True:
            job = ctx.job_store.next_queued()
            if job is None:
                await asyncio.sleep(0.1)
                continue
            if is_cancel_requested(job):
                # Cancelled before it started: no segment ever runs.
                job["status"] = "cancelled"
                job["finished_at"] = now_iso()
                ctx.log_bus.publish(job["id"], "任务已取消（尚未开始）", level="warn")
                continue
            try:
                await self._run_job(job)
            except Exception as exc:  # noqa: BLE001 - defensive: worker must survive
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = now_iso()
                ctx.log_bus.publish(job["id"], f"任务失败：{exc}", level="error")

    def ensure(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    worker = JobWorker(ctx)

    def _segment_plan(
        engine_id: str, text: str, voice_id: str | None
    ) -> tuple[Engine, dict | None, list[str]]:
        """Validate the job inputs up front and cut the segment plan.

        The voice checks mirror the ones run_generation enforces anyway —
        here they fail fast BEFORE anything is queued, instead of failing
        the first segment minutes later."""
        engine = ctx.require_engine(engine_id)
        voice = None
        if voice_id:
            voice = ctx.require_voice(voice_id)
            if not voice.get("reference"):
                raise HTTPException(
                    status_code=409,
                    detail="该音色没有参考样本，无法参与生成；请删除后重新创建",
                )
            if not ctx.voice_store.reference_path(voice_id).is_file():
                raise HTTPException(
                    status_code=409,
                    detail="voice reference sample is missing; delete and recreate the voice",
                )
        max_chars = engine.capabilities().max_chars_per_request
        segments = split_text(text, max_chars) if max_chars else [text]
        return engine, voice, segments

    @router.post("/generations/jobs", dependencies=[Depends(ctx.require_auth)])
    async def create_generation_job(body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text is required")
        params = body.get("params")
        if params is not None and not isinstance(params, dict):
            raise HTTPException(status_code=422, detail="params must be an object")
        engine, voice, segments = _segment_plan(
            body.get("engine_id") or "", text, body.get("voice_id")
        )
        job = new_job(
            engine.engine_id,
            body.get("voice_id"),
            voice["name"] if voice else None,
            text,
            segments,
            params,
        )
        job["created_at"] = now_iso()
        ctx.job_store.add(job)
        worker.ensure()
        max_chars = engine.capabilities().max_chars_per_request
        ctx.log_bus.publish(
            job["id"],
            f"任务已入队：{len(segments)} 个分段（每段 ≤ {max_chars or len(text)} 字符）",
        )
        return public_view(job)

    @router.get("/jobs", dependencies=[Depends(ctx.require_auth)])
    async def list_jobs(limit: int = 50) -> dict:
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        return {"jobs": [public_view(j) for j in ctx.job_store.list(limit)]}

    @router.get("/jobs/{job_id}", dependencies=[Depends(ctx.require_auth)])
    async def get_job(job_id: str) -> dict:
        job = ctx.job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return public_view(job)

    @router.post("/jobs/{job_id}/cancel", dependencies=[Depends(ctx.require_auth)])
    async def cancel_job(job_id: str) -> dict:
        job = ctx.job_store.request_cancel(job_id)
        if job is None:
            # Unknown id, or already terminal: distinguish for the client.
            if ctx.job_store.get(job_id) is None:
                raise HTTPException(status_code=404, detail="job not found")
            raise HTTPException(status_code=409, detail="任务已结束，无法取消")
        ctx.log_bus.publish(job_id, "收到取消请求，将在当前分段结束后停止…", level="warn")
        return public_view(job)

    return router
