"""中文回归会话 + 固化能力矩阵（issue #16）。"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext
from ..generations import now_iso
from ..pipeline import run_generation
from ..registry import InstallableEngine
from ..regression import (
    REGRESSION_ITEMS,
    aggregate_session,
    build_capability_matrix,
    char_error_rate,
    load_matrix_data,
    make_item_record,
    sample_peak_vram,
    wav_duration_seconds,
)


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    registry = ctx.registry
    regression_store = ctx.regression_store
    audio_dir = ctx.audio_dir
    _require_voice = ctx.require_voice

    def _regression_worker(
        session_id: str, plan: list[dict], session_voice_id: str | None, asr_check: bool
    ) -> asyncio.Task[None]:
        """Schedule one regression session on the MAIN event loop (issue #28).

        The worker used to run in a daemon thread via ``asyncio.run``, i.e. a
        second event loop publishing onto asyncio.Queues created by the main
        loop without ``call_soon_threadsafe`` — a cross-loop delivery the
        asyncio debug mode flags. Running as a task on the main loop keeps
        LogBus publishing, per-engine locks and store access on one loop;
        blocking work (synthesis, ASR, ffprobe) already hops to executors
        inside the pipeline. The caller holds ``ctx.regression_lock`` for the
        whole run, so sessions are serialized process-wide.
        """

        async def _run():
            for step in plan:
                item = next(i for i in REGRESSION_ITEMS if i["id"] == step["item_id"])
                record_item = make_item_record(
                    step["engine_id"], item, status="running"
                )

                def _flush():
                    fresh = regression_store.get(session_id)
                    if fresh is None:
                        return
                    fresh["items"] = [
                        record_item  # noqa: B023 - invoked within the same iteration
                        if i.get("item_id") == step["item_id"]  # noqa: B023 - invoked within the same iteration
                        and i.get("engine_id") == step["engine_id"]  # noqa: B023 - invoked within the same iteration
                        else i
                        for i in fresh["items"]
                    ]
                    regression_store.update(fresh)

                try:
                    if step.get("local"):
                        with sample_peak_vram() as vram:
                            record = await run_generation(
                                ctx,
                                engine_id=step["engine_id"],
                                text=item["text"],
                                voice_id=session_voice_id,
                                generation_id=uuid.uuid4().hex,
                            )
                        record_item["peak_vram_bytes"] = vram.get("peak_vram_bytes")
                    else:
                        record = await run_generation(
                            ctx,
                            engine_id=step["engine_id"],
                            text=item["text"],
                            voice_id=session_voice_id,
                            generation_id=uuid.uuid4().hex,
                        )
                    record_item["generation_id"] = record["id"]
                    record_item["audio_url"] = record.get("audio_url")
                    # 口径：run_generation 的 duration_seconds 是端到端墙钟
                    # （含引擎加载与归一化）——与 docs/evaluation/README 一致。
                    record_item["wall_seconds"] = record.get("duration_seconds")
                    audio_file = record.get("audio_file")
                    if audio_file:
                        seconds = wav_duration_seconds(audio_dir / audio_file)
                        record_item["audio_seconds"] = round(seconds, 3) if seconds else None
                        if seconds and record_item["wall_seconds"]:
                            record_item["rtf"] = round(
                                seconds / record_item["wall_seconds"], 3
                            )
                        if asr_check:
                            try:
                                transcript = await asyncio.get_running_loop().run_in_executor(
                                    None,
                                    lambda: ctx.local_transcriber().transcribe(
                                        str(audio_dir / audio_file), lambda m: None  # noqa: B023 - awaited within the same iteration
                                    ),
                                )
                                record_item["asr_text"] = transcript
                                record_item["cer"] = char_error_rate(item["expect"], transcript)
                            except Exception as exc:  # noqa: BLE001 - ASR is additive
                                record_item["asr_text"] = None
                                record_item["error"] = f"asr: {exc}"
                    record_item["status"] = "ok"
                except Exception as exc:  # noqa: BLE001 - recorded per item
                    record_item["status"] = "failed"
                    detail = getattr(exc, "detail", None) or str(exc)
                    record_item["error"] = detail
                _flush()
            fresh = regression_store.get(session_id)
            if fresh is not None:
                fresh["status"] = "completed"
                fresh["finished_at"] = now_iso()
                regression_store.update(fresh)

        async def _guarded():
            try:
                await _run()
            except asyncio.CancelledError:
                # Cancelled via session delete: the record is (being) removed;
                # mark failed only if it is still readable.
                fresh = regression_store.get(session_id)
                if fresh is not None:
                    fresh["status"] = "failed"
                    fresh["error"] = "cancelled"
                    fresh["finished_at"] = now_iso()
                    regression_store.update(fresh)
                raise
            except Exception as exc:  # noqa: BLE001 - a crashed run must NOT read
                # as "completed" evidence: the matrix only merges sessions whose
                # status is exactly "completed".
                fresh = regression_store.get(session_id)
                if fresh is not None:
                    fresh["status"] = "failed"
                    fresh["error"] = str(exc)
                    fresh["finished_at"] = now_iso()
                    regression_store.update(fresh)
            finally:
                ctx.regression_tasks.pop(session_id, None)
                ctx.regression_lock.release()

        task = asyncio.create_task(_guarded(), name=f"regression-{session_id[:8]}")
        ctx.regression_tasks[session_id] = task
        return task

    @router.post("/regression/run", dependencies=[Depends(ctx.require_auth)])
    async def run_regression(body: dict) -> dict:
        """Start a Chinese-regression run: engines × categories, background."""
        voice_id = body.get("voice_id")
        if voice_id is not None:
            _require_voice(voice_id)
        engine_ids = body.get("engine_ids") or [
            e.engine_id for e in registry.list()
        ]
        for engine_id in engine_ids:
            ctx.require_engine(engine_id)
        asr_check = bool(body.get("asr_check", False))
        if asr_check:
            tr = ctx.local_transcriber()
            if not tr.is_installed():
                raise HTTPException(
                    status_code=409,
                    detail="本地转写工具未安装，无法做 ASR 可懂度复核；"
                    "可先到「引擎」页的转写工具卡片安装，或关闭 asr_check 重试",
                )

        plan = []
        for engine_id in engine_ids:
            engine = registry.get(engine_id)
            # 本地可安装引擎（MLX / CUDA）才做峰值显存采样；云端引擎的
            # "本地显存"口径不存在，采样无意义。
            is_local = isinstance(engine, InstallableEngine)
            for item in REGRESSION_ITEMS:
                plan.append({
                    "engine_id": engine_id,
                    "item_id": item["id"],
                    "category": item["category"],
                    "local": is_local,
                })
        # Issue #28: one regression run at a time. A second POST used to
        # start a second event loop contending for the same stores and
        # per-engine locks; refuse it instead of queueing a heavier fight.
        if not ctx.regression_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="已有回归运行在进行，请等待完成或删除该会话后再试",
            )
        try:
            session = {
                "id": uuid.uuid4().hex,
                "created_at": now_iso(),
                "status": "running",
                "voice_id": voice_id,
                "engines": list(engine_ids),
                "asr_check": asr_check,
                "items": [
                    make_item_record(p["engine_id"], {"id": p["item_id"], "category": p["category"]})
                    for p in plan
                ],
            }
            regression_store.create(session)
            _regression_worker(session["id"], plan, voice_id, asr_check)
        except BaseException:
            # The task owns the release only once it exists; a failure
            # before that must not wedge /regression/run with 409s.
            ctx.regression_lock.release()
            raise
        return session

    @router.get("/regression/sessions", dependencies=[Depends(ctx.require_auth)])
    async def list_regression_sessions(limit: int = 20) -> dict:
        sessions = regression_store.list(limit)
        return {"sessions": [{**s, "summary": aggregate_session(s)} for s in sessions]}

    @router.get("/regression/{session_id}", dependencies=[Depends(ctx.require_auth)])
    async def get_regression_session(session_id: str) -> dict:
        session = regression_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="regression session not found")
        return {**session, "summary": aggregate_session(session)}

    @router.delete("/regression/{session_id}", dependencies=[Depends(ctx.require_auth)])
    async def delete_regression_session(session_id: str) -> dict:
        # Issue #28: deleting a running session also cancels its in-flight
        # task. The task is awaited BEFORE the record is removed, so the
        # cancelled run can never re-materialize a "failed" record after the
        # client saw the delete succeed, and the serialization lock is
        # guaranteed released when this handler returns.
        task = ctx.regression_tasks.get(session_id)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        session = regression_store.delete(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="regression session not found")
        return {"deleted": session_id}

    @router.get("/capability-matrix", dependencies=[Depends(ctx.require_auth)])
    async def capability_matrix() -> dict:
        """The fixated capability matrix: curated data + runtime declarations
        + the latest regression evidence, ready for the UI to render."""
        return build_capability_matrix(registry, load_matrix_data(), regression_store)

    return router
