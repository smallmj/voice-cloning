"""Direct unit tests for the shared generation pipeline (issue #20).

The point of extracting ``run_generation`` out of the ``create_app``
closure: the synthesis pipeline can now be exercised as a plain function
over an ``AppContext`` — no FastAPI app, no sidecar subprocess.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from voiceclone_sidecar.context import AppContext
from voiceclone_sidecar.engines.fake import FakeEngine
from voiceclone_sidecar.pipeline import run_generation
from voiceclone_sidecar.registry import Registry


@pytest.fixture()
def ctx(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    registry = Registry()
    registry.register(FakeEngine(output_dir=audio_dir))
    return AppContext(
        registry=registry,
        token="t",
        audio_dir=audio_dir,
        data_root=tmp_path,
    )


def test_run_generation_is_a_plain_function(ctx):
    """A full synthesis run with no app object anywhere."""
    record = asyncio.run(run_generation(ctx, "fake", "你好世界"))
    assert record["status"] == "succeeded"
    assert record["audio_url"].startswith("/audio/")
    assert record["normalized_text"]  # normalization layer applied
    stored = ctx.generation_store.get(record["id"])
    assert stored is not None and stored["status"] == "succeeded"


def test_run_generation_unknown_engine_404(ctx):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_generation(ctx, "nope", "text"))
    assert exc.value.status_code == 404


def test_run_generation_unknown_voice_404(ctx):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_generation(ctx, "fake", "text", voice_id="missing"))
    assert exc.value.status_code == 404


def test_run_generation_without_voice_works(ctx):
    record = asyncio.run(run_generation(ctx, "fake", "hello"))
    assert record["voice_id"] is None
    assert record["status"] == "succeeded"
