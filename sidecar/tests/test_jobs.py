"""Contract tests for issue #13: long-text segmentation, job queue, cancel,
and rate-limit absorption.

Runs against the real sidecar process (conftest), which registers the
fake-slow and fake-rate-limited test engines alongside the regular fakes.
Long text is segmented per the engine's declared max_chars_per_request, each
segment becomes a REAL generation record, and the segment audio is stitched
into one complete artifact.
"""

from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import httpx

from voiceclone_sidecar.segmentation import concat_wav_files, split_text

TERMINAL = {"succeeded", "failed", "cancelled"}


def _wait_terminal(client: httpx.Client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in TERMINAL:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _create_job(client: httpx.Client, engine_id: str, text: str, voice_id=None, params=None) -> dict:
    body = {"engine_id": engine_id, "text": text}
    if voice_id:
        body["voice_id"] = voice_id
    if params:
        body["params"] = params
    r = client.post("/generations/jobs", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# -- unit-level: segmentation -------------------------------------------------


def test_split_text_prefers_sentence_boundaries_and_round_trips():
    text = "第一句话。第二句话！第三句话？" * 10
    segments = split_text(text, 40)
    assert len(segments) > 1
    assert all(len(s) <= 40 for s in segments)
    assert "".join(segments) == text  # exact round-trip
    # Every segment ends at a sentence boundary unless hard-split.
    for seg in segments[:-1]:
        assert seg[-1] in "。！？!?\n，,、；;：:"


def test_split_text_hard_splits_a_single_monster_sentence():
    segments = split_text("a" * 250, 100)
    assert segments == ["a" * 100, "a" * 100, "a" * 50]


def test_split_text_short_text_is_one_segment():
    assert split_text("你好。", 100) == ["你好。"]


def test_concat_wav_files_resamples_differing_rates(tmp_path: Path):
    def write(name: str, rate: int, seconds: float) -> Path:
        p = tmp_path / name
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x01\x02" * int(rate * seconds))
        return p

    a = write("a.wav", 22050, 0.5)
    b = write("b.wav", 44100, 0.5)
    out = tmp_path / "out.wav"
    rate = concat_wav_files([a, b], out)
    assert rate == 22050
    with wave.open(str(out), "rb") as w:
        # 0.5s @22050 + 0.5s resampled from 44100 -> ~22050 frames total.
        assert abs(w.getnframes() - 22050) < 500
        assert w.getnframes() > 20000


# -- contract: capabilities declaration ----------------------------------------


def test_engines_declare_max_chars_per_request(client):
    engines = client.get("/engines").json()["engines"]
    by_id = {e["id"]: e for e in engines}
    assert by_id["fake"]["capabilities"]["max_chars_per_request"] == 100
    assert by_id["qwen3-tts-vc-cloud"]["capabilities"]["max_chars_per_request"] == 2000
    assert by_id["qwen3-tts-vd-cloud"]["capabilities"]["max_chars_per_request"] == 2000


# -- contract: job lifecycle ----------------------------------------------------


def test_job_splits_long_text_and_concatenates_audio(client):
    text = "第一句话。第二句话！第三句话？Third sentence here. " * 5  # > 100 chars
    job = _create_job(client, "fake", text)
    assert job["status"] == "queued"
    assert job["segment_count"] == len(job["segments"]) > 1
    for seg in job["segments"]:
        assert len(seg["text"]) <= 100
        assert "".join(s["text"] for s in job["segments"]) == text

    job = _wait_terminal(client, job["id"])
    assert job["status"] == "succeeded", job
    assert all(seg["status"] == "succeeded" for seg in job["segments"])
    assert all(seg["generation_id"] for seg in job["segments"])

    # Each segment is a REAL generation record in durable history.
    for seg in job["segments"]:
        rec = client.get(f"/generations/{seg['generation_id']}")
        assert rec.status_code == 200
        assert rec.json()["text"] == seg["text"]

    # The concatenated artifact is one playable WAV longer than any segment.
    audio = client.get(job["audio_url"])
    assert audio.status_code == 200
    assert audio.content[:4] == b"RIFF"
    with wave.open(io.BytesIO(audio.content), "rb") as w:
        assert w.getnframes() > 0
        assert w.getframerate() == job["sample_rate"]


def test_job_applies_engine_params_to_every_segment(client):
    """Queueing must not alter single-segment behavior: user-facing engine
    parameters ride along on EVERY segment, exactly as one plain generation
    would receive them."""
    job = _create_job(client, "fake", "第一句话。" * 30, params={"fake_mode": "fast"})
    assert job["params"] == {"fake_mode": "fast"}
    job = _wait_terminal(client, job["id"])
    assert job["status"] == "succeeded", job
    for seg in job["segments"]:
        record = client.get(f"/generations/{seg['generation_id']}").json()
        assert record["params"]["fake_mode"] == "fast"


def test_single_segment_text_via_plain_generations_is_unchanged(client):
    """The existing synchronous path still works exactly as before —
    segmentation/queueing never changes short-text behavior."""
    r = client.post("/generations", json={"engine_id": "fake", "text": "你好，世界。"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "succeeded"
    assert r.json()["audio_url"]


def test_queued_job_can_be_cancelled_before_it_starts(client):
    # Occupies the single worker for a while: 4 slow segments.
    blocker = _create_job(client, "fake-slow", "第一句。第二句。第三句。第四句。" * 10)
    r = client.post("/generations", json={"engine_id": "fake", "text": "你好。"})
    assert r.status_code == 200  # fast path does not queue behind the job

    queued = _create_job(client, "fake", "第一句话。" * 30)
    cancel = client.post(f"/jobs/{queued['id']}/cancel")
    assert cancel.status_code == 200, cancel.text

    job = _wait_terminal(client, queued["id"])
    assert job["status"] == "cancelled"
    assert all(seg["status"] == "pending" for seg in job["segments"]), job["segments"]
    assert job["audio_url"] is None  # nothing ran, nothing to keep
    _wait_terminal(client, blocker["id"])


def test_running_job_can_be_cancelled_and_keeps_completed_audio(client):
    job = _create_job(client, "fake-slow", "第一句。第二句。第三句。第四句。第五句。第六句。" * 10)
    # Let at least one segment finish, then cancel mid-run.
    time.sleep(2.5)
    assert client.post(f"/jobs/{job['id']}/cancel").status_code == 200

    job = _wait_terminal(client, job["id"])
    assert job["status"] == "cancelled"
    statuses = [seg["status"] for seg in job["segments"]]
    assert "succeeded" in statuses  # some segments finished
    assert "skipped" in statuses  # the rest were skipped, not run
    assert job["audio_url"], "completed segments must be kept as audio"

    # Cancelling again on a terminal job is refused, not silently accepted.
    r = client.post(f"/jobs/{job['id']}/cancel")
    assert r.status_code == 409


def test_job_failure_is_recorded_per_segment(client):
    job = _create_job(client, "fake-broken-rebuild", "第一句。" * 30)
    job = _wait_terminal(client, job["id"])
    # No key configured for the fake cloud engine: the first segment fails
    # and the job stops with a clear, segment-local error.
    assert job["status"] == "failed"
    failed = [s for s in job["segments"] if s["status"] == "failed"]
    assert failed and failed[0]["error"]


# -- contract: rate-limit absorption ---------------------------------------------


def _create_voice(client: httpx.Client) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(3.5 * 16000))
    r = client.post(
        "/voices",
        data={"name": "限流测试音色", "description": ""},
        files={"file": ("ref.wav", buf.getvalue(), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_rate_limit_errors_are_absorbed_by_retry(client):
    r = client.put("/settings/keys", json={"engine_id": "fake-rate-limited", "key": "sk-test"})
    assert r.status_code == 200, r.text
    voice_id = _create_voice(client)

    # The vendor throttles this engine's first two calls, then lets it
    # through. The pipeline retries with backoff; the generation succeeds
    # and the retry is visible only as a log line.
    r = client.post(
        "/generations",
        json={"engine_id": "fake-rate-limited", "text": "你好，世界。", "voice_id": voice_id},
    )
    assert r.status_code == 200, r.text
    record = r.json()
    assert record["status"] == "succeeded"
    assert any("限流" in line["message"] for line in record["logs"])
