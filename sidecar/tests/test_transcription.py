"""Contract tests: reference diagnostics + transcription (issue #8)."""

from __future__ import annotations

import io
import math
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from voiceclone_sidecar import transcription
from voiceclone_sidecar.runtime import uvman

from .test_voices import create_voice


def sine_wav_bytes(seconds: float = 4.0, freq: float = 220.0, rate: int = 16000) -> bytes:
    frames = int(rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / rate)))
                for i in range(frames)
            )
        )
    return buf.getvalue()


# --- diagnostics --------------------------------------------------------------


def test_diagnose_upload_returns_diagnostics_with_advice(client):
    r = client.post(
        "/diagnose",
        files={"file": ("ref.wav", sine_wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    ids = {d["id"] for d in data["diagnostics"]}
    assert {"snr", "speaker", "clipping", "silence"} <= ids
    for d in data["diagnostics"]:
        assert d["status"] in ("good", "warn", "bad")
        assert d["message"].strip()
        assert d["advice"].strip()


def test_diagnose_rejects_empty_upload(client):
    r = client.post("/diagnose", files={"file": ("ref.wav", b"", "audio/wav")})
    assert r.status_code == 422


def test_voice_diagnose_never_modifies_the_reference(client):
    """The app must not touch user material: byte-identical before/after."""
    voice = create_voice(client, seconds=4.0)
    r0 = client.get(f"/voices/{voice['id']}/reference")
    before = r0.content
    r = client.get(f"/voices/{voice['id']}/diagnose")
    assert r.status_code == 200, r.text
    assert r.json()["diagnostics"]
    r1 = client.get(f"/voices/{voice['id']}/reference")
    assert r1.content == before


def test_voice_diagnose_unknown_voice(client):
    assert client.get("/voices/nope/diagnose").status_code == 404


# --- issue #18: ASR honesty — the test double leaves production ---------------


def test_default_registry_without_test_env_excludes_fake_engine(monkeypatch, tmp_path):
    """Production (no VOICECLONE_TEST_ENGINES) must not register the fake
    engine at all: it cannot appear in engine lists or the transcription
    provider list, and no path can select it."""
    monkeypatch.delenv("VOICECLONE_TEST_ENGINES", raising=False)
    from voiceclone_sidecar.registry import default_registry

    registry = default_registry(output_dir=tmp_path / "audio")
    assert registry.get("fake") is None


def test_default_registry_without_test_env_has_no_engine_transcribers(monkeypatch, tmp_path):
    """The transcription provider list has no test double in production.
    Since issue #33 the real cloud vendor (qwen3-tts-vc-cloud via DashScope
    Qwen-ASR) is the engine transcriber; the fake must never be."""
    monkeypatch.delenv("VOICECLONE_TEST_ENGINES", raising=False)
    from voiceclone_sidecar.registry import default_registry
    from voiceclone_sidecar.transcription import engine_transcribers

    registry = default_registry(output_dir=tmp_path / "audio")
    ids = [e.engine_id for e in engine_transcribers(registry)]
    assert ids == ["qwen3-tts-vc-cloud"]


def test_default_registry_with_test_env_keeps_fake_engine(monkeypatch, tmp_path):
    """The test-engine switch keeps the double available for contract tests."""
    monkeypatch.setenv("VOICECLONE_TEST_ENGINES", "1")
    from voiceclone_sidecar.registry import default_registry

    registry = default_registry(output_dir=tmp_path / "audio")
    assert registry.get("fake") is not None


def test_placeholder_transcript_helper():
    from voiceclone_sidecar.transcription import (
        FAKE_TRANSCRIPT_PLACEHOLDER,
        is_placeholder_transcript,
    )

    assert is_placeholder_transcript(FAKE_TRANSCRIPT_PLACEHOLDER)
    assert is_placeholder_transcript(f"  {FAKE_TRANSCRIPT_PLACEHOLDER}\n")
    assert not is_placeholder_transcript("真正的参考音频转写内容。")
    assert not is_placeholder_transcript("")
    assert not is_placeholder_transcript(None)


def test_placeholder_transcript_in_library_is_flagged(client):
    """A placeholder already written into the voice library by an earlier
    version is recognized and surfaced to the UI as an actionable flag."""
    voice = create_voice(client, seconds=4.0)
    client.put("/transcription/provider", json={"provider": "fake"})
    r = client.post(f"/voices/{voice['id']}/transcribe", json={})
    assert r.status_code == 200, r.text
    client.put("/transcription/provider", json={"provider": "local"})

    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["reference"]["transcript"].strip()
    assert stored["reference"]["transcript_placeholder"] is True

    listed = client.get("/voices").json()["voices"]
    match = next(v for v in listed if v["id"] == voice["id"])
    assert match["reference"]["transcript_placeholder"] is True


def test_real_transcript_is_not_flagged(client, sidecar):
    """A genuine transcript must not raise the placeholder warning."""
    from voiceclone_sidecar.voices import VoiceStore

    voice = create_voice(client, seconds=4.0)
    store = VoiceStore(Path(sidecar["data_dir"]))
    store.set_transcript(voice["id"], "这是真实参考音频的转写内容。")
    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["reference"]["transcript_placeholder"] is False


# --- transcription providers + settings ---------------------------------------


def test_providers_lists_local_and_fake_engine(client):
    r = client.get("/transcription/providers")
    assert r.status_code == 200
    data = r.json()
    assert data["default"] == "local"
    assert data["provider"] == "local"
    assert data["local"]["supported"] is True
    assert any(e["id"] == "fake" for e in data["engines"])


def test_set_provider_rejects_unknown(client):
    r = client.put("/transcription/provider", json={"provider": "no-such-vendor"})
    assert r.status_code == 422


def test_set_provider_accepts_registered_engine_and_persists(client):
    r = client.put("/transcription/provider", json={"provider": "fake"})
    assert r.status_code == 200, r.text
    assert client.get("/transcription/providers").json()["provider"] == "fake"
    # restore default for other tests
    client.put("/transcription/provider", json={"provider": "local"})


def test_local_tool_status_reports_uninstalled(client):
    """Assumes the local tool is not installed on the machine running the
    tests — the status endpoint reads the REAL runtime root, so once it has
    been installed (e.g. by a manual smoke test) this contract moves to a
    clean machine / CI."""
    if client.get("/transcription/local/status").json()["installed"]:
        pytest.skip("local transcription tool is installed on this machine")
    r = client.get("/transcription/local/status")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "transcribe-local"
    assert data["installed"] is False


def test_local_tool_status_progress_contract(client):
    """Issue #42: the transcription tool status endpoint exposes the SAME
    unified progress structure engine installs do (issue #35) — the engine
    page renders both with one shared helper. With no live install the
    field is present and null (a stale byte count must never look alive)."""
    data = client.get("/transcription/local/status").json()
    assert set(data) >= {"id", "installed", "installing", "steps", "progress"}
    assert data["progress"] is None


def test_local_tool_install_state_carries_progress(tmp_path):
    """Unit seam for the live case (cross-process HTTP tests cannot fake a
    running install): a mid-download install state carries the byte-level
    ``progress`` through ``install_state`` so the endpoint can expose it."""
    if transcription.platform_config() is None:
        pytest.skip("local transcription is unsupported on this platform")

    from voiceclone_sidecar.runtime import installer

    tool = transcription.LocalTranscriber(root=tmp_path)
    # Exactly what a live install persists mid-download (installer._persist_progress):
    # progress sits in the state file between the running and cleared moments.
    installer.save_state(
        tool._ctx(),
        {
            "steps": {"weights": {"status": "running"}},
            "progress": {"file": "weights:model.bin", "done_bytes": 1024, "total_bytes": 4096},
        },
    )
    state = tool.install_state()
    assert state["progress"] == {
        "file": "weights:model.bin",
        "done_bytes": 1024,
        "total_bytes": 4096,
    }


def test_transcribe_via_engine_provider_stores_transcript(client):
    """Cloud transcription reuses a registered engine — no new vendor."""
    client.put("/transcription/provider", json={"provider": "fake"})
    voice = create_voice(client, seconds=4.0)
    r = client.post(f"/voices/{voice['id']}/transcribe", json={})
    assert r.status_code == 200, r.text
    assert r.json()["reference"]["transcript"].strip()
    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["reference"]["transcript"] == r.json()["reference"]["transcript"]
    client.put("/transcription/provider", json={"provider": "local"})


def test_transcribe_with_uninstalled_local_tool_is_actionable(client):
    """Same machine-state caveat as test_local_tool_status_reports_uninstalled:
    the uninstalled-guidance contract is only reachable when the tool is
    genuinely not installed on this machine's runtime root."""
    if client.get("/transcription/local/status").json()["installed"]:
        pytest.skip("local transcription tool is installed on this machine")
    voice = create_voice(client, seconds=4.0)
    r = client.post(f"/voices/{voice['id']}/transcribe", json={})
    assert r.status_code == 409
    assert "安装" in r.json()["detail"]  # the error says what to do next


def test_transcribe_unknown_voice(client):
    assert client.post("/voices/nope/transcribe", json={}).status_code == 404


# --- reference-text auto-fill at generation time --------------------------------


def test_generation_auto_fills_ref_text_from_transcript(client):
    voice = create_voice(client, seconds=4.0)
    client.put("/transcription/provider", json={"provider": "fake"})
    r = client.post(
        "/generations",
        json={"engine_id": "fake-ref-text", "text": "你好", "voice_id": voice["id"]},
    )
    assert r.status_code == 200, r.text
    record = r.json()
    assert record["status"] == "succeeded"
    assert record["params"]["ref_text"].strip()
    # The transcript was persisted, so a rerun never re-transcribes by hand.
    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["reference"]["transcript"] == record["params"]["ref_text"]
    client.put("/transcription/provider", json={"provider": "local"})


def test_generation_with_ref_text_engine_fails_actionably_without_transcription(client):
    """Local provider uninstalled + no transcript ⇒ a 409 that says what to do,
    never a silent send-to-cloud fallback."""
    client.put("/transcription/provider", json={"provider": "local"})
    voice = create_voice(client, seconds=4.0)
    r = client.post(
        "/generations",
        json={"engine_id": "fake-ref-text", "text": "你好", "voice_id": voice["id"]},
    )
    assert r.status_code == 409, r.text
    assert "转写" in r.json()["detail"]


def test_ref_text_engine_without_voice_still_generates(client):
    r = client.post("/generations", json={"engine_id": "fake-ref-text", "text": "无音色"})
    assert r.status_code == 200, r.text


def test_engines_expose_requires_reference_text(client):
    engines = {e["id"]: e for e in client.get("/engines").json()["engines"]}
    assert engines["fake"]["capabilities"]["requires_reference_text"] is False
    assert engines["fake-ref-text"]["capabilities"]["requires_reference_text"] is True


def test_generation_does_not_reuse_placeholder_transcript(client):
    """The legacy fake engine's placeholder transcript (issue #18) is NOT a
    transcript: a ref-text-conditioned generation must not reuse it silently
    (it degrades FireRedTTS3/VoxCPM2 clones to near-silence/gibberish).
    Instead the pipeline re-transcribes — and fails with an actionable 409
    when no transcription provider is available, rather than shipping a
    broken clone."""
    voice = create_voice(client, seconds=4.0)
    # Transcribe via the fake engine provider: the placeholder gets stored.
    client.put("/transcription/provider", json={"provider": "fake"})
    client.post(f"/voices/{voice['id']}/transcribe", json={})
    stored = client.get(f"/voices/{voice['id']}").json()["reference"]["transcript"]
    assert stored == "这是一段用于测试的转写文本。"
    # Switch to the (uninstalled) local provider and generate: the stored
    # placeholder must NOT be reused as-is — the pipeline must try to
    # re-transcribe and fail loudly.
    client.put("/transcription/provider", json={"provider": "local"})
    r = client.post(
        "/generations",
        json={"engine_id": "fake-ref-text", "text": "你好", "voice_id": voice["id"]},
    )
    assert r.status_code == 409, r.text
    assert "自动转写失败" in r.json()["detail"]
    client.put("/transcription/provider", json={"provider": "local"})


def test_transcribe_unknown_voice_returns_404(client):
    r = client.post("/voices/nope/transcribe", json={})
    assert r.status_code == 404
    assert r.json()["detail"] == "voice not found"


# --- local tool install: the weights step's mirror fallback ---------------------


def test_local_weights_step_retries_through_hf_mirror(tmp_path, monkeypatch):
    """Regression: the weights step called its snapshot helper with no argument
    while the helper took a required one, so the download died with a TypeError
    before touching the network. Observed on a real machine as a 'failed'
    weights step in the transcribe-local install state; no test covered it."""
    if transcription.platform_config() is None:
        pytest.skip("local transcription is unsupported on this platform")

    seen: list[dict] = []

    def fake_run(cmd, **kwargs):
        seen.append(kwargs.get("env") or {})
        # Non-zero on every attempt, so the step must try HF then hf-mirror
        # and finally raise a plain RuntimeError.
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: tmp_path / "uv")
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(uvman, "venv_python", lambda *a, **k: tmp_path / "python")
    monkeypatch.setattr(transcription.subprocess, "run", fake_run)
    # _snapshot merges os.environ, so an ambient HF_ENDPOINT would make the
    # first attempt look like a mirror attempt. Start from a clean slate.
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    tool = transcription.LocalTranscriber(root=tmp_path, env={})
    steps = {s.id: s for s in tool.install_steps()}
    assert "weights" in steps

    with pytest.raises(RuntimeError, match="model download failed"):
        steps["weights"].run(lambda m: None, lambda *a: None)

    assert len(seen) == 2, "expected one plain attempt plus one hf-mirror retry"
    assert seen[0].get("HF_ENDPOINT") != "https://hf-mirror.com"
    assert seen[1].get("HF_ENDPOINT") == "https://hf-mirror.com"


# --- real cloud provider (issue #33) ----------------------------------------------
#
# The production provider list is exercised against the REAL default registry
# (no fake registered: VOICECLONE_TEST_ENGINES is not set in-process), while
# the vendor's HTTP leg is an httpx.MockTransport injected into the
# already-registered qwen3-tts-vc-cloud engine — the full route runs.


@pytest.fixture()
def cloud_client(tmp_path):
    from types import SimpleNamespace

    import httpx
    from fastapi.testclient import TestClient

    from voiceclone_sidecar.main import create_app
    from voiceclone_sidecar.registry import default_registry
    from voiceclone_sidecar.secrets import KeyStore, MemoryBackend

    store = KeyStore(backend=MemoryBackend())
    store.set("qwen3-tts-vc-cloud", "sk-test")
    registry = default_registry(key_store=store, env={})
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "output": {"choices": [{"message": {"content": [{"text": " 云端转写结果。 "}]}}]}
            },
        )

    # Swap in a vc engine built on the ADR-0015 constructor seam (injectable
    # httpx client); the registry has no unregister, so replace the entry.
    from voiceclone_sidecar.engines.qwen_tts_cloud import (
        BASE_URL,
        SYNTH_PATH,
        Qwen3TtsVcCloudEngine,
    )

    registry._engines["qwen3-tts-vc-cloud"] = Qwen3TtsVcCloudEngine(
        output_dir=tmp_path / "audio",
        key_store=store,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    app = create_app(
        registry,
        token="t",
        audio_dir=tmp_path / "audio",
        data_dir=tmp_path / "data",
        key_store=store,
    )
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer t"})
    try:
        yield SimpleNamespace(
            client=client,
            seen=seen,
            base_url_seen=f"{BASE_URL}{SYNTH_PATH}",
            registry=registry,
        )
    finally:
        client.put("/transcription/provider", json={"provider": "local"})


def test_production_providers_include_the_real_cloud_vendor(cloud_client):
    data = cloud_client.client.get("/transcription/providers").json()
    ids = {e["id"] for e in data["engines"]}
    assert "qwen3-tts-vc-cloud" in ids
    assert "fake" not in ids, "issue #18: the test double must stay out"
    entry = next(e for e in data["engines"] if e["id"] == "qwen3-tts-vc-cloud")
    assert entry["requires_key"] is True
    assert entry["key_configured"] is True


def test_cloud_transcribe_writes_transcript_and_feeds_ref_text(cloud_client):
    from voiceclone_sidecar.engines.fake import FakeRefTextEngine

    voice = cloud_client.client.post(
        "/voices",
        data={"name": "云转写", "description": "d"},
        files={"file": ("ref.wav", sine_wav_bytes(4.0), "audio/wav")},
    ).json()

    r = cloud_client.client.put("/transcription/provider", json={"provider": "qwen3-tts-vc-cloud"})
    assert r.status_code == 200, r.text

    r = cloud_client.client.post(f"/voices/{voice['id']}/transcribe", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reference"]["transcript"] == "云端转写结果。"
    assert body["reference"]["transcript_placeholder"] is False
    assert cloud_client.seen, "the vendor must have been called"
    assert cloud_client.seen[0].url == cloud_client.base_url_seen

    # The stored transcript is what a requires_reference_text engine gets.
    cloud_client.registry.register(FakeRefTextEngine(output_dir=None))
    r = cloud_client.client.post(
        "/generations",
        json={"engine_id": "fake-ref-text", "text": "你好", "voice_id": voice["id"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["params"]["ref_text"] == "云端转写结果。"


def test_cloud_transcribe_without_key_fails_actionably_and_never_falls_back(
    cloud_client,
):
    voice = cloud_client.client.post(
        "/voices",
        data={"name": "无密钥", "description": "d"},
        files={"file": ("ref.wav", sine_wav_bytes(4.0), "audio/wav")},
    ).json()
    cloud_client.client.put("/transcription/provider", json={"provider": "qwen3-tts-vc-cloud"})
    r = cloud_client.client.delete("/settings/keys/qwen3-tts-vc-cloud")
    assert r.status_code == 200

    r = cloud_client.client.post(f"/voices/{voice['id']}/transcribe", json={})
    assert r.status_code == 409, r.text
    assert "API Key" in r.json()["detail"], "the hint must be actionable"
    # No silent fallback: the stored transcript is untouched and the provider
    # choice stays where the user put it.
    assert (
        cloud_client.client.get(f"/voices/{voice['id']}").json()["reference"].get("transcript")
        is None
    )
    assert (
        cloud_client.client.get("/transcription/providers").json()["provider"]
        == "qwen3-tts-vc-cloud"
    )
    assert cloud_client.seen == []


def test_darwin_model_repo_not_the_removed_one():
    """Regression (install 'no reaction'): `mlx-community/whisper-small` was
    removed from the Hub (404 on huggingface.co AND hf-mirror, verified
    2026-09), so the weights step failed within a second of every install
    attempt and the engine-page card looked dead. The config must point at a
    repo that actually resolves (`whisper-small-mlx`, the surviving rename
    with the same config.json + weights.npz payload)."""
    cfg = transcription.PLATFORM_CONFIG.get("darwin")
    if cfg is None:
        pytest.skip("darwin config not present on this platform")
    assert cfg["model_repo"] == "mlx-community/whisper-small-mlx"
