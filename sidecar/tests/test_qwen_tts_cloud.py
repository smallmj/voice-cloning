"""Unit tests for the Alibaba Model Studio (百炼) Qwen3-TTS-VC cloud engine.

The vendor's HTTP contract is exercised through an httpx.MockTransport, so
these tests assert the exact request shapes documented by Aliyun and the
engine's handling of responses — without any network access.
"""

from __future__ import annotations

import base64
import json
import wave

import httpx
import pytest

from voiceclone_sidecar.engines.cloud_base import CloudEngineError
from voiceclone_sidecar.engines.qwen_tts_cloud import (
    BASE_URL,
    ENROLL_MODEL,
    PRICE_PER_10K_CHARS,
    SYNTH_PATH,
    TARGET_MODEL,
    Qwen3TtsVcCloudEngine,
    billed_chars,
    billing_cost,
)
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.secrets import KeyStore, MemoryBackend


def make_wav_bytes(seconds: float = 0.5, rate: int = 24000) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class Harness:
    """Builds an engine around a scripted MockTransport and records requests."""

    def __init__(self, responses: list[httpx.Response], key: str | None = "sk-test") -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._responses.pop(0)

        self.store = KeyStore(backend=MemoryBackend())
        if key:
            self.store.set("qwen3-tts-vc-cloud", key)
        self.engine = Qwen3TtsVcCloudEngine(
            output_dir=None,
            key_store=self.store,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


@pytest.fixture()
def wav_file(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(make_wav_bytes())
    return p


def test_billed_chars_counts_cjk_as_two():
    assert billed_chars("你好") == 4
    assert billed_chars("你好a") == 5
    assert billed_chars("") == 0


def test_billing_cost_uses_aliyun_rate():
    # 10000 billed chars -> 0.8 CNY (price is an explicit argument since
    # ADR-0015 moved billing_cost into the vendor-neutral cloud base)
    assert billing_cost("你" * 5000, PRICE_PER_10K_CHARS) == 0.8
    assert billing_cost("a" * 10000, PRICE_PER_10K_CHARS) == 0.8


def test_missing_key_fails_with_actionable_message(tmp_path):
    engine = Qwen3TtsVcCloudEngine(key_store=KeyStore(backend=MemoryBackend()))
    with pytest.raises(CloudEngineError, match="API Key"):
        engine.synthesize(
            GenerationRequest(generation_id="g1", text="hi", params={"voice_id": "v"}),
            lambda m: None,
        )


def test_enrollment_request_shape_and_voice_extraction(wav_file):
    harness = Harness(
        [httpx.Response(200, json={"output": {"voice": "qwen3-tts-vc-2026-01-22-vctest-abc"}})]
    )
    extra = harness.engine.bind_reference(wav_file, "参考文本", lambda m: None)
    assert extra == {
        "voice_id": "qwen3-tts-vc-2026-01-22-vctest-abc",
        "target_model": TARGET_MODEL,
    }

    req = harness.requests[0]
    assert req.url == f"{BASE_URL}/services/audio/tts/customization"
    assert req.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == ENROLL_MODEL
    assert body["input"]["action"] == "create"
    assert body["input"]["target_model"] == TARGET_MODEL
    assert body["input"]["text"] == "参考文本"
    data_url = body["input"]["audio"]["data"]
    assert data_url.startswith("data:audio/wav;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == wav_file.read_bytes()


def test_enrollment_rejects_unsupported_container(tmp_path):
    flac = tmp_path / "ref.flac"
    flac.write_bytes(b"not really flac")
    harness = Harness([])
    with pytest.raises(CloudEngineError, match="WAV / MP3 / M4A"):
        harness.engine.bind_reference(flac, None, lambda m: None)


def test_enrollment_surfaces_vendor_errors(wav_file):
    harness = Harness(
        [httpx.Response(401, json={"code": "InvalidApiKey", "message": "bad key"})]
    )
    with pytest.raises(CloudEngineError, match="API Key 无效或无权限"):
        harness.engine.bind_reference(wav_file, None, lambda m: None)


def test_synthesize_downloads_audio_and_reports_cost(tmp_path):
    audio = make_wav_bytes(rate=16000)
    harness = Harness(
        [
            httpx.Response(200, json={"output": {"audio": {"url": "https://cdn.example/a.wav"}}}),
            httpx.Response(200, content=audio),
        ]
    )
    harness.engine.output_dir = tmp_path
    result = harness.engine.synthesize(
        GenerationRequest(generation_id="gen1", text="你好，世界！", params={"voice_id": "v1"}),
        lambda m: None,
    )
    # Request shape.
    synth_req = harness.requests[0]
    assert synth_req.url == f"{BASE_URL}{SYNTH_PATH}"
    body = json.loads(synth_req.content)
    assert body == {"model": TARGET_MODEL, "input": {"text": "你好，世界！", "voice": "v1"}}
    # Artifact is a real local WAV with the vendor's sample rate.
    assert result.sample_rate == 16000
    with wave.open(result.audio_path, "rb") as w:
        assert w.getframerate() == 16000
    # Billing: 6 CJK chars -> 12 billed chars -> 0.00096 CNY.
    assert result.cost == round(12 / 10000 * 0.8, 4)
    assert result.model_version == TARGET_MODEL


def test_synthesize_language_param_only_for_known_choices(tmp_path):
    audio = make_wav_bytes(seconds=0.1)
    harness = Harness(
        [
            httpx.Response(200, json={"output": {"audio": {"url": "https://cdn.example/a.wav"}}}),
            httpx.Response(200, content=audio),
        ]
    )
    harness.engine.output_dir = tmp_path
    harness.engine.synthesize(
        GenerationRequest(
            generation_id="g",
            text="hello",
            params={"voice_id": "v", "language_type": "auto"},
        ),
        lambda m: None,
    )
    body = json.loads(harness.requests[0].content)
    assert "language_type" not in body["input"]  # auto -> omitted

    harness2 = Harness(
        [
            httpx.Response(200, json={"output": {"audio": {"url": "https://cdn.example/a.wav"}}}),
            httpx.Response(200, content=audio),
        ]
    )
    harness2.engine.output_dir = tmp_path
    harness2.engine.synthesize(
        GenerationRequest(
            generation_id="g",
            text="hello",
            params={"voice_id": "v", "language_type": "English"},
        ),
        lambda m: None,
    )
    body = json.loads(harness2.requests[0].content)
    assert body["input"]["language_type"] == "English"


def test_synthesize_requires_cloud_voice_id():
    harness = Harness([])
    with pytest.raises(CloudEngineError, match="云端绑定"):
        harness.engine.synthesize(
            GenerationRequest(generation_id="g", text="hi", params={}),
            lambda m: None,
        )


def test_param_specs_drive_the_ui_contract():
    specs = Qwen3TtsVcCloudEngine().param_specs()
    # Issue #23 / ADR-0018: the canonical name is `language`; the wire key
    # stays the vendor's `language_type`, and "auto" maps to "send nothing".
    assert [s.to_dict()["name"] for s in specs] == ["language"]
    spec = specs[0]
    assert spec.to_dict()["choices"][0] == "auto"
    assert spec.to_dict()["wire_path"] == "language_type"
    assert spec.to_dict()["layer"] == "canonical"
    assert spec.to_wire("auto") is None
    assert spec.to_wire("Japanese") == "Japanese"


# --- voice health check (issue #12) -----------------------------------------


def test_check_voice_alive_returns_true():
    harness = Harness(
        [httpx.Response(200, json={"output": {"audio": {"url": "https://cdn.example/a.wav"}}})]
    )
    assert harness.engine.check_voice("vc-abc", lambda m: None) is True
    body = json.loads(harness.requests[0].content)
    # The probe is a minimal synthesis against the target model with the
    # bound voice — the cheapest documented "is this voice alive" call.
    assert body["model"] == TARGET_MODEL
    assert body["input"]["voice"] == "vc-abc"
    assert body["input"]["text"] == "好"


def test_check_voice_missing_returns_false():
    harness = Harness(
        [
            httpx.Response(
                400,
                json={"code": "InvalidParameter", "message": "voice 不存在或已被删除"},
            )
        ]
    )
    assert harness.engine.check_voice("vc-dead", lambda m: None) is False


def test_check_voice_unrelated_failure_raises():
    harness = Harness(
        [httpx.Response(401, json={"code": "InvalidApiKey", "message": "无效 API Key"})]
    )
    with pytest.raises(CloudEngineError, match="健康检查"):
        harness.engine.check_voice("vc-abc", lambda m: None)


def test_voice_missing_error_never_fires_without_voice_mention():
    # A generic vendor error (e.g. text too long) must NOT be read as a
    # dead voice — the health check raises instead of rebuilding.
    from voiceclone_sidecar.engines.cloud_base import voice_missing_error

    assert voice_missing_error(
        httpx.Response(400, json={"code": "InvalidParameter", "message": "voice xxx 已删除"})
    )
    assert not voice_missing_error(
        httpx.Response(400, json={"code": "InvalidParameter", "message": "文本过长"})
    )
    assert not voice_missing_error(httpx.Response(500, content=b"boom"))
