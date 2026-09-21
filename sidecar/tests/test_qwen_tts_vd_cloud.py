"""Unit tests for the Alibaba Model Studio (百炼) Qwen voice-design engine.

The vendor contract (help.aliyun.com 声音设计 API, verified 2026-09) is
exercised through an httpx.MockTransport:

- Create: POST /services/audio/tts/customization with
  ``model: qwen-voice-design``, ``input.action: create``,
  ``input.target_model`` + ``input.voice_prompt`` + ``input.preview_text``.
  The response carries ``output.voice`` plus a base64 ``preview_audio``.
- Synthesis: POST /services/aigc/multimodal-generation/generation with
  ``model: qwen3-tts-vd-2026-01-26`` and the designed ``voice``.
"""

from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import pytest

from voiceclone_sidecar.engines.qwen_tts_vd_cloud import (
    BASE_URL,
    DESIGN_MODEL,
    ENROLL_PATH,
    SYNTH_PATH,
    TARGET_MODEL,
    CloudEngineError,
    Qwen3TtsVdCloudEngine,
)
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.secrets import KeyStore, MemoryBackend


def wav_b64(seconds: float = 1.0, rate: int = 24000) -> tuple[str, bytes]:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), raw


class Harness:
    def __init__(self, responses: list[httpx.Response], key: str | None = "sk-test") -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._responses.pop(0)

        self.store = KeyStore(backend=MemoryBackend())
        if key:
            self.store.set("qwen3-tts-vd-cloud", key)
        self.engine = Qwen3TtsVdCloudEngine(
            output_dir=None,
            key_store=self.store,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_capabilities_declare_design_but_not_cloning():
    caps = Harness([]).engine.capabilities()
    assert caps.voice_design is True
    assert caps.voice_cloning is False


def test_missing_key_fails_with_actionable_message():
    engine = Qwen3TtsVdCloudEngine(key_store=KeyStore(backend=MemoryBackend()))
    with pytest.raises(CloudEngineError, match="API Key"):
        engine.design_voice("描述", "文本", lambda m: None)


def test_design_request_shape_and_response_parsing(tmp_path):
    b64, _raw = wav_b64()
    harness = Harness(
        [
            httpx.Response(
                200,
                json={
                    "output": {
                        "voice": "qwen3-tts-vd-2026-01-26-vdtest-abc",
                        "preview_audio": {"data": b64, "sample_rate": 24000},
                    }
                },
            )
        ]
    )
    out_dir = tmp_path / "audio"
    out_dir.mkdir()
    harness.engine.output_dir = out_dir
    extra = harness.engine.design_voice(
        "低沉缓慢的男声", "欢迎收听今晚的节目。", lambda m: None
    )
    assert extra["voice_id"] == "qwen3-tts-vd-2026-01-26-vdtest-abc"
    assert extra["transcript"] == "欢迎收听今晚的节目。"
    sample = extra["sample_audio_path"]
    with wave.open(sample, "rb") as w:
        assert w.getframerate() == 24000

    req = harness.requests[0]
    assert req.url == f"{BASE_URL}{ENROLL_PATH}"
    assert req.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == DESIGN_MODEL
    assert body["input"]["action"] == "create"
    assert body["input"]["target_model"] == TARGET_MODEL
    assert body["input"]["voice_prompt"] == "低沉缓慢的男声"
    assert body["input"]["preview_text"] == "欢迎收听今晚的节目。"


def test_design_failure_surfaces_vendor_error():
    harness = Harness([httpx.Response(400, json={"code": "InvalidParameter", "message": "bad"})])
    with pytest.raises(CloudEngineError, match="InvalidParameter"):
        harness.engine.design_voice("d", "t", lambda m: None)


def test_design_failure_without_voice_field():
    harness = Harness([httpx.Response(200, json={"output": {}})])
    with pytest.raises(CloudEngineError, match="voice"):
        harness.engine.design_voice("d", "t", lambda m: None)


def test_synthesis_uses_vd_model_and_designed_voice(tmp_path):
    _dl_b64, dl_raw = wav_b64()
    harness = Harness(
        [
            httpx.Response(200, json={"output": {"audio": {"url": "https://x/a.wav"}}}),
            httpx.Response(200, content=dl_raw),
        ]
    )
    out_dir = tmp_path / "audio"
    out_dir.mkdir()
    harness.engine.output_dir = out_dir
    result = harness.engine.synthesize(
        GenerationRequest(
            generation_id="g1", text="你好", params={"voice_id": "vd-voice-1"}
        ),
        lambda m: None,
    )
    req = harness.requests[0]
    assert req.url == f"{BASE_URL}{SYNTH_PATH}"
    body = json.loads(req.content)
    assert body["model"] == TARGET_MODEL
    assert body["input"]["voice"] == "vd-voice-1"
    assert result.audio_path.endswith("g1.wav")
    assert result.model_version == TARGET_MODEL


# --- voice health check (issue #12) -----------------------------------------


def test_check_voice_alive_and_missing():
    harness = Harness(
        [httpx.Response(200, json={"output": {"audio": {"url": "https://cdn.example/a.wav"}}})]
    )
    assert harness.engine.check_voice("vd-abc", lambda m: None) is True
    body = json.loads(harness.requests[0].content)
    assert body["model"] == TARGET_MODEL
    assert body["input"]["voice"] == "vd-abc"

    harness2 = Harness(
        [
            httpx.Response(
                400, json={"code": "InvalidParameter", "message": "音色不存在或已被删除"}
            )
        ]
    )
    assert harness2.engine.check_voice("vd-dead", lambda m: None) is False


def test_check_voice_unrelated_failure_raises():
    harness = Harness(
        [httpx.Response(401, json={"code": "InvalidApiKey", "message": "无效 API Key"})]
    )
    with pytest.raises(CloudEngineError, match="健康检查"):
        harness.engine.check_voice("vd-abc", lambda m: None)


# --- issue #38: parameter gap audit -----------------------------------------


def test_language_is_declared_but_not_exposed_until_verified():
    # The qwen3-tts-vd request schema has the same three input knobs as the
    # VC model (text / voice / language_type), but language_type is not
    # verified for the pinned vd version — per ADR-0018 an unverified
    # parameter is DATA (exposed=False + reason), never exposed on a guess.
    specs = Qwen3TtsVdCloudEngine(key_store=KeyStore(backend=MemoryBackend())).param_specs()
    assert [s.name for s in specs] == ["language"]
    spec = specs[0]
    assert spec.exposed is False
    assert spec.not_exposed_reason == "unverified"
    assert spec.to_wire("English") == "English"
    assert spec.to_wire("auto") is None
