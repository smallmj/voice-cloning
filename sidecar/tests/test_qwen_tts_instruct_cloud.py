"""Unit + contract tests for the qwen3-tts-instruct-flash cloud engine (issue #68).

Vendor contract (help.aliyun.com qwen3-tts-instruct-flash + DashScope
非实时语音合成 docs, 厂商口径): synthesis posts to the same
multimodal-generation endpoint as the other Qwen-TTS cloud engines with
``input.instructions`` (natural-language 语气/语速/方言 direction) and
``input.language_type``. All HTTP is exercised through httpx.MockTransport;
the registration assertions run through ``default_registry`` (issue #22:
contract tests must ride the production construction path).
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import httpx
import pytest

from voiceclone_sidecar.engines.cloud_base import CloudEngineError
from voiceclone_sidecar.engines.qwen_tts_instruct_cloud import (
    BASE_URL,
    SYNTH_PATH,
    TARGET_MODEL,
    Qwen3TtsInstructCloudEngine,
)
from voiceclone_sidecar.key_store import KeyStore, MemoryBackend
from voiceclone_sidecar.registry import GenerationRequest, default_registry

ENGINE_ID = "qwen3-tts-instruct-cloud"
MATRIX_PATH = (
    Path(__file__).resolve().parent.parent / "voiceclone_sidecar" / "data" / "capability_matrix.json"
)


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
            self.store.set(ENGINE_ID, key)
        self.engine = Qwen3TtsInstructCloudEngine(
            output_dir=None,
            key_store=self.store,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def wav_bytes(seconds: float = 0.2, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def audio_response(audio: bytes) -> httpx.Response:
    return httpx.Response(200, content=audio)


# -- registration (contract: default_registry production path) -----------------


def test_engine_registered_in_default_registry():
    registry = default_registry(output_dir=None, key_store=KeyStore(backend=MemoryBackend()))
    engine = registry.get(ENGINE_ID)
    assert engine is not None
    assert engine.display_name
    assert engine.requires_key is True
    assert any(e.engine_id == ENGINE_ID for e in registry.list())


def test_missing_key_raises_actionable_vendor_message():
    engine = Qwen3TtsInstructCloudEngine(key_store=KeyStore(backend=MemoryBackend()))
    with pytest.raises(CloudEngineError, match="阿里百炼"):
        engine.synthesize(
            GenerationRequest(generation_id="g1", text="你好", params={"voice_id": "v1"}),
            lambda m: None,
        )


# -- param specs (ADR-0018) -----------------------------------------------------


def test_param_specs_instructions_language_voice():
    engine = Qwen3TtsInstructCloudEngine(key_store=KeyStore(backend=MemoryBackend()))
    specs = {s.name: s.to_dict() for s in engine.param_specs()}

    ins = specs["instructions"]
    assert ins["kind"] == "textarea"
    assert ins["exposed"] is True
    assert ins["layer"] == "engine"
    assert ins["wire_path"] == "instructions"
    assert ins["applies_to"]["model"] == TARGET_MODEL

    lang = specs["language"]
    assert lang["kind"] == "select"
    assert lang["default"] == "auto"
    assert "auto" in lang["choices"] and "Chinese" in lang["choices"]
    assert lang["layer"] == "canonical"
    assert lang["wire_path"] == "language_type"
    assert lang["wire_map"] is True  # to_wire adapter present (ADR-0018)

    voice = specs["voice"]
    assert voice["exposed"] is False
    assert voice["not_exposed_reason"] == "server-injected"

    # The closed list is exactly these three (ADR-0018: nothing undeclared).
    assert set(specs) == {"instructions", "language", "voice"}


def test_capabilities_no_cloning_but_cloud_disclosures():
    engine = Qwen3TtsInstructCloudEngine(key_store=KeyStore(backend=MemoryBackend()))
    caps = engine.capabilities()
    assert caps.voice_cloning is False
    assert caps.emotion is False  # 情感走 instructions 指令，不是 API 枚举
    assert caps.commercial_license is True
    assert caps.max_chars_per_request == 2000


# -- synthesis wire shape --------------------------------------------------------


def test_synthesize_sends_instructions_and_language(tmp_path):
    audio = wav_bytes()
    h = Harness([httpx.Response(200, json={"output": {"audio": {"url": "https://x/out.wav"}}}), audio_response(audio)])
    result = h.engine.synthesize(
        GenerationRequest(
            generation_id="gen1",
            text="用兴奋的语气说这句话",
            params={
                "voice_id": "voice-123",
                "instructions": "用兴奋的语气、稍快的语速说",
                "language": "Chinese",
            },
        ),
        lambda m: None,
    )
    synth = json.loads(h.requests[0].content)
    assert h.requests[0].url.path.endswith(SYNTH_PATH)
    assert synth["model"] == TARGET_MODEL
    assert synth["input"]["text"] == "用兴奋的语气说这句话"
    assert synth["input"]["voice"] == "voice-123"
    assert synth["input"]["instructions"] == "用兴奋的语气、稍快的语速说"
    assert synth["input"]["language_type"] == "Chinese"
    assert result.audio_path.endswith("gen1.wav")
    assert Path(result.audio_path).read_bytes() == audio
    assert result.cost is not None


def test_synthesize_auto_language_and_blank_instructions_send_nothing(tmp_path):
    audio = wav_bytes()
    h = Harness([httpx.Response(200, json={"output": {"audio": {"url": "https://x/out.wav"}}}), audio_response(audio)])
    h.engine.synthesize(
        GenerationRequest(
            generation_id="gen2",
            text="你好",
            params={"voice_id": "v1", "instructions": "   ", "language": "auto"},
        ),
        lambda m: None,
    )
    body = json.loads(h.requests[0].content)["input"]
    assert "instructions" not in body
    assert "language_type" not in body


def test_synthesize_without_voice_binding_raises():
    h = Harness([])
    with pytest.raises(CloudEngineError, match="云端绑定"):
        h.engine.synthesize(
            GenerationRequest(generation_id="g", text="你好", params={"instructions": "x"}),
            lambda m: None,
        )


def test_synthesize_vendor_error_surfaces():
    h = Harness([httpx.Response(400, json={"code": "InvalidParameter", "message": "bad instructions"})])
    with pytest.raises(CloudEngineError, match="InvalidParameter"):
        h.engine.synthesize(
            GenerationRequest(generation_id="g", text="你好", params={"voice_id": "v1"}),
            lambda m: None,
        )


def test_synthesize_overlong_instruction_raises():
    h = Harness([])
    with pytest.raises(CloudEngineError, match="过长"):
        h.engine.synthesize(
            GenerationRequest(
                generation_id="g", text="你好", params={"voice_id": "v1", "instructions": "长" * 2001}
            ),
            lambda m: None,
        )


# -- capability matrix (issue #68: vendor 口径 entries) ---------------------------


def test_capability_matrix_has_instruct_engine_entry():
    matrix = json.loads(MATRIX_PATH.read_text())
    entry = next(e for e in matrix["engines"] if e["engine_id"] == ENGINE_ID)
    fields = {ev["field"] for ev in entry["evidence"]}
    assert {"instructions 参数", "语种参数", "音色"} <= fields
    for ev in entry["evidence"]:
        assert ev["verification"] == "vendor"
        assert ev["source"] and ev["date"]
