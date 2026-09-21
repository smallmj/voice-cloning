"""Unit tests for the MiniMax cloud engine (issue #27).

The vendor's HTTP contract is exercised through an httpx.MockTransport, so
these tests assert the exact request shapes documented by MiniMax
(docs/research/2026-09-minimax-api-and-local-tts.md) and the engine's
handling of responses — without any network access.

Facts under test (all verified 2026-09-18 against official docs):
- cloning = file upload (multipart, purpose=voice_clone) + /v1/voice_clone;
- the cloned voice is INACTIVE until a real T2A synthesis — previewing
  during voice_clone does NOT activate it — so the engine silently
  synthesizes a short sentence right after cloning and records
  ``activated_at``; on activation failure the binding extra carries
  status "unactivated" (the pipeline persists it);
- ``check_voice`` can only be a probe synthesis (get_voice is POST and
  only lists voices that have already synthesized at least once);
- sync T2A for short text with output_format=url and format=wav (the
  default mp3 would break the pipeline's peak/self-check on WAV);
- long text goes to /v1/t2a_async_v2 whose payload keys DIFFER from the
  sync request (audio_sample_rate, english_normalization; no
  output_format / stream / force_cbr);
- error code 1002 is a rate limit (must carry the retry markers), 2038
  means the account lacks cloning permission (real-name/paid tier);
- the region (.io / .cn) is an engine-level configurable base URL.
"""

from __future__ import annotations

import io
import json
import wave

import httpx
import pytest

from voiceclone_sidecar.engine_config import EngineConfig
from voiceclone_sidecar.engines.cloud_base import CloudEngineError
from voiceclone_sidecar.engines.minimax_cloud import (
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    MODEL_CHOICES,
    PRICES_PER_M_CHARS,
    SYNC_CHAR_LIMIT,
    TARGET_MODEL,
    MiniMaxCloudEngine,
)
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.secrets import KeyStore, MemoryBackend


def tmp_dir():
    import tempfile
    return pathlib.Path(tempfile.mkdtemp())


import pathlib  # noqa: E402 - test helpers above, imports after


def make_wav_bytes(seconds: float = 0.5, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def ok_base() -> dict:
    return {"base_resp": {"status_code": 0, "status_msg": "success"}}


def err_base(code: int, msg: str) -> dict:
    return {"base_resp": {"status_code": code, "status_msg": msg}}


class Harness:
    """Builds an engine around a scripted MockTransport and records requests."""

    def __init__(self, responses: list[httpx.Response], key: str | None = "mk-test",
                 config: EngineConfig | None = None, output_dir=None) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._responses.pop(0)

        self.store = KeyStore(backend=MemoryBackend())
        if key:
            self.store.set("minimax-speech-cloud", key)
        self.engine = MiniMaxCloudEngine(
            output_dir=output_dir,
            key_store=self.store,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            config=config,
        )


@pytest.fixture()
def wav_file(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(make_wav_bytes())
    return p


def test_missing_key_fails_with_actionable_message():
    engine = MiniMaxCloudEngine(key_store=KeyStore(backend=MemoryBackend()))
    with pytest.raises(CloudEngineError, match="API Key"):
        engine.synthesize(
            GenerationRequest(generation_id="g1", text="hi", params={"voice_id": "v"}),
            lambda m: None,
        )


def test_enroll_uploads_then_clones_then_activates(wav_file):
    _ = make_wav_bytes(0.2)
    harness = Harness([
        httpx.Response(200, json={"file": {"file_id": 42}, **ok_base()}),
        httpx.Response(200, json=ok_base()),
        # activation probe: sync T2A returning a URL that then gets downloaded
        httpx.Response(200, json={"data": {"audio": "https://cdn.example/a"}, **ok_base()}),
        httpx.Response(200, content=make_wav_bytes(0.1)),
    ])
    extra = harness.engine.bind_reference(wav_file, None, lambda m: None)

    assert extra["voice_id"][0].isalpha()
    assert 8 <= len(extra["voice_id"]) <= 256
    assert extra["voice_id"][-1] not in "-_"
    assert extra["activated_at"]
    assert "activation_error" not in extra

    # 1: multipart upload with purpose=voice_clone
    up = harness.requests[0]
    assert up.url.path == "/v1/files/upload"
    assert "multipart/form-data" in up.headers["content-type"]
    assert b'name="purpose"' in up.content and b"voice_clone" in up.content
    assert wav_file.read_bytes()[:4] in up.content  # the reference bytes ride along
    # 2: voice_clone carries the file id and the app-chosen voice id
    clone = harness.requests[1]
    assert clone.url.path == "/v1/voice_clone"
    body = json.loads(clone.content)
    assert body["file_id"] == 42
    assert body["voice_id"] == extra["voice_id"]
    # 3: activation is a REAL sync synthesis (previewing does not activate)
    act = harness.requests[2]
    assert act.url.path == "/v1/t2a_v2"
    act_body = json.loads(act.content)
    assert act_body["voice_setting"]["voice_id"] == extra["voice_id"]
    assert len(act_body["text"]) <= 30


def test_activation_failure_marks_binding_unactivated(wav_file):
    harness = Harness([
        httpx.Response(200, json={"file": {"file_id": 7}, **ok_base()}),
        httpx.Response(200, json=ok_base()),
        httpx.Response(200, json=err_base(2013, "invalid param")),
    ])
    extra = harness.engine.bind_reference(wav_file, None, lambda m: None)
    assert extra["status"] == "unactivated"
    assert extra.get("activated_at") is None
    assert "2013" in extra["activation_error"]


def test_enroll_rejects_unsupported_container(tmp_path):
    flac = tmp_path / "ref.flac"
    flac.write_bytes(b"not really flac")
    harness = Harness([])
    with pytest.raises(CloudEngineError, match="WAV / MP3 / M4A"):
        harness.engine.bind_reference(flac, None, lambda m: None)


def test_check_voice_probe_synthesis():
    alive = Harness([
        httpx.Response(200, json={"data": {"audio": "https://x/y"}, **ok_base()}),
    ])
    assert alive.engine.check_voice("vc-alive", lambda m: None) is True
    body = json.loads(alive.requests[0].content)
    assert body["voice_setting"]["voice_id"] == "vc-alive"

    dead = Harness([
        httpx.Response(200, json=err_base(2049, "voice not exist")),
    ])
    assert dead.engine.check_voice("vc-dead", lambda m: None) is False


def test_check_voice_non_verdicts_raise():
    throttled = Harness([httpx.Response(200, json=err_base(1002, "rate limit"))])
    with pytest.raises(CloudEngineError, match="限流"):
        throttled.engine.check_voice("v", lambda m: None)
    bad_key = Harness([httpx.Response(401, json=err_base(1004, "invalid api key"))])
    with pytest.raises(CloudEngineError, match="API Key"):
        bad_key.engine.check_voice("v", lambda m: None)


def test_rate_limit_error_carries_retry_markers():
    harness = Harness([httpx.Response(200, json=err_base(1002, "too many requests"))])
    with pytest.raises(CloudEngineError) as excinfo:
        harness.engine.check_voice("v", lambda m: None)
    msg = str(excinfo.value)
    assert "限流" in msg or "rate limit" in msg


def test_2038_maps_to_realname_hint(wav_file):
    harness = Harness([
        httpx.Response(200, json={"file": {"file_id": 1}, **ok_base()}),
        httpx.Response(200, json=err_base(2038, "No cloning permission, please check account verification status")),
    ])
    with pytest.raises(CloudEngineError, match="实名"):
        harness.engine.bind_reference(wav_file, None, lambda m: None)


def test_sync_synthesis_requests_wav_via_url(tmp_path):
    wav = make_wav_bytes(0.2)
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": "https://cdn/x.wav"},
                       "extra_info": {"usage_characters": 2}, **ok_base()}),
        httpx.Response(200, content=wav),
    ])
    result = harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="你好",
                          params={"voice_id": "vc-x"}),
        lambda m: None,
    )
    body = json.loads(harness.requests[0].content)
    assert harness.requests[0].url.path == "/v1/t2a_v2"
    assert body["model"] == TARGET_MODEL
    assert body["voice_setting"]["voice_id"] == "vc-x"
    assert body["audio_setting"]["format"] == "wav"  # never the mp3 default
    assert body["output_format"] == "url"
    assert "stream" not in body

    import pathlib
    written = pathlib.Path(result.audio_path)
    assert written.read_bytes() == wav
    assert result.sample_rate == 24000
    assert result.model_version == TARGET_MODEL
    assert result.cost is not None


def test_sync_response_hex_audio_is_decoded_to_wav(tmp_path):
    wav = make_wav_bytes(0.2)
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": wav.hex()}, **ok_base()}),
    ], output_dir=tmp_path)
    result = harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="hi", params={"voice_id": "v"}),
        lambda m: None,
    )
    import pathlib
    assert pathlib.Path(result.audio_path).read_bytes() == wav


def test_non_wav_audio_is_rejected_loudly():
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": "https://cdn/x"}, **ok_base()}),
        httpx.Response(200, content=b"ID3 mp3 data pretending"),
    ])
    with pytest.raises(CloudEngineError, match="WAV"):
        harness.engine.synthesize(
            GenerationRequest(generation_id="g1", text="hi", params={"voice_id": "v"}),
            lambda m: None,
        )


def test_long_text_routes_to_async_with_distinct_payload_keys(monkeypatch, tmp_path):
    monkeypatch.setattr("voiceclone_sidecar.engines.minimax_cloud.ASYNC_POLL_SECONDS", 0.0)
    long_text = "这是一个用于触发异步长文本合成的句子。" * 200  # > SYNC_CHAR_LIMIT
    assert len(long_text) > SYNC_CHAR_LIMIT
    wav = make_wav_bytes(0.2)
    harness = Harness([
        httpx.Response(200, json={"task_id": "t-1", **ok_base()}),
        httpx.Response(200, json={"task_id": "t-1", "status": "Processing", **ok_base()}),
        # "SUCCESS" (upper case, not in the enum) must still match: the
        # status comparison is case-insensitive on purpose.
        httpx.Response(200, json={"task_id": "t-1", "status": "SUCCESS",
                                  "file_id": 99, **ok_base()}),
        httpx.Response(200, json={"file": {"download_url": "https://cdn/async"}, **ok_base()}),
        httpx.Response(200, content=wav),
    ], output_dir=tmp_path)
    result = harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text=long_text, params={"voice_id": "v"}),
        lambda m: None,
    )

    create = harness.requests[0]
    assert create.url.path == "/v1/t2a_async_v2"
    body = json.loads(create.content)
    # keys that ONLY exist on the async payload — and the sync-only keys that must NOT
    assert body["audio_setting"]["audio_sample_rate"] == 24000
    assert "sample_rate" not in body["audio_setting"]
    assert "output_format" not in body
    assert "stream" not in body
    assert "force_cbr" not in body
    assert body["audio_setting"]["format"] == "wav"

    # status comparison is case-insensitive ("Processing" in the docs example)
    query = harness.requests[1]
    assert query.url.path == "/v1/query/t2a_async_query_v2"
    assert query.url.params["task_id"] == "t-1"

    retrieve = harness.requests[3]
    assert retrieve.url.path == "/v1/files/retrieve"
    assert retrieve.url.params["file_id"] == "99"

    import pathlib
    assert pathlib.Path(result.audio_path).read_bytes() == wav


def test_async_failure_raises_with_task_state(monkeypatch, tmp_path):
    monkeypatch.setattr("voiceclone_sidecar.engines.minimax_cloud.ASYNC_POLL_SECONDS", 0.0)
    harness = Harness([
        httpx.Response(200, json={"task_id": "t-2", **ok_base()}),
        httpx.Response(200, json={"task_id": "t-2", "status": "failed", **ok_base()}),
    ], output_dir=tmp_path)
    with pytest.raises(CloudEngineError, match="failed"):
        harness.engine.synthesize(
            GenerationRequest(generation_id="g1", text="字" * 3000,
                              params={"voice_id": "v"}),
            lambda m: None,
        )


def test_speed_and_emotion_and_language_boost_wire_into_payload(tmp_path):
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": make_wav_bytes(0.1).hex()}, **ok_base()}),
    ], output_dir=tmp_path)
    harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="hi", params={
            "voice_id": "v", "speed": 1.2, "emotion": "happy",
            "language_boost": "Chinese",
        }),
        lambda m: None,
    )
    body = json.loads(harness.requests[0].content)
    assert body["voice_setting"]["speed"] == 1.2
    assert body["voice_setting"]["emotion"] == "happy"
    assert body["language_boost"] == "Chinese"


def test_system_voice_override_param(tmp_path):
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": make_wav_bytes(0.1).hex()}, **ok_base()}),
    ], output_dir=tmp_path)
    harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="hi", params={
            "voice_id": "vc-bound", "system_voice": "male-qn-qingse",
        }),
        lambda m: None,
    )
    body = json.loads(harness.requests[0].content)
    assert body["voice_setting"]["voice_id"] == "male-qn-qingse"


def test_design_voice_returns_id_and_saves_sample(tmp_path):
    wav = make_wav_bytes(0.2)
    harness = Harness([
        httpx.Response(200, json={"voice_id": "ttv-voice-1",
                                  "trial_audio": wav.hex(), **ok_base()}),
    ], output_dir=tmp_path)
    extra = harness.engine.design_voice("低沉的男声", "你好，世界。", lambda m: None)
    body = json.loads(harness.requests[0].content)
    assert harness.requests[0].url.path == "/v1/voice_design"
    assert body["prompt"] == "低沉的男声"
    assert body["preview_text"] == "你好，世界。"
    assert extra["voice_id"] == "ttv-voice-1"
    import pathlib
    sample = pathlib.Path(extra["sample_audio_path"])
    assert sample.read_bytes() == wav
    assert extra["sample_rate"] == 24000


def test_region_base_url_is_engine_level_config(tmp_path):
    cfg = EngineConfig(
        engine_id="minimax-speech-cloud",
        env={BASE_URL_ENV: "https://api.minimax.cn"},
    )
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": make_wav_bytes(0.1).hex()}, **ok_base()}),
    ], config=cfg, output_dir=tmp_path)
    harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="hi", params={"voice_id": "v"}),
        lambda m: None,
    )
    assert harness.requests[0].url.host == "api.minimax.cn"


def test_default_base_url_is_international():
    assert DEFAULT_BASE_URL == "https://api.minimax.io"


def test_pricing_table_covers_all_models():
    assert set(PRICES_PER_M_CHARS) == set(MODEL_CHOICES)


def test_english_normalization_is_async_only():
    harness = Harness([
        httpx.Response(200, json={"task_id": "t-3", **ok_base()}),
        httpx.Response(200, json={"task_id": "t-3", "status": "success",
                                  "file_id": 5, **ok_base()}),
        httpx.Response(200, json={"file": {"download_url": "https://cdn/z"}, **ok_base()}),
        httpx.Response(200, content=make_wav_bytes(0.2)),
    ], output_dir=tmp_dir())
    harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="字" * 3000, params={
            "voice_id": "v", "english_normalization": True,
        }),
        lambda m: None,
    )
    body = json.loads(harness.requests[0].content)
    assert body["english_normalization"] is True

    sync = Harness([
        httpx.Response(200, json={"data": {"audio": make_wav_bytes(0.1).hex()}, **ok_base()}),
    ], output_dir=tmp_dir())
    sync.engine.synthesize(
        GenerationRequest(generation_id="g2", text="hi", params={
            "voice_id": "v", "english_normalization": True,
        }),
        lambda m: None,
    )
    assert "english_normalization" not in json.loads(sync.requests[0].content)


def test_auto_params_are_never_sent_to_the_vendor():
    # "auto" = do not send (ADR-0018). Forwarding it verbatim makes the
    # vendor reject the whole request with "invalid params: emotion".
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": make_wav_bytes(0.1).hex()}, **ok_base()}),
    ], output_dir=tmp_dir())
    harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="hi", params={
            "voice_id": "v", "emotion": "auto", "language_boost": "auto",
        }),
        lambda m: None,
    )
    body = json.loads(harness.requests[0].content)
    assert "emotion" not in body["voice_setting"]
    assert "language_boost" not in body


def test_unknown_emotion_value_is_dropped_not_forwarded():
    harness = Harness([
        httpx.Response(200, json={"data": {"audio": make_wav_bytes(0.1).hex()}, **ok_base()}),
    ], output_dir=tmp_dir())
    harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="hi", params={
            "voice_id": "v", "emotion": "whisper",  # 2.8 does not support it
        }),
        lambda m: None,
    )
    assert "emotion" not in json.loads(harness.requests[0].content)["voice_setting"]


def test_1008_maps_to_topup_hint():
    harness = Harness([httpx.Response(200, json=err_base(1008, "insufficient balance"))])
    with pytest.raises(CloudEngineError, match="充值"):
        harness.engine.check_voice("v", lambda m: None)


def test_boolean_string_false_is_not_truthy(monkeypatch, tmp_path):
    # The UI may deliver the checkbox as the string "false": the async
    # payload must NOT include english_normalization in that case.
    monkeypatch.setattr("voiceclone_sidecar.engines.minimax_cloud.ASYNC_POLL_SECONDS", 0.0)
    harness = Harness([
        httpx.Response(200, json={"task_id": "t-4", **ok_base()}),
        httpx.Response(200, json={"task_id": "t-4", "status": "success",
                                  "file_id": 6, **ok_base()}),
        httpx.Response(200, json={"file": {"download_url": "https://cdn/w"}, **ok_base()}),
        httpx.Response(200, content=make_wav_bytes(0.2)),
    ], output_dir=tmp_path)
    harness.engine.synthesize(
        GenerationRequest(generation_id="g1", text="字" * 3000, params={
            "voice_id": "v", "english_normalization": "false",
        }),
        lambda m: None,
    )
    assert "english_normalization" not in json.loads(harness.requests[0].content)


def test_disclosure_note_names_the_tos_language():
    # Issue #21/#27: the ToS "改进算法或增强服务" reservation must be part of
    # the visible data-usage disclosure the settings/generate pages render.
    engine = MiniMaxCloudEngine(key_store=KeyStore(backend=MemoryBackend()))
    assert "改进算法" in engine.data_usage_note
    assert "实名" in engine.data_usage_note  # 2038 limitation is disclosed too
