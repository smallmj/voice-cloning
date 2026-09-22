"""Contract tests for voice design (issue #11): text description -> voice.

A designed voice is a first-class voice (ADR-0001) whose origin is
``"designed"`` instead of ``"cloned"``. The design engine mints a preview
sample that becomes the voice's reference, so the designed voice can bind
to other engines, generate, and be deleted exactly like a cloned one.
"""

from __future__ import annotations

import wave


def test_design_voice_with_capable_engine_creates_first_class_voice(client):
    r = client.post(
        "/voices/design",
        json={
            "engine_id": "fake",
            "name": "深夜电台主播",
            "description": "低沉缓慢的男声",
            "voice_prompt": "低沉、缓慢、有磁性的年轻男声，适合深夜电台",
            "preview_text": "欢迎收听今晚的节目。",
        },
    )
    assert r.status_code == 200, r.text
    voice = r.json()
    assert voice["id"]
    assert voice["name"] == "深夜电台主播"
    assert voice["origin"] == "designed"
    # Provenance: which engine designed it, from what description.
    assert voice["design"]["engine_id"] == "fake"
    assert voice["design"]["voice_prompt"].startswith("低沉")
    assert voice["design"]["preview_text"] == "欢迎收听今晚的节目。"
    # The preview sample became the reference (the source of truth for
    # every other engine), with the preview text as its transcript.
    ref = voice["reference"]
    assert ref["format"] == "wav"
    assert ref["duration_seconds"] > 0
    assert ref["sha256"]
    assert ref["transcript"] == "欢迎收听今晚的节目。"
    # The designing engine got a binding carrying its minted voice id.
    assert voice["bindings"]["fake"]["voice_id"].startswith("fake-voice-")


def test_design_rejects_engine_without_voice_design_capability(client):
    r = client.post(
        "/voices/design",
        json={
            "engine_id": "qwen3-tts-vc-cloud",
            "name": "x",
            "voice_prompt": "描述",
            "preview_text": "文本",
        },
    )
    assert r.status_code == 422
    assert "音色设计" in r.json()["detail"]


def test_design_rejects_unknown_engine(client):
    r = client.post(
        "/voices/design",
        json={"engine_id": "nope", "name": "x", "voice_prompt": "d", "preview_text": "t"},
    )
    assert r.status_code == 404


def test_design_rejects_missing_fields(client):
    for payload in (
        {"engine_id": "fake", "name": "", "voice_prompt": "d", "preview_text": "t"},
        {"engine_id": "fake", "name": "n", "voice_prompt": "", "preview_text": "t"},
        {"engine_id": "fake", "name": "n", "voice_prompt": "d", "preview_text": ""},
    ):
        r = client.post("/voices/design", json=payload)
        assert r.status_code == 422, payload


def test_engine_list_exposes_voice_design_capability(client):
    engines = client.get("/engines").json()["engines"]
    by_id = {e["id"]: e for e in engines}
    # The fake engine declares design support; a cloning-only engine does not.
    assert by_id["fake"]["capabilities"]["voice_design"] is True
    # A cloud cloning-only engine does not declare voice design.
    assert by_id["qwen3-tts-vc-cloud"]["capabilities"]["voice_design"] is False


def test_designed_voice_lists_and_generates_like_any_voice(client):
    r = client.post(
        "/voices/design",
        json={
            "engine_id": "fake",
            "name": "设计音色",
            "voice_prompt": "清亮的女声",
            "preview_text": "你好，世界。",
        },
    )
    voice = r.json()
    listed = client.get("/voices").json()["voices"]
    match = [v for v in listed if v["id"] == voice["id"]]
    assert match and match[0]["origin"] == "designed"

    # The designed voice is usable for generation on another engine: its
    # reference sample is handed over like any cloned voice's.
    gen = client.post(
        "/generations",
        json={"engine_id": "fake", "text": "设计音色也能正常生成。", "voice_id": voice["id"]},
    )
    assert gen.status_code == 200, gen.text
    assert gen.json()["status"] == "succeeded"
    assert gen.json()["voice_name"] == "设计音色"


def test_uploaded_voice_reports_cloned_origin(client):
    files = {"file": ("ref.wav", _wav_bytes(), "audio/wav")}
    r = client.post("/voices", data={"name": "复刻音色"}, files=files)
    assert r.status_code == 200
    assert r.json()["origin"] == "cloned"


def test_designed_voice_can_be_deleted(client):
    r = client.post(
        "/voices/design",
        json={
            "engine_id": "fake",
            "name": "待删除",
            "voice_prompt": "d",
            "preview_text": "t",
        },
    )
    voice = r.json()
    assert client.delete(f"/voices/{voice['id']}").status_code == 200
    assert client.get(f"/voices/{voice['id']}").status_code == 404


def _wav_bytes(seconds: float = 5.0, rate: int = 16000) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


# --- VoiceStore unit level ----------------------------------------------------

def test_voicestore_designed_record_and_legacy_origin_default(tmp_path):
    import json

    from voiceclone_sidecar.voices import VoiceStore

    store = VoiceStore(tmp_path)
    voice = store.create_designed(
        "设计音色", "说明", engine_id="fake", voice_prompt="描述", preview_text="文本"
    )
    assert voice["origin"] == "designed"
    assert voice["reference"] is None
    assert voice["design"]["engine_id"] == "fake"

    # Legacy records (written before issue #11, stored without the origin
    # field) read back as cloned — now exercised through the JSON→SQLite
    # migration of ADR-0017.
    legacy_dir = tmp_path / "legacy-library"
    legacy_dir.mkdir()
    (legacy_dir / "voices.json").write_text(json.dumps({"voices": [{
        "id": "legacy1",
        "name": "旧音色",
        "description": "",
        "created_at": "2026-01-01T00:00:00Z",
        "reference": None,
        "avatar": None,
        "bindings": {},
    }]}, ensure_ascii=False), encoding="utf-8")
    store2 = VoiceStore(legacy_dir)
    assert store2.get("legacy1")["origin"] == "cloned"


def test_failed_design_leaves_no_orphaned_voice(client):
    # A plain OSError (not a CloudEngineError) must still clean up.
    r = client.post(
        "/voices/design",
        json={
            "engine_id": "fake-failing-design",
            "name": "会失败",
            "voice_prompt": "d",
            "preview_text": "t",
        },
    )
    assert r.status_code == 500
    listed = client.get("/voices").json()["voices"]
    assert all(v["name"] != "会失败" for v in listed)


# --- issue #57: VoxCPM2 原生 Voice Design（/voices/design 端到端） -------------


def _voxcpm2_design_app(tmp_path):
    """In-process app with the real VoxCPM2 MPS engine whose worker seam is
    faked (no model weights, per the testing decisions of spec #54)."""

    from voiceclone_sidecar.engines import fake as fake_mod
    from voiceclone_sidecar.engines.voxcpm2_mps import VoxCPM2MpsEngine
    from voiceclone_sidecar.main import create_app
    from voiceclone_sidecar.registry import Registry

    engine = VoxCPM2MpsEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})
    payloads: list[dict] = []

    class _Sup:
        def request(self, payload: dict, on_log=None, timeout_s=None) -> dict:
            payloads.append(payload)
            out = tmp_path / "audio" / f"design-{len(payloads)}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            fake_mod.write_tone_wav(out, 0.6)
            return {"audio_path": str(out), "sample_rate": 48000}

    engine.is_installed = lambda: True
    engine._get_supervisor = lambda: _Sup()
    registry = Registry()
    registry.register(engine)
    return create_app(registry, token="t", audio_dir=tmp_path / "audio"), payloads


def test_design_voice_e2e_voxcpm2(tmp_path):
    app, payloads = _voxcpm2_design_app(tmp_path)
    from fastapi.testclient import TestClient

    with TestClient(app, headers={"Authorization": "Bearer t"}) as c:
        r = c.post(
            "/voices/design",
            json={
                "engine_id": "voxcpm2-mps",
                "name": "设计的音色",
                "description": "低沉缓慢的男声",
                "voice_prompt": "低沉、缓慢、有磁性的男声",
                "preview_text": "欢迎收听今晚的节目。",
            },
        )
        assert r.status_code == 200, r.text
        voice = r.json()
        assert voice["origin"] == "designed"
        assert voice["design"]["engine_id"] == "voxcpm2-mps"
        assert voice["design"]["preview_text"] == "欢迎收听今晚的节目。"
        # ADR-0010: the preview sample became the reference, transcript attached.
        ref = voice["reference"]
        assert ref["format"] == "wav" and ref["sha256"]
        assert ref["transcript"] == "欢迎收听今晚的节目。"
        assert voice["bindings"]["voxcpm2-mps"]["voice_id"].startswith("voxcpm2-design-")

        # The design call reached the worker with the official prefix syntax,
        # no reference audio, and verbatim (never-normalized) description.
        assert len(payloads) == 1
        assert payloads[0]["design"] is True
        assert "ref_audio" not in payloads[0]
        assert payloads[0]["text"] == "(低沉、缓慢、有磁性的男声)欢迎收听今晚的节目。"

        # ADR-0010 hardening: later synthesis of the designed voice runs the
        # existing pure reference-cloning channel — ref_audio is the sample,
        # and the design flag is gone (pipeline untouched).
        gen = c.post(
            "/generations",
            json={"engine_id": "voxcpm2-mps", "text": "设计音色也能生成。", "voice_id": voice["id"]},
        )
        assert gen.status_code == 200, gen.text
        assert gen.json()["status"] == "succeeded"
        clone_payloads = payloads[1:]
        assert clone_payloads and all("design" not in p for p in clone_payloads)
        assert all(p.get("ref_audio") for p in clone_payloads)


def test_capability_matrix_includes_voxcpm2_voice_design(client):
    """/capability-matrix 合并运行时声明：voice_design=True 如实上报。"""
    engines = client.get("/engines").json()["engines"]
    by_id = {e["id"]: e for e in engines}
    if "voxcpm2-mps" not in by_id:  # registered only on Apple Silicon
        import pytest

        pytest.skip("voxcpm2-mps not registered on this platform")
    assert by_id["voxcpm2-mps"]["capabilities"]["voice_design"] is True


# --- capability/implementation contract (spec #54 user story 16) --------------


def test_every_voice_design_capable_engine_implements_design_voice(tmp_path):
    """契约：任何声明 voice_design=True 的引擎必须自己实现 design_voice —
    声明为真、调用即 NotImplementedError 的错位声明（issue #57）不允许再出现。
    遍历默认注册表的全部引擎。"""
    from voiceclone_sidecar.registry import Engine, default_registry

    base_impl = Engine.design_voice
    registry = default_registry(output_dir=tmp_path, env={})
    checked = 0
    for engine in registry.list():
        if engine.capabilities().voice_design:
            assert type(engine).design_voice is not base_impl, (
                f"engine {engine.engine_id} declares voice_design=True but does "
                "not implement design_voice"
            )
            checked += 1
    assert checked >= 1  # the fake engine at minimum
