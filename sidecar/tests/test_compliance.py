"""Compliance guardrails contract tests (issue #15): first-use voice consent
and the AI-generated-content marker on real generation artifacts. Runs
against the real sidecar process (conftest)."""

from __future__ import annotations

import io
import json
import wave


def test_consent_defaults_to_unconfirmed(client):
    r = client.get("/consent")
    assert r.status_code == 200
    body = r.json()
    assert body["acknowledged"] is False
    assert body["confirmed_at"] is None


def test_consent_put_requires_true(client):
    r = client.put("/consent", json={"acknowledged": False})
    assert r.status_code == 422
    r = client.put("/consent", json={})
    assert r.status_code == 422


def test_consent_put_persists_to_data_root(client, sidecar):
    r = client.put("/consent", json={"acknowledged": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["acknowledged"] is True
    assert body["confirmed_at"]

    # GET reflects the durable state…
    assert client.get("/consent").json()["acknowledged"] is True
    # …which lives in the portable data root (issue #14 layout, SQLite
    # app_state table per ADR-0017), not the renderer's localStorage — so
    # backup/restore carries it along.
    from voiceclone_sidecar.db import T_APP_STATE, Database, library_db_path
    db = Database(library_db_path(sidecar["data_dir"]), migrate=False)
    try:
        assert db.get_kv(T_APP_STATE, "voice_consent")["version"]
    finally:
        db.close()


def test_generation_artifact_carries_aigc_marker(client):
    r = client.post("/generations", json={"engine_id": "fake", "text": "标记测试。"})
    assert r.status_code == 200, r.text
    record = r.json()
    assert record["status"] == "succeeded"
    assert record.get("aigc_marked") is True

    audio = client.get(record["audio_url"])
    assert audio.status_code == 200
    # The artifact is still a valid WAV…
    with wave.open(io.BytesIO(audio.content)) as w:
        assert w.getnframes() > 0
    # …whose LIST/INFO ICMT chunk declares AI-generated content.
    import tempfile
    from pathlib import Path

    from voiceclone_sidecar.aigc import read_info_comment

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio.content)
        comment = read_info_comment(Path(f.name))
    assert comment, "no ICMT chunk in artifact"
    assert "AI-generated audio" in comment
    assert "engine=fake" in comment
