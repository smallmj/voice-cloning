"""Blind comparison, LUFS-normalized fairness, preference profile (issue #10)."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

from voiceclone_sidecar.compare import (
    CompareError,
    CompareStore,
    detect_language,
)
from voiceclone_sidecar.loudness import TARGET_LUFS, integrated_lufs, read_wav_mono

TEST_ENGINES = ["fake-loud", "fake-quiet"]  # ~33 dB of output-level difference


def _engines(client) -> list[str]:
    engines = client.get("/engines").json()["engines"]
    return [e["id"] for e in engines]


def test_level_skewed_seams_are_registered(client):
    ids = _engines(client)
    for e in TEST_ENGINES:
        assert e in ids


def _make_voice(client) -> str:
    import io
    import struct

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        frames = bytearray()
        for i in range(int(3.0 * 22050)):
            v = 0.2 * math.sin(2 * math.pi * 220 * i / 22050)
            frames += struct.pack("<h", int(v * 32767))
        w.writeframes(bytes(frames))
    buf.seek(0)
    r = client.post(
        "/voices",
        files={"file": ("ref.wav", buf, "audio/wav")},
        data={"name": "对比音色", "description": "issue-10"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_compare_normalizes_loudness_and_blinds_identity(client, sidecar):
    voice_id = _make_voice(client)
    r = client.post(
        "/compare",
        json={
            "voice_id": voice_id,
            "text": "同一音色，同一文本，公平对比。",
            "engine_ids": TEST_ENGINES,
        },
    )
    assert r.status_code == 200, r.text
    session = r.json()
    assert len(session["entries"]) == 2

    # Blind: engine identity must NOT be reachable from the response.
    body = str(session)
    assert "fake-loud" not in body and "fake-quiet" not in body
    assert {e["label"] for e in session["entries"]} == {"A", "B"}

    # Blind entries must not leak the loudness fingerprints either.
    for e in session["entries"]:
        for hidden in ("original_lufs", "gain_db", "achieved_lufs", "peak_limited"):
            assert hidden not in e

    # Loudness fairness: both normalized files measure the same LUFS.
    lufs_values = []
    for e in session["entries"]:
        assert e["normalized_audio_url"]
        # URL carries only session id + blind label — no generation id.
        assert session["id"] in e["normalized_audio_url"]
        path = sidecar["audio_dir"] / e["normalized_audio_url"].rsplit("/", 1)[-1]
        assert path.is_file()
        samples, rate = read_wav_mono(path)
        lufs_values.append(integrated_lufs(samples, rate))
    assert abs(lufs_values[0] - lufs_values[1]) <= 1.0, lufs_values

    # On reveal, the measurement lineage shows the raw artifacts differ
    # enormously (the premise of the test) yet landed on one loudness.
    revealed = client.get(f"/compare/{session['id']}?reveal=true").json()
    originals = sorted(e["original_lufs"] for e in revealed["entries"])
    assert originals[-1] - originals[0] > 20.0, originals
    achieved = {e["achieved_lufs"] for e in revealed["entries"]}
    assert all(abs(a - TARGET_LUFS) <= 0.5 for a in achieved), achieved


def test_compare_reveals_identity_on_request(client, sidecar):
    voice_id = _make_voice(client)
    session = client.post(
        "/compare",
        json={
            "voice_id": voice_id,
            "text": "揭示身份检查。Hello mixed text.",
            "engine_ids": TEST_ENGINES,
        },
    ).json()
    revealed = client.get(f"/compare/{session['id']}?reveal=true").json()
    assert {e["engine_id"] for e in revealed["entries"]} == set(TEST_ENGINES)
    # And the default view stays blind.
    blind = client.get(f"/compare/{session['id']}").json()
    assert all("engine_id" not in e for e in blind["entries"])


def test_compare_scores_and_preferences(client):
    voice_id = _make_voice(client)
    session = client.post(
        "/compare",
        json={
            "voice_id": voice_id,
            "text": "偏好画像统计测试文本。",
            "engine_ids": TEST_ENGINES,
            "text_type": "narration",
        },
    ).json()

    labels = sorted(e["label"] for e in session["entries"])
    scores = {labels[0]: 5, labels[1]: 2}
    r = client.post(f"/compare/{session['id']}/scores", json={"scores": scores})
    assert r.status_code == 200, r.text
    scored = r.json()
    assert scored["scored"] is True

    # A fully scored session is auto-revealed (identity no longer sensitive).
    assert {e["engine_id"] for e in scored["entries"]} == set(TEST_ENGINES)
    for e in scored["entries"]:
        assert e["score"] == scores[e["label"]]

    # Invalid ratings are rejected.
    bad = client.post(
        f"/compare/{session['id']}/scores", json={"scores": {labels[0]: 9}}
    )
    assert bad.status_code == 422

    profile = client.get("/preferences").json()
    cell = next(
        c
        for c in profile["cells"]
        if c["text_type"] == "narration" and c["language"] == "zh"
    )
    by_engine = {e["engine_id"]: e for e in cell["engines"]}
    assert sum(e["score_count"] for e in by_engine.values()) == 2
    best = cell["engines"][0]
    assert best["average_score"] == 5.0
    assert best["engine_id"] == scored["entries"][[e["label"] for e in scored["entries"]].index(labels[0])]["engine_id"]


def test_compare_requires_voice_and_two_engines(client):
    r = client.post(
        "/compare",
        json={"voice_id": "missing", "text": "x", "engine_ids": TEST_ENGINES},
    )
    assert r.status_code == 404

    voice_id = _make_voice(client)
    r = client.post(
        "/compare",
        json={"voice_id": voice_id, "text": "x", "engine_ids": ["fake"]},
    )
    assert r.status_code == 422

    r = client.post(
        "/compare",
        json={
            "voice_id": voice_id,
            "text": "x",
            "engine_ids": TEST_ENGINES,
            "text_type": "not-a-type",
        },
    )
    assert r.status_code == 422


def test_compare_failed_leg_does_not_kill_session(client):
    # The BYOK cloud-shaped fake requires a configured key; without one its
    # leg fails while the other legs still compare.
    client.delete("/settings/keys/fake-key")
    voice_id = _make_voice(client)
    r = client.post(
        "/compare",
        json={
            "voice_id": voice_id,
            "text": "部分失败仍然可用。",
            "engine_ids": ["fake-key", *TEST_ENGINES],
        },
    )
    assert r.status_code == 200
    blind = r.json()
    # Blind mode hides WHICH engine failed, only how many.
    assert blind["failed_count"] == 1 and blind["failed"] == []
    revealed = client.get(f"/compare/{blind['id']}?reveal=true").json()
    assert [f["engine_id"] for f in revealed["failed"]] == ["fake-key"]
    assert len(revealed["entries"]) == 2


def test_compare_delete_removes_session_but_keeps_generations(client, sidecar):
    voice_id = _make_voice(client)
    session = client.post(
        "/compare",
        json={
            "voice_id": voice_id,
            "text": "删除会话保留历史。",
            "engine_ids": TEST_ENGINES,
        },
    ).json()
    revealed = client.get(f"/compare/{session['id']}?reveal=true").json()
    norm_files = [e["normalized_file"] for e in revealed["entries"]]
    r = client.delete(f"/compare/{session['id']}")
    assert r.status_code == 200
    for name in norm_files:
        assert not (sidecar["audio_dir"] / name).exists()
    # Original generation records remain browsable history.
    r = client.get("/generations", params={"limit": 100})
    ids = {rec["id"] for rec in r.json()["records"]}
    # Entry generation ids were blind-stripped from the earlier response, so
    # just assert history is non-empty and intact.
    assert len(ids) >= 2


# --- pure unit behavior ------------------------------------------------------


def test_detect_language():
    assert detect_language("你好，世界。") == "zh"
    assert detect_language("Hello, world!") == "en"
    assert detect_language("你好 world") == "zh"
    assert detect_language("12345") == "zh"  # no letters -> default zh


def test_compare_store_scores_validation(tmp_path: Path):
    _ = CompareStore(tmp_path)
    session = {
        "id": "s1",
        "created_at": "t",
        "entries": [
            {"label": "A", "engine_id": "x"},
            {"label": "B", "engine_id": "y"},
        ],
        "scored": False,
    }
    out = CompareStore.set_scores(session, {"A": 3})
    assert out["entries"][0]["score"] == 3
    assert out["scored"] is False  # B still unscored
    out = CompareStore.set_scores(out, {"B": 5})
    assert out["scored"] is True

    with pytest.raises(CompareError):
        CompareStore.set_scores(session, {"Z": 3})
    with pytest.raises(CompareError):
        CompareStore.set_scores(session, {"A": 0})
    with pytest.raises(CompareError):
        CompareStore.set_scores(session, {"A": "5"})


def test_preferences_are_read_only_over_sessions(tmp_path: Path):
    store = CompareStore(tmp_path)
    store.create(
        {
            "id": "s1",
            "created_at": "t",
            "language": "en",
            "text_type": "dialogue",
            "entries": [
                {"label": "A", "engine_id": "e1", "score": 4},
                {"label": "B", "engine_id": "e2", "score": 2},
            ],
        }
    )
    before = store.preference_profile()
    store.preference_profile()
    assert store.preference_profile() == before  # pure read, stable output
    cell = next(
        c for c in before["cells"] if c["language"] == "en" and c["text_type"] == "dialogue"
    )
    assert cell["engines"][0]["engine_id"] == "e1"

    # Unscored entries never leak into the profile.
    store.create({"id": "s2", "created_at": "t", "language": "en", "text_type": "dialogue",
                  "entries": [{"label": "A", "engine_id": "e1"}]})
    profile = store.preference_profile()
    cell = next(
        c for c in profile["cells"] if c["language"] == "en" and c["text_type"] == "dialogue"
    )
    assert all(e["score_count"] == 1 for e in cell["engines"])


# -- local residency invariant (multi-local compare must not pile up models) --


def _residency_violations(events):
    """Walk [{kind, engine}] events; return any second model loaded while
    another resident-worker model has not been unloaded yet."""
    violations = []
    resident = None
    for ev in events:
        kind, engine = ev["kind"], ev["engine"]
        if kind == "load":
            if resident is not None and resident != engine:
                violations.append((resident, engine))
            resident = engine
        elif kind == "unload" and resident == engine:
            resident = None
    return violations


def test_compare_never_keeps_two_local_models_resident(client, sidecar):
    """Three local models compared with a cloud one locked a machine up:
    each local engine keeps its model resident, so running legs back to
    back left several full models in memory at once. The generation path
    must evict the previous local model before the next one loads."""
    events_path = sidecar["audio_dir"] / "fake-residency.jsonl"
    events_path.unlink(missing_ok=True)
    voice_id = _make_voice(client)
    r = client.post(
        "/compare",
        json={
            "voice_id": voice_id,
            "text": "两个本地引擎先后生成，不允许同时驻留。",
            "engine_ids": ["fake-resident-a", "fake-resident-b"],
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["entries"]) == 2
    import json

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["kind"] for e in events].count("load") == 2, events
    assert _residency_violations(events) == [], events
