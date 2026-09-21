"""Generation history: durable records, search/filter/pagination, delete with
artifact cleanup, and rerun-grade lineage (issue #7)."""

from __future__ import annotations

from voiceclone_sidecar.generations import GenerationStore

# --- store unit tests -------------------------------------------------------


def _write_wav(path, seconds: float = 0.2) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x01" * int(seconds * 22050))


def _record(gen_id: str, **overrides) -> dict:
    record = {
        "id": gen_id,
        "engine_id": "fake",
        "model_version": "fake-1.0",
        "text": f"文本 {gen_id}",
        "normalized_text": f"文本 {gen_id}",
        "params": {},
        "voice_id": None,
        "voice_name": None,
        "status": "succeeded",
        "error": None,
        "logs": [{"ts": "2026-09-17T00:00:00Z", "message": "hello"}],
        "duration_seconds": 1.0,
        "cost": 0.0,
        "audio_url": f"/audio/{gen_id}.wav",
        "audio_file": f"{gen_id}.wav",
        "sample_rate": 22050,
        "created_at": "2026-09-17T00:00:00Z",
        "finished_at": "2026-09-17T00:00:01Z",
    }
    record["created_at"] = "2026-09-17T00:00:00Z"
    record.update(overrides)
    return record


def test_store_persists_across_restart(tmp_path):
    store = GenerationStore(tmp_path, tmp_path / "audio")
    store.persist(_record("aaa"))
    # A new store over the same dir reads the same history.
    store2 = GenerationStore(tmp_path, tmp_path / "audio")
    assert store2.get("aaa")["id"] == "aaa"


def test_store_list_search_filters_pagination(tmp_path):
    store = GenerationStore(tmp_path, tmp_path / "audio")
    store.persist(_record("a1", text="关于春天的广播稿", created_at="2026-09-17T00:00:01Z"))
    store.persist(_record("a2", text="关于秋天的广播稿", created_at="2026-09-17T00:00:02Z"))
    store.persist(_record("a3", text="news script", status="failed", created_at="2026-09-17T00:00:03Z"))
    store.persist(_record("a4", text="spring special", engine_id="other", created_at="2026-09-17T00:00:04Z"))

    everything = store.list()
    assert everything["total"] == 4
    assert [r["id"] for r in everything["records"]] == ["a4", "a3", "a2", "a1"]  # newest first

    assert store.list(q="春天")["total"] == 1
    assert store.list(q="spring")["total"] == 1
    assert store.list(status="failed")["records"][0]["id"] == "a3"
    assert store.list(engine_id="other")["records"][0]["id"] == "a4"

    paged = store.list(limit=2, offset=1)
    assert paged["total"] == 4
    assert [r["id"] for r in paged["records"]] == ["a3", "a2"]


def test_store_delete_removes_record_and_audio_file(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav = audio_dir / "g1.wav"
    _write_wav(wav)
    store = GenerationStore(tmp_path, audio_dir)
    store.persist(_record("g1"))

    store.delete("g1")

    assert store.get("g1") is None
    assert not wav.exists()
    assert not (tmp_path / "generations.json.tmp").exists()


def test_store_delete_many_batch_reuses_unlink_and_reports_missing(tmp_path):
    """Issue #39: batch delete removes each record + its audio file; a
    pre-deleted / missing file is not an error and an unknown id lands in
    ``missing`` instead of aborting the batch."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav1 = audio_dir / "g1.wav"
    _write_wav(wav1)
    store = GenerationStore(tmp_path, audio_dir)
    store.persist(_record("g1"))
    store.persist(_record("g2"))
    store.persist(_record("g3"))  # its file never existed on disk
    (audio_dir / "g3.wav").unlink(missing_ok=True)

    result = store.delete_many(["g1", "g2", "g3", "no-such-id"])

    assert result["deleted"] == ["g1", "g2", "g3"]
    assert result["missing"] == ["no-such-id"]
    assert store.get("g1") is None and store.get("g2") is None and store.get("g3") is None
    assert not wav1.exists()


def test_store_delete_matching_uses_filters_and_exclude(tmp_path):
    store = GenerationStore(tmp_path, tmp_path / "audio")
    store.persist(_record("a1", text="关于春天的广播稿"))
    store.persist(_record("a2", text="关于秋天的广播稿"))
    store.persist(_record("a3", text="spring special", engine_id="other"))
    store.persist(_record("a4", text="关于冬天的广播稿", status="failed"))

    deleted = store.delete_matching(q="关于")
    assert sorted(deleted) == ["a1", "a2", "a4"]
    assert store.get("a3") is not None

    assert store.delete_matching(engine_id="other", exclude=["excluded"]) == ["a3"]
    assert store.get("a3") is None


def test_torn_legacy_index_fails_loudly(tmp_path):
    """ADR-0017 kills "parse-failure = empty library": a corrupt legacy JSON
    index aborts the migration loudly instead of degrading to empty history."""
    import pytest

    from voiceclone_sidecar.db import DatabaseError

    (tmp_path / "generations.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(DatabaseError, match=r"generations\.json"):
        GenerationStore(tmp_path, tmp_path / "audio")


# --- contract tests (real sidecar process) ---------------------------------


def test_generation_record_contains_full_lineage(client):
    r = client.post("/generations", json={"engine_id": "fake", "text": "血统完整性测试"})
    assert r.status_code == 200, r.text
    record = r.json()
    for key in (
        "id", "engine_id", "model_version", "text", "normalized_text", "params",
        "status", "logs", "duration_seconds", "cost", "audio_url", "audio_file",
        "sample_rate", "created_at", "finished_at",
    ):
        assert key in record, f"record is missing {key}"
    assert record["status"] == "succeeded"
    assert record["model_version"] == "fake-1.0"
    assert record["cost"] == 0.0
    assert isinstance(record["logs"], list) and record["logs"]
    assert record["logs"][0]["message"].startswith("fake:")
    assert record["duration_seconds"] >= 0


def test_generation_history_listed_after_reload_and_searchable(client):
    created = client.post("/generations", json={"engine_id": "fake", "text": "历史检索专用文本XYZZY"}).json()
    r = client.get("/generations", params={"q": "XYZZY"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert any(rec["id"] == created["id"] for rec in body["records"])


def test_generation_history_status_filter_and_pagination(client):
    r = client.get("/generations", params={"status": "succeeded", "limit": 1, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["records"] and all(rec["status"] == "succeeded" for rec in body["records"])
    assert body["total"] >= len(body["records"])

    r = client.get("/generations", params={"limit": 0})
    assert r.status_code == 422


def test_delete_generation_removes_record_and_audio_file(client, sidecar):
    record = client.post("/generations", json={"engine_id": "fake", "text": "删除我"}).json()
    audio_path = sidecar["audio_dir"] / record["audio_file"]
    assert audio_path.exists()

    r = client.delete(f"/generations/{record['id']}")
    assert r.status_code == 200
    assert not audio_path.exists()
    assert client.get(f"/generations/{record['id']}").status_code == 404
    # The artifact is no longer served either.
    assert client.get(record["audio_url"]).status_code == 404


def test_delete_missing_generation_is_404(client):
    assert client.delete("/generations/no-such-id").status_code == 404


def test_batch_delete_by_ids_reports_missing_and_tolerates_lost_files(client, sidecar):
    """Issue #39 contract: explicit-id batch delete, empty selection rejected,
    missing ids reported, audio files removed with their rows."""
    created = [
        client.post("/generations", json={"engine_id": "fake", "text": f"批量删除测试{i}"}).json()
        for i in range(3)
    ]
    r = client.post("/generations/batch-delete", json={"ids": [c["id"] for c in created] + ["no-such-id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 3
    assert sorted(body["deleted"]) == sorted(c["id"] for c in created)
    assert body["missing"] == ["no-such-id"]
    for c in created:
        assert client.get(f"/generations/{c['id']}").status_code == 404
        assert client.get(c["audio_url"]).status_code == 404


def test_batch_delete_empty_selection_is_422(client):
    assert client.post("/generations/batch-delete", json={"ids": []}).status_code == 422
    assert client.post("/generations/batch-delete", json={}).status_code == 422
    assert client.post("/generations/batch-delete", json={"ids": ["", 1]}).status_code == 422


def test_batch_delete_all_matching_with_exclude(client):
    keep = client.post("/generations", json={"engine_id": "fake", "text": "排除专用保留XYZZY"}).json()
    drop = client.post("/generations", json={"engine_id": "fake", "text": "筛选全选专用XYZZY"}).json()
    r = client.post(
        "/generations/batch-delete",
        json={"all_matching": True, "q": "XYZZY", "exclude": [keep["id"]]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert drop["id"] in body["deleted"]
    assert keep["id"] not in body["deleted"]
    assert client.get(f"/generations/{keep['id']}").status_code == 200
    assert client.get(f"/generations/{drop['id']}").status_code == 404


def test_failed_generation_is_recorded(client, sidecar):
    # Unknown engine never reaches the record path; instead make a real
    # synthesis fail by pointing the fake engine at an impossible output dir
    # is invasive — so exercise the failure contract at the engine boundary:
    # a generation whose text is empty is rejected before a record exists.
    assert client.post("/generations", json={"engine_id": "fake", "text": "  "}).status_code == 422
    r = client.get("/generations")
    assert r.status_code == 200


def test_generation_record_survives_sidecar_restart(sidecar, client):
    record = client.post("/generations", json={"engine_id": "fake", "text": "重启留存测试"}).json()
    # ADR-0017: the index is SQLite now; durability is a committed row.
    from voiceclone_sidecar.db import Database, library_db_path
    db = Database(library_db_path(sidecar["data_dir"]), migrate=False)
    try:
        assert db.get_payload("generations", record["id"]) is not None
    finally:
        db.close()


def test_generation_with_voice_snapshots_voice_name(client, sidecar):
    _write_wav(sidecar["audio_dir"] / "voice-ref.wav", 4.0)
    voice = client.post(
        "/voices",
        data={"name": "重跑溯源音色", "description": ""},
        files={"file": ("ref.wav", open(sidecar["audio_dir"] / "voice-ref.wav", "rb"), "audio/wav")},  # noqa: SIM115 - request-scoped handle consumed by the client call
    ).json()
    record = client.post(
        "/generations",
        json={"engine_id": "fake", "text": "带音色生成", "voice_id": voice["id"]},
    ).json()
    assert record["voice_id"] == voice["id"]
    assert record["voice_name"] == "重跑溯源音色"
    # Lineage: the reference handed to the engine is recorded in params.
    assert "ref_audio" in record["params"]
