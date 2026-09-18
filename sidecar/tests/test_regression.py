"""能力矩阵固化：中文回归集 + 回归运行 + 矩阵聚合（issue #16）。"""

from __future__ import annotations

import wave

import pytest

from voiceclone_sidecar.regression import (
    REGRESSION_CATEGORIES,
    REGRESSION_ITEMS,
    RegressionStore,
    aggregate_session,
    char_error_rate,
    load_matrix_data,
    wav_duration_seconds,
)


# ---------------------------------------------------------------------------
# 固化的中文回归集
# ---------------------------------------------------------------------------


def test_regression_set_covers_all_five_categories():
    covered = {item["category"] for item in REGRESSION_ITEMS}
    assert covered == set(REGRESSION_CATEGORIES)


def test_regression_items_are_unique_and_complete():
    ids = [item["id"] for item in REGRESSION_ITEMS]
    assert len(ids) == len(set(ids))
    for item in REGRESSION_ITEMS:
        assert item["text"].strip(), item["id"]
        assert item["expect"].strip(), item["id"]
        assert item["category"] in REGRESSION_CATEGORIES


def test_each_category_has_at_least_three_items():
    for category in REGRESSION_CATEGORIES:
        items = [i for i in REGRESSION_ITEMS if i["category"] == category]
        assert len(items) >= 3, category


def test_regression_set_includes_hard_cases():
    texts = "".join(i["text"] for i in REGRESSION_ITEMS)
    # 数字：阿拉伯数字原文直接出现在回归集里（交给归一化层改写）
    assert any(i["category"] == "numbers" and any(c.isdigit() for c in i["text"]) for i in REGRESSION_ITEMS)
    # 多音字：典型的多音字必须被覆盖
    for ch in ("重", "行", "长"):
        assert ch in texts, ch
    # 中英混读：至少一条同时含汉字与拉丁字母
    assert any(
        i["category"] == "mixed"
        and any("\u4e00" <= c <= "\u9fff" for c in i["text"])
        and any(c.isascii() and c.isalpha() for c in i["text"])
        for i in REGRESSION_ITEMS
    )


# ---------------------------------------------------------------------------
# 纯函数：CER 与 WAV 时长
# ---------------------------------------------------------------------------


def test_char_error_rate_identical_text_is_zero():
    assert char_error_rate("今天天气不错", "今天天气不错") == 0.0


def test_char_error_rate_counts_edits():
    # 参考 6 字，假设漏 1 换 1 → 2 次编辑 / 6
    assert char_error_rate("一二三四五六", "一三四七五六") == pytest.approx(2 / 6)


def test_char_error_rate_handles_empty_and_none():
    assert char_error_rate("", "") is None
    assert char_error_rate("abc", "") == pytest.approx(1.0)
    assert char_error_rate("", "xyz") is None


def test_wav_duration_seconds(tmp_path):
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050)  # 1 second
    assert wav_duration_seconds(p) == pytest.approx(1.0, abs=1e-3)


def test_wav_duration_missing_file_returns_none(tmp_path):
    assert wav_duration_seconds(tmp_path / "nope.wav") is None


def test_peak_vram_sampler_degrades_without_nvidia_smi(monkeypatch):
    import shutil

    from voiceclone_sidecar.regression import sample_peak_vram

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with sample_peak_vram() as holder:
        pass
    assert holder == {"peak_vram_bytes": None}


# ---------------------------------------------------------------------------
# RegressionStore
# ---------------------------------------------------------------------------


def _session(engines=("fake",)) -> dict:
    return {
        "id": "sess-1",
        "created_at": "2026-09-18T00:00:00Z",
        "voice_id": None,
        "engines": list(engines),
        "status": "completed",
        "items": [
            {
                "item_id": REGRESSION_ITEMS[0]["id"],
                "category": REGRESSION_ITEMS[0]["category"],
                "engine_id": "fake",
                "status": "ok",
                "generation_id": "g1",
                "audio_url": "/audio/g1.wav",
                "audio_seconds": 2.0,
                "wall_seconds": 1.0,
                "rtf": 0.5,
                "peak_vram_bytes": None,
                "asr_text": None,
                "cer": None,
                "error": None,
            },
            {
                "item_id": "date-01",
                "category": "dates",
                "engine_id": "fake",
                "status": "failed",
                "generation_id": None,
                "audio_url": None,
                "audio_seconds": None,
                "wall_seconds": None,
                "rtf": None,
                "peak_vram_bytes": None,
                "asr_text": None,
                "cer": None,
                "error": "boom",
            },
        ],
    }


def test_store_roundtrip(tmp_path):
    store = RegressionStore(tmp_path)
    session = _session()
    store.create(session)
    reloaded = RegressionStore(tmp_path)
    assert reloaded.get("sess-1")["id"] == "sess-1"
    assert reloaded.list()[0]["id"] == "sess-1"
    assert reloaded.get("nope") is None


def test_store_update_and_delete(tmp_path):
    store = RegressionStore(tmp_path)
    session = _session()
    store.create(session)
    session["status"] = "running"
    store.update(session)
    assert store.get("sess-1")["status"] == "running"
    store.delete("sess-1")
    assert store.get("sess-1") is None


# ---------------------------------------------------------------------------
# 聚合：每引擎 × 每类别的成功率与 RTF
# ---------------------------------------------------------------------------


def test_aggregate_session_counts_and_mean_rtf():
    session = _session()
    summary = aggregate_session(session)
    fake = summary["engines"]["fake"]
    assert fake["ok"] == 1 and fake["failed"] == 1
    assert fake["rtf_mean"] == pytest.approx(0.5)
    by_cat = fake["by_category"][REGRESSION_ITEMS[0]["category"]]
    assert by_cat["ok"] == 1 and by_cat["failed"] == 0


def test_aggregate_empty_session_is_safe():
    summary = aggregate_session({"engines": ["fake"], "items": []})
    fake = summary["engines"]["fake"]
    assert fake["ok"] == 0 and fake["failed"] == 0 and fake["rtf_mean"] is None


# ---------------------------------------------------------------------------
# 固化的能力矩阵数据
# ---------------------------------------------------------------------------


def test_matrix_data_lists_all_shipped_engines():
    data = load_matrix_data()
    ids = {e["engine_id"] for e in data["engines"]}
    assert ids == {
        "fake",
        "qwen3-tts-vc-cloud",
        "qwen3-tts-vd-cloud",
        "qwen3-tts-mlx",
        "indextts-25-cuda",
        "indextts-25-mps",
    }


def test_matrix_data_every_fact_carries_verification():
    data = load_matrix_data()
    allowed = {"measured", "verified", "vendor", "unverified", "unknown"}
    for engine in data["engines"]:
        for fact in engine.get("evidence", []):
            assert fact["verification"] in allowed, (engine["engine_id"], fact)
        for perf in engine.get("performance", []):
            assert perf["verification"] in allowed, (engine["engine_id"], perf)


def test_matrix_data_marks_rtx3090_benchmark_as_pending():
    data = load_matrix_data()
    indextts = next(e for e in data["engines"] if e["engine_id"] == "indextts-25-cuda")
    perf = indextts["performance"]
    assert perf, "IndexTTS-2.5 必须有性能条目（哪怕标注为待测）"
    on_3090 = [p for p in perf if "3090" in p.get("env", "")]
    assert on_3090, "必须有 RTX 3090 条目占位"
    assert all(
        p["verification"] in {"unverified", "unknown"} and p["rtf"] is None
        for p in on_3090
    ), "3090 实测未跑之前，3090 的性能数字不得标为已核实"


# ---------------------------------------------------------------------------
# 端到端：真实 sidecar 进程上的回归端点与矩阵端点
# ---------------------------------------------------------------------------


def test_regression_run_completes_on_fake_engines(client):
    engines = [e["id"] for e in client.get("/engines").json()["engines"]]
    engines = [e for e in engines if e.startswith("fake-loud") or e == "fake"]
    r = client.post("/regression/run", json={"engine_ids": engines})
    assert r.status_code == 200, r.text
    session_id = r.json()["id"]

    # 后台运行：轮询直到完成（fake 引擎是秒级）。
    import time

    deadline = time.monotonic() + 60
    while True:
        s = client.get(f"/regression/{session_id}").json()
        if s["status"] == "completed":
            break
        assert time.monotonic() < deadline, s
        time.sleep(0.3)

    assert len(s["items"]) == len(engines) * len(REGRESSION_ITEMS)
    ok_items = [i for i in s["items"] if i["status"] == "ok"]
    assert ok_items, [i for i in s["items"] if i["status"] != "ok"]
    for i in ok_items:
        assert i["audio_url"], i
        assert i["audio_seconds"] and i["audio_seconds"] > 0
        assert i["rtf"] and i["rtf"] > 0

    summary = s["summary"]["engines"]
    assert all(summary[e]["ok"] > 0 for e in engines)


def test_regression_session_listing_and_delete(client):
    r = client.post("/regression/run", json={"engine_ids": ["fake"]})
    session_id = r.json()["id"]
    import time

    deadline = time.monotonic() + 60
    while client.get(f"/regression/{session_id}").json()["status"] != "completed":
        assert time.monotonic() < deadline
        time.sleep(0.3)
    listed = client.get("/regression/sessions").json()["sessions"]
    assert any(s["id"] == session_id for s in listed)
    assert client.delete(f"/regression/{session_id}").status_code == 200
    assert client.get(f"/regression/{session_id}").status_code == 404


def test_capability_matrix_endpoint_shape(client):
    r = client.get("/capability-matrix")
    assert r.status_code == 200, r.text
    data = r.json()
    ids = {e["engine_id"] for e in data["engines"]}
    assert "fake" in ids and "indextts-25-cuda" in ids
    for e in data["engines"]:
        # 固化事实 + 运行时声明同时在场
        assert "evidence" in e and "runtime" in e
    fake = next(e for e in data["engines"] if e["engine_id"] == "fake")
    assert fake["runtime"]["capabilities"]["voice_cloning"] is True
    # 类别与回归集条目随矩阵下发，界面据此渲染
    assert {c["id"] for c in data["categories"]} == {
        "numbers", "dates", "amounts", "mixed", "polyphones",
    }
    assert len(data["regression_items"]) >= 15


def test_regression_run_rejects_unknown_engine(client):
    r = client.post("/regression/run", json={"engine_ids": ["nope"]})
    assert r.status_code == 404
