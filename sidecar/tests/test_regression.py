"""能力矩阵固化：中文回归集 + 回归运行 + 矩阵聚合（issue #16）。"""

from __future__ import annotations

import json
import time
import wave

import httpx
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

from .test_contract import auth_headers

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
        "qwen3-tts-vc-cloud",
        "qwen3-tts-vd-cloud",
        "minimax-speech-cloud",
        "qwen3-tts-mlx",
        "indextts-25-cuda",
        "indextts-25-mps",
        # issue #25: three new local engine families
        "fireredtts3-mps",
        "voxcpm2-mps",
        "voxcpm2-cuda",
        "dots-tts-cuda",
    }


def test_matrix_data_has_no_fake_rows():
    """issue #41: 矩阵是用户可见的能力矩阵，不得出现测试替身（fake）条目。"""
    data = load_matrix_data()
    for engine in data["engines"]:
        blob = json.dumps(engine, ensure_ascii=False)
        assert "fake" not in blob.lower(), engine["engine_id"]


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
    # issue #41: 测试替身（fake）不再是矩阵条目，用户可见矩阵只含真实引擎。
    assert "fake" not in ids and "indextts-25-cuda" in ids
    for e in data["engines"]:
        # 固化事实 + 运行时声明同时在场
        assert "evidence" in e and "runtime" in e
    indextts = next(e for e in data["engines"] if e["engine_id"] == "indextts-25-cuda")
    assert indextts["evidence"], "固化事实必须随矩阵下发"
    # 类别与回归集条目随矩阵下发，界面据此渲染
    assert {c["id"] for c in data["categories"]} == {
        "numbers", "dates", "amounts", "mixed", "polyphones",
    }
    assert len(data["regression_items"]) >= 15


def test_regression_run_rejects_unknown_engine(client):
    r = client.post("/regression/run", json={"engine_ids": ["nope"]})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 运行串行化与取消（issue #28）
# ---------------------------------------------------------------------------


def test_regression_run_serializes_concurrent_sessions(client):
    r1 = client.post("/regression/run", json={"engine_ids": ["fake-slow"]})
    assert r1.status_code == 200, r1.text
    sid1 = r1.json()["id"]
    # 第二个 POST 不再起第二个事件循环抢单：直接 409。
    r2 = client.post("/regression/run", json={"engine_ids": ["fake"]})
    assert r2.status_code == 409, r2.text
    # 删除运行中的会话会取消任务并立即释放串行锁。
    # DELETE awaits the cancelled task, so the serialization lock is
    # already released when the 200 comes back — no polling needed.
    assert client.delete(f"/regression/{sid1}").status_code == 200
    r3 = client.post("/regression/run", json={"engine_ids": ["fake"]})
    assert r3.status_code == 200, r3.text
    client.delete(f"/regression/{r3.json()['id']}")


def test_regression_run_keeps_logs_ws_and_jobs_responsive(client, sidecar, ws_url):
    """验收（issue #28）：回归运行期间 /ws/logs 实时且 /jobs 响应。

    旧实现里回归 worker 在第二个事件循环上运行，重活期间会拖住主循环；
    现在它是主循环上的一个任务 + executor，所以运行中这两个通道必须活着。
    """
    import asyncio
    import json as jsonlib

    import websockets

    async def run():
        async with websockets.connect(
            ws_url, additional_headers=auth_headers(sidecar)
        ) as ws:
            r = await asyncio.get_running_loop().run_in_executor(
                None, lambda: client.post("/regression/run", json={"engine_ids": ["fake"]})
            )
            assert r.status_code == 200, r.text
            session_id = r.json()["id"]

            # 运行期间 /jobs 必须及时响应（旧架构下会被重活拖住）。
            loop = asyncio.get_running_loop()
            jobs = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: client.get("/jobs")), timeout=5
            )
            assert jobs.status_code == 200

            # 日志必须实时流到 /ws/logs（回归 item 走 run_generation 管线）。
            saw_log = False
            deadline = asyncio.get_running_loop().time() + 30
            while asyncio.get_running_loop().time() < deadline:
                event = jsonlib.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                if event.get("type") == "log":
                    saw_log = True
                    break
            assert saw_log, "回归运行期间没有日志实时流出 /ws/logs"

            deadline = time.monotonic() + 60
            while client.get(f"/regression/{session_id}").json()["status"] != "completed":
                assert time.monotonic() < deadline
                await asyncio.sleep(0.3)

    asyncio.run(asyncio.wait_for(run(), timeout=90))


def test_regression_run_clean_under_asyncio_debug(tmp_path_factory):
    """验收（issue #28）：asyncio debug 下回归运行无跨线程/跨循环告警。

    旧实现在 daemon 线程里 asyncio.run 起第二个 loop，对主 loop 创建的
    LogBus 队列 put_nowait；debug 模式会记 Non-thread-safe / cross-loop
    告警。现在 worker 就在主循环上，起一个 PYTHONASYNCIODEBUG=1 的真实
    sidecar 跑一遍回归，stderr 必须干净。
    """
    import json as jsonlib
    import os
    import secrets
    import socket
    import subprocess
    import sys
    import time as time_mod

    SIDECAR_DIR = tmp_path_factory.mktemp("sidecar-root")
    token = secrets.token_urlsafe(16)
    proc = subprocess.Popen(
        [sys.executable, "-m", "voiceclone_sidecar", "--port", "0", "--token", token,
         "--audio-dir", str(SIDECAR_DIR / "audio")],
        env={**os.environ, "PYTHONASYNCIODEBUG": "1", "VOICECLONE_TEST_ENGINES": "1",
             "VOICECLONE_KEY_BACKEND": "memory"},
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        handshake = jsonlib.loads(proc.stdout.readline())
        assert handshake["event"] == "ready"
        base = f"http://127.0.0.1:{handshake['port']}"
        deadline = time_mod.monotonic() + 15
        while True:
            try:
                socket.create_connection(("127.0.0.1", handshake["port"]), timeout=0.5).close()
                break
            except OSError:
                assert time_mod.monotonic() < deadline
                time_mod.sleep(0.1)
        with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {token}"}) as c:
            r = c.post("/regression/run", json={"engine_ids": ["fake"]})
            assert r.status_code == 200, r.text
            sid = r.json()["id"]
            deadline = time_mod.monotonic() + 60
            while c.get(f"/regression/{sid}").json()["status"] != "completed":
                assert time_mod.monotonic() < deadline
                time_mod.sleep(0.3)
        proc.terminate()
        _, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    low = stderr.lower()
    for marker in ("non-thread-safe", "call_soon_threadsafe was not called",
                   "took ", "execute took"):
        assert marker not in low, f"asyncio debug 告警：{marker!r} 出现在 stderr"
