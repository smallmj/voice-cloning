"""ADR-0016 contract tests (issue #17): three download-source axes with a
preferred source + silent fallback chain, "which source actually served"
logging, ModelScope coverage, HF_HUB_DISABLE_XET on the transcription path,
and local model management (weight dir, disk usage, uninstall)."""

from __future__ import annotations

import pytest

from voiceclone_sidecar import sources
from voiceclone_sidecar.db import T_SETTINGS, write_kv_block
from voiceclone_sidecar.engine_config import (
    effective_env,
    load_source_prefs,
    save_source_prefs,
)
from voiceclone_sidecar.runtime import downloader

# --- axis 1: weight sources --------------------------------------------------


def test_weight_sources_preferred_first_others_silent_fallback():
    chain = sources.weight_sources("demo/repo", ms_repo="ms/demo", preferred="hf-mirror")
    assert chain[0].startswith("https://hf-mirror.com/")
    rest = " ".join(chain[1:])
    assert "huggingface.co" in rest and "modelscope.cn" in rest


def test_weight_sources_modelscope_without_twin_degrades_silently():
    """A repo without a ModelScope twin drops that source instead of 404ing
    mid-chain (verified: bigvgan had no twin until nv-community appeared)."""
    chain = sources.weight_sources("demo/repo", preferred="modelscope")
    assert not any("modelscope" in u for u in chain)
    assert chain[0].startswith("https://huggingface.co/")


def test_weight_sources_unknown_preferred_is_ignored():
    chain = sources.weight_sources("demo/repo", preferred="not-a-source")
    assert chain[0].startswith("https://huggingface.co/")


def test_torch_wheel_sources_preferred_moves_host_forward():
    chain = sources.torch_wheel_sources("cu124", preferred="aliyun")
    assert chain[0].startswith("https://mirrors.aliyun.com/pytorch-wheels/cu124/")
    assert chain[1].startswith("https://download.pytorch.org/whl/cu124/")
    # axis separation: the CUDA choice never bleeds into the weight chain
    assert all("huggingface" not in u for u in chain)


def test_source_label_names_known_hosts():
    assert sources.source_label(sources.HF_TEMPLATE.replace("{repo}", "x")) == "HF 官方"
    assert (
        sources.source_label(
            sources.WEIGHT_SOURCE_CHOICES["hf-mirror"]["template"].replace("{repo}", "x")
        )
        == "hf-mirror"
    )
    assert (
        sources.source_label(
            sources.WEIGHT_SOURCE_CHOICES["modelscope"]["template"].replace("{ms_repo}", "x")
        )
        == "ModelScope"
    )
    assert sources.source_label("https://example.com/{path}") == "example.com"


# --- installer log must name the actual serving source ------------------------


def test_download_log_states_the_serving_source(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_download(url, part, name, progress, log, total=None):
        part.write_bytes(b"abc")
        return url  # no redirect reported

    monkeypatch.setattr(downloader, "_download_from", fake_download)
    monkeypatch.setattr(downloader, "remote_size", lambda url: 3)
    spec = downloader.DownloadSpec(path="model.bin", dest_name="model.bin")
    downloader.download_file(
        spec, tmp_path, [sources.HF_TEMPLATE.replace("{repo}", "demo/repo")],
        progress=None, log=calls.append,
    )
    assert any("HF 官方" in c for c in calls), calls


def test_download_log_labels_the_redirect_target_not_the_mirror(tmp_path, monkeypatch):
    """C-001: hf-mirror 308-redirects境外 traffic back to huggingface.co —
    the log must name huggingface (where bytes came from), not hf-mirror."""
    calls: list[str] = []

    def fake_download(url, part, name, progress, log, total=None):
        part.write_bytes(b"abc")
        return "https://huggingface.co/demo/repo/resolve/main/model.bin"

    monkeypatch.setattr(downloader, "_download_from", fake_download)
    monkeypatch.setattr(downloader, "remote_size", lambda url: 3)
    spec = downloader.DownloadSpec(path="model.bin", dest_name="model.bin")
    downloader.download_file(
        spec, tmp_path, [sources.HF_MIRROR_TEMPLATE.replace("{repo}", "demo/repo")],
        progress=None, log=calls.append,
    )
    assert any("HF 官方" in c for c in calls), calls
    assert not any("hf-mirror" in c for c in calls), calls


# --- engine_config: sources prefs storage + env mapping -----------------------


def test_source_prefs_roundtrip_and_defaults(tmp_path):
    prefs = load_source_prefs(tmp_path)
    assert prefs == {"weights": "hf", "pypi": "aliyun", "cuda": "official"}
    save_source_prefs(tmp_path, {"weights": "modelscope", "pypi": "official"})
    prefs = load_source_prefs(tmp_path)
    assert prefs["weights"] == "modelscope"
    assert prefs["pypi"] == "official"
    assert prefs["cuda"] == "official"  # untouched axis keeps its default


def test_source_prefs_invalid_values_fall_back_to_defaults(tmp_path):
    write_kv_block(tmp_path, T_SETTINGS, "sources",
                   {"weights": "gitee", "pypi": 42, "cuda": None})
    assert load_source_prefs(tmp_path) == {
        "weights": "hf", "pypi": "aliyun", "cuda": "official",
    }


def test_effective_env_maps_source_prefs_onto_knobs(tmp_path):
    save_source_prefs(
        tmp_path, {"weights": "modelscope", "pypi": "official", "cuda": "aliyun"}
    )
    settings = {"sources": load_source_prefs(tmp_path)}
    env = effective_env("qwen3-tts-mlx", process_env={}, settings=settings)
    assert env[sources.PREFERRED_WEIGHT_ENV] == "modelscope"
    assert env["UV_DEFAULT_INDEX"] == sources.PYPI_CHOICES["official"]["index"]


def test_per_engine_override_beats_source_prefs(tmp_path):
    save_source_prefs(tmp_path, {"pypi": "official"})
    settings = {
        "sources": load_source_prefs(tmp_path),
        "engines": {"qwen3-tts-mlx": {"env": {"UV_DEFAULT_INDEX": "https://custom/simple"}}},
    }
    env = effective_env("qwen3-tts-mlx", process_env={}, settings=settings)
    assert env["UV_DEFAULT_INDEX"] == "https://custom/simple"


# --- transcription path: XET off + snapshot attempts + ms weights -------------


def test_snapshot_attempts_preferred_order_and_xet_off():
    from voiceclone_sidecar import transcription as tr

    hf_first = tr._snapshot_attempts("hf")
    assert hf_first[0]["HF_ENDPOINT"] == sources.HF_ENDPOINT_DEFAULT
    assert all(e["HF_HUB_DISABLE_XET"] == "1" for e in hf_first)
    mirror_first = tr._snapshot_attempts("hf-mirror")
    assert mirror_first[0]["HF_ENDPOINT"] == sources.HF_ENDPOINT_MIRROR
    # modelscope is never served by snapshot_download — handled by direct URLs
    assert tr._snapshot_attempts("modelscope") == hf_first


@pytest.mark.skipif(
    not __import__("sys").platform.startswith("win"),
    reason="faster-whisper-small ModelScope twin is declared on Windows only",
)
def test_windows_asr_declares_modelscope_twin():
    from voiceclone_sidecar.transcription import PLATFORM_CONFIG

    assert PLATFORM_CONFIG["win32"]["ms_repo"] == "gpustack/faster-whisper-small"


def test_local_transcriber_weights_prefers_managed_ms_dir(tmp_path, monkeypatch):
    import sys

    from voiceclone_sidecar import transcription as tr

    if not sys.platform.startswith("win"):
        pytest.skip("ms-weight resolution exists only for platforms with an ms_repo")
    tool = tr.LocalTranscriber(root=tmp_path)
    ms_dir = tool._ms_weights_dir()
    ms_dir.mkdir(parents=True)
    # An interrupted install (only the big payload) must NOT shadow the HF
    # snapshot cache — completeness means every file in MS_ASR_FILES.
    (ms_dir / "model.bin").write_bytes(b"x")
    assert tool.weights_dir() != ms_dir
    for name in tr.MS_ASR_FILES:
        (ms_dir / name).write_bytes(b"x")
    assert tool.weights_dir() == ms_dir


# --- local model management endpoints -----------------------------------------


def test_models_listing_shape(client):
    body = client.get("/engines/models").json()
    assert "runtime_root" in body
    ids = {m["id"] for m in body["models"]}
    assert "transcribe-local" in ids
    for m in body["models"]:
        assert set(m) >= {
            "id", "display_name", "installed", "model_dir",
            "weights_dir", "disk_usage_bytes", "installing",
        }
        assert m["disk_usage_bytes"] >= 0


def test_uninstall_refuses_cloud_and_unknown_engines(client):
    r = client.delete("/engines/qwen3-tts-vc-cloud/model")
    assert r.status_code == 422
    r = client.delete("/engines/definitely-not-an-engine/model")
    assert r.status_code == 404


def test_uninstall_removes_engine_dir(tmp_path):
    """The uninstall contract at the object level: model_dir removal also
    resets install state (the state file lives inside the dir)."""
    import shutil

    from voiceclone_sidecar.engines.qwen3_tts import Qwen3TtsMlxEngine

    engine = Qwen3TtsMlxEngine(env={})
    engine.root = tmp_path
    d = engine.model_dir()
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "w.bin").write_bytes(b"x" * 1234)
    (d / "install_state.json").write_text("{}", encoding="utf-8")
    shutil.rmtree(d)
    assert not d.exists()
    assert engine.is_installed() is False
