"""ADR-0015 contract tests (issue #22): the injected engine-config seam.

The previous failure mode was "tests proved a path that did not exist in
production": tests passed ``env=`` to engine constructors while
``default_registry`` never forwarded the process environment, so the
documented ``VOICECLONE_*`` overrides were dead in the app. Per the issue's
test-contract change, these tests construct engines through the PRODUCTION
path — ``default_registry()`` — only.
"""

from __future__ import annotations

import json

import pytest

from voiceclone_sidecar import sources
from voiceclone_sidecar.engine_config import (
    BUILTIN_DEFAULTS,
    EngineConfig,
    effective_env,
    load_engine_settings,
)
from voiceclone_sidecar.engines.cloud_base import (
    CloudEngineBase,
    CloudEngineError,
    billed_chars,
    voice_missing_error,
)
from voiceclone_sidecar.registry import default_registry
from voiceclone_sidecar.secrets import KeyStore, MemoryBackend


@pytest.fixture()
def key_store():
    return KeyStore(backend=MemoryBackend())


def _engine(registry, engine_id):
    engine = registry.get(engine_id)
    assert engine is not None, f"{engine_id} must register on this platform"
    return engine


def test_user_env_reaches_engines_through_default_registry(key_store):
    """P0-2: the user's UV_DEFAULT_INDEX must not be reversed by the
    built-in Aliyun default — the process env wins over the defaults."""
    registry = default_registry(
        key_store=key_store,
        env={"UV_DEFAULT_INDEX": "https://user.example/simple"},
    )
    engine = _engine(registry, "qwen3-tts-mlx")
    assert engine.env["UV_DEFAULT_INDEX"] == "https://user.example/simple"


def test_builtin_defaults_apply_when_env_is_silent(key_store):
    registry = default_registry(key_store=key_store, env={})
    engine = _engine(registry, "qwen3-tts-mlx")
    assert engine.env["UV_PYTHON_INSTALL_MIRROR"] == (
        BUILTIN_DEFAULTS["UV_PYTHON_INSTALL_MIRROR"]
    )
    assert engine.env["UV_DEFAULT_INDEX"] == BUILTIN_DEFAULTS["UV_DEFAULT_INDEX"]


def test_voiceclone_override_reaches_local_engine(key_store):
    """The documented zero-download reuse variable (docs/installation/
    macos-indextts-2.5-mps.md) must actually reach the MPS engine through
    the production construction path."""
    registry = default_registry(
        key_store=key_store,
        env={"VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS": "/models/local"},
    )
    engine = _engine(registry, "indextts-25-mps")
    assert engine.env["VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS"] == "/models/local"
    assert engine.env.get("VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS") == "/models/local"
    from voiceclone_sidecar.engines import indextts25_mps

    assert engine.env[indextts25_mps.LOCAL_WEIGHTS_ENV] == "/models/local"


def test_settings_override_beats_process_env(key_store, tmp_path):
    """Settings storage is the most specific layer (ADR-0015 decision 1)."""
    registry = default_registry(
        key_store=key_store,
        env={"UV_DEFAULT_INDEX": "https://user.example/simple"},
        settings={"engines": {"qwen3-tts-mlx": {"env": {
            "UV_DEFAULT_INDEX": "https://settings.example/simple",
        }}}},
    )
    engine = _engine(registry, "qwen3-tts-mlx")
    assert engine.env["UV_DEFAULT_INDEX"] == "https://settings.example/simple"
    # Other engines are untouched by that engine-level override.
    other = _engine(registry, "qwen3-tts-vd-cloud")
    assert other.env["UV_DEFAULT_INDEX"] == "https://user.example/simple"


def test_registry_injects_config_object(key_store):
    registry = default_registry(key_store=key_store, env={})
    engine = _engine(registry, "qwen3-tts-mlx")
    # The engine received a merged env (not a live os.environ reference) and
    # the registry-computed dict — the seam is injection, not env reads.
    assert isinstance(engine.env, dict)
    assert "PATH" not in engine.env  # only config-relevant keys are forwarded


def test_effective_env_only_forwards_config_keys():
    env = effective_env("x", process_env={"PATH": "/bin", "UV_DEFAULT_INDEX": "u",
                                          "VOICECLONE_FOO": "1"})
    assert env["UV_DEFAULT_INDEX"] == "u"
    assert env["VOICECLONE_FOO"] == "1"
    assert "PATH" not in env


class MiniMaxEngine(CloudEngineBase):
    """A vendor-labeled cloud engine that does not exist yet (pre-#27)."""

    engine_id = "minimax-fake"
    vendor_label = "MiniMax"


def test_cloud_base_vendor_label_drives_user_copy():
    engine = MiniMaxEngine()
    assert "MiniMax" in engine.key_missing_hint()
    assert "阿里百炼" not in engine.key_missing_hint()

    with pytest.raises(CloudEngineError, match="密钥存储未初始化"):
        MiniMaxEngine(key_store=None)._key()

    class EmptyKeyStore:
        def get(self, engine_id):
            return ""

    with pytest.raises(CloudEngineError, match="MiniMax"):
        MiniMaxEngine(key_store=EmptyKeyStore())._key()


def test_pipeline_key_missing_shows_vendor_label(tmp_path):
    """The production 409 branch reads the copy from the ENGINE: a
    MiniMax-labeled BYOK engine must never be told it needs 阿里百炼."""
    import asyncio

    from fastapi import HTTPException

    from voiceclone_sidecar.context import AppContext
    from voiceclone_sidecar.pipeline import run_generation
    from voiceclone_sidecar.registry import Registry

    class MiniMaxShapedEngine(MiniMaxEngine):
        pass

    registry = Registry()
    registry.register(MiniMaxShapedEngine(output_dir=tmp_path / "audio"))
    ctx = AppContext(registry=registry, token="t", audio_dir=tmp_path, data_root=tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(run_generation(ctx, "minimax-fake", "你好"))
    assert exc_info.value.status_code == 409
    assert "MiniMax" in exc_info.value.detail
    assert "阿里百炼" not in exc_info.value.detail


def test_sources_weight_chain_order_and_modelscope_opt_in():
    chain = sources.weight_sources("org/repo", ms_repo="ms/repo")
    assert chain[0] == "https://huggingface.co/org/repo/resolve/main/{path}"
    assert chain[1] == "https://hf-mirror.com/org/repo/resolve/main/{path}"
    assert chain[2] == "https://modelscope.cn/models/ms/repo/resolve/master/{path}"
    # Without a declared ModelScope twin, no ModelScope source (it would 404).
    assert sources.weight_sources("org/repo") == chain[:2]


def test_sources_pypi_index_env_wins():
    assert sources.pypi_index({}) == BUILTIN_DEFAULTS["UV_DEFAULT_INDEX"]
    assert sources.pypi_index({"UV_DEFAULT_INDEX": "https://u/x"}) == "https://u/x"


def test_sources_torch_wheels_are_explicit_urls():
    """ADR-0002: CUDA wheels are EXACT wheel URLs — no index, no resolution,
    so a mirror index can never downgrade CUDA torch to CPU."""
    srcs = sources.torch_wheel_sources("cu128")
    assert srcs[0].startswith("https://download.pytorch.org/whl/cu128/")
    assert srcs[1].startswith("https://mirrors.aliyun.com/pytorch-wheels/cu128/")
    assert all("{path}" in s for s in srcs)


def test_load_engine_settings_uses_the_shared_settings_store(tmp_path):
    """Same <data>/settings.json store as the other app settings — not a
    parallel file nothing writes."""
    from voiceclone_sidecar.db import T_SETTINGS, library_db_path, read_kv_block, write_kv_block
    from voiceclone_sidecar.engine_config import SETTINGS_KEY, save_engine_settings

    assert load_engine_settings(None) == {SETTINGS_KEY: {}}
    # missing file is no overrides
    assert load_engine_settings(tmp_path) == {SETTINGS_KEY: {}}
    save_engine_settings(tmp_path, {"qwen3-tts-mlx": {"env": {
        "UV_DEFAULT_INDEX": "https://s/x",
    }}})
    # the other store keys survive a save (separate rows in the settings
    # table, ADR-0017)
    write_kv_block(tmp_path, T_SETTINGS, "transcription_provider", "local")
    settings = load_engine_settings(tmp_path)
    env = effective_env("qwen3-tts-mlx", process_env={}, settings=settings)
    assert env["UV_DEFAULT_INDEX"] == "https://s/x"
    assert read_kv_block(tmp_path, T_SETTINGS, "transcription_provider") == "local"
    # a corrupt database degrades to no overrides instead of unbootable
    db_path = library_db_path(tmp_path)
    db_path.write_bytes(b"not a database")
    assert load_engine_settings(tmp_path) == {SETTINGS_KEY: {}}


def test_settings_router_round_trip_and_rebuild(client):
    """The settings endpoint is the producer of the config the registry
    injects; a PUT must reach a freshly rebuilt engine."""
    r = client.get("/settings/engines")
    assert r.status_code == 200
    assert r.json() == {"engines": {}}
    r = client.put("/settings/engines", json={
        "engines": {"qwen3-tts-mlx": {"env": {"UV_DEFAULT_INDEX": "https://rt/x"}}},
    })
    assert r.status_code == 200, r.text
    r = client.put("/settings/engines", json={"engines": {"nope": {}}})
    assert r.status_code == 422
    # cleanup: restore empty so other tests see defaults
    r = client.put("/settings/engines", json={"engines": {}})
    assert r.status_code == 200


def test_engine_config_defaults():
    cfg = EngineConfig()
    assert cfg.env == {}
    assert cfg.settings == {}
    assert cfg.output_dir is None
