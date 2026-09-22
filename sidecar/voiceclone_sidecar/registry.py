"""Engine protocol and registry.

The registry is the ONLY extension point: adding an engine must not require
touching any generation logic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from .capabilities import Capabilities, ParamSpec


@dataclass(frozen=True)
class GenerationRequest:
    generation_id: str
    text: str
    params: dict


@dataclass(frozen=True)
class GenerationResult:
    """Path is the engine-written audio file; the sidecar serves it.

    ``model_version`` and ``cost`` are lineage facts the engine itself knows
    best; engines that cannot report them leave the defaults and the record
    stores null (the contract keeps the keys present either way).
    """

    audio_path: str
    sample_rate: int
    model_version: str | None = None
    cost: float | None = None


class Engine(abc.ABC):
    """A synthesizing backend: a cloud API or a local model."""

    engine_id: str
    display_name: str
    # BYOK engines need a per-user API key from the OS key store (ADR-0003);
    # the sidecar refuses generation while the key is not configured.
    requires_key: bool = False
    # Billing and data-usage disclosure shown verbatim in the settings page.
    # The billing note must state the practical Chinese unit price (issue #9).
    billing_note: str | None = None
    data_usage_note: str | None = None

    def key_missing_hint(self) -> str:
        """User-facing message for the pipeline's key-missing (409) path.

        Vendor copy lives on the engine, never in the pipeline: a MiniMax
        engine on this path must show MiniMax text (ADR-0015 decision 4).
        BYOK cloud engines override this with their vendor label.
        """
        return f"引擎 {self.engine_id} 需要配置 API Key；请在「设置」页配置后再生成"

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    def param_specs(self) -> list[ParamSpec]:
        """The full, closed list of user-facing parameters this engine
        accepts. The UI renders exactly this list — nothing else."""
        return []

    @abc.abstractmethod
    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        """Run one text -> audio generation. Use ``log(msg)`` to emit
        progress lines; they are streamed to the UI over WebSocket."""

    def full_synthesis_text(self, normalized_text: str, params: dict | None = None) -> str:
        """The COMPLETE text this engine will synthesize for the given
        ALREADY-normalized text (ADR-0008: the pipeline normalizes centrally)
        and generation params — including any engine-adapter text shaping
        such as VoxCPM2's control-instruction prefix (US19, issue #54).

        The /normalize preview calls this when the UI supplies engine
        context, so the preview shows exactly what the engine will hear.
        Engines whose adapters only map wire parameters (default) return the
        text unchanged; engines that reshape the text override this with the
        SAME helper their ``synthesize`` path runs, so preview and synthesis
        can never drift apart."""
        return normalized_text

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        """Prepare the reference on this engine and return binding extras.

        Local engines hand the file over at generation time and don't need
        this; cloud engines enroll the sample and return e.g. a voice_id.
        Failures surface the vendor's error message to the user.
        """
        raise NotImplementedError(
            f"engine {self.engine_id} does not support explicit reference binding"
        )

    def check_voice(self, voice_id: str, log) -> bool:
        """Probe whether a previously minted binding voice_id still exists.

        Cloud vendors silently recycle enrolled voices (some after 7 days
        idle, some after a year); issue #12 asks the engine BEFORE every
        generation so a dead binding is rebuilt from the local reference
        instead of failing mid-run. Engines that mint no vendor-side voice
        (local engines) don't need this; the sidecar never calls it on an
        engine that does not implement it. Returns True when the voice is
        alive, False when it is definitively gone; any other failure (bad
        key, vendor outage) raises.
        """
        raise NotImplementedError(
            f"engine {self.engine_id} does not support cloud voice health checks"
        )

    def design_voice(self, description: str, preview_text: str, log) -> dict:
        """Create a voice from a text description (issue #11) and return
        ``{"voice_id", "sample_audio_path", "transcript"}``.

        ``sample_audio_path`` is an engine-written preview WAV that becomes
        the designed voice's reference sample — the source of truth every
        other engine binds from (ADR-0001). Only engines declaring the
        ``voice_design`` capability need to implement this; the sidecar
        never calls it on an engine that does not declare the capability.
        """
        raise NotImplementedError(
            f"engine {self.engine_id} does not support voice design"
        )


class InstallableEngine(Engine):
    """An engine whose environment/weights must be installed before use.

    Implement this (in addition to Engine) for engines backed by a bundled
    runtime: the sidecar exposes GET /engines/{id}/status and
    POST /engines/{id}/install from exactly this surface.
    """

    @abc.abstractmethod
    def is_installed(self) -> bool: ...

    @abc.abstractmethod
    def install_state(self) -> dict:
        """{"installed": bool, "steps": {step_id: {status, error, ...}}}"""

    @abc.abstractmethod
    def install(self, log, progress=None) -> dict: ...

    def model_dir(self):
        """Directory holding this engine's weights + venv (issue #17).

        Local model management (weight-dir path, disk usage, uninstall)
        operates on exactly this directory; the install state file lives
        inside it, so removing it also resets the install lifecycle.
        """
        from pathlib import Path

        from .runtime import paths

        root = getattr(self, "root", None) or paths.runtime_root()
        return paths.engine_dir(Path(root), self.engine_id)


class Registry:
    """Real engine registry. Engines register here; nothing else knows them."""

    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {}

    def register(self, engine: Engine) -> None:
        if engine.engine_id in self._engines:
            raise ValueError(f"engine already registered: {engine.engine_id}")
        self._engines[engine.engine_id] = engine

    def get(self, engine_id: str) -> Engine | None:
        return self._engines.get(engine_id)

    def list(self) -> list[Engine]:
        return list(self._engines.values())


def default_registry(output_dir=None, key_store=None, env=None, settings=None) -> Registry:
    """Registry with the engines shipped in this package.

    Every real engine is constructed with an injected :class:`EngineConfig`
    (ADR-0015): its ``env`` is merged through :func:`effective_env` —
    built-in China-friendly defaults < the process environment's
    ``VOICECLONE_*`` / ``UV_*`` variables < per-engine settings overrides —
    so the documented ``VOICECLONE_*`` variables and the settings store both
    actually reach the engines. ``env=None`` means ``os.environ``.

    The fake engine is a TEST SEAM, not a product engine (issue #18): it is
    registered only under VOICECLONE_TEST_ENGINES=1 so production engine
    lists and the transcription provider list never offer a test double —
    the double's transcribe() writes a fixed test text into the voice
    library, which real engines then use as ref_text and silently degrade
    clones. Real local engines register only where they can run (platform +
    architecture); cloud-API engines are always registered — they are usable
    the moment the user brings their own API key (ADR-0003).

    Contract tests MUST construct engines through this factory (issue #22):
    the previous "tests pass env= directly while the registry never did" gap
    produced green tests for a path that did not exist in production.
    """
    import os
    import sys

    if key_store is None:
        from .key_store import KeyStore

        key_store = KeyStore()

    from .engine_config import SETTINGS_KEY, EngineConfig, effective_env

    def config_for(engine_cls) -> EngineConfig:
        # The engine class is the source of its own id — no parallel string
        # literal that could silently drift from the class attribute.
        engine_id = engine_cls.engine_id
        return EngineConfig(
            engine_id=engine_id,
            output_dir=output_dir,
            env=effective_env(engine_id, process_env=env, settings=settings),
            settings=(settings or {}).get(SETTINGS_KEY, {}).get(engine_id, {}),
        )

    from .engines.qwen_tts_cloud import Qwen3TtsVcCloudEngine

    registry = Registry()
    registry.register(Qwen3TtsVcCloudEngine(key_store=key_store, config=config_for(Qwen3TtsVcCloudEngine)))

    from .engines.qwen_tts_vd_cloud import Qwen3TtsVdCloudEngine

    registry.register(Qwen3TtsVdCloudEngine(key_store=key_store, config=config_for(Qwen3TtsVdCloudEngine)))

    # Issue #27: the MiniMax cloud engine — cloning + sync/async synthesis
    # + system voices + voice design in ONE adapter (its designed and
    # cloned voices are synthesized by the same speech-2.8 model). Always
    # registered like every BYOK cloud engine; the region (.io/.cn) is an
    # engine-level base-URL setting, not a separate engine.
    from .engines.minimax_cloud import MiniMaxCloudEngine

    registry.register(MiniMaxCloudEngine(key_store=key_store, config=config_for(MiniMaxCloudEngine)))

    if sys.platform == "darwin" and os.uname().machine == "arm64":
        from .engines.qwen3_tts import Qwen3TtsMlxEngine

        registry.register(Qwen3TtsMlxEngine(config=config_for(Qwen3TtsMlxEngine)))

        from .engines.indextts25_mps import IndexTts25MpsEngine

        registry.register(IndexTts25MpsEngine(config=config_for(IndexTts25MpsEngine)))

        # Issue #25 (ADR-0019): FireRedTTS3-Base, macOS MPS only (patched).
        from .engines.fireredtts3_mps import FireRedTts3MpsEngine

        registry.register(FireRedTts3MpsEngine(config=config_for(FireRedTts3MpsEngine)))

        # Issue #25: VoxCPM2, official MPS support.
        from .engines.voxcpm2_mps import VoxCPM2MpsEngine

        registry.register(VoxCPM2MpsEngine(config=config_for(VoxCPM2MpsEngine)))

    if sys.platform == "win32":
        from .engines.indextts25 import IndexTts25CudaEngine

        registry.register(IndexTts25CudaEngine(config=config_for(IndexTts25CudaEngine)))

        # Issue #25: VoxCPM2 on Windows CUDA.
        from .engines.voxcpm2_cuda import VoxCPM2CudaEngine

        registry.register(VoxCPM2CudaEngine(config=config_for(VoxCPM2CudaEngine)))

        # Issue #25: dots.tts on Windows CUDA (pynini-free patched install).
        from .engines.dots_tts_cuda import DotsTtsCudaEngine

        registry.register(DotsTtsCudaEngine(config=config_for(DotsTtsCudaEngine)))

    if os.environ.get("VOICECLONE_TEST_ENGINES") == "1":
        # Contract-test seam only (issue #18): the fake runs the exact same
        # code path as real engines and keeps the contract suite runnable
        # without models/keys — but it must never surface in production
        # lists, and its transcribe() must never reach a user's voice
        # library. Everything below exists only under the test switch.
        from .engines.fake import FakeEngine, FakeInstructEngine, FakeRefTextEngine

        registry.register(FakeEngine(output_dir=output_dir))
        registry.register(FakeRefTextEngine(output_dir=output_dir))
        # US19 seam (issue #54): an engine that reshapes the text, so the
        # /normalize engine-context preview is contract-tested end to end.
        registry.register(FakeInstructEngine(output_dir=output_dir))

        # Contract-test seam: a BYOK cloud-shaped engine for the issue-#9
        # surface (keys, billing disclosure, cloud bindings, voice_id).
        from .engines.fake import FakeKeyEngine

        registry.register(FakeKeyEngine(output_dir=output_dir))

        # Contract-test seams (issue #10): two output-level extremes. The
        # comparison feature must bring both to the same LUFS.
        from .engines.fake import FakeLoudEngine, FakeQuietEngine

        registry.register(FakeLoudEngine(output_dir=output_dir))
        registry.register(FakeQuietEngine(output_dir=output_dir))

        # Contract-test seam (issue #11): design always fails with a plain
        # OSError — the design endpoint must clean up the voice regardless.
        from .engines.fake import FakeFailingDesignEngine

        registry.register(FakeFailingDesignEngine(output_dir=output_dir))

        # Contract-test seams (issue #12): one whose cloud voice is silently
        # recycled after each synthesis (health check must rebuild the
        # binding), and one whose rebuild itself always fails (the binding
        # must be marked unavailable with a clear reason, others untouched).
        from .engines.fake import FakeBrokenRebuildEngine, FakeVanishingVoiceEngine

        registry.register(FakeVanishingVoiceEngine(output_dir=output_dir))
        registry.register(FakeBrokenRebuildEngine(output_dir=output_dir))

        # Contract-test seams (issue #13): a slow engine (segments long enough
        # to observe queue/cancel states) and one the vendor throttles on its
        # first calls (rate-limit retry must absorb it).
        from .engines.fake import (
            FakeBrokenEngine,
            FakeRateLimitedEngine,
            FakeSlowEngine,
        )

        registry.register(FakeSlowEngine(output_dir=output_dir))
        registry.register(FakeRateLimitedEngine(output_dir=output_dir))
        # Test seam (issue #19): always fails with a path-bearing message.
        registry.register(FakeBrokenEngine(output_dir=output_dir))

        # Contract-test seams (local residency): two engines that model a
        # resident local worker (model stays in memory; unload() releases
        # it). The generation path must never let two of these be resident
        # at once - the multi-local blind-compare scenario that locked a
        # machine up.
        from .engines.fake import FakeResidentEngine, FakeResidentEngineTwo

        registry.register(FakeResidentEngine(output_dir=output_dir))
        registry.register(FakeResidentEngineTwo(output_dir=output_dir))
    return registry
